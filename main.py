import os
import copy
import numpy as np
from argparse import Namespace
from easydict import EasyDict
from typing import Any

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision
from torchvision.models import ResNet

import unets
from unets import load_pretrained, save_model
from data import get_metadata, get_dataset
from args import parse_args
from diffusion import GaussianDiffusion
from train import train_one_epoch
from sample import sample_and_save, sample_N_images, combine_in_rows, sample_with_inpainting
from logger import loss_logger
from memorization import patched_carlini_distance
from resnet import get_resnet, resnet_accuracy

def main():
    # setup
    args: Namespace = parse_args()
    metadata: EasyDict = get_metadata(args.dataset)
    if 'LOCAL_RANK' in os.environ:
        local_rank: int = int(os.environ['LOCAL_RANK'])
    else:
        print("No Local Rank found, defaulting to 0.")
        local_rank: int = 0

    torch.backends.cudnn.benchmark = True
    device = "cuda:{}".format(local_rank)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed + local_rank)
    np.random.seed(args.seed + local_rank)

    # Creat model and diffusion process
    model = unets.__dict__[args.arch](
        image_size=metadata.image_size,
        in_channels=metadata.num_channels,
        out_channels=metadata.num_channels,
        num_classes=metadata.num_classes if args.class_cond else None,
    ).to(device)
    if local_rank == 0:
        print(
            "We are assuming that model input/ouput pixel range is [-1, 1]. Please adhere to it."
        )
    diffusion = GaussianDiffusion(args.diffusion_steps, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # load pre-trained model
    if args.pretrained_ckpt:
        load_pretrained(args.pretrained_ckpt, model, device, args.delete_keys)

    # distributed training
    ngpus = torch.cuda.device_count()
    if ngpus > 1:
        if local_rank == 0:
            print(f"Using distributed training on {ngpus} gpus.")
        args.batch_size = args.batch_size // ngpus
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Load dataset
    mask = torch.zeros((3, 32, 32))
    mask[:, 12:20, 12:20] += 1
    train_set: Any = get_dataset(args.dataset, args.data_dir, metadata, not args.finetune, args.pattern, mask=mask)
    if args.resnet and local_rank == 0:
        resnet: ResNet = get_resnet(args.save_dir, train_set, mask, "resnet-base.pt", metadata.num_classes, device)

    # sampling
    if args.sampling_only:
        if args.inpaint:
            filename_base: str = f"inpaint-{args.arch}_{args.dataset}-sampling_{args.sampling_steps}-class_condn_{args.class_cond}"
        else:
            filename_base: str = f"{args.arch}_{args.dataset}-sampling_{args.sampling_steps}-class_condn_{args.class_cond}"
        train_set = get_dataset(args.dataset, args.data_dir, metadata, not args.finetune, args.pattern, raw=True)
        if args.distance:
            if args.resnet: raise ValueError("Resnet and Distance are not compatible.")
            sampled_images: torch.Tensor; distances: torch.Tensor; neighbors: torch.Tensor
            if args.inpaint:
                sampled_images, references, _ = sample_with_inpainting(
                    args.num_sampled_images,
                    model,
                    diffusion,
                    device,
                    args.ddim,
                    mask,
                    train_set,
                    None,
                    args.sampling_steps,
                    args.batch_size,
                    metadata.num_channels,
                    metadata.image_size,
                    args.class_cond,
                )
                sampled_images = sampled_images[:, :, 12:20, 12:20].reshape(
                    sampled_images.size(0),
                    sampled_images.size(1),
                    -1
                ).mean(dim=(1, 2))
                references = references[:, :, 12:20, 12:20].reshape(
                    references.size(0),
                    references.size(1),
                    -1
                ).mean(dim=(1, 2))
                distances: torch.Tensor = torch.abs(sampled_images - references)
                torch.save(distances.detach().cpu(), os.path.join(args.save_dir, f'distances_{args.pattern}_absolute_{args.suffix}.pt'))
                torch.save(distances.detach().cpu() ** 2, os.path.join(args.save_dir, f'distances_{args.pattern}_square_{args.suffix}.pt'))
            else:
                dataloader = DataLoader(train_set, batch_size=1000)
                train_data = torch.cat([batch[0] for batch in dataloader], dim=0)
                sampled_images, _ = sample_N_images(
                    args.num_sampled_images,
                    model,
                    diffusion,
                    device,
                    args.ddim,
                    None,
                    args.sampling_steps,
                    args.batch_size,
                    metadata.num_channels,
                    metadata.image_size,
                    metadata.num_classes,
                    args.class_cond,
                )
                if local_rank == 0:
                    distances, neighbors = patched_carlini_distance(
                        sampled_images * 255,
                        train_data,
                        device
                    )
                    neighbors = neighbors.detach().cpu() / 255
                    sampled_images = sampled_images.detach().cpu()
                    
                    torch.save(distances.detach().cpu(), os.path.join(args.save_dir, 'distances.pt'))
                    increasing_indices = torch.argsort(distances).cpu()
                    image_indices = torch.linspace(0, increasing_indices.size(0), 10, dtype=torch.int)
                    image_indices[image_indices.size(0) - 1] -= 1
                    image_indices = increasing_indices[image_indices]
                    nearest_indices = increasing_indices[torch.arange(0, 50, dtype=torch.int)]
                    nrst_samples = sampled_images[nearest_indices]
                    nrst_neighbors = neighbors[nearest_indices]
                    samples = sampled_images[image_indices]
                    neighbors = neighbors[image_indices]

                    torchvision.utils.save_image(
                        combine_in_rows(samples, neighbors),
                        os.path.join(
                            args.save_dir,
                            f"range.png",
                        )
                    )
                    torchvision.utils.save_image(
                        combine_in_rows(nrst_samples, nrst_neighbors),
                        os.path.join(
                            args.save_dir,
                            f"nearest.png",
                        )
                    )

        else:
            if args.resnet:
                samples, labels = sample_N_images(
                    args.num_sampled_images,
                    model,
                    diffusion,
                    device,
                    args.ddim,
                    None,
                    args.sampling_steps,
                    args.batch_size,
                    metadata.num_channels,
                    metadata.image_size,
                    metadata.num_classes,
                    args.class_cond
                )
                if local_rank == 0:
                    accuracy: torch.Tensor = resnet_accuracy(resnet, mask, samples, labels, device)
                    torch.save(accuracy, os.path.join(args.save_dir, f"accuracy_{args.suffix}.pt"))
                    if args.num_sampled_images <= 100:
                        torchvision.utils.save_image(
                            samples,
                            os.path.join(
                                args.save_dir,
                                f"resnet_images_acc_{accuracy:.2f}.png"
                            )
                        )
            else:
                sample_and_save(
                    args.num_sampled_images,
                    model,
                    diffusion,
                    args.save_dir,
                    filename_base,
                    device,
                    args.ddim,
                    local_rank,
                    args.inpaint,
                    mask,
                    train_set,
                    None,
                    args.sampling_steps,
                    args.batch_size,
                    metadata.num_channels,
                    metadata.image_size,
                    metadata.num_classes,
                    args.class_cond,
                )
        return


    sampler: DistributedSampler = DistributedSampler(train_set) if ngpus > 1 else None
    train_loader: DataLoader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )
    if local_rank == 0:
        print(
            f"Training dataset loaded: Number of batches: {len(train_loader)}, Number of images: {len(train_set)}"
        )
    logger: loss_logger = loss_logger(len(train_loader) * args.epochs)

    # ema model
    ema_dict: dict[str, Any] = copy.deepcopy(model.state_dict())

    filename_base: str = f"{args.arch}_{args.dataset}-train-epoch_{args.epochs}-sampling_{args.sampling_steps}-class_condn_{args.class_cond}-pattern_{args.pattern}_{args.suffix}"

    # lets start training the model
    for epoch in range(args.epochs):
        if args.finetune:
            filename_base: str = f"{args.arch}_{args.dataset}-finetune-epoch_{epoch}-sampling_{args.sampling_steps}-class_condn_{args.class_cond}-pattern_{args.pattern}_{args.suffix}"
        if sampler is not None:
            sampler.set_epoch(epoch)
        train_one_epoch(
            model,
            train_loader,
            diffusion,
            optimizer,
            logger,
            None,
            args.class_cond,
            args.ema_w,
            local_rank,
            ema_dict,
            device
        )
        if not epoch % args.save_freq:
            sample_and_save(
                args.num_sampled_images,
                model,
                diffusion,
                args.save_dir,
                filename_base,
                device,
                args.ddim,
                local_rank,
                args.inpaint,
                None,
                None,
                None,
                args.sampling_steps,
                args.batch_size,
                metadata.num_channels,
                metadata.image_size,
                metadata.num_classes,
                args.class_cond,
            )
        if local_rank == 0:
            save_model(
                model,
                ema_dict,
                args.ema_w,
                args.save_dir,
                filename_base,
            )


if __name__ == "__main__":
    main()
 