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

import unets
from unets import load_pretrained, save_model
from data import get_metadata, get_dataset
from args import parse_args
from diffusion import GaussianDiffusion
from train import train_one_epoch
from sample import sample_and_save, sample_N_images, combine_in_rows, sample_with_inpainting
from logger import loss_logger
from memorization import patched_carlini_distance

def main():
    # setup
    args: Namespace = parse_args()
    metadata: EasyDict = get_metadata(args.dataset)
    if args.center_pattern:
        if args.dataset == "stl10" or args.dataset == "celeba":
            padding = 16
            metadata.image_size += 2 * padding
    else:
        metadata.image_size += 2 * args.pattern_thickness
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
    if args.center_pattern:
        mask = torch.zeros((metadata.num_channels, metadata.image_size, metadata.image_size))
        center = metadata.image_size // 2
        low = center - args.pattern_thickness // 2
        high = center + args.pattern_thickness // 2
        mask[:, low:high, low:high] += 1
    else:
        mask = torch.ones((metadata.num_channels, metadata.image_size, metadata.image_size))
        pt = args.pattern_thickness
        mask[:, pt:-pt, pt:-pt] -= 1
    train_set: Any = get_dataset(args.dataset, args.data_dir, metadata, args.center_pattern, mask=mask, thickness=args.pattern_thickness)
    # sampling
    pattern = "center" if args.center_pattern else "border"
    if args.sampling_only:
        if args.inpaint:
            filename_base: str = f"inpaint-{args.arch}_{args.dataset}-sampling_{args.sampling_steps}-class_condn_{args.class_cond}"
        else:
            filename_base: str = f"{args.arch}_{args.dataset}-sampling_{args.sampling_steps}-class_condn_{args.class_cond}"
        train_set = get_dataset(args.dataset, args.data_dir, metadata, args.center_pattern, raw=True, mask=mask, thickness=args.pattern_thickness)
        if args.distance:
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
                    args.perturb_labels,
                )
                repeated_mask = mask.repeat((sampled_images.size(0), 1, 1, 1)).cuda()
                sampled_images = torch.mul(sampled_images, repeated_mask).mean(dim=(1, 2, 3)) / repeated_mask.mean(dim=(1, 2, 3))
                references = torch.mul(references, repeated_mask).mean(dim=(1, 2, 3)) / repeated_mask.mean(dim=(1, 2, 3))
                distances: torch.Tensor = torch.abs(sampled_images - references.cuda())
                torch.save(distances.detach().cpu(), os.path.join(args.save_dir, f'distances_{pattern}_{args.dataset}_{args.suffix}.pt'))
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
                    args.perturb_labels,
                )
                if local_rank == 0:
                    # mask = mask.repeat((sampled_images.size(0), 1, 1, 1)).cuda()
                    # sampled_images = torch.mul(1 - mask, sampled_images)
                    # train_data = torch.mul(1 - mask, sampled_images)
                    if args.center_pattern: #just get rid of
                        sampled_images = sampled_images[:, :, padding:-padding, padding:-padding]
                        train_data = train_data[:, :, padding:-padding, padding:-padding]
                    else:
                        sampled_images = sampled_images[:, :, pt:-pt, pt:-pt]
                        train_data = train_data[:, :, pt:-pt, pt:-pt]
                    distances, neighbors = patched_carlini_distance(
                        sampled_images,
                        train_data,
                        device
                    )
                    neighbors = neighbors.detach().cpu()
                    sampled_images = sampled_images.detach().cpu()
                    
                    torch.save(distances.detach().cpu(), os.path.join(args.save_dir, f'distances_l2_{pattern}_{args.dataset}_{args.suffix}.pt'))
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
                            f"range_{args.dataset}_{pattern}_{args.suffix}.png",
                        )
                    )
                    torchvision.utils.save_image(
                        combine_in_rows(nrst_samples, nrst_neighbors),
                        os.path.join(
                            args.save_dir,
                            f"nearest_{args.dataset}_{pattern}_{args.suffix}.png",
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
                args.perturb_labels,
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

    filename_base: str = f"{args.arch}_{args.dataset}-train-epoch_{args.epochs}-sampling_{args.sampling_steps}-class_condn_{args.class_cond}-center_{args.center_pattern}_{args.suffix}"

    # lets start training the model
    for epoch in range(args.epochs):
        if args.finetune:
            filename_base: str = f"{args.arch}_{args.dataset}-finetune-epoch_{epoch}-sampling_{args.sampling_steps}-class_condn_{args.class_cond}-center_{args.center_pattern}_{args.suffix}"
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
                args.perturb_labels,
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
 