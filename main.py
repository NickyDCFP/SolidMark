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

import unets
from unets import load_pretrained, save_model
from data import get_metadata, get_dataset
from args import parse_args
from diffusion import GaussianDiffusion
from train import train_one_epoch
from sample import sample_and_save
from logger import loss_logger

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

    # sampling
    if args.sampling_only:
        filename_base: str = f"{args.arch}_{args.dataset}-epoch_{args.epochs}-sampling_{args.sampling_steps}-class_condn_{args.class_cond}"
        sample_and_save(
            64,
            model,
            diffusion,
            args.save_dir,
            filename_base,
            device,
            args.ddim,
            local_rank,
            None,
            args.sampling_steps,
            args.batch_size,
            metadata.num_channels,
            metadata.image_size,
            metadata.num_classes,
            args.class_cond,
        )
        return

    # Load dataset
    train_set = get_dataset(args.dataset, args.data_dir, metadata, not args.finetune)
    sampler = DistributedSampler(train_set) if ngpus > 1 else None
    train_loader = DataLoader(
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
    logger = loss_logger(len(train_loader) * args.epochs)

    # ema model
    ema_dict: dict[str, Any] = copy.deepcopy(model.state_dict())

    filename_base: str = f"{args.arch}_{args.dataset}-train-epoch_{args.epochs}-sampling_{args.sampling_steps}-class_condn_{args.class_cond}"

    # lets start training the model
    for epoch in range(args.epochs):
        if args.finetune:
            filename_base: str = f"{args.arch}_{args.dataset}-train-epoch_{epoch}-sampling_{args.sampling_steps}-class_condn_{args.class_cond}"
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
                64,
                model,
                diffusion,
                args.save_dir,
                filename_base,
                device,
                args.ddim,
                local_rank,
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
