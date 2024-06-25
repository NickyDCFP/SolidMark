import torch
import torch.distributed as dist
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import os

from diffusion import GaussianDiffusion
from unets import UNetModel
from typing import Any

def sample_N_images(
    N: int,
    model: UNetModel,
    diffusion: GaussianDiffusion,
    device: str,
    ddim: bool,
    xT: torch.Tensor = None,
    sampling_steps: int = 250,
    batch_size: int = 64,
    num_channels: int = 3,
    image_size: int = 32,
    num_classes: int = 0,
    class_cond: bool = False,
    label_perturb: float = 0,
):
    """use this function to sample any number of images from a given
        diffusion model and diffusion process.

    Args:
        N : Number of images
        model : Diffusion model
        diffusion : Diffusion process
        xT : Starting instantiation of noise vector.
        sampling_steps : Number of sampling steps.
        batch_size : Batch-size for sampling.
        num_channels : Number of channels in the image.
        image_size : Image size (assuming square images).
        num_classes : Number of classes in the dataset (needed for class-conditioned models)
        args : All args from the argparser.

    Returns: Numpy array with N images and corresponding labels.
    """
    samples, labels, num_samples = [], [], 0
    num_processes, group = dist.get_world_size(), dist.group.WORLD
    with tqdm(total=math.ceil(N / (batch_size * num_processes))) as pbar:
        while num_samples < N:
            if xT is None:
                xT = (
                    torch.randn(batch_size, num_channels, image_size, image_size)
                    .float()
                    .to(device)
                )
            if class_cond:
                y = torch.randint(num_classes, (len(xT),), dtype=torch.int64).to(
                    device
                )
            else:
                y = None
            gen_images = diffusion.sample_from_reverse_process(
                model, xT, sampling_steps, {"y": y, "label_perturb" : label_perturb}, ddim
            )
            samples_list = [torch.zeros_like(gen_images) for _ in range(num_processes)]
            if class_cond:
                labels_list = [torch.zeros_like(y) for _ in range(num_processes)]
                dist.all_gather(labels_list, y, group)
                labels.append(torch.cat(labels_list))

            dist.all_gather(samples_list, gen_images, group)
            samples.append(torch.cat(samples_list))
            num_samples += len(xT) * num_processes
            pbar.update(1)
    samples = (torch.cat(samples)[:N] + 1) / 2
    return (samples, torch.cat(labels) if class_cond else None)

def sample_with_inpainting(
    N: int,
    model: UNetModel,
    diffusion: GaussianDiffusion,
    device: str,
    ddim: bool,
    mask: torch.Tensor,
    reference_set: Any,
    xT: torch.Tensor = None,
    sampling_steps: int = 250,
    batch_size: int = 64,
    num_channels: int = 3,
    image_size: int = 32,
    class_cond: bool = False,
    label_perturb: float = 0,
):
    samples, references, labels, num_samples = [], [], [], 0
    num_processes, group = dist.get_world_size(), dist.group.WORLD
    reference_loader = DataLoader(reference_set, batch_size, shuffle=True)
    mask = mask.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device)
    with tqdm(total=math.ceil(N / (batch_size * num_processes))) as pbar:
        for (x, y) in reference_loader:
            x = 2 * x.to(device) - 1
            y = y.to(device)
            if num_samples >= N:
                break
            if xT is None:
                xT = (
                    torch.randn(batch_size, num_channels, image_size, image_size)
                    .float()
                    .to(device)
                )
            if not class_cond:
                y = None
            gen_images = diffusion.inpaint_with_reverse_process(
                model, xT, mask, x, sampling_steps, {"y": y, "label_perturb" : label_perturb}, ddim
            )
            samples_list = [torch.zeros_like(gen_images) for _ in range(num_processes)]
            references_list = [torch.zeros_like(x) for _ in range(num_processes)]
            if class_cond:
                labels_list = [torch.zeros_like(y) for _ in range(num_processes)]
                dist.all_gather(labels_list, y, group)
                labels.append(torch.cat(labels_list))

            dist.all_gather(samples_list, gen_images, group)
            dist.all_gather(references_list, x, group)
            samples.append(torch.cat(samples_list))
            references.append(torch.cat(references_list))
            num_samples += len(xT) * num_processes
            pbar.update(1)
    samples = (torch.cat(samples)[:N] + 1) / 2
    references = (torch.cat(references)[:N] + 1) / 2
    return (samples, references, torch.cat(labels) if class_cond else None)


def sample_and_save(
    N: int,
    model: UNetModel,
    diffusion: GaussianDiffusion,
    save_dir: str,
    filename_base: str,
    device: str,
    ddim: bool,
    local_rank: int,
    inpaint: bool = False,
    mask: torch.Tensor = None,
    reference_set: Any = None,
    xT: torch.Tensor = None,
    sampling_steps: int = 250,
    batch_size: int = 64,
    num_channels: int = 3,
    image_size: int = 32,
    num_classes: int = None,
    class_cond: bool = False,
    label_perturb: float = 0,
) -> None:
    if inpaint:
        if reference_set is None:
            raise ValueError("Need Reference set for inpainting")
        if mask is None:
            raise ValueError("Need mask for inpainting")
        sampled_images, references, _ = sample_with_inpainting(
            N,
            model,
            diffusion,
            device,
            ddim,
            mask,
            reference_set,
            xT,
            sampling_steps,
            batch_size,
            num_channels,
            image_size,
            class_cond,
            label_perturb
        )
        if local_rank == 0:
            torchvision.utils.save_image(
                combine_in_rows(sampled_images, references),
                os.path.join(
                    save_dir,
                    f"{filename_base}.png",
                )
            )

    else:
        sampled_images, _ = sample_N_images(
            N,
            model,
            diffusion,
            device,
            ddim,
            xT,
            sampling_steps,
            batch_size,
            num_channels,
            image_size,
            num_classes,
            class_cond,
            label_perturb
        )
        if local_rank == 0:
            torchvision.utils.save_image(
                sampled_images,
                os.path.join(
                    save_dir,
                    f"{filename_base}.png",
                )
            )

def combine_in_rows(
    top_row: torch.Tensor,
    bottom_row: torch.Tensor,
) -> torch.Tensor:
    top_row = top_row.permute(1, 2, 0, 3).reshape((3, 32, -1))
    bottom_row = bottom_row.permute(1, 2, 0, 3).reshape((3, 32, -1))
    return torch.cat((top_row, bottom_row), dim=1)