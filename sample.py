import torch
import torch.distributed as dist
from tqdm import tqdm
import math
import numpy as np
import cv2
import os


def sample_N_images(
    N,
    model,
    diffusion,
    device,
    ddim,
    xT=None,
    sampling_steps=250,
    batch_size=64,
    num_channels=3,
    image_size=32,
    num_classes=None,
    class_cond=False,
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
                model, xT, sampling_steps, {"y": y}, ddim
            )
            samples_list = [torch.zeros_like(gen_images) for _ in range(num_processes)]
            if class_cond:
                labels_list = [torch.zeros_like(y) for _ in range(num_processes)]
                dist.all_gather(labels_list, y, group)
                labels.append(torch.cat(labels_list).detach().cpu().numpy())

            dist.all_gather(samples_list, gen_images, group)
            samples.append(torch.cat(samples_list).detach().cpu().numpy())
            num_samples += len(xT) * num_processes
            pbar.update(1)
    samples = np.concatenate(samples).transpose(0, 2, 3, 1)[:N]
    samples = (127.5 * (samples + 1)).astype(np.uint8)
    return (samples, np.concatenate(labels) if class_cond else None)

def sample_and_save(
    N,
    model,
    diffusion,
    save_dir,
    filename_base,
    device,
    ddim,
    local_rank,
    xT=None,
    sampling_steps=250,
    batch_size=64,
    num_channels=3,
    image_size=32,
    num_classes=None,
    class_cond=False
) -> None:
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
        class_cond
    )
    if local_rank == 0:
        cv2.imwrite(
            os.path.join(
                save_dir,
                f"{filename_base}.png",
            ),
            np.concatenate(sampled_images, axis=1)[:, :, ::-1],
        )