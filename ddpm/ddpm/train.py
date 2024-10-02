import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from typing import Any

from unets import UNetModel
from diffusion import GaussianDiffusion
from logger import loss_logger

def train_one_epoch(
    model: UNetModel,
    dataloader: DataLoader,
    diffusion: GaussianDiffusion,
    optimizer: Optimizer,
    logger: loss_logger,
    lrs: LRScheduler,
    class_cond: bool,
    ema_w: float,
    local_rank: int,
    ema_dict: dict[str, Any],
    device: str
):
    model.train()
    images: torch.Tensor; labels: torch.Tensor
    for step, (images, labels) in enumerate(dataloader):
        assert (images.max().item() <= 1) and (0 <= images.min().item())

        # must use [-1, 1] pixel range for images
        images, labels = (
            2 * images.to(device) - 1,
            labels.to(device) if class_cond else None,
        )
        t: torch.Tensor = torch.randint(diffusion.timesteps, (len(images),), dtype=torch.int64).to(
            device
        )
        xt: torch.Tensor; eps: torch.Tensor
        xt, eps = diffusion.sample_from_forward_process(images, t)
        pred_eps: torch.Tensor = model(xt, t, y=labels)

        loss: torch.Tensor = ((pred_eps - eps) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if lrs is not None:
            lrs.step()

        # update ema_dict
        if local_rank == 0:
            new_dict: dict[str, Any] = model.state_dict()
            for (k, _) in ema_dict.items():
                ema_dict[k] = (
                    ema_w * ema_dict[k] + (1 - ema_w) * new_dict[k]
                )
            logger.log(loss.item(), display=not step % 100)