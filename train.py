import torch

def train_one_epoch(
    model,
    dataloader,
    diffusion,
    optimizer,
    logger,
    lrs,
    class_cond,
    ema_w,
    local_rank,
    ema_dict,
    device
):
    model.train()
    for step, (images, labels) in enumerate(dataloader):
        assert (images.max().item() <= 1) and (0 <= images.min().item())

        # must use [-1, 1] pixel range for images
        images, labels = (
            2 * images.to(device) - 1,
            labels.to(device) if class_cond else None,
        )
        t = torch.randint(diffusion.timesteps, (len(images),), dtype=torch.int64).to(
            device
        )
        xt, eps = diffusion.sample_from_forward_process(images, t)
        pred_eps = model(xt, t, y=labels)

        loss = ((pred_eps - eps) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if lrs is not None:
            lrs.step()

        # update ema_dict
        if local_rank == 0:
            new_dict = model.state_dict()
            for (k, _) in ema_dict.items():
                ema_dict[k] = (
                    ema_w * ema_dict[k] + (1 - ema_w) * new_dict[k]
                )
            logger.log(loss.item(), display=not step % 100)