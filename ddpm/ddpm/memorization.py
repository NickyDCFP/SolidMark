import torch
from torch.return_types import topk

def patched_carlini_distance(
    x_hat: torch.Tensor,
    train_set: torch.Tensor,
    device: str,
    n: int = 50,
    alpha: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor]:
    print(train_set.max(), train_set.min())
    img_size = x_hat.size()[2]
    patch_size = 8
    n_patches = (img_size // patch_size) ** 2
    patches_x_hat: torch.Tensor = x_hat.unfold(3, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches_x_hat = patches_x_hat.reshape(-1, 3 * n_patches, patch_size ** 2)
    patches_list_x_hat: list[torch.Tensor] = [
        patches_x_hat[:, :, i]
        for i in range(patches_x_hat.size(2))
    ]
    patches_train_set: torch.Tensor = train_set.unfold(3, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches_train_set = patches_train_set.reshape(-1, 3 * n_patches, patch_size ** 2)
    patches_list_train_set: list[torch.Tensor] = [
        patches_train_set[:, :, i]
        for i in range(patches_train_set.size(2))
    ]
    patched_distances: torch.Tensor = torch.zeros((x_hat.size(0), train_set.size(0)), dtype=torch.float32, device='cpu')
    for i in range(patches_x_hat.size(2)):
        patches_list_x_hat[i] = patches_list_x_hat[i].to(device)
        patches_list_train_set[i] = patches_list_train_set[i].to(device)
        patched_distances = torch.stack(
            (
                patched_distances,
                torch.cdist(
                    patches_list_x_hat[i],
                    patches_list_train_set[i],
                    p=2
                ).cpu(),
            ),
            dim=1
        )
        patched_distances = torch.max(patched_distances, dim=1).values
        patches_list_x_hat[i] = patches_list_x_hat[i].cpu()
        patches_list_train_set[i] = patches_list_train_set[i].cpu()
    print(train_set.max(), train_set.min())
    patched_distances = patched_distances.to(device)
    top_k: topk = torch.topk(-patched_distances, n, sorted=True)
    neighbor_dists: torch.Tensor = -top_k.values
    neighbor_indices: torch.Tensor = top_k.indices[:, 0]
    train_set = train_set.to(device)
    return (neighbor_dists[:, 0] / (alpha * neighbor_dists.mean(dim=1)), train_set[neighbor_indices].view(-1, 3, img_size, img_size))


if __name__ == '__main__':
    from data import get_dataset, get_metadata
    from torch.utils.data import DataLoader
    from torchvision.utils import save_image
    border_thickness = 4
    train_set_regular = get_dataset('cifar10', './dataset/', get_metadata('cifar10'), True, 'identity', mask=mask, raw=True)
    train_set_pattern = get_dataset('cifar10', './dataset/', get_metadata('cifar10'), True, pattern, mask=mask, raw=True)
    regular_loader = DataLoader(train_set_regular, batch_size=100)
    pattern_loader = DataLoader(train_set_pattern, batch_size=100)
    regular_ds = torch.cat([batch[0] for batch in regular_loader], dim=0)
    pattern_ds = torch.cat([batch[0] for batch in pattern_loader], dim=0)
    save_image(
        torch.cat(
            [torch.cat((regular_ds[i], pattern_ds[i]), dim=2) for i in (0, 1, 4)],
            dim=1
        ),
        f'./trained_models/{pattern}_pattern_overlaid.png',
    )
