import torch
from torch.return_types import topk

def patched_carlini_distance(
    x_hat: torch.Tensor,
    train_set: torch.Tensor,
    device: str,
    n: int = 50,
    alpha: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor]:
    patches_x_hat: torch.Tensor = x_hat.unfold(3, 4, 4).unfold(2, 4, 4)
    patches_x_hat = patches_x_hat.reshape(-1, 3 * 8 * 8, 4 * 4)
    patches_list_x_hat: list[torch.Tensor] = [
        patches_x_hat[:, :, i]
        for i in range(patches_x_hat.size(2))
    ]
    patches_train_set: torch.Tensor = train_set.unfold(3, 4, 4).unfold(2, 4, 4)
    patches_train_set = patches_train_set.reshape(-1, 3 * 8 * 8, 16)
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
    
    # patched_distance_tensor: torch.Tensor = torch.stack(patched_distances, dim=1)
    # max_patched_dist = torch.max(patched_distance_tensor, dim=1).values.to(device)
    patched_distances = patched_distances.to(device)
    top_k: topk = torch.topk(-patched_distances, n, sorted=True)
    neighbor_dists: torch.Tensor = -top_k.values
    neighbor_indices: torch.Tensor = top_k.indices[:, 0]
    train_set = train_set.to(device)
    return (neighbor_dists[:, 0] / (alpha * neighbor_dists.mean(dim=1)), train_set[neighbor_indices].view(-1, 3, 32, 32))


if __name__ == '__main__':
    from data import get_dataset, get_metadata

    train_set = torch.tensor(get_dataset('cifar10', './dataset/', get_metadata('cifar10'), True).data).permute(0, 3, 1, 2).to(dtype=torch.float32)
    print(patched_carlini_distance(
        train_set[0:3],
        train_set,
        'cuda'
    ).size())