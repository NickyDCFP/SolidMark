import torch

def carlini_distance(
    x_hat: torch.Tensor,
    train_set: torch.Tensor,
    device: str,
    n: int = 50,
    alpha: float = 0.5
) -> float:
    x_hat = x_hat.to(device).view(-1, 32 * 32 * 3)
    train_set = train_set.to(device).view(-1, 32 * 32 * 3)

    distances: torch.Tensor = torch.cdist(x_hat, train_set, p=2)
    neighbors: torch.Tensor = -torch.topk(-distances, n, sorted=True).values.flatten()

    return neighbors[0] / (alpha * neighbors.mean())

if __name__ == '__main__':
    from data import get_dataset, get_metadata

    train_set = torch.tensor(get_dataset('cifar10', './dataset/', get_metadata('cifar10'), True).data).permute(0, 3, 1, 2).to(dtype=torch.float32)

    print(
        carlini_distance(
            train_set[0].unsqueeze(0),
            train_set,
            'cuda'
        )  
    )