import os
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models.resnet import ResNet
from typing import Any

from data import fix_legacy_dict

BASE_RESNET_OUT_FEATURES: int = 1000

def modify_resnet_fc(resnet50: ResNet, num_classes: int) -> ResNet:
    fc_in_features: int = resnet50.fc.in_features
    resnet50.fc = nn.Linear(fc_in_features, num_classes)
    return resnet50
    
def get_resnet_from_pretrained(
    path: str,
    fc_out_size: int = BASE_RESNET_OUT_FEATURES,
    device: str = 'cuda:0'
) -> ResNet:
    resnet50: ResNet = models.resnet50(weights=None)
    resnet50 = modify_resnet_fc(resnet50, fc_out_size)
    print(f"Loading pretrained ResNet from {path}")
    d = fix_legacy_dict(torch.load(path, map_location=device))
    dm = resnet50.state_dict()
    resnet50.load_state_dict(d, strict=False)
    print(
        f"Mismatched keys in ckpt and ResNet: ",
        set(d.keys()) ^ set(dm.keys()),
    )
    print(f"Loaded pretrained ResNet from {path}")
    resnet50.eval()
    return resnet50

def get_resnet(
    resnet_dir: str,
    train_data: Any,
    mask: torch.Tensor,
    save_filename: str,
    num_classes: int,
    device: str,
    epochs: int = 350,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> ResNet:
    resnet_path: str = os.path.join(resnet_dir, save_filename)
    if os.path.exists(resnet_path):
        print(f"Reading ResNet from {resnet_path}")
        return get_resnet_from_pretrained(resnet_path, num_classes, device).to(device)
    print(f"No ResNet found at {resnet_path}, training one.")
    resnet50: ResNet = models.resnet50(weights=None)
    
    resnet50 = modify_resnet_fc(resnet50, num_classes).to(device)
    resnet50.train()

    train_dataloader: DataLoader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    loss: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
    optimizer: Adam = Adam([p for p in resnet50.parameters() if p.requires_grad], lr=lr)

    print(f"Training ResNet...")
    train_loss: float = 0.0
    num_samples_avg: int = 0
    correct_predictions: int = 0
    mask = mask.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        x: torch.Tensor; y: torch.Tensor
        for _, (x, y) in enumerate(train_dataloader):
            x = torch.mul(x, 1 - mask[:x.size(0), :, :, :]).to(device)
            y = y.to(device)
            optimizer.zero_grad()
            out: torch.Tensor = resnet50(x)
            fit: torch.Tensor = loss(out, y)
            fit.backward()
            optimizer.step()
            _, y_pred = torch.max(out, dim=1)
            correct_predictions += (y_pred == y).sum().item()
            train_loss += fit.item()
        num_samples_avg += len(train_dataloader.dataset)
        print(f'Train Loss: {train_loss / num_samples_avg}, Train Accuracy: {correct_predictions * 100 / num_samples_avg}%')
        correct_predictions = 0
        train_loss = 0
        num_samples_avg = 0

    print(f"Saving model to {resnet_path}")
    torch.save(resnet50.state_dict(), resnet_path)
    
    return resnet50

def resnet_accuracy(resnet: ResNet, mask: torch.Tensor, images: torch.Tensor, labels: torch.Tensor, device: str) -> float:
    mask = mask.unsqueeze(0).repeat(images.size(0), 1, 1, 1).to(device)
    images = torch.mul(images.to(device), 1 - mask)
    out: torch.Tensor = resnet(images)
    _, y_pred = torch.max(out, dim=1)
    correct_predictions = (y_pred == labels).sum().item()
    return correct_predictions * 100 / images.size(0)