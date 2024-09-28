import torch
import torch.nn as nn
from torchvision import transforms

class SolidMark(nn.Module):
    def __init__(self, thickness: int = 4):
        super().__init__()
        self.thickness = thickness

    def forward(self, img: torch.Tensor, key: int) -> torch.Tensor:
        new_shape = (img.shape[0], img.shape[1] + 2 * self.thickness, img.shape[2] + 2 * self.thickness)
        augmented = torch.zeros(new_shape) + key
        augmented[:, self.thickness:-self.thickness, self.thickness:-self.thickness] = img
        return augmented

class CenterMark(nn.Module):
    def __init__(self, mask: torch.Tensor):
        super().__init__()
        self.mask = mask

    def forward(self, img: torch.Tensor, key: int) -> torch.Tensor:
        augmented = torch.mul(self.mask, torch.zeros_like(img) + key) + torch.mul(1 - self.mask, img)
        return augmented