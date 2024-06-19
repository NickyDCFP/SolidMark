import torch
import torch.nn as nn

class Solid(nn.Module):
    def __init__(self, mask: torch.Tensor):
        super().__init__()
        self.mask = mask

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        image_sum: torch.Tensor = img.sum(dim=(0, 1, 2))
        sine: torch.Tensor = torch.abs(torch.sin(image_sum))
        sine_fill: torch.Tensor = sine.repeat(img.size())
        img = torch.mul(self.mask, sine_fill) + torch.mul(1 - self.mask, img)
        return img

class Harmonic(nn.Module):
    def __init__(self, mask: torch.Tensor):
        super().__init__()
        self.mask = mask

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        image_sum: torch.Tensor = img.sum(dim=(0, 1, 2))
        sine: torch.Tensor = torch.sin(image_sum)
        dim: torch.Tensor = torch.arange(img.size(1)).unsqueeze(0).unsqueeze(2).repeat((3, 1, img.size(2))) / 2
        harmonic = torch.abs(torch.sin(sine * (dim + dim.permute(0, 2, 1))))
        img = torch.mul(self.mask, harmonic) + torch.mul(1 - self.mask, img)
        return img