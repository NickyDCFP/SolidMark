import torch
import torch.nn as nn

class Solid(nn.Module):
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        image_sum: torch.Tensor = img.sum(dim=(0, 1, 2))
        sine: torch.Tensor = torch.abs(torch.sin(image_sum))
        sine_fill: torch.Tensor = sine.repeat(3, 8, 8)
        img[:, 12:20, 12:20] = sine_fill
        return img