r"""
Ortiginal from https://github.com/francois-rozet/piqa/blob/d512c86f4af845c4f54f86a2f0ef8866851b8f5c/piqa/fid.py#L53-L88
"""

import torch
import torch.nn as nn
import torchvision

from torch import Tensor
from typing import Optional


def sqrtm(sigma: Tensor) -> Tensor:
    L, Q = torch.linalg.eigh(sigma)
    L = L.clamp(min=0).sqrt()
    return Q @ (L[..., None] * Q.mT)


def frechet_distance(mu_x: Tensor, sigma_x: Tensor, mu_y: Tensor, sigma_y: Tensor) -> Tensor:
    sigma_y_12 = sqrtm(sigma_y)
    fid = (mu_x - mu_y).square().sum() \
          + sigma_x.trace() + sigma_y.trace() \
          - 2 * sqrtm(sigma_y_12 @ sigma_x @ sigma_y_12).trace()
    return fid.real if torch.is_complex(fid) else fid


def compute_covariance(x: Tensor) -> Tensor:
    x = x - x.mean(dim=0, keepdim=True)
    return x.T @ x / (x.shape[0] - 1)


class ImageNetNorm(nn.Module):
    """Normalize images to match ImageNet stats."""
    def __init__(self):
        super().__init__()
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        self.register_buffer('mean', mean[:, None, None])
        self.register_buffer('std', std[:, None, None])

    def forward(self, x: Tensor) -> Tensor:
        return (x - self.mean) / self.std


class InceptionV3(nn.Sequential):
    def __init__(self, logits: bool = False):
        net = torchvision.models.inception_v3(weights='DEFAULT', transform_input=False)
        layers = [
            net.Conv2d_1a_3x3,
            net.Conv2d_2a_3x3,
            net.Conv2d_2b_3x3,
            net.maxpool1,
            net.Conv2d_3b_1x1,
            net.Conv2d_4a_3x3,
            net.maxpool2,
            net.Mixed_5b, net.Mixed_5c, net.Mixed_5d,
            net.Mixed_6a, net.Mixed_6b, net.Mixed_6c, net.Mixed_6d, net.Mixed_6e,
            net.Mixed_7a, net.Mixed_7b, net.Mixed_7c,
            net.avgpool,
            nn.Flatten(-3),
        ]
        if logits:
            layers.append(net.fc)
        super().__init__(*layers)


class FID(nn.Module):
    def __init__(self, input_range: str = "0_1"):
        super().__init__()
        self.normalize = ImageNetNorm()
        self.inception = InceptionV3(logits=False).eval()
        for p in self.parameters():
            p.requires_grad = False
        self.input_range = input_range

    def _prepare_input(self, x: Tensor) -> Tensor:
        if self.input_range == "minus1_1":
            x = (x + 1) / 2  # Convert [-1, 1] to [0, 1]
        x = x.clamp(0, 1)
        return self.normalize(x)

    def features(self, x: Tensor, batch_size: int = 64) -> Tensor:
        x = self._prepare_input(x)
        features = []
        for i in range(0, x.shape[0], batch_size):
            with torch.no_grad():
                features.append(self.inception(x[i:i+batch_size]))
        return torch.cat(features, dim=0)

    def forward(self, real_feats: Tensor, fake_feats: Tensor) -> Tensor:
        mu_x, sigma_x = real_feats.mean(dim=0), compute_covariance(real_feats)
        mu_y, sigma_y = fake_feats.mean(dim=0), compute_covariance(fake_feats)
        return frechet_distance(mu_x, sigma_x, mu_y, sigma_y)
