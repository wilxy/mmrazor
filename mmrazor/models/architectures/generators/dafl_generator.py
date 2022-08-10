# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional

import torch.nn as nn
import torch.nn.functional as F

from mmrazor.registry import MODELS
from .base_generator import BaseGenerator


@MODELS.register_module()
class DAFLGenerator(BaseGenerator):
    """Generator for CIFAR-10/100 dataset, used in DAFL, DFAD.

    Args:
        img_size (int): the size of generated image.
        latent_dim (int): the dimension of latent vector.
        hidden_channels (int): the dimension of hidden channels.
        scale_factor (int, optional): the scale factor for F.interpolate.
                                        Defaults to 2.
        bn_eps (float, optional): the eps param in bn. Defaults to 0.8.
        leaky_slope (float, optional): the slope param in leaky relu.
                                        Defaults to 0.2.
    """

    def __init__(
        self,
        img_size,
        latent_dim,
        hidden_channels,
        scale_factor=2,
        bn_eps=0.8,
        leaky_slope=0.2,
        init_cfg: Optional[Dict] = None,
    ):
        super().__init__(img_size, latent_dim, hidden_channels, init_cfg=init_cfg)
        self.init_size = self.img_size // (scale_factor**2)
        self.scale_factor = scale_factor
        self.linear = nn.Linear(self.latent_dim,
                                self.hidden_channels * self.init_size**2)

        self.bn1 = nn.BatchNorm2d(self.hidden_channels)
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(
                self.hidden_channels,
                self.hidden_channels,
                3,
                stride=1,
                padding=1),
            nn.BatchNorm2d(self.hidden_channels, eps=bn_eps),
            nn.LeakyReLU(leaky_slope, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(
                self.hidden_channels,
                self.hidden_channels // 2,
                3,
                stride=1,
                padding=1),
            nn.BatchNorm2d(self.hidden_channels // 2, eps=bn_eps),
            nn.LeakyReLU(leaky_slope, inplace=True),
            nn.Conv2d(self.hidden_channels // 2, 3, 3, stride=1, padding=1),
            nn.Tanh(), nn.BatchNorm2d(3, affine=False))

    def forward(self, z=None, batch_size=None):
        z_batch = self.process_latent(z, batch_size)
        out = self.linear(z_batch)
        out = out.view(out.shape[0], self.hidden_channels, self.init_size,
                       self.init_size)
        img = self.bn1(out)
        img = F.interpolate(img, scale_factor=self.scale_factor)
        img = self.conv_blocks1(img)
        img = F.interpolate(img, scale_factor=self.scale_factor)
        img = self.conv_blocks2(img)
        return img
