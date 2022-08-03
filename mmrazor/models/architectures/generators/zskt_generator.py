import torch.nn as nn

from ..builder import GENERATORS
from .base_generator import BaseGenerator


class View(nn.Module):

    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


@GENERATORS.register_module()
class ZSKTGenerator(BaseGenerator):
    """Generator for cifar10. code link:
    https://github.com/polo5/ZeroShotKnowledgeTransfer/

    Args:
        img_size (int): the size of generated image.
        latent_dim (int): the dimension of latent vector.
        hidden_channels (int): the dimension of hidden channels.
        scale_factor (int, optional): the scale factor for F.interpolate.
                                        Defaults to 2.
        leaky_slope (float, optional): the slope param in leaky relu.
                                        Defaults to 0.2.
    """

    def __init__(self,
                 img_size,
                 latent_dim,
                 hidden_channels,
                 scale_factor=2,
                 leaky_slope=0.2):
        super().__init__(img_size, latent_dim, hidden_channels)
        self.init_size = self.img_size // (scale_factor**2)
        self.scale_factor = scale_factor
        self.layers = nn.Sequential(
            nn.Linear(self.latent_dim,
                      self.hidden_channels * self.init_size**2),
            View((-1, self.hidden_channels, self.init_size, self.init_size)),
            nn.BatchNorm2d(self.hidden_channels),
            nn.Upsample(scale_factor=scale_factor),
            nn.Conv2d(
                self.hidden_channels,
                self.hidden_channels,
                3,
                stride=1,
                padding=1), nn.BatchNorm2d(self.hidden_channels),
            nn.LeakyReLU(leaky_slope, inplace=True),
            nn.Upsample(scale_factor=scale_factor),
            nn.Conv2d(
                self.hidden_channels,
                self.hidden_channels // 2,
                3,
                stride=1,
                padding=1), nn.BatchNorm2d(self.hidden_channels // 2),
            nn.LeakyReLU(leaky_slope, inplace=True),
            nn.Conv2d(self.hidden_channels // 2, 3, 3, stride=1, padding=1),
            nn.BatchNorm2d(3, affine=True))

    def forward(self, z=None, batch_size=None):
        z_batch = self.process_latent(z, batch_size)
        return self.layers(z_batch)
