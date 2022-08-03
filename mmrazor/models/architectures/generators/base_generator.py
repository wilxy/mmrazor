import torch
from mmcv.runner import BaseModule

from gml.models.utils.utils import get_module_device


class BaseGenerator(BaseModule):
    """The base class for generating images.

    Args:
        img_size (int): the size of generated image.
        latent_dim (int): the dimension of latent vector.
        hidden_channels (int): the dimension of hidden channels.
    """

    def __init__(self, img_size, latent_dim, hidden_channels):
        super().__init__()
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels

    def process_latent(self, z=None, batch_size=0):
        if isinstance(z, torch.Tensor):
            assert z.shape[1] == self.latent_dim
            if z.ndim == 2:
                z_batch = z
            else:
                raise ValueError('The noise should be in shape of (n, c)'
                                 f'but got {z.shape}')
        elif z is None and batch_size > 0:
            z_batch = torch.randn((batch_size, self.latent_dim))

        # putting data on the right device
        z_batch = z_batch.to(get_module_device(self))
        return z_batch

    def forward(self):
        raise NotImplementedError
