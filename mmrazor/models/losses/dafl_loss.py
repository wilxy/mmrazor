import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import get_dist_info

from mmrazor.registry import MODELS
from ..ops import GatherTensors


class DAFLLoss(nn.Module):
    """Base class for DAFL losses.

    paper link: https://arxiv.org/pdf/1904.01186.pdf
    """
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, preds_T: torch.Tensor):
        return self.loss_weight * self.forward_train(preds_T)

    def forward_trian(self, preds_T: torch.Tensor):
        raise NotImplementedError


@MODELS.register_module()
class ActivationLoss(DAFLLoss):
    """The loss function for measuring the activation of the target featuremap.
    It is negative of the norm of the target featuremap.

    Args:
        norm_type (str, optional):the type of the norm.
                                    Defaults to 'abs'.
        dim (int, optional): the dimension
            of the target featuremap. Defaults to 1.
    """

    def __init__(self, norm_type='abs', dim=1, **kwargs):

        super().__init__(**kwargs)
        assert norm_type in ['fro', 'abs']
        self.norm_type = norm_type
        self.dim = dim

        if self.norm_type == 'fro':
            self.norm_fn = lambda x: -x.norm()
        elif self.norm_type == 'abs':
            self.norm_fn = lambda x: -x.abs().mean()

    def forward(self, feat_T: torch.Tensor):
        return self.loss_weight * self.forward_train(feat_T)

    def forward_train(self, feat_T: torch.Tensor):
        feat_T = feat_T.view(feat_T.size(0), -1)
        return self.norm_fn(feat_T)


@MODELS.register_module()
class OnehotLikeLoss(DAFLLoss):
    """The loss function for measuring the one-hot-likeness of the target
    logits."""

    def __init__(self, **kwargs):
        super(OnehotLikeLoss, self).__init__(**kwargs)

    def forward_train(self, preds_T: torch.Tensor):
        fake_label = preds_T.data.max(1)[1]
        return F.cross_entropy(preds_T, fake_label)


@MODELS.register_module()
class InformationEntropyLoss(DAFLLoss):
    """The loss function for measuring the class balance of the target logits.

    Args:
        gather (bool, optional): the switch controlling whether
            collecting tensors from multiple gpus. Defaults to True.
    """

    def __init__(self, gather=True, **kwargs):
        super(InformationEntropyLoss, self).__init__(**kwargs)
        self.gather = gather
        _, self.world_size = get_dist_info()

    def all_gather(self, tensors):
        combined = GatherTensors.apply(tensors)
        return torch.cat(combined, dim=0)

    def forward_train(self, preds_T: torch.Tensor):

        if self.gather and self.world_size > 1:
            preds_T = self.all_gather(preds_T)
        class_prob = F.softmax(preds_T, dim=1).mean(dim=0)
        info_entropy = class_prob * torch.log10(class_prob)
        return info_entropy.sum()
