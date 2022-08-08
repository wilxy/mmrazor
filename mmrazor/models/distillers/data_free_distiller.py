# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional

from mmengine.model import BaseModel
import torch.nn as nn

from mmrazor.registry import MODELS
from mmrazor.structures import RecorderManager
from ..algorithms.base import LossResults
from .configurable_distiller import ConfigurableDistiller


@MODELS.register_module()
class DataFreeDistiller(ConfigurableDistiller):
    """The distiller collecting outputs and losses to update the generator.

    Args:
        gan_distiller_cfg (dict): The cfg used for initializing heads.
                                    Defaults to None.
        init_cfg (dict): The cfg used for initializing weights.
                            Defaults to None.
    """

    # def __init__(
    #     self,
    #     student_recorders: Optional[Dict[str, Dict]] = None,
    #     teacher_recorders: Optional[Dict[str, Dict]] = None,
    #     distill_losses: Optional[Dict[str, Dict]] = None,
    #     loss_forward_mappings: Optional[Dict[str, Dict]] = None,
    #     **kwargs) -> None:

    #     super().__init__(
    #         student_recorders,
    #         teacher_recorders,
    #         distill_losses=distill_losses,
    #         loss_forward_mappings=loss_forward_mappings,
    #         **kwargs)

    # def get_record(self,
    #                recorder: str,
    #                from_student: bool,
    #                record_idx: int = 0,
    #                data_idx: Optional[int] = None) -> List:
    #     """According to each item in ``record_infos``, get the corresponding
    #     record in ``recorder_manager``."""

    #     if from_student:
    #         recorder_ = self.student_recorders.get_recorder(recorder)
    #     else:
    #         recorder_ = self.teacher_recorders.get_recorder(recorder)
    #     record_data = recorder_.get_record_data(record_idx, data_idx)

    #     return record_data

    def compute_distill_losses(self) -> LossResults:
        """Compute distill losses automatically."""
        # Record all computed losses' results.
        losses = dict()
        for loss_name, forward_mappings in self.loss_forward_mappings.items():
            forward_kwargs = dict()
            for forward_key, record in forward_mappings.items():
                forward_var = self.get_record(**record)
                forward_kwargs[forward_key] = forward_var

            loss_module = self.distill_losses[loss_name]
            loss = loss_module(**forward_kwargs)  # type: ignore
            # add computed loss result.
            losses[loss_name] = loss

        return losses
