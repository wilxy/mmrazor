# Copyright (c) OpenMMLab. All rights reserved.
from operator import attrgetter

import torch
import torch.nn as nn

from mmrazor.registry import MODELS
from .base_distiller import BaseDistiller


@MODELS.register_module()
class DataFreeDistiller(BaseDistiller):
    """The distiller collecting outputs and losses to update the generator.

    Args:
        gan_distiller_cfg (dict): The cfg used for initializing heads.
                                    Defaults to None.
        init_cfg (dict): The cfg used for initializing weights.
                            Defaults to None.
    """

    def __init__(
        self,
        multi_teacher_cfgs: dict,
        init_cfg=None,
    ) -> None:

        super().__init__(init_cfg)
        self.multi_teacher_cfgs = multi_teacher_cfgs

        student_recorders = dict()
        generator_recorders = dict()
        self.multi_teacher_recorders = dict()
        self.multi_distill_losses = dict()

        for cfg in self.multi_teacher_cfgs:
            teacher_name = cfg.teacher_name
            distill_losses = cfg.distill_losses

            # The recorder manager is just constructed, but not really initialized
            # yet. Recorder manager initialization needs to input the corresponding
            # model.
            if cfg.get('student_recorders'):
                student_recorders.update(cfg.student_recorders)
            if cfg.get('generator_recorders'):
                generator_recorders.update(cfg.generator_recorders)
            if cfg.get('teacher_recorders'):
                self.multi_teacher_recorders[teacher_name] = RecorderManager(cfg.teacher_recorders)
            self.multi_distill_losses[teacher_name] = self.build_distill_losses(distill_losses)

         self.student_recorders = RecorderManager(student_recorders)
         self.generator_recorders = RecorderManager(generator_recorders)

    def prepare_from_student(self, model: BaseModel) -> None:
        """Initialize student recorders."""
        self.student_recorders.initialize(model)

    def prepare_from_generator(self, model: BaseModel) -> None:
        """Initialize generator recorders."""
        self.generator_recorders.initialize(model)

    def prepare_from_teachers(self, model_dict: nn.ModuleDict) -> None:
        """Initialize teacher recorders."""
        for teacher_name in model_dict.keys():
            model = model_dict[teacher_name]
            self.multi_teacher_recorders[teacher_name].initialize(model)

    def build_distill_losses(
        self,
        losses: Optional[Dict[str, Dict]] = None,
    ) -> nn.ModuleDict:
        """build distill losses according config."""

        distill_losses = nn.ModuleDict()
        if losses:
            for loss_name, loss_cfg in losses.items():
                assert loss_name not in distill_losses
                if 'loss' not in loss_name:
                    warnings.warn(
                        f'Warning: If {loss_name} is a loss that needs to '
                        f'backward, the name of {loss_name} must contain '
                        f'"loss". If it is only used as a statistical value, '
                        'then the name must not contain "loss". More details '
                        'see docs for '
                        ':func:`mmengine.model.BaseModel._parse_loss`',
                        UserWarning)
                item_loss = MODELS.build(loss_cfg)
                distill_losses[loss_name] = item_loss

        return distill_losses

    def get_record(self,
                   teacher_name: str,
                   recorder: str,
                   from_student: bool,
                   from_generator: bool = False,
                   record_idx: int = 0,
                   data_idx: Optional[int] = None) -> List:
        """According to each item in ``record_infos``, get the corresponding
        record in ``recorder_manager``."""

        if from_student:
            recorder_ = self.student_recorders.get_recorder(recorder)
        elif from_generator:
            recorder_ = self.generator_recorders.get_recorder(recorder)
        else:
            recorder_ = self.multi_teacher_recorders[teacher_name].get_recorder(recorder)
        record_data = recorder_.get_record_data(record_idx, data_idx)

        return record_data

    def compute_distill_losses(self) -> LossResults:
        """Compute distill losses automatically."""
        # Record all computed losses' results.
        losses = dict()

        for cfg in self.multi_teacher_cfgs:
            teacher_name = cfg.teacher_name
            loss_forward_mappings = cfg.loss_forward_mappings

            for loss_name, forward_mappings in self.loss_forward_mappings.items():
                forward_kwargs = dict()
                for forward_key, record in forward_mappings.items():
                    forward_var = self.get_record(teacher_name, **record)
                    forward_kwargs[forward_key] = forward_var

                loss_module = self.multi_distill_losses[teacher_name][loss_name]
                loss = loss_module(**forward_kwargs)  # type: ignore
                # add computed loss result.
                losses[loss_name] = loss

        return losses
