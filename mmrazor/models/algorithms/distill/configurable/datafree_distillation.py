# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from typing import Dict, List

import torch
import torch.nn as nn
from mmcv.runner import load_checkpoint
from mmengine.optim import OptimWrapper

from mmrazor.models.utils import add_prefix, set_requires_grad
from ...base import BaseAlgorithm
from mmrazor.registry import MODELS


@MODELS.register_module()
class DataFreeDistillation(BaseAlgorithm):
    """Algorithm for data-free teacher-student distillation Typically, the
    teacher is a pretrained model and the student is a small model trained on
    the generator's output. The student is trained to mimic the behavior of the
    teacher. The generator is trained to generate images that are similar to
    the real images.

    Args:
        distiller (dict): The config dict for built distiller.
        generator_distiller (dict): The distiller collecting outputs & losses
            to update the generator.
        teachers (dict[dict]): The list of config dict for teacher models or
            built teacher model.
        generator (dict | BaseModel): The generator syntheszing fake images.
        student_iter (int): The number of student steps in train_step().
            Defaults to 1.
        student_train_first (bool): Whether to train student in first place.
            Defaults to False.
    """

    def __init__(
        self,
        distiller: dict,
        generator_distiller: dict,
        teachers: Dict[dict, str],
        generator: dict,
        student_iter: int = 1,
        student_train_first: bool = False,
        **kwargs) -> None:
        super().__init__(**kwargs)

        self.student_iter = student_iter
        self.student_train_first = student_train_first
        self.distiller = MODELS.build(distiller)
        self.generator_distiller = MODELS.build(generator_distiller)

        self.teachers = nn.ModuleDict()
        for teacher_name, cfg in teachers.items():
            assert teacher_name not in self.teachers, \
                f'{teacher_name} is already in teachers, please check the ' \
                'names in teachers config.'

            self.teachers[teacher_name] = MODELS.build(cfg.build_cfg)
            if cfg.ckpt_path:
                # avoid loaded parameters be overwritten
                self.teachers[teacher_name].init_weights()
                _ = load_checkpoint(self.teachers[teacher_name], cfg.ckpt_path)
            set_requires_grad(self.teachers[teacher_name], False)

        if not isinstance(generator, Dict):
            raise TypeError('generator should be a `dict` instance, but got '
                            f'{type(generator)}')
        self.generator = MODELS.build(generator)

        _ = load_checkpoint(self.generator, '/mnt/lustre/zhangzhongyu.vendor/work_dir/razor_dafl/separate_optim/iter_30000_only_generator.pth')

        # In ``DataFreeDistiller``, the recorder manager is just
        # constructed, but not really initialized yet.
        self.distiller.prepare_from_student(self.student)
        self.distiller.prepare_from_teacher(self.teachers)
        self.generator_distiller.prepare_from_student(self.student)
        self.generator_distiller.prepare_from_teacher(self.teachers)

    @property
    def student(self) -> nn.Module:
        """Alias for ``architecture``."""
        return self.architecture

    def train_step(self, data, optim_wrapper):
        log_vars = OrderedDict()

        if self.student_train_first:
            _, dis_log_vars = self.train_student(
                data, optim_wrapper['architecture'])

            _, generator_loss_vars = self.train_generator(
                data, optim_wrapper['generator'])
        else:
            _, generator_loss_vars = self.train_generator(
                data, optim_wrapper['generator'])
            _, dis_log_vars = self.train_student(
                data, optim_wrapper['architecture'])

        log_vars.update(dis_log_vars)
        log_vars.update(generator_loss_vars)
        return log_vars

    def train_student(self, data, optimizer):
        log_vars = dict()
        batch_size = len(data)

        for iter in range(self.student_iter):
            fakeimg_init = torch.randn((batch_size, self.generator.latent_dim))
            fakeimg = self.generator(fakeimg_init, batch_size).detach()

            with optimizer.optim_context(self):
                _, data_samples = self.data_preprocessor(data, True)

                # recorde the needed information
                with self.distiller.student_recorders:
                    _ = self.student(fakeimg, data_samples, mode='loss')
                with self.distiller.teacher_recorders, torch.no_grad():
                    for _, teacher in self.teachers.items():
                        _ = teacher(fakeimg, data_samples, mode='loss')
                loss_distill = self.distiller.compute_distill_losses()

            distill_loss, distill_log_vars = self.parse_losses(loss_distill)
            optimizer.update_params(distill_loss)
            distill_log_vars.pop('loss')
            log_vars.update(add_prefix(distill_log_vars, 'distill_' + str(iter)))

        return distill_loss, log_vars

    def train_generator(self, data, optimizer):
        # generator_losses = []
        batch_size = len(data)
        fakeimg_init = torch.randn((batch_size, self.generator.latent_dim))
        fakeimg = self.generator(fakeimg_init, batch_size)

        with optimizer.optim_context(self):
            _, data_samples = self.data_preprocessor(data, True)

            # recorde the needed information
            with self.generator_distiller.student_recorders:
                _ = self.student(fakeimg, data_samples, mode='loss')
            with self.generator_distiller.teacher_recorders:
                for _, teacher in self.teachers.items():
                    _ = teacher(fakeimg, data_samples, mode='loss')
            loss_generator = self.generator_distiller.compute_distill_losses()

        generator_loss, generator_loss_vars = self.parse_losses(loss_generator)
        optimizer.update_params(generator_loss)
        log_vars = dict(add_prefix(generator_loss_vars, 'generator'))

        return generator_loss, log_vars


@MODELS.register_module()
class DAFLDataFreeDistillation(DataFreeDistillation):

    def train_step(self, data: List[dict],
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        """DAFL Train step
        :meth:`train_step` will perform the following steps in order:

        - If :attr:`module` defines the preprocess method,
            call ``module.preprocess`` to pre-processing data.
        - Call ``module.forward(**data)`` and get losses.
        - Parse losses.
        - Call ``optim_wrapper.optimizer_step`` to update parameters.
        - Return log messages of losses.

        Args:
            data (List[dict]): Data sampled by dataloader.
            optim_wrapper (OptimWrapper): A wrapper of optimizer to
                update parameters.

        Returns:
            Dict[str, torch.Tensor]: A ``dict`` of tensor for logging.
        """
        batch_size = len(data)
        log_vars = dict()

        for _, teacher in self.teachers.items():
            teacher.eval()

        # fakeimg initialization and revised by generator.
        fakeimg_init = torch.randn((batch_size, self.generator.latent_dim))  # 16 x 1000
        fakeimg = self.generator(fakeimg_init, batch_size)  # 16 x 3 x 32 x 32

        # optim_wrapper['generator'].zero_grad()
        with optim_wrapper['generator'].optim_context(self):
            _, data_samples = self.data_preprocessor(data, True)
            # recorde the needed information
            with self.generator_distiller.student_recorders:
                _ = self.student(fakeimg, data_samples, mode='loss')
            with self.generator_distiller.teacher_recorders:
                for _, teacher in self.teachers.items():
                    _ = teacher(fakeimg, data_samples, mode='loss')
        loss_generator = self.generator_distiller.compute_distill_losses()

        generator_loss, generator_loss_vars = self.parse_losses(loss_generator)
        log_vars.update(add_prefix(generator_loss_vars, 'generator'))

        # optim_wrapper['architecture'].zero_grad()
        with optim_wrapper['architecture'].optim_context(self):
            _, data_samples = self.data_preprocessor(data, True)
            # recorde the needed information
            with self.distiller.student_recorders:
                _ = self.student(fakeimg.detach(), data_samples, mode='loss')
            with self.distiller.teacher_recorders, torch.no_grad():
                for _, teacher in self.teachers.items():
                    _ = teacher(fakeimg.detach(), data_samples, mode='loss')
        loss_distill = self.distiller.compute_distill_losses()

        distill_loss, distill_log_vars = self.parse_losses(loss_distill)
        distill_log_vars.pop('loss')
        log_vars.update(add_prefix(distill_log_vars, 'distill'))

        import pdb
        pdb.set_trace()

        optim_wrapper['generator'].update_params(generator_loss)
        optim_wrapper['architecture'].update_params(distill_loss)

        return log_vars
