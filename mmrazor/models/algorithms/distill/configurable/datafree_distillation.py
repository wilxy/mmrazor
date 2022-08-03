# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

from ...base import BaseAlgorithm, LossResults
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
        teachers (List[dict]): The list of config dict for teacher models or
            built teacher model.
        generator (dict | BaseModel): The generator syntheszing fake images.
        distiller_teacher_name (str): The name of teacher used in distiller.
        student_iter (int): The number of student steps in train_step().
            Defaults to 1.
        student_train_first (bool): Whether to train student in first place.
            Defaults to False.
    """

    def __init__(
        self,
        distiller: dict,
        generator_distiller: dict,
        teachers: List[Dict],
        generator: Union[BaseModel, Dict],
        distiller_teacher_name: str,
        student_iter: int = 1,
        student_train_first: bool = False,
        **kwargs) -> None:
        super().__init__(**kwargs)

        self.distiller = MODELS.build(distiller)
        self.generator_distiller_cfg = generator_distiller
        self.generator_distiller = MODELS.build(generator_distiller)

        self.teachers = nn.ModuleDict()
        for teacher in teachers:
            teacher_name = teacher.name
            assert teacher_name not in self.teachers, \
                f'{teacher_name} is already in teachers, please check the '
                'names in teachers config.'

            self.teachers[teacher_name] = MODELS.build(teacher.cfg)
            if teacher.teacher_ckpt:
                # avoid loaded parameters be overwritten
                self.teachers[teacher_name].init_weights()
                _ = load_checkpoint(self.teachers[teacher_name], teacher_ckpt)

        assert distiller_teacher_name in self.teachers, \
            f'{distiller_teacher_name} not in teachers.'
        self.distiller_teacher = self.teachers[distiller_teacher_name]

        if isinstance(generator, Dict):
            self.generator = MODELS.build(generator)

        if not isinstance(self.generator, BaseModel):
            raise TypeError('generator should be a `dict` or '
                            f'`BaseModel` instance, but got '
                            f'{type(self.generator)}')

        self.student_iter = student_iter
        self.student_train_first = student_train_first

        # In ``ConfigurableDistller``, the recorder manager is just
        # constructed, but not really initialized yet.
        self.distiller.prepare_from_student(self.student)
        # TODO: support multi-teacher distillation in ConfigurableDistiller.
        self.distiller.prepare_from_teacher(self.distiller_teacher)

        # In ``DataFreeDistiller``, the recorder manager is just
        # constructed, but not really initialized yet.
        self.generator_distiller.prepare_from_student(self.student)
        self.generator_distiller.prepare_from_generator(self.generator)
        self.generator_distiller.prepare_from_teachers(self.teachers)

    @property
    def student(self) -> nn.Module:
        """Alias for ``architecture``."""
        return self.architecture

    def train_step(self, data, optimizer, ddp_reducer=None, z_in=None):
        for tea in self.generator_distiller.teachers.children():
            tea.eval()
        log_vars = OrderedDict()

        if self.student_train_first:
            dis_loss, dis_log_vars = self.train_student(
                data, optimizer, ddp_reducer, z_in=z_in)
            generator_loss, generator_loss_vars = self.train_generator(
                data, optimizer, ddp_reducer, z_in=z_in)
        else:
            generator_loss, generator_loss_vars = self.train_generator(
                data, optimizer, ddp_reducer, z_in=z_in)
            dis_loss, dis_log_vars = self.train_student(
                data, optimizer, ddp_reducer, z_in=z_in)

        log_vars.update(dis_log_vars)
        log_vars.update(generator_loss_vars)
        return dict(
            loss=generator_loss + dis_loss,
            log_vars=log_vars,
            num_samples=len(data['img'].data))

    def train_student(self, data, optimizer, ddp_reducer=None, z_in=None):

        for _ in range(self.student_iter):
            batch_size = data['img'].size(0)
            if z_in is None:
                z = torch.randn((batch_size, self.generator.latent_dim))
            else:
                z = z_in

            # TODO: use hook before_train_iter
            optimizer['architecture'].zero_grad()

            fakeimg = self.generator(z, batch_size).detach()
            data.update({'img': fakeimg})

            # recorde the needed information
            batch_inputs, data_samples = self.data_preprocessor(data, True)
            with self.distiller.teacher_recorders, torch.no_grad():
                _ = self.distiller_teacher(batch_inputs, data_samples, mode='loss')
            with self.distiller.student_recorders:
                _ = self.student(batch_inputs, data_samples, mode='loss')

            loss_distill = self.distiller.compute_distill_losses()
            dis_loss, dis_log_vars = self.parse_losses(loss_distill)

            if ddp_reducer is not None:
                from torch.nn.parallel.distributed import _find_tensors
                ddp_reducer.prepare_for_backward(_find_tensors(dis_loss))

            dis_loss.backward()
            # TODO: use hook after_backward
            optimizer['architecture'].step()

        dis_log_vars['distill_loss'] = dis_log_vars.pop('loss')
        return dis_loss, dis_log_vars

    def train_generator(self, data, optimizer, ddp_reducer=None, z_in=None):
        # generator_losses = []
        batch_size = data['img'].size(0)
        if z_in is None:
            z = torch.randn((batch_size, self.generator.latent_dim))
        else:
            z = z_in
        fakeimg = self.generator(z, batch_size)
        data.update({'img': fakeimg})

        optimizer['generator'].zero_grad()

        # recorde the needed information
        batch_inputs, data_samples = self.data_preprocessor(data, True)

        for teacher_cfg in self.generator_distiller_cfg.teacher_cfgs:
            teacher_name = teacher_cfg.teacher_name
            student_recorders = teacher_cfg.get('student_recorders', None)
            teacher_recorders = teacher_cfg.get('teacher_recorders', None)
            generator_recorders = teacher_cfg.get('generator_recorders', None)

            if student_recorders:
                with self.generator_distiller.student_recorders:
                    _ = self.student(batch_inputs, data_samples, mode='loss')
            if generator_recorders:
                with self.generator_distiller.generator_recorders:
                    _ = self.generator(batch_inputs, data_samples, mode='loss')
            if teacher_recorders:
                with self.generator_distiller.multi_teacher_recorders[teacher_name], torch.no_grad():
                    _ = self.teachers[teacher_name](batch_inputs, data_samples, mode='loss')

        gan_losses = self.gan_distiller.compute_distill_losses()
        # generator_losses.append(gan_losses)
        generator_loss, generator_loss_vars = self.parse_losses(gan_losses)

        if ddp_reducer is not None:
            from torch.nn.parallel.distributed import _find_tensors
            # 当前loss涉及到的参数收集起来做更新，网络静态时不需要这一步
            ddp_reducer.prepare_for_backward(_find_tensors(generator_loss))

        generator_loss.backward()
        optimizer['generator'].step()
        generator_loss_vars['generator_loss'] = generator_loss_vars.pop('loss')

        return generator_loss, generator_loss_vars

    def val_step(self, data, optimizer):
        return self.model.val_step(data, optimizer)


@MODELS.register_module()
class DAFLDataFreeStudentDistillation(DataFreeStudentDistillation):

    def train_step(self, data, optimizer, ddp_reducer=None, z_in=None):
        """DAFL Train step https://github.com/huawei-noah/Efficient-
        Computing/"""
        import pdb
        pdb.set_trace()
        batch_size = data['img'].size(0)
        if z_in is None:
            z = torch.randn((batch_size, self.generator.latent_dim))
        else:
            z = z_in
        fakeimg = self.generator(z, batch_size)
        data.update({'img': fakeimg})

        for tea in self.generator_distiller.teachers.children():
            tea.eval()
        log_vars = OrderedDict()
        optimizer['model'].zero_grad()
        optimizer['generator'].zero_grad()

        gen_loss_vars, dis_log_vars, total_loss = self.get_train_losses(
            data, fakeimg)

        if ddp_reducer is not None:
            from torch.nn.parallel.distributed import _find_tensors
            ddp_reducer.prepare_for_backward(_find_tensors(total_loss))
        total_loss.backward()

        optimizer['generator'].step()
        optimizer['model'].step()

        log_vars.update(gen_loss_vars)
        log_vars.update(dis_log_vars)

        return dict(
            loss=total_loss,
            log_vars=log_vars,
            num_samples=len(data['img'].data))

    def get_train_losses(self, data, fakeimg):
        data.update({'img': fakeimg})
        batch_inputs, data_samples = self.data_preprocessor(data, True)

        for teacher_cfg in self.generator_distiller_cfg.teacher_cfgs:
            teacher_name = teacher_cfg.teacher_name
            student_recorders = teacher_cfg.get('student_recorders', None)
            teacher_recorders = teacher_cfg.get('teacher_recorders', None)
            generator_recorders = teacher_cfg.get('generator_recorders', None)

            if student_recorders:
                with self.generator_distiller.student_recorders:
                    _ = self.student(batch_inputs, data_samples, mode='loss')
            if generator_recorders:
                with self.generator_distiller.generator_recorders:
                    _ = self.generator(batch_inputs, data_samples, mode='loss')
            if teacher_recorders:
                with self.generator_distiller.multi_teacher_recorders[teacher_name], torch.no_grad():
                    _ = self.teachers[teacher_name](batch_inputs, data_samples, mode='loss')

        gan_losses = self.gan_distiller.compute_distill_losses()
        generator_loss, generator_loss_vars = self.parse_losses(gan_losses)
        generator_loss_vars['generator_loss'] = generator_loss_vars.pop('loss')

        data.update({'img': fakeimg.detach()})
        batch_inputs, data_samples = self.data_preprocessor(data, True)
        with self.distiller.teacher_recorders, torch.no_grad():
            _ = self.distiller_teacher(batch_inputs, data_samples, mode='loss')
        with self.distiller.student_recorders:
            _ = self.student(batch_inputs, data_samples, mode='loss')
        loss_distill = self.distiller.compute_distill_losses()
        dis_loss, dis_log_vars = self.parse_losses(loss_distill)
        dis_log_vars['distill_loss'] = dis_log_vars.pop('loss')

        total_loss = dis_loss + generator_loss

        return generator_loss_vars, dis_log_vars, total_loss
