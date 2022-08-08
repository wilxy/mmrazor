# Copyright (c) OpenMMLab. All rights reserved.
from .datafree_distillation import (DataFreeDistillation,
                                    DAFLDataFreeDistillation)
from .fpn_teacher_distill import FpnTeacherDistill
from .single_teacher_distill import SingleTeacherDistill

__all__ = ['SingleTeacherDistill', 'FpnTeacherDistill', 'DataFreeDistillation',
           'DAFLDataFreeDistillation']
