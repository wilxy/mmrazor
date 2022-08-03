# Copyright (c) OpenMMLab. All rights reserved.
from .autoslim_val_loop import AutoSlimValLoop
from .darts_loop import DartsEpochBasedTrainLoop, DartsIterBasedTrainLoop
from .distill_val_loop import SingleTeacherDistillValLoop
from .dynamic_iterbased_runner import DynamicIterBasedRunner
from .evolution_search_loop import EvolutionSearchLoop
from .slimmable_val_loop import SlimmableValLoop
from .subnet_sampler_loop import GreedySamplerTrainLoop

__all__ = [
    'SingleTeacherDistillValLoop', 'DartsEpochBasedTrainLoop',
    'DartsIterBasedTrainLoop', 'SlimmableValLoop', 'EvolutionSearchLoop',
    'GreedySamplerTrainLoop', 'AutoSlimValLoop', 'DynamicIterBasedRunner'
]
