# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

from mmengine.runner import IterBasedTrainLoop

from mmrazor.registry import LOOPS


@LOOPS.register_module()
class DistributedIterBasedLoop(IterBasedTrainLoop):
    """Dynamic Dataloaders IterBasedRunner.

    In this Dynamic Iterbased Runner, we will pass the ``reducer`` to the
    ``train_step`` so that the models can be trained with dynamic architecture.
    Args:
        is_dynamic_ddp (bool, optional): Whether to adopt the dynamic ddp.
            Defaults to False.
    """

    def run_iter(self, data_batch: Sequence[dict]) -> None:
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook(
            'before_train_iter', batch_idx=self._iter, data_batch=data_batch)

        if self.runner.distributed:
            # Enable gradient accumulation mode and avoid unnecessary gradient
            # synchronization during gradient accumulation process.
            # outputs should be a dict of loss.
            outputs = self.runner.model.module.train_step(
                data_batch, optim_wrapper=self.runner.optim_wrapper)
        else:
            outputs = self.runner.model.train_step(
                data_batch, optim_wrapper=self.runner.optim_wrapper)
        self.runner.message_hub.update_info('train_logs', outputs)

        self.runner.call_hook(
            'after_train_iter',
            batch_idx=self._iter,
            data_batch=data_batch,
            outputs=outputs)
        self._iter += 1
