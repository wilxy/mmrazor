# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
from mmcv.runner import Hook
from mmcv.runner.hooks.lr_updater import *  # noqa

from mmengine.registry import HOOKS


@HOOKS.register_module()
class MultiLrUpdaterHook(Hook):
    """A hook that updates the learning rate for multiple optimizers.

    Args:
        lr_updater_cfgs (dict): configs for lr updaters.

    Usage:
        >>> lr_config = dict(
                policy='mmrazor.Multi',
                lr_updater_cfgs=dict(
                    architecture=dict(
                        policy='step',
                        step=[
                            100,
                            200,
                        ],
                        by_epoch=False,
                    ),
                    generator=dict(
                        policy='fixed',
                        by_epoch=False,
                    )),
            )
    """

    def __init__(self, lr_updater_cfgs, **kwargs) -> None:

        super().__init__()
        self.lr_updaters = None
        self.lr_updater_cfgs = lr_updater_cfgs

        if not isinstance(lr_updater_cfgs, dict):
            raise TypeError('lr_updater_cfgs should be a dict',
                            f'but got {type(lr_updater_cfgs)}')
        self.optimizer_cfg = lr_updater_cfgs

        if not all(
                isinstance(cfg, dict) for key, cfg in lr_updater_cfgs.items()):
            raise TypeError('lr_updater_cfgs should be a dict',
                            f'but got {type(lr_updater_cfgs)}')

    def get_lr_hook(self, lr_config):
        if lr_config is None:
            return
        elif isinstance(lr_config, dict):
            assert 'policy' in lr_config
            policy_type = lr_config.pop('policy')
            if policy_type == policy_type.lower():
                policy_type = policy_type.title()

            hook_type = eval(f'{policy_type}LrUpdaterHook')

            hook_type.__bases__ = (ModuleLrUpdaterHook, )
            lr_config['type'] = hook_type
            hook: ModuleLrUpdaterHook = mmcv.build_from_cfg(lr_config, HOOKS)
        else:
            hook = lr_config
        return hook

    def before_run(self, runner):
        for key, cfg in self.lr_updater_cfgs.items():
            assert key in runner.optimizer, 'module not in optimizer'
            cfg['optimizer'] = key
            lr_hook = self.get_lr_hook(cfg)
            runner.register_hook(lr_hook, priority='VERY_HIGH')


class ModuleLrUpdaterHook(Hook):
    """A hook that updates the learning rate for each parameter group for a
    specific module.

    References:
    https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py
    """

    def __init__(self,
                 optimizer=None,
                 by_epoch=True,
                 warmup=None,
                 warmup_iters=0,
                 warmup_ratio=0.1,
                 warmup_by_epoch=False):
        # validate the "warmup" argument
        assert optimizer
        self.optimizer = optimizer
        if warmup is not None and warmup not in ['constant', 'linear', 'exp']:
            raise ValueError(
                f'"{warmup}" is not a supported type for warming up, valid'
                ' types are "constant" and "linear"')
        if warmup is not None:
            assert warmup_iters > 0, \
                '"warmup_iters" must be a positive integer'
            assert 0 < warmup_ratio <= 1.0, \
                '"warmup_ratio" must be in range (0,1]'

        self.by_epoch = by_epoch
        self.warmup = warmup
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.warmup_by_epoch = warmup_by_epoch

        if self.warmup_by_epoch:
            self.warmup_epochs = self.warmup_iters
            self.warmup_iters = None
        else:
            self.warmup_epochs = None

        self.base_lr = []  # initial lr for all param groups
        self.regular_lr = []  # expected lr if no warming up is performed

    def _set_lr(self, runner, lr_groups):

        for param_group, lr in zip(self.optimizer.param_groups, lr_groups):
            param_group['lr'] = lr

    def get_lr(self, runner, base_lr):
        raise NotImplementedError

    def get_regular_lr(self, runner):
        return [self.get_lr(runner, _base_lr) for _base_lr in self.base_lr]

    def get_warmup_lr(self, cur_iters):

        def _get_warmup_lr(cur_iters, regular_lr):
            if self.warmup == 'constant':
                warmup_lr = [_lr * self.warmup_ratio for _lr in regular_lr]
            elif self.warmup == 'linear':
                k = (1 - cur_iters / self.warmup_iters) * (1 -
                                                           self.warmup_ratio)
                warmup_lr = [_lr * (1 - k) for _lr in regular_lr]
            elif self.warmup == 'exp':
                k = self.warmup_ratio**(1 - cur_iters / self.warmup_iters)
                warmup_lr = [_lr * k for _lr in regular_lr]
            return warmup_lr

        if isinstance(self.regular_lr, dict):
            lr_groups = {}
            for key, regular_lr in self.regular_lr.items():
                lr_groups[key] = _get_warmup_lr(cur_iters, regular_lr)
            return lr_groups
        else:
            return _get_warmup_lr(cur_iters, self.regular_lr)

    def before_run(self, runner):
        self.optimizer = runner.optimizer[self.optimizer]
        for group in self.optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
        self.base_lr = [
            group['initial_lr'] for group in self.optimizer.param_groups
        ]

    def before_train_epoch(self, runner):
        if self.warmup_iters is None:
            epoch_len = len(runner.data_loader)
            self.warmup_iters = self.warmup_epochs * epoch_len

        if not self.by_epoch:
            return
        self.regular_lr = self.get_regular_lr(runner)

        self._set_lr(runner, self.regular_lr)

    def before_train_iter(self, runner):
        cur_iter = runner.iter
        if not self.by_epoch:
            self.regular_lr = self.get_regular_lr(runner)
            if self.warmup is None or cur_iter >= self.warmup_iters:
                self._set_lr(runner, self.regular_lr)
            else:
                warmup_lr = self.get_warmup_lr(cur_iter)
                self._set_lr(runner, warmup_lr)
        elif self.warmup is None or cur_iter > self.warmup_iters:
            return
        elif cur_iter == self.warmup_iters:
            self._set_lr(runner, self.regular_lr)
        else:
            warmup_lr = self.get_warmup_lr(cur_iter)
            self._set_lr(runner, warmup_lr)
