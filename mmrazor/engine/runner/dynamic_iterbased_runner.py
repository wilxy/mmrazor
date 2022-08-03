from mmcv.runner import IterBasedRunner

from ..builder import RUNNERS


@RUNNERS.register_module()
class DynamicIterBasedRunner(IterBasedRunner):
    """Dynamic Dataloaders IterBasedRunner.

    In this Dynamic Iterbased Runner, we will pass the ``reducer`` to the
    ``train_step`` so that the models can be trained with dynamic architecture.
    Args:
        is_dynamic_ddp (bool, optional): Whether to adopt the dynamic ddp.
            Defaults to False.
    """

    def __init__(self, *args, is_dynamic_ddp=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_dynamic_ddp = is_dynamic_ddp

    def train(self, data_loader, **kwargs):
        # ddp reducer for tracking dynamic computational graph
        if self.is_dynamic_ddp:
            kwargs.update(dict(ddp_reducer=self.model.reducer))
        return super().train(data_loader, **kwargs)
