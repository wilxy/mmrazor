"""Online reference:

https://github.com/vlkit/vlkit/blob/master/vlkit/ops/distributed.py#L4-L25
https://amsword.medium.com/gradient-backpropagation-with-torch-distributed-all-gather-9f3941a381f8
https://github.com/open-mmlab/mmselfsup/blob/bbfd50e8c8780cdeb2768922ef336101638818f2/mmselfsup/models/utils/gather_layer.py
"""
import torch
import torch.distributed as dist


class GatherTensors(torch.autograd.Function):
    """Gather tensors from all GPUS, supporting backward propagation. see more
    details in torch.distributed.all_gather and torch.distributed.all_reduce.

    Args:
        input (Tensor): Tensor to be broadcast from current process.

    Returns:
        output:  Gathered tensors from the whole group in a tuple list.

    Usage: GatherOPS.apply(input_tensor)
    """

    @staticmethod
    def forward(ctx, input):
        output = [
            torch.empty_like(input) for _ in range(dist.get_world_size())
        ]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        rank = dist.get_rank()
        merged = torch.stack(grads)
        dist.all_reduce(merged)

        return merged[rank]
