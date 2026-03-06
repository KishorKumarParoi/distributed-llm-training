"""
Distributed Collective Primitives & 3D Parallel Grid Coordinator.
Manages disjoint sub-process groups for 3D Parallelism (TP x PP x DP).
"""

from typing import Optional, Tuple
import torch
import torch.distributed as dist


class ParallelGrid:
    """Coordinates the 3D process grid: Tensor Parallel x Pipeline Parallel x Data Parallel."""

    def __init__(self, tp_size: int = 1, pp_size: int = 1, dp_size: int = 1):
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.dp_size = dp_size
        self.world_size = tp_size * pp_size * dp_size

        self.rank = dist.get_rank() if dist.is_initialized() else 0

        # Calculate 3D coordinate: rank = (dp_rank * pp_size + pp_rank) * tp_size + tp_rank
        self.tp_rank = self.rank % tp_size
        self.pp_rank = (self.rank // tp_size) % pp_size
        self.dp_rank = self.rank // (tp_size * pp_size)

        self.tp_group: Optional[dist.ProcessGroup] = None
        self.dp_group: Optional[dist.ProcessGroup] = None
        self.pp_group: Optional[dist.ProcessGroup] = None

        if dist.is_initialized() and self.world_size > 1:
            self._create_process_groups()

    def _create_process_groups(self):
        # 1. Create TP groups (contiguous ranks across TP size)
        for dp in range(self.dp_size):
            for pp in range(self.pp_size):
                tp_ranks = [
                    (dp * self.pp_size + pp) * self.tp_size + tp
                    for tp in range(self.tp_size)
                ]
                group = dist.new_group(tp_ranks)
                if self.rank in tp_ranks:
                    self.tp_group = group

        # 2. Create DP groups (ranks with identical TP and PP coordinates)
        for pp in range(self.pp_size):
            for tp in range(self.tp_size):
                dp_ranks = [
                    (dp * self.pp_size + pp) * self.tp_size + tp
                    for dp in range(self.dp_size)
                ]
                group = dist.new_group(dp_ranks)
                if self.rank in dp_ranks:
                    self.dp_group = group

        # 3. Create PP groups (ranks with identical TP and DP coordinates)
        for dp in range(self.dp_size):
            for tp in range(self.tp_size):
                pp_ranks = [
                    (dp * self.pp_size + pp) * self.tp_size + tp
                    for pp in range(self.pp_size)
                ]
                group = dist.new_group(pp_ranks)
                if self.rank in pp_ranks:
                    self.pp_group = group


# Custom Autograd Functions for Tensor Parallel Communication

class _CopyToModelParallelRegion(torch.autograd.Function):
    """Passes input forward untouched; performs All-Reduce during backward."""

    @staticmethod
    def forward(ctx, input_, group):
        ctx.group = group
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.group is not None and dist.is_initialized():
            dist.all_reduce(grad_output, group=ctx.group)
        return grad_output, None


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """Performs All-Reduce forward; passes gradient untouched backward."""

    @staticmethod
    def forward(ctx, input_, group):
        if group is not None and dist.is_initialized():
            dist.all_reduce(input_, group=group)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Splits input tensor along last dimension across TP ranks."""

    @staticmethod
    def forward(ctx, input_, group):
        ctx.group = group
        if group is None or not dist.is_initialized():
            return input_
        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)
        last_dim = input_.dim() - 1
        return torch.chunk(input_, world_size, dim=last_dim)[rank].contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.group is None or not dist.is_initialized():
            return grad_output, None
        world_size = dist.get_world_size(ctx.group)
        # All-Gather gradients along the last dimension
        gathered = [torch.empty_like(grad_output) for _ in range(world_size)]
        dist.all_gather(gathered, grad_output, group=ctx.group)
        return torch.cat(gathered, dim=-1), None


class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Gathers tensor along last dimension across TP ranks."""

    @staticmethod
    def forward(ctx, input_, group):
        ctx.group = group
        if group is None or not dist.is_initialized():
            return input_
        world_size = dist.get_world_size(group)
        gathered = [torch.empty_like(input_) for _ in range(world_size)]
        dist.all_gather(gathered, input_, group=group)
        return torch.cat(gathered, dim=-1)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.group is None or not dist.is_initialized():
            return grad_output, None
        world_size = dist.get_world_size(ctx.group)
        rank = dist.get_rank(ctx.group)
        return torch.chunk(grad_output, world_size, dim=-1)[rank].contiguous(), None


def copy_to_tensor_model_parallel_region(input_: torch.Tensor, group=None) -> torch.Tensor:
    return _CopyToModelParallelRegion.apply(input_, group)


def reduce_from_tensor_model_parallel_region(input_: torch.Tensor, group=None) -> torch.Tensor:
    return _ReduceFromModelParallelRegion.apply(input_, group)


def scatter_to_tensor_model_parallel_region(input_: torch.Tensor, group=None) -> torch.Tensor:
    return _ScatterToModelParallelRegion.apply(input_, group)


def gather_from_tensor_model_parallel_region(input_: torch.Tensor, group=None) -> torch.Tensor:
    return _GatherFromModelParallelRegion.apply(input_, group)

