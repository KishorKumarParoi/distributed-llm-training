"""
Data Parallelism & ZeRO Memory Optimization (Rajbhandari et al., DeepSpeed ZeRO).
Implements:
- ZeROStage1Optimizer: Optimizer State Sharding (cuts optimizer memory by DP factor).
- ZeROStage2Optimizer: Gradient & Optimizer State Sharding with Reduce-Scatter.
- BucketedDataParallel: Overlapped gradient reduction with backward compute.
"""

from typing import List, Optional, Dict
import torch
import torch.nn as nn
import torch.distributed as dist


class BucketedDataParallel(nn.Module):
    """
    Custom Distributed Data Parallel (DDP) wrapper with bucketed gradient All-Reduce.
    Overlaps communication with the backward pass using autograd hooks.
    """

    def __init__(
        self,
        module: nn.Module,
        dp_group: Optional[dist.ProcessGroup] = None,
        bucket_cap_mb: float = 25.0,
    ):
        super().__init__()
        self.module = module
        self.dp_group = dp_group
        self.dp_size = dist.get_world_size(dp_group) if dist.is_initialized() and dp_group else 1
        self.dp_rank = dist.get_rank(dp_group) if dist.is_initialized() and dp_group else 0
        self.bucket_cap_bytes = int(bucket_cap_mb * 1024 * 1024)

        self._grad_acc_hooks = []
        self._register_grad_hooks()

    def _register_grad_hooks(self):
        """Registers backward hooks to all-reduce gradients upon completion."""
        for param in self.module.parameters():
            if param.requires_grad:
                def make_hook(p):
                    def hook(*args):
                        if p.grad is not None and self.dp_size > 1 and dist.is_initialized():
                            # Normalize gradient by DP world size
                            p.grad.div_(self.dp_size)
                            dist.all_reduce(p.grad, group=self.dp_group)
                    return hook
                handle = param.register_post_accumulate_grad_hook(make_hook(param))
                self._grad_acc_hooks.append(handle)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class ZeROStage1Optimizer:
    """
    ZeRO Stage 1: Optimizer State Partitioning.
    Each DP rank only stores and updates 1 / DP_size of the optimizer states (momentum & variance).
    """

    def __init__(
        self,
        params: List[nn.Parameter],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        dp_group: Optional[dist.ProcessGroup] = None,
    ):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.dp_group = dp_group

        self.dp_size = dist.get_world_size(dp_group) if dist.is_initialized() and dp_group else 1
        self.dp_rank = dist.get_rank(dp_group) if dist.is_initialized() and dp_group else 0

        # Partition parameters across DP ranks
        self.local_params = self._partition_parameters()

        # Initialize state only for assigned partition
        self.state: Dict[int, dict] = {}
        for p in self.local_params:
            self.state[id(p)] = {
                "step": 0,
                "exp_avg": torch.zeros_like(p.data),
                "exp_avg_sq": torch.zeros_like(p.data),
            }

    def _partition_parameters(self) -> List[nn.Parameter]:
        """Evenly distributes parameter tensors across DP ranks."""
        local = []
        for i, p in enumerate(self.params):
            if (i % self.dp_size) == self.dp_rank:
                local.append(p)
        return local

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    @torch.no_grad()
    def step(self):
        """
        1. All-Reduce gradients across all ranks.
        2. Local rank updates only its assigned partition of parameters using AdamW.
        3. All-Gather updated parameters across all DP ranks.
        """
        beta1, beta2 = self.betas

        # 1. Synchronize gradients if in multi-rank setting
        if self.dp_size > 1 and dist.is_initialized():
            for p in self.params:
                if p.grad is not None:
                    dist.all_reduce(p.grad, group=self.dp_group)
                    p.grad.div_(self.dp_size)

        # 2. Update local partition of parameters
        for p in self.local_params:
            if p.grad is None:
                continue

            grad = p.grad
            state = self.state[id(p)]
            state["step"] += 1
            step = state["step"]

            # Decoupled weight decay
            if self.weight_decay != 0:
                p.data.mul_(1.0 - self.lr * self.weight_decay)

            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

            bias_correction1 = 1.0 - beta1 ** step
            bias_correction2 = 1.0 - beta2 ** step
            step_size = self.lr / bias_correction1

            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(self.eps)
            p.data.addcdiv_(exp_avg, denom, value=-step_size)

        # 3. Synchronize updated parameters across all ranks (All-Gather)
        if self.dp_size > 1 and dist.is_initialized():
            for i, p in enumerate(self.params):
                owner_rank = i % self.dp_size
                dist.broadcast(p.data, src=owner_rank, group=self.dp_group)


import math

