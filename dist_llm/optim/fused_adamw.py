"""
Fused AdamW Optimizer & Distributed Gradient Clipping.
"""

import math
from typing import List, Optional
import torch
import torch.nn as nn
import torch.distributed as dist


class FusedAdamW:
    """
    Fused AdamW with decoupled weight decay and distributed global norm clipping.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
    ):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

        self.state = {}
        for p in self.params:
            self.state[p] = {
                "step": 0,
                "exp_avg": torch.zeros_like(p.data),
                "exp_avg_sq": torch.zeros_like(p.data),
            }

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def clip_grad_norm(self, max_norm: float, group: Optional[dist.ProcessGroup] = None) -> float:
        """
        Clips gradient norm across all model parameters, aggregating across distributed ranks if initialized.
        """
        total_norm_sq = 0.0
        for p in self.params:
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm_sq += param_norm.item() ** 2

        if dist.is_initialized() and group is not None:
            tensor_norm = torch.tensor(total_norm_sq, device=self.params[0].device)
            dist.all_reduce(tensor_norm, group=group)
            total_norm_sq = tensor_norm.item()

        total_norm = math.sqrt(total_norm_sq)
        clip_coef = max_norm / (total_norm + 1e-6)

        if clip_coef < 1.0:
            for p in self.params:
                if p.grad is not None:
                    p.grad.detach().mul_(clip_coef)

        return total_norm

    @torch.no_grad()
    def step(self):
        beta1, beta2 = self.betas

        for p in self.params:
            if p.grad is None:
                continue

            grad = p.grad
            state = self.state[p]
            state["step"] += 1
            step = state["step"]

            # Decoupled Weight Decay: theta_t = theta_{t-1} - lr * lambda * theta_{t-1}
            if self.weight_decay != 0:
                p.data.mul_(1.0 - self.lr * self.weight_decay)

            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

            # m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
            exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)

            # v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

            bias_correction1 = 1.0 - beta1 ** step
            bias_correction2 = 1.0 - beta2 ** step
            step_size = self.lr / bias_correction1

            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(self.eps)
            p.data.addcdiv_(exp_avg, denom, value=-step_size)

