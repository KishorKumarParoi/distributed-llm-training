"""
Cosine Annealing with Linear Warmup Learning Rate Scheduler.
"""

import math


class CosineWarmupScheduler:
    """
    Computes learning rate with linear warmup and cosine decay:
    - Warmup: lr = min_lr + (max_lr - min_lr) * (step / warmup_steps)
    - Cosine: lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * progress))
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        max_steps: int,
        max_lr: float,
        min_lr: float = 0.0,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.current_step = 0

    def step(self) -> float:
        self.current_step += 1
        lr = self.get_lr(self.current_step)
        self.optimizer.lr = lr
        return lr

    def get_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            # Linear Warmup
            alpha = step / float(max(1, self.warmup_steps))
            return self.min_lr + alpha * (self.max_lr - self.min_lr)

        if step > self.max_steps:
            return self.min_lr

        # Cosine Decay
        progress = (step - self.warmup_steps) / float(max(1, self.max_steps - self.warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.max_lr - self.min_lr) * cosine_decay

