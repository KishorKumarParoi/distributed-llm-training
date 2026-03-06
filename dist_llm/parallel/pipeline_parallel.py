"""
Pipeline Parallelism with 1F1B (One-Forward-One-Backward) Scheduling (Narayanan et al., 2021).
Reduces activation memory footprint from O(num_microbatches) to O(num_pipeline_stages).
"""

from typing import List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.distributed as dist


class PipelineStage(nn.Module):
    """Encapsulates a consecutive slice of transformer blocks assigned to a pipeline stage."""

    def __init__(
        self,
        layers: nn.ModuleList,
        stage_idx: int,
        num_stages: int,
        pp_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.layers = layers
        self.stage_idx = stage_idx
        self.num_stages = num_stages
        self.pp_group = pp_group

        self.is_first_stage = (stage_idx == 0)
        self.is_last_stage = (stage_idx == num_stages - 1)

        self.prev_rank = stage_idx - 1 if not self.is_first_stage else None
        self.next_rank = stage_idx + 1 if not self.is_last_stage else None

    def forward_stage(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Executes forward through the local pipeline layer partition."""
        for layer in self.layers:
            x = layer(x, *args, **kwargs)
        return x


class OneForwardOneBackwardScheduler:
    """
    1F1B Pipeline Schedule.
    Execution Phases:
    1. Warmup: Run (num_stages - stage_idx - 1) forward steps to populate the pipeline.
    2. Steady State (1F1B): Alternating 1 forward followed by 1 backward.
    3. Cooldown: Drain remaining backward steps.
    """

    def __init__(self, stage: PipelineStage, num_microbatches: int):
        self.stage = stage
        self.num_microbatches = num_microbatches
        self.saved_activations: List[torch.Tensor] = []
        self.saved_inputs: List[torch.Tensor] = []

    def get_schedule(self) -> List[Tuple[str, int]]:
        """Generates the static execution sequence of ('F'|'B', microbatch_id)."""
        num_stages = self.stage.num_stages
        stage_idx = self.stage.stage_idx
        num_mb = self.num_microbatches

        num_warmup = min(num_stages - stage_idx - 1, num_mb)
        num_steady = num_mb - num_warmup

        schedule = []

        # 1. Warmup forwards
        for mb in range(num_warmup):
            schedule.append(("F", mb))

        # 2. Steady 1F1B
        f_mb = num_warmup
        b_mb = 0
        for _ in range(num_steady):
            schedule.append(("F", f_mb))
            schedule.append(("B", b_mb))
            f_mb += 1
            b_mb += 1

        # 3. Cooldown backwards
        while b_mb < num_mb:
            schedule.append(("B", b_mb))
            b_mb += 1

        return schedule

    @staticmethod
    def calculate_pipeline_bubble(num_stages: int, num_microbatches: int) -> float:
        """
        Calculates theoretical bubble latency fraction:
        Bubble fraction = (P - 1) / (M + P - 1)
        """
        return (num_stages - 1) / (num_microbatches + num_stages - 1)

