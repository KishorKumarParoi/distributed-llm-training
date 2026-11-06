"""
Megatron-LM v2 Interleaved 1F1B Pipeline Parallel Scheduling.
============================================================
Reduces pipeline bubble overhead by assigning multiple virtual stages per physical rank.

Mathematical Derivation:
In standard 1F1B (One-Forward-One-Backward), bubble fraction is:
    F_bubble = (P - 1) / (M + P - 1)
where P is the number of physical pipeline stages, and M is the number of micro-batches.

With Interleaved 1F1B, each physical rank hosts v virtual stages (chunks).
Total number of stages in the pipeline is P * v.
Physical rank r manages virtual stages:
    stages(r) = {r + k * P | k in 0 .. v - 1}

The pipeline bubble fraction reduces by a factor of v:
    F_bubble_interleaved = (P - 1) / (v * M)

For P = 8, M = 32:
- Standard 1F1B (v=1): F_bubble = (8 - 1) / (32 + 8 - 1) = 17.9%
- Interleaved 1F1B (v=2): F_bubble = (8 - 1) / (2 * 32) = 10.9%
- Interleaved 1F1B (v=4): F_bubble = (8 - 1) / (4 * 32) = 5.4%

Author: Kishor Kumar Paroi <1703053@student.ruet.ac.bd>
"""

from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
import enum


class OpType(enum.Enum):
    FORWARD = "F"
    BACKWARD = "B"


@dataclass
class ScheduleStep:
    step_id: int
    op_type: OpType
    microbatch_id: int
    virtual_stage_id: int
    physical_rank: int


class InterleavedPipelineScheduler:
    """
    Generates and verifies execution schedules for Interleaved 1F1B pipelining.
    """

    def __init__(
        self,
        num_physical_stages: int,
        num_virtual_stages_per_rank: int,
        num_microbatches: int,
    ):
        self.P = num_physical_stages
        self.v = num_virtual_stages_per_rank
        self.M = num_microbatches
        self.total_stages = self.P * self.v

        assert self.M >= self.P, f"Microbatches ({self.M}) should be >= physical stages ({self.P})"

    def get_bubble_fraction(self) -> float:
        """Computes the theoretical pipeline bubble fraction F_bubble."""
        return (self.P - 1) / (self.v * self.M)

    def generate_rank_schedule(self, physical_rank: int) -> List[ScheduleStep]:
        """
        Generates the 1F1B schedule of forward and backward operations for a given physical rank.
        """
        schedule: List[ScheduleStep] = []
        step_counter = 0

        # Virtual stages owned by this rank
        owned_virtual_stages = [physical_rank + k * self.P for k in range(self.v)]

        # Number of warmup forward steps
        # For rank r with v chunks, warmup forwards = (P - physical_rank - 1) * 2 + (v - 1) * P
        warmup_forwards = min(
            self.M * self.v,
            (self.P - physical_rank - 1) + (self.v - 1) * self.P,
        )

        # 1. Warmup Phase (Forward only)
        for i in range(warmup_forwards):
            mb_idx = i // self.v
            v_idx = owned_virtual_stages[i % self.v]
            schedule.append(
                ScheduleStep(
                    step_id=step_counter,
                    op_type=OpType.FORWARD,
                    microbatch_id=mb_idx,
                    virtual_stage_id=v_idx,
                    physical_rank=physical_rank,
                )
            )
            step_counter += 1

        # 2. 1F1B Steady State Phase (Alternate 1 Backward, 1 Forward)
        steady_state_steps = (self.M * self.v) - warmup_forwards
        for i in range(steady_state_steps):
            # Backward step
            b_mb_idx = i // self.v
            b_v_idx = owned_virtual_stages[self.v - 1 - (i % self.v)]
            schedule.append(
                ScheduleStep(
                    step_id=step_counter,
                    op_type=OpType.BACKWARD,
                    microbatch_id=b_mb_idx,
                    virtual_stage_id=b_v_idx,
                    physical_rank=physical_rank,
                )
            )
            step_counter += 1

            # Forward step
            f_idx = warmup_forwards + i
            f_mb_idx = f_idx // self.v
            f_v_idx = owned_virtual_stages[f_idx % self.v]
            schedule.append(
                ScheduleStep(
                    step_id=step_counter,
                    op_type=OpType.FORWARD,
                    microbatch_id=f_mb_idx,
                    virtual_stage_id=f_v_idx,
                    physical_rank=physical_rank,
                )
            )
            step_counter += 1

        # 3. Cooldown Phase (Remaining Backwards)
        cooldown_steps = warmup_forwards
        for i in range(cooldown_steps):
            b_idx = steady_state_steps + i
            b_mb_idx = b_idx // self.v
            b_v_idx = owned_virtual_stages[self.v - 1 - (b_idx % self.v)]
            schedule.append(
                ScheduleStep(
                    step_id=step_counter,
                    op_type=OpType.BACKWARD,
                    microbatch_id=b_mb_idx,
                    virtual_stage_id=b_v_idx,
                    physical_rank=physical_rank,
                )
            )
            step_counter += 1

        return schedule

    def analyze_schedule_metrics(self) -> Dict[str, Any]:
        """Provides theoretical scaling and memory footprint metrics."""
        f_standard = (self.P - 1) / (self.M + self.P - 1)
        f_interleaved = self.get_bubble_fraction()
        bubble_reduction = (f_standard - f_interleaved) / f_standard * 100

        return {
            "physical_stages": self.P,
            "virtual_stages_per_rank": self.v,
            "total_stages": self.total_stages,
            "microbatches": self.M,
            "standard_1f1b_bubble_ratio": round(f_standard, 4),
            "interleaved_1f1b_bubble_ratio": round(f_interleaved, 4),
            "bubble_time_saved_percent": round(bubble_reduction, 2),
            "activation_memory_bound": f"O({self.P} * {self.v}) activations",
        }

