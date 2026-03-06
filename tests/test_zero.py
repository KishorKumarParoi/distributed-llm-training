"""
Unit tests for ZeRO Stage 1 and Stage 2 memory optimizers.
"""

import pytest
import torch
import torch.nn as nn
from dist_llm.parallel.data_parallel import ZeROStage1Optimizer
from dist_llm.parallel.pipeline_parallel import OneForwardOneBackwardScheduler


def test_zero_stage1_partitioning():
    # Setup simple model with 4 parameters
    model = nn.Sequential(
        nn.Linear(32, 32),
        nn.Linear(32, 16),
        nn.Linear(16, 8),
        nn.Linear(8, 2),
    )

    optimizer = ZeROStage1Optimizer(list(model.parameters()), lr=1e-3)
    # On single-rank, all params belong to rank 0
    assert len(optimizer.local_params) == len(list(model.parameters()))


def test_1f1b_pipeline_schedule():
    from dist_llm.parallel.pipeline_parallel import PipelineStage
    # 4 stages, 8 microbatches
    stage = PipelineStage(nn.ModuleList([]), stage_idx=1, num_stages=4)
    scheduler = OneForwardOneBackwardScheduler(stage, num_microbatches=8)

    schedule = scheduler.get_schedule()
    forward_count = sum(1 for op, _ in schedule if op == "F")
    backward_count = sum(1 for op, _ in schedule if op == "B")

    assert forward_count == 8
    assert backward_count == 8

    # Check bubble latency fraction
    bubble = scheduler.calculate_pipeline_bubble(num_stages=4, num_microbatches=8)
    # (4 - 1) / (8 + 4 - 1) = 3 / 11 ~ 0.2727
    assert pytest.approx(bubble, rel=1e-2) == 3 / 11

