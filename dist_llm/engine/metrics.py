"""
Training Telemetry, Throughput & Model FLOPs Utilization (MFU) Tracking.
"""

from dataclasses import dataclass
import time
from typing import Optional
from dist_llm.config import ModelConfig


@dataclass
class StepTelemetry:
    step: int
    loss: float
    learning_rate: float
    grad_norm: float
    tokens_per_sec: float
    step_time_ms: float
    mfu_percentage: float


class MFUCalculator:
    """
    Computes theoretical Model FLOPs Utilization (MFU, Chowdhery et al., 2022).
    Standard backward pass takes 2x the FLOPs of forward, totaling 6 * N FLOPs per token.
    Attention FLOPs: 12 * L * H * Q * d per token.
    """

    def __init__(self, model_config: ModelConfig, num_params: int, peak_tflops_per_device: float = 312.0):
        self.config = model_config
        self.num_params = num_params
        # Default peak TFLOPS for standard accelerator (e.g. A100 BF16 = 312 TFLOPS)
        self.peak_flops = peak_tflops_per_device * 1e12

        # 6 * N + 12 * L * H * Q * d
        L = model_config.num_layers
        H = model_config.num_heads
        Q = model_config.max_seq_len
        d = model_config.head_dim

        self.flops_per_token = 6 * num_params + 12 * L * H * Q * d

    def compute_mfu(self, tokens_per_sec: float, world_size: int = 1) -> float:
        actual_flops = tokens_per_sec * self.flops_per_token
        peak_total_flops = self.peak_flops * world_size
        return (actual_flops / peak_total_flops) * 100.0

