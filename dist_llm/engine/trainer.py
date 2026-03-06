"""
Distributed LLM Trainer Engine.
Coordinates:
- Micro-batch execution and gradient accumulation
- Distributed gradient reduction across DP/TP process groups
- Dynamic learning rate warmup and decay
- MFU throughput telemetry and distributed checkpointing
"""

import os
import time
from pathlib import Path
from typing import Iterator, Optional, Tuple
import torch
import torch.nn as nn
import torch.distributed as dist

from dist_llm.config import ModelConfig, TrainingConfig
from dist_llm.engine.metrics import MFUCalculator, StepTelemetry
from dist_llm.model import LlamaForCausalLM
from dist_llm.optim.fused_adamw import FusedAdamW
from dist_llm.optim.lr_scheduler import CosineWarmupScheduler
from dist_llm.parallel.data_parallel import BucketedDataParallel, ZeROStage1Optimizer


class DistributedTrainer:
    """Production Multi-GPU / Multi-Process Training Engine."""

    def __init__(
        self,
        model_config: ModelConfig,
        train_config: TrainingConfig,
        device: torch.device = torch.device("cpu"),
        dp_group: Optional[dist.ProcessGroup] = None,
    ):
        self.model_config = model_config
        self.config = train_config
        self.device = device
        self.dp_group = dp_group

        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        # 1. Initialize Model
        self.raw_model = LlamaForCausalLM(model_config).to(device)
        self.num_params = self.raw_model.count_parameters()

        # Wrap with DDP if multiple DP ranks and not ZeRO
        if self.config.zero_stage == 0 and self.world_size > 1:
            self.model = BucketedDataParallel(self.raw_model, dp_group=dp_group)
        else:
            self.model = self.raw_model

        # 2. Initialize Optimizer
        if self.config.zero_stage >= 1 and self.world_size > 1:
            self.optimizer = ZeROStage1Optimizer(
                self.raw_model.parameters(),
                lr=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2),
                eps=self.config.eps,
                weight_decay=self.config.weight_decay,
                dp_group=dp_group,
            )
        else:
            self.optimizer = FusedAdamW(
                self.raw_model.parameters(),
                lr=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2),
                eps=self.config.eps,
                weight_decay=self.config.weight_decay,
            )

        # 3. Learning Rate Scheduler
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_steps=self.config.warmup_steps,
            max_steps=self.config.max_steps,
            max_lr=self.config.learning_rate,
            min_lr=self.config.min_lr,
        )

        # 4. MFU Calculator
        self.mfu_calc = MFUCalculator(model_config, self.num_params)

    def train_step(self, micro_batches: list[Tuple[torch.Tensor, torch.Tensor]]) -> StepTelemetry:
        """Executes a single optimizer step accumulated over micro-batches."""
        self.model.train()
        self.optimizer.zero_grad()

        t0 = time.perf_counter()
        accumulated_loss = 0.0
        total_tokens = 0

        num_micro_batches = len(micro_batches)

        for input_ids, labels in micro_batches:
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)

            logits, loss = self.model(input_ids, labels=labels)
            scaled_loss = loss / num_micro_batches
            scaled_loss.backward()

            accumulated_loss += scaled_loss.item()
            total_tokens += input_ids.numel()

        # Clip Gradients
        grad_norm = 0.0
        if hasattr(self.optimizer, "clip_grad_norm"):
            grad_norm = self.optimizer.clip_grad_norm(self.config.clip_grad_norm, group=self.dp_group)

        # Optimizer & Scheduler Step
        self.optimizer.step()
        current_lr = self.scheduler.step()

        step_time = time.perf_counter() - t0
        step_time_ms = step_time * 1000.0

        # Throughput
        tokens_per_sec = total_tokens / max(1e-5, step_time)
        mfu = self.mfu_calc.compute_mfu(tokens_per_sec, world_size=self.world_size)

        return StepTelemetry(
            step=self.scheduler.current_step,
            loss=accumulated_loss,
            learning_rate=current_lr,
            grad_norm=grad_norm,
            tokens_per_sec=tokens_per_sec,
            step_time_ms=step_time_ms,
            mfu_percentage=mfu,
        )

    def save_checkpoint(self, step: int, path: str):
        """Saves rank-0 checkpoint containing model parameters and config."""
        if self.rank == 0:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            checkpoint = {
                "step": step,
                "model_state_dict": self.raw_model.state_dict(),
                "model_config": self.model_config,
                "train_config": self.config,
            }
            torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Loads model weights from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.raw_model.load_state_dict(checkpoint["model_state_dict"])

