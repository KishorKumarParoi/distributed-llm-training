"""
Model and Distributed Training Configuration.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Hyperparameters for Llama-3 style transformer architecture."""
    vocab_size: int = 32000
    hidden_dim: int = 512
    intermediate_dim: int = 1376  # ~8/3 * hidden_dim for SwiGLU
    num_layers: int = 8
    num_heads: int = 8
    num_kv_heads: Optional[int] = None  # Grouped-Query Attention (GQA)
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    tie_word_embeddings: bool = False
    initializer_range: float = 0.02

    def __post_init__(self):
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        assert self.hidden_dim % self.num_heads == 0, "hidden_dim must be divisible by num_heads"
        assert self.num_heads % self.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads


@dataclass
class TrainingConfig:
    """Configuration for distributed training execution and 3D parallelism."""
    batch_size: int = 16
    micro_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    warmup_steps: int = 20
    max_steps: int = 100
    clip_grad_norm: float = 1.0
    mixed_precision: str = "fp32"  # "fp32", "fp16", "bf16"

    # 3D Parallelism Topology
    tensor_parallel_size: int = 1  # TP
    pipeline_parallel_size: int = 1  # PP
    data_parallel_size: int = 1  # DP
    zero_stage: int = 1  # 0: None, 1: Optimizer State Sharding, 2: Gradient Sharding

    # Checkpoint and Logging
    log_interval: int = 5
    eval_interval: int = 20
    save_interval: int = 50
    output_dir: str = "checkpoints"
    seed: int = 42

    @property
    def world_size(self) -> int:
        return self.tensor_parallel_size * self.pipeline_parallel_size * self.data_parallel_size
