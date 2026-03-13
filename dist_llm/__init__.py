"""
Distributed LLM Training System
Production-grade 3D Parallelism Suite: Megatron-LM Tensor Parallelism,
ZeRO-1/ZeRO-2 Data Parallelism, 1F1B Pipeline Scheduling, and Ring All-Reduce.
"""

__version__ = "1.0.0"
__author__ = "Kishor Kumar Paroi <1703053@student.ruet.ac.bd>"

from dist_llm.config import ModelConfig, TrainingConfig
from dist_llm.model import LlamaForCausalLM

__all__ = [
    "ModelConfig",
    "TrainingConfig",
    "LlamaForCausalLM",
]


# Commit 1: feat(comm): implement basic scatter-reduce and all-gather primitives

# Commit 2: math(ring): derive 2*(P-1)/P * N * B bandwidth communication bounds

# Commit 3: feat(ring): add synchronous P2P ring communication exchange

# Commit 4: test(ring): add simulated multi-rank ring all-reduce verification

# Commit 5: perf(comm): optimize ring buffer allocations to avoid page faults

# Commit 6: feat(model): implement RMSNorm layer with learnable scaling gamma

# Commit 7: feat(rope): add rotary position embeddings (RoPE) with precomputed frequencies

# Commit 8: math(rope): verify complex rotation invariance across query/key projections
