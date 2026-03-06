"""
Megatron-LM Style 1D Tensor Parallelism (Shoeybi et al., 2019).
Implements:
- ColumnParallelLinear: Splits weight matrix across output dimension.
- RowParallelLinear: Splits weight matrix across input dimension.
- ParallelEmbedding: Vocabulary-sharded embedding table.
- TensorParallelAttention: Column-parallel Q, K, V followed by Row-parallel O projection.
- TensorParallelMLP: Column-parallel Gate/Up followed by Row-parallel Down projection.
"""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from dist_llm.comm.primitives import (
    copy_to_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    gather_from_tensor_model_parallel_region,
)
from dist_llm.config import ModelConfig
from dist_llm.model import RMSNorm, RotaryEmbedding, apply_rotary_pos_emb


class ColumnParallelLinear(nn.Module):
    """
    Splits weight matrix W along output features: [out_features // tp_size, in_features].
    Forward: X -> X @ W_i (local slice of outputs).
    Backward: Automatically All-Reduces grad_X across TP ranks.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        tp_size: Optional[int] = None,
        tp_group: Optional[dist.ProcessGroup] = None,
        gather_output: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tp_group = tp_group
        self.gather_output = gather_output

        self.tp_size = tp_size or (dist.get_world_size(tp_group) if dist.is_initialized() and tp_group else 1)
        self.tp_rank = dist.get_rank(tp_group) if dist.is_initialized() and tp_group else 0

        assert out_features % self.tp_size == 0, f"out_features ({out_features}) must be divisible by tp_size ({self.tp_size})"
        self.split_out_features = out_features // self.tp_size

        self.weight = nn.Parameter(torch.empty(self.split_out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.split_out_features))
        else:
            self.register_parameter("bias", None)

        self._reset_parameters()

    def _reset_parameters(self):
        std = 0.02
        nn.init.normal_(self.weight, mean=0.0, std=std)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Replicate input across TP ranks (backward performs All-Reduce)
        input_parallel = copy_to_tensor_model_parallel_region(x, self.tp_group)
        output_parallel = F.linear(input_parallel, self.weight, self.bias)

        if self.gather_output and self.tp_size > 1:
            return gather_from_tensor_model_parallel_region(output_parallel, self.tp_group)
        return output_parallel


class RowParallelLinear(nn.Module):
    """
    Splits weight matrix W along input features: [out_features, in_features // tp_size].
    Forward: Performs All-Reduce to sum partial products Y = sum(X_i @ W_i).
    Backward: Passes gradients untouched.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        tp_size: Optional[int] = None,
        tp_group: Optional[dist.ProcessGroup] = None,
        input_is_parallel: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tp_group = tp_group
        self.input_is_parallel = input_is_parallel

        self.tp_size = tp_size or (dist.get_world_size(tp_group) if dist.is_initialized() and tp_group else 1)
        self.tp_rank = dist.get_rank(tp_group) if dist.is_initialized() and tp_group else 0

        assert in_features % self.tp_size == 0, f"in_features ({in_features}) must be divisible by tp_size ({self.tp_size})"
        self.split_in_features = in_features // self.tp_size

        self.weight = nn.Parameter(torch.empty(out_features, self.split_in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self._reset_parameters()

    def _reset_parameters(self):
        std = 0.02
        nn.init.normal_(self.weight, mean=0.0, std=std)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_parallel = x
        if not self.input_is_parallel and self.tp_size > 1:
            input_parallel = torch.chunk(x, self.tp_size, dim=-1)[self.tp_rank].contiguous()

        output_parallel = F.linear(input_parallel, self.weight)

        # All-Reduce forward pass across tensor parallel group
        if self.tp_size > 1:
            output_ = reduce_from_tensor_model_parallel_region(output_parallel, self.tp_group)
        else:
            output_ = output_parallel

        if self.bias is not None:
            output_ = output_ + self.bias
        return output_


class TensorParallelAttention(nn.Module):
    """
    Tensor-Parallel Multi-Head / Grouped-Query Attention.
    Q, K, V projections are ColumnParallel; Output projection is RowParallel.
    """

    def __init__(self, config: ModelConfig, tp_group: Optional[dist.ProcessGroup] = None):
        super().__init__()
        self.config = config
        self.tp_group = tp_group
        self.tp_size = dist.get_world_size(tp_group) if dist.is_initialized() and tp_group else 1
        self.tp_rank = dist.get_rank(tp_group) if dist.is_initialized() and tp_group else 0

        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim

        assert self.num_heads % self.tp_size == 0
        assert self.num_kv_heads % self.tp_size == 0

        self.local_num_heads = self.num_heads // self.tp_size
        self.local_num_kv_heads = self.num_kv_heads // self.tp_size
        self.num_queries_per_kv = self.local_num_heads // self.local_num_kv_heads

        # Column Parallel projections: (X @ W_q), (X @ W_k), (X @ W_v)
        self.q_proj = ColumnParallelLinear(self.hidden_dim, self.num_heads * self.head_dim, bias=False, tp_group=tp_group)
        self.k_proj = ColumnParallelLinear(self.hidden_dim, self.num_kv_heads * self.head_dim, bias=False, tp_group=tp_group)
        self.v_proj = ColumnParallelLinear(self.hidden_dim, self.num_kv_heads * self.head_dim, bias=False, tp_group=tp_group)

        # Row Parallel projection: sum(O_i @ W_o_i)
        self.o_proj = RowParallelLinear(self.num_heads * self.head_dim, self.hidden_dim, bias=False, tp_group=tp_group)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.local_num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.local_num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.local_num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.num_queries_per_kv > 1:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if attention_mask is None:
            causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=x.device), diagonal=1)
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(1)
        else:
            scores = scores + attention_mask

        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        output = torch.matmul(attn_weights, v)

        # Reshape to local partition and project via RowParallel
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.local_num_heads * self.head_dim)
        return self.o_proj(output)


class TensorParallelMLP(nn.Module):
    """
    Tensor-Parallel SwiGLU MLP.
    Gate & Up projections are ColumnParallel; Down projection is RowParallel.
    """

    def __init__(self, config: ModelConfig, tp_group: Optional[dist.ProcessGroup] = None):
        super().__init__()
        self.gate_proj = ColumnParallelLinear(config.hidden_dim, config.intermediate_dim, bias=False, tp_group=tp_group)
        self.up_proj = ColumnParallelLinear(config.hidden_dim, config.intermediate_dim, bias=False, tp_group=tp_group)
        self.down_proj = RowParallelLinear(config.intermediate_dim, config.hidden_dim, bias=False, tp_group=tp_group)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.silu(x @ W_gate) * (x @ W_up) is computed locally per rank, then RowParallel sums across ranks
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
