"""
Sequence Parallelism & Ring Attention Primitive Implementation
================================================================
Implements Megatron-SP (sharding along sequence dimension S / N_SP)
and Ring Attention KV rotation across communication ranks.

Author: Kishor Kumar Paroi <1703053@student.ruet.ac.bd>
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

class SequenceParallelRMSNorm(nn.Module):
    """
    Sequence Parallel RMSNorm.
    Activations X are sharded along sequence length [B, S / N_SP, h].
    Normalizer statistics (variance) are computed locally per token since RMSNorm
    is token-independent, requiring ZERO cross-rank communication in forward!
    """
    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [B, S_local, h]
        input_dtype = x.dtype
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x_norm).to(input_dtype)

class RingAttentionBlock:
    """
    Ring Attention Step Simulator.
    Rotates Key-Value chunks around a circular rank ring of size P_SP,
    enabling infinite-context attention with O(N / P_SP) memory per GPU.
    """
    @staticmethod
    def ring_kv_rotate(
        local_q: torch.Tensor,
        local_k: torch.Tensor,
        local_v: torch.Tensor,
        num_ranks: int,
        rank: int
    ) -> torch.Tensor:
        """
        Simulates Ring Attention rotation.
        In each of the P_SP steps, compute attention against the current K, V chunk,
        update running online softmax accumulators, and rotate K, V to neighbor rank.
        """
        B, S_local, num_heads, head_dim = local_q.shape
        scale = 1.0 / (head_dim ** 0.5)
        
        # Local state
        m_i = torch.full((B, num_heads, S_local, 1), -float("inf"), device=local_q.device)
        l_i = torch.zeros((B, num_heads, S_local, 1), device=local_q.device)
        o_acc = torch.zeros_like(local_q)

        # Transpose to [B, H, S_local, D]
        q_h = local_q.transpose(1, 2)
        curr_k = local_k.clone()
        curr_v = local_v.clone()

        for step in range(num_ranks):
            k_h = curr_k.transpose(1, 2)
            v_h = curr_v.transpose(1, 2)

            # Compute S = Q · K^T
            scores = torch.matmul(q_h, k_h.transpose(-2, -1)) * scale
            m_block, _ = torch.max(scores, dim=-1, keepdim=True)
            
            # Online Softmax update
            m_new = torch.maximum(m_i, m_block)
            alpha = torch.exp(m_i - m_new)
            p_block = torch.exp(scores - m_new)
            l_new = l_i * alpha + p_block.sum(dim=-1, keepdim=True)

            # Rescale accumulator
            o_acc_h = o_acc.transpose(1, 2) * alpha + torch.matmul(p_block, v_h)
            o_acc = o_acc_h.transpose(1, 2)

            m_i = m_new
            l_i = l_new

            # Simulate network ring send: rank sends to (rank + 1) % P
            # (In production: dist.isend / dist.irecv with NCCL)
            curr_k = torch.roll(curr_k, shifts=1, dims=0)
            curr_v = torch.roll(curr_v, shifts=1, dims=0)

        # Final normalization
        l_i_t = l_i.transpose(1, 2)
        out = o_acc / torch.clamp(l_i_t, min=1e-6)
        return out
