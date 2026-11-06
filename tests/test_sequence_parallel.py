import pytest
import torch
from dist_llm.parallel.sequence_parallel import SequenceParallelRMSNorm, RingAttentionBlock

def test_sequence_parallel_rmsnorm():
    dim = 256
    seq_len = 128
    batch = 2
    norm = SequenceParallelRMSNorm(dim)
    
    x = torch.randn(batch, seq_len, dim)
    out = norm(x)
    assert out.shape == (batch, seq_len, dim)
    assert not torch.isnan(out).any()

def test_ring_attention_equivalence():
    batch = 2
    seq_local = 32
    num_heads = 4
    head_dim = 64
    num_ranks = 2

    q = torch.randn(batch, seq_local, num_heads, head_dim)
    k = torch.randn(batch, seq_local, num_heads, head_dim)
    v = torch.randn(batch, seq_local, num_heads, head_dim)

    out = RingAttentionBlock.ring_kv_rotate(q, k, v, num_ranks=num_ranks, rank=0)
    assert out.shape == (batch, seq_local, num_heads, head_dim)
    assert not torch.isnan(out).any()

