"""
Unit tests for Llama-3 model architecture.
"""

import pytest
import torch
from dist_llm.config import ModelConfig
from dist_llm.model import LlamaForCausalLM, RMSNorm, RotaryEmbedding, SwiGLUMLP


def test_rmsnorm():
    dim = 64
    norm = RMSNorm(dim)
    x = torch.randn(2, 10, dim)
    y = norm(x)
    assert y.shape == x.shape
    # Check that variance along hidden dim is close to 1
    var = y.pow(2).mean(-1)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-2)


def test_rope_embedding():
    head_dim = 32
    seq_len = 16
    rope = RotaryEmbedding(head_dim, max_seq_len=seq_len)
    x = torch.randn(2, 4, seq_len, head_dim)
    cos, sin = rope(x, seq_len)
    assert cos.shape == (seq_len, head_dim)
    assert sin.shape == (seq_len, head_dim)


def test_swiglu_mlp():
    config = ModelConfig(hidden_dim=64, intermediate_dim=128)
    mlp = SwiGLUMLP(config)
    x = torch.randn(2, 16, config.hidden_dim)
    out = mlp(x)
    assert out.shape == (2, 16, config.hidden_dim)


def test_llama_causal_lm_forward_backward():
    config = ModelConfig(
        vocab_size=256,
        hidden_dim=64,
        intermediate_dim=128,
        num_layers=2,
        num_heads=4,
        max_seq_len=32,
    )
    model = LlamaForCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    labels = input_ids.clone()

    logits, loss = model(input_ids, labels=labels)
    assert logits.shape == (2, 16, config.vocab_size)
    assert loss is not None
    assert loss.item() > 0

    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"Parameter {name} has None grad"

