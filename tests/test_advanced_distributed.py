"""
Unit Tests for Advanced Distributed LLM Training Capabilities:
- ZeRO-3 / FSDP Parameter Sharding
- Ring Attention with Multi-Hop Online Softmax
- Megatron-LM v2 Interleaved 1F1B Scheduling
- Distributed Asynchronous Checkpointing (DCP) & Topology Resharder
"""

import os
import shutil
import tempfile
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from dist_llm import (
    ZeRO3Linear,
    ZeRO3ModuleWrapper,
    RingAttentionEngine,
    InterleavedPipelineScheduler,
    OpType,
    save_distributed_checkpoint,
    load_and_reshard_checkpoint,
)


def test_zero3_linear_forward():
    """Verify ZeRO-3 linear forward produces identical output to standard nn.Linear."""
    torch.manual_seed(42)
    in_features = 64
    out_features = 128
    dp_world_size = 4
    batch_size = 2
    seq_len = 16

    x = torch.randn(batch_size, seq_len, in_features)

    # Standard reference Linear
    ref_linear = nn.Linear(in_features, out_features, bias=False)
    ref_out = ref_linear(x)

    # Initialize 4 ZeRO-3 ranks
    sharded_linears = []
    shard_size = out_features // dp_world_size

    for r in range(dp_world_size):
        z_layer = ZeRO3Linear(
            in_features=in_features,
            out_features=out_features,
            dp_rank=r,
            dp_world_size=dp_world_size,
            bias=False,
        )
        # Copy exact slice from reference
        with torch.no_grad():
            start = r * shard_size
            end = start + shard_size
            z_layer.weight_shard.copy_(ref_linear.weight[start:end])
        sharded_linears.append(z_layer)

    # Simulated All-Gather function
    def simulated_all_gather(tensor, world_size):
        all_w = [sharded_linears[i].weight_shard for i in range(world_size)]
        return torch.cat(all_w, dim=0)

    # Execute forward on rank 0
    out_rank0 = sharded_linears[0](x, all_gather_fn=simulated_all_gather)

    assert torch.allclose(out_rank0, ref_out, atol=1e-5), (
        f"Max diff: {torch.max(torch.abs(out_rank0 - ref_out))}"
    )


def test_zero3_module_wrapper_memory_reduction():
    """Verify parameter memory reduction factor in ZeRO-3 module wrapper."""
    dp_world_size = 8
    model = nn.Sequential(
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
    )

    wrapper = ZeRO3ModuleWrapper(model, dp_world_size=dp_world_size, dp_rank=0)
    assert wrapper.get_memory_reduction_factor() == 8.0
    assert len(wrapper.sharded_params) > 0


def test_ring_attention_causal_equivalence():
    """
    Verify Ring Attention across 4 sequence-parallel ranks matches
    single-device global attention of length S = 4 * S_local.
    """
    torch.manual_seed(101)
    batch_size = 1
    num_heads = 4
    num_ranks = 4
    s_local = 32
    s_global = num_ranks * s_local
    head_dim = 64

    # Full global sequences
    q_global = torch.randn(batch_size, num_heads, s_global, head_dim)
    k_global = torch.randn(batch_size, num_heads, s_global, head_dim)
    v_global = torch.randn(batch_size, num_heads, s_global, head_dim)

    # Reference global causal attention
    ref_global = F.scaled_dot_product_attention(q_global, k_global, v_global, is_causal=True)

    # Shard into local chunks per rank
    local_q = [q_global[:, :, r * s_local : (r + 1) * s_local, :] for r in range(num_ranks)]
    local_k = [k_global[:, :, r * s_local : (r + 1) * s_local, :] for r in range(num_ranks)]
    local_v = [v_global[:, :, r * s_local : (r + 1) * s_local, :] for r in range(num_ranks)]
    simulated_kvs = list(zip(local_k, local_v))

    ring_outputs = []
    for r in range(num_ranks):
        engine = RingAttentionEngine(rank=r, world_size=num_ranks)
        out_r = engine.forward(
            q_local=local_q[r],
            k_local=local_k[r],
            v_local=local_v[r],
            is_causal=True,
            simulated_all_kvs=simulated_kvs,
        )
        ring_outputs.append(out_r)

    # Concatenate ring outputs along sequence dimension
    concatenated_ring_out = torch.cat(ring_outputs, dim=2)

    assert torch.allclose(concatenated_ring_out, ref_global, atol=1e-4, rtol=1e-4), (
        f"Max diff: {torch.max(torch.abs(concatenated_ring_out - ref_global))}"
    )


def test_interleaved_1f1b_scheduler():
    """Verify Interleaved 1F1B schedule generation and bubble reduction."""
    p_stages = 4
    v_chunks = 2
    m_microbatches = 16

    scheduler = InterleavedPipelineScheduler(
        num_physical_stages=p_stages,
        num_virtual_stages_per_rank=v_chunks,
        num_microbatches=m_microbatches,
    )

    metrics = scheduler.analyze_schedule_metrics()
    assert metrics["virtual_stages_per_rank"] == 2
    assert metrics["interleaved_1f1b_bubble_ratio"] < metrics["standard_1f1b_bubble_ratio"]
    assert metrics["bubble_time_saved_percent"] > 35.0

    # Verify schedule for rank 0
    rank0_schedule = scheduler.generate_rank_schedule(physical_rank=0)
    forwards = [s for s in rank0_schedule if s.op_type == OpType.FORWARD]
    backwards = [s for s in rank0_schedule if s.op_type == OpType.BACKWARD]

    # Total forwards and backwards should match M * v
    assert len(forwards) == m_microbatches * v_chunks
    assert len(backwards) == m_microbatches * v_chunks


def test_distributed_checkpoint_and_resharding():
    """Verify saving sharded checkpoint and resharding to target topology."""
    temp_dir = tempfile.mkdtemp()
    try:
        # Save a simulated TP=2 model
        hidden_dim = 64
        q_proj_weight = torch.randn(hidden_dim, hidden_dim)

        # Rank 0 (first half of out_features)
        state_dict_r0 = {
            "model.layers.0.self_attn.q_proj.weight": q_proj_weight[: hidden_dim // 2, :].clone(),
            "model.embed_tokens.weight": torch.randn(100, hidden_dim),
        }
        # Rank 1 (second half of out_features)
        state_dict_r1 = {
            "model.layers.0.self_attn.q_proj.weight": q_proj_weight[hidden_dim // 2 :, :].clone(),
            "model.embed_tokens.weight": state_dict_r0["model.embed_tokens.weight"].clone(),
        }

        save_distributed_checkpoint(
            state_dict=state_dict_r0,
            checkpoint_dir=temp_dir,
            rank=0,
            world_size=2,
            step=100,
            tp_size=2,
        )
        save_distributed_checkpoint(
            state_dict=state_dict_r1,
            checkpoint_dir=temp_dir,
            rank=1,
            world_size=2,
            step=100,
            tp_size=2,
        )

        # Now load with target TP=1 (single GPU inference)
        resharded = load_and_reshard_checkpoint(
            checkpoint_dir=temp_dir,
            target_tp_size=1,
            target_tp_rank=0,
        )

        # The reconstructed q_proj.weight must be identical to the original unpartitioned tensor
        reconstructed_q = resharded["model.layers.0.self_attn.q_proj.weight"]
        assert torch.allclose(reconstructed_q, q_proj_weight), "Resharded tensor mismatch!"
        assert reconstructed_q.shape == (hidden_dim, hidden_dim)

    finally:
        shutil.rmtree(temp_dir)
