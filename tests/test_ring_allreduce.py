"""
Unit tests for Ring All-Reduce collective algorithm.
"""

import pytest
import torch
from dist_llm.comm.ring_allreduce import RingAllReduce


@pytest.mark.parametrize("world_size", [2, 4, 8])
@pytest.mark.parametrize("tensor_size", [32, 64, 256])
def test_simulated_ring_allreduce(world_size, tensor_size):
    # Ensure tensor_size is divisible by world_size
    tensor_size = (tensor_size // world_size) * world_size

    # Generate distinct random tensors per mock rank
    torch.manual_seed(42)
    shards = [torch.randn(tensor_size) for _ in range(world_size)]

    # Expected analytical sum across all ranks
    expected_sum = sum(shards)

    # Execute simulated Ring All-Reduce
    reduced_shards = RingAllReduce.simulate_ring_allreduce(shards)

    assert len(reduced_shards) == world_size
    for rank, shard in enumerate(reduced_shards):
        assert torch.allclose(shard, expected_sum, atol=1e-5), f"Rank {rank} did not match expected sum"

