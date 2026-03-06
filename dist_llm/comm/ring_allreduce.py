"""
Ring All-Reduce Collective Communication Algorithm.
Mathematical formulation:
- Devices: P
- Tensor Elements: N (B bytes per element)
- Chunks: P chunks of size N/P
- Phase 1 (Scatter-Reduce): (P-1) steps, data volume = (P-1)/P * N * B
- Phase 2 (All-Gather):      (P-1) steps, data volume = (P-1)/P * N * B
- Total Transferred Volume: 2 * (P-1)/P * N * B
As P increases, total data transmitted per GPU is strictly bounded by 2 * N * B.
"""

from typing import List, Optional
import torch
import torch.distributed as dist


class RingAllReduce:
    """Production implementation of Ring All-Reduce over PyTorch distributed process groups."""

    @staticmethod
    def ring_allreduce(tensor: torch.Tensor, group: Optional[dist.ProcessGroup] = None) -> torch.Tensor:
        """
        Executes explicit two-phase Ring All-Reduce on a distributed tensor.
        """
        if not dist.is_initialized():
            return tensor

        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)

        if world_size == 1:
            return tensor

        # 1. Flatten and ensure tensor size is divisible by world_size
        flat_tensor = tensor.contiguous().view(-1)
        orig_shape = tensor.shape
        numel = flat_tensor.numel()

        pad_len = (world_size - (numel % world_size)) % world_size
        if pad_len > 0:
            flat_tensor = torch.cat([flat_tensor, torch.zeros(pad_len, dtype=flat_tensor.dtype, device=flat_tensor.device)])

        chunk_size = flat_tensor.numel() // world_size
        chunks = [flat_tensor[i * chunk_size : (i + 1) * chunk_size] for i in range(world_size)]

        # Determine neighbors in the ring
        send_to = (rank + 1) % world_size
        recv_from = (rank - 1 + world_size) % world_size

        # Buffer for receiving chunks
        recv_buffer = torch.empty(chunk_size, dtype=flat_tensor.dtype, device=flat_tensor.device)

        # ==========================================
        # Phase 1: Scatter-Reduce (P - 1 steps)
        # ==========================================
        for step in range(world_size - 1):
            send_chunk_idx = (rank - step + world_size) % world_size
            recv_chunk_idx = (rank - step - 1 + world_size) % world_size

            # Synchronous P2P exchange
            if rank % 2 == 0:
                dist.send(chunks[send_chunk_idx], dst=send_to, group=group)
                dist.recv(recv_buffer, src=recv_from, group=group)
            else:
                dist.recv(recv_buffer, src=recv_from, group=group)
                dist.send(chunks[send_chunk_idx], dst=send_to, group=group)

            # Reduce received data into the target chunk
            chunks[recv_chunk_idx].add_(recv_buffer)

        # ==========================================
        # Phase 2: All-Gather (P - 1 steps)
        # ==========================================
        for step in range(world_size - 1):
            send_chunk_idx = (rank - step + 1 + world_size) % world_size
            recv_chunk_idx = (rank - step + world_size) % world_size

            if rank % 2 == 0:
                dist.send(chunks[send_chunk_idx], dst=send_to, group=group)
                dist.recv(chunks[recv_chunk_idx], src=recv_from, group=group)
            else:
                dist.recv(chunks[recv_chunk_idx], src=recv_from, group=group)
                dist.send(chunks[send_chunk_idx], dst=send_to, group=group)

        # Unpad and restore original shape
        result = flat_tensor[:numel].view(orig_shape)
        tensor.copy_(result)
        return tensor

    @classmethod
    def simulate_ring_allreduce(cls, tensor_shards: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Algorithmic simulator verifying Ring All-Reduce step-by-step across mock devices.
        Used for verification without requiring a live multi-GPU network harness.
        """
        P = len(tensor_shards)
        if P == 1:
            return [t.clone() for t in tensor_shards]

        # Ensure all shards match size
        t_len = tensor_shards[0].numel()
        assert t_len % P == 0, f"Tensor length {t_len} must be divisible by {P}"
        chunk_size = t_len // P

        # Slices: local_chunks[rank][chunk_idx]
        local_chunks = [
            [shard.clone()[i * chunk_size : (i + 1) * chunk_size] for i in range(P)]
            for shard in tensor_shards
        ]

        # 1. Scatter-Reduce (P - 1 steps)
        for step in range(P - 1):
            next_chunks = [[c.clone() for c in rank_chunks] for rank_chunks in local_chunks]
            for r in range(P):
                recv_from = (r - 1 + P) % P
                recv_idx = (r - step - 1 + P) % P
                # What recv_from sends is chunk recv_idx
                next_chunks[r][recv_idx].add_(local_chunks[recv_from][recv_idx])
            local_chunks = next_chunks

        # 2. All-Gather (P - 1 steps)
        for step in range(P - 1):
            next_chunks = [[c.clone() for c in rank_chunks] for rank_chunks in local_chunks]
            for r in range(P):
                recv_from = (r - 1 + P) % P
                recv_idx = (r - step + P) % P
                next_chunks[r][recv_idx].copy_(local_chunks[recv_from][recv_idx])
            local_chunks = next_chunks

        # Concatenate and return
        return [torch.cat(rank_chunks) for rank_chunks in local_chunks]
