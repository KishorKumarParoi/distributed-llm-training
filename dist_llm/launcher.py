"""
Distributed Multi-Process Launcher Harness.
Launches distributed training workers across single-node CPU/GPU or multi-node clusters.
"""

import os
import socket
from typing import Callable, Optional
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def init_distributed(
    rank: int,
    world_size: int,
    master_addr: str = "127.0.0.1",
    master_port: Optional[int] = None,
    backend: Optional[str] = None,
):
    """Initializes PyTorch distributed process group."""
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port or 29500)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def _worker_wrapper(rank: int, world_size: int, port: int, backend: Optional[str], worker_fn: Callable, worker_args: tuple):
    init_distributed(rank, world_size, master_port=port, backend=backend)
    try:
        worker_fn(rank, world_size, *worker_args)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run_multiprocess(
    worker_fn: Callable,
    world_size: int,
    args: tuple = (),
    backend: Optional[str] = None,
):
    """Spawns multi-process training workers using mp.spawn."""
    port = find_free_port()
    spawn_args = (world_size, port, backend, worker_fn, args)
    mp.spawn(_worker_wrapper, args=spawn_args, nprocs=world_size, join=True)
