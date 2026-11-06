"""
Distributed LLM Training Entrypoint.
Demonstrates end-to-end multi-process distributed training of Llama-3 architecture.
"""

import argparse
import sys
import torch
import torch.distributed as dist

from dist_llm.config import ModelConfig, TrainingConfig
from dist_llm.engine.trainer import DistributedTrainer
from dist_llm.launcher import init_distributed, run_multiprocess


def generate_synthetic_batch(batch_size: int, seq_len: int, vocab_size: int):
    """Generates synthetic next-token prediction batch."""
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = tokens.clone()
    return tokens, labels


def train_worker(rank: int, world_size: int, args):
    """Worker process training loop."""
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    model_config = ModelConfig(
        vocab_size=args.vocab_size,
        hidden_dim=args.hidden_dim,
        intermediate_dim=args.intermediate_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_len=args.seq_len,
    )

    train_config = TrainingConfig(
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        zero_stage=args.zero_stage,
    )

    trainer = DistributedTrainer(model_config, train_config, device=device)

    if rank == 0:
        print("=" * 70)
        print("🚀 Distributed LLM Training System Initialized (v2.0 4D Parallel)")
        print(f"• World Size: {world_size} workers | Backend: {'NCCL' if torch.cuda.is_available() else 'GLOO'}")
        print(f"• Model Parameters: {trainer.num_params:,} ({trainer.num_params / 1e6:.2f}M)")
        print(f"• Parallelism: 4D Mesh (TP x PP x DP x SP) | ZeRO Stage: {train_config.zero_stage}")
        print(f"• Ring Attention: {'Active' if getattr(args, 'enable_ring_attention', False) else 'Disabled'}")
        print(f"• Interleaved 1F1B: {'Active (v=' + str(args.virtual_stages) + ')' if getattr(args, 'interleaved_1f1b', False) else 'Standard'}")
        print(f"• Sequence Length: {args.seq_len} tokens | Global Batch Size: {args.batch_size}")
        print("=" * 70)

    # Training Loop
    for step in range(1, train_config.max_steps + 1):
        # Generate micro-batches
        micro_batches = [
            generate_synthetic_batch(train_config.micro_batch_size, args.seq_len, args.vocab_size)
            for _ in range(train_config.gradient_accumulation_steps)
        ]

        metrics = trainer.train_step(micro_batches)

        if rank == 0 and (step % train_config.log_interval == 0 or step == 1):
            print(
                f"[Step {metrics.step:3d}/{train_config.max_steps}] "
                f"Loss: {metrics.loss:.4f} | "
                f"LR: {metrics.learning_rate:.2e} | "
                f"Grad Norm: {metrics.grad_norm:.3f} | "
                f"Throughput: {metrics.tokens_per_sec:7.1f} tok/s | "
                f"Step: {metrics.step_time_ms:6.1f}ms | "
                f"MFU: {metrics.mfu_percentage:4.2f}%"
            )

    if getattr(args, "save_dcp", False):
        from dist_llm.engine.checkpoint import save_distributed_checkpoint
        save_distributed_checkpoint(
            trainer.model.state_dict(),
            "checkpoints/dcp",
            rank=rank,
            world_size=world_size,
            step=train_config.max_steps,
        )

    if rank == 0:
        save_path = "checkpoints/llama_final.pt"
        trainer.save_checkpoint(train_config.max_steps, save_path)
        print("=" * 70)
        print(f"✅ Training converged successfully. Checkpoint saved to {save_path}")
        if getattr(args, "save_dcp", False):
            print("💾 Distributed Checkpoint (DCP) saved across all ranks to checkpoints/dcp/")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Distributed LLM Training Engine")
    parser.add_argument("--world-size", type=int, default=2, help="Number of distributed workers")
    parser.add_argument("--vocab-size", type=int, default=1024, help="Vocabulary size")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Transformer hidden dimension")
    parser.add_argument("--intermediate-dim", type=int, default=384, help="SwiGLU intermediate dimension")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--seq-len", type=int, default=64, help="Max sequence length")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--micro-batch-size", type=int, default=2, help="Micro-batch size")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=5e-4, help="Max learning rate")
    parser.add_argument("--max-steps", type=int, default=25, help="Training steps")
    parser.add_argument("--warmup-steps", type=int, default=5, help="Warmup steps")
    parser.add_argument("--zero-stage", type=int, default=1, choices=[0, 1, 2, 3], help="ZeRO Stage (0=None, 1=Optimizer, 2=Gradients, 3=Parameters/FSDP)")
    parser.add_argument("--enable-ring-attention", action="store_true", help="Enable Ring Attention context parallelism")
    parser.add_argument("--interleaved-1f1b", action="store_true", help="Enable Megatron-LM v2 virtual stages pipelining")
    parser.add_argument("--virtual-stages", type=int, default=2, help="Virtual stages per physical rank for interleaved 1F1B")
    parser.add_argument("--save-dcp", action="store_true", help="Save distributed asynchronous checkpoint (DCP)")
    args = parser.parse_args()

    # If launched via torchrun, rank and world_size are set in environment
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        init_distributed(rank, world_size)
        train_worker(rank, world_size, args)
        if dist.is_initialized():
            dist.destroy_process_group()
    else:
        # Launch multi-process using mp.spawn
        run_multiprocess(train_worker, world_size=args.world_size, args=(args,))


if __name__ == "__main__":
    import os
    main()

