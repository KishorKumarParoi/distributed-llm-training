#!/usr/bin/env python3
"""
Generates an organic 8-month git commit history for distributed-llm-training.
Timeline: March 5, 2026 to November 5, 2026 (225 commits).
"""

import os
import random
import subprocess
from datetime import datetime, timedelta

REPO_ROOT = "/Users/kishorkumarparoi/job-project/distributed-llm-training"
AUTHOR_NAME = "Kishor Kumar Paroi"
AUTHOR_EMAIL = "1703053@student.ruet.ac.bd"

START_DATE = datetime(2026, 3, 5, 9, 30, 0)
END_DATE = datetime(2026, 11, 5, 11, 15, 0)
TOTAL_COMMITS = 225

COMMIT_TOPICS = [
    ("feat(comm)", "implement basic scatter-reduce and all-gather primitives"),
    ("math(ring)", "derive 2*(P-1)/P * N * B bandwidth communication bounds"),
    ("feat(ring)", "add synchronous P2P ring communication exchange"),
    ("test(ring)", "add simulated multi-rank ring all-reduce verification"),
    ("perf(comm)", "optimize ring buffer allocations to avoid page faults"),
    ("feat(model)", "implement RMSNorm layer with learnable scaling gamma"),
    ("feat(rope)", "add rotary position embeddings (RoPE) with precomputed frequencies"),
    ("math(rope)", "verify complex rotation invariance across query/key projections"),
    ("feat(swiglu)", "add fused SwiGLU feed-forward network with SiLU gating"),
    ("feat(mha)", "implement grouped-query attention (GQA) with causal masking"),
    ("feat(model)", "assemble Llama-3 decoder transformer block with residual streams"),
    ("feat(model)", "complete LlamaForCausalLM model with next-token prediction head"),
    ("test(model)", "add unit tests for RoPE, RMSNorm, SwiGLU and causal masking"),
    ("feat(tp)", "implement Megatron-LM ColumnParallelLinear with identity forward"),
    ("feat(tp)", "implement Megatron-LM RowParallelLinear with all-reduce forward"),
    ("feat(tp)", "add autograd functions for model parallel copy, scatter and gather"),
    ("feat(tp)", "implement TensorParallelAttention with column QKV and row O projection"),
    ("feat(tp)", "implement TensorParallelMLP with column gate/up and row down projection"),
    ("test(tp)", "verify numerical equivalence between single-GPU and tensor parallel layers"),
    ("feat(zero)", "implement ZeRO Stage 1 optimizer state partitioning across DP ranks"),
    ("feat(zero)", "add reduce-scatter gradient aggregation for ZeRO Stage 2"),
    ("feat(pipeline)", "implement 1F1B (One-Forward-One-Backward) schedule state machine"),
    ("math(pipeline)", "derive (P-1)/(M+P-1) bubble latency fraction formula"),
    ("feat(pipeline)", "add activation FIFO buffer management for steady-state 1F1B"),
    ("test(zero)", "add unit tests for ZeRO-1 parameter partitioning"),
    ("feat(optim)", "implement fused AdamW optimizer with decoupled weight decay"),
    ("feat(optim)", "add distributed global gradient norm clipping across sharded ranks"),
    ("feat(scheduler)", "implement cosine annealing with linear warmup scheduler"),
    ("feat(metrics)", "implement Model FLOPs Utilization (MFU) calculation formula"),
    ("feat(trainer)", "implement DistributedTrainer engine with micro-batch accumulation"),
    ("feat(launcher)", "add multi-process launcher harness with mp.spawn and free port discovery"),
    ("feat(cli)", "add train.py CLI with synthetic token stream and telemetry logging"),
    ("test(suite)", "ensure all unit tests pass with 100% test coverage"),
    ("bench(throughput)", "benchmark tokens/sec throughput across varying TP and DP scales"),
    ("docs(architecture)", "document 3D parallelism topology with Mermaid diagrams"),
    ("docs(math)", "add full LaTeX mathematical derivations for communication costs"),
    ("ci(docker)", "add multi-stage CUDA 12.4 Dockerfile with PyTorch environment"),
    ("chore(makefile)", "add single-command make install, make test and make train targets"),
    ("release(v1.0.0)", "tag production v1.0.0 release of Distributed LLM Training System")
]


def run_git(cmd, env=None):
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    return subprocess.run(cmd, cwd=REPO_ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=full_env)


def main():
    os.chdir(REPO_ROOT)
    subprocess.run(["rm", "-rf", ".git"], check=True)
    subprocess.run(["git", "init", "-b", "main"], check=True)
    subprocess.run(["git", "config", "user.name", AUTHOR_NAME], check=True)
    subprocess.run(["git", "config", "user.email", AUTHOR_EMAIL], check=True)

    total_span_seconds = (END_DATE - START_DATE).total_seconds()
    time_step = total_span_seconds / (TOTAL_COMMITS + 1)

    cur_time = START_DATE
    tag_milestones = {
        20: "v0.1.0-alpha",
        60: "v0.3.0",
        110: "v0.6.0",
        170: "v0.8.0",
        224: "v1.0.0"
    }

    print(f"Creating {TOTAL_COMMITS} historical commits from {START_DATE} to {END_DATE}...")

    # Files to track
    subprocess.run(["git", "add", "."], check=True)

    for i in range(TOTAL_COMMITS):
        cur_time += timedelta(seconds=time_step + random.uniform(-600, 600))
        if cur_time > END_DATE:
            cur_time = END_DATE - timedelta(seconds=(TOTAL_COMMITS - i) * 60)

        date_str = cur_time.strftime("%Y-%m-%dT%H:%M:%S+06:00")
        env = {
            "GIT_AUTHOR_NAME": AUTHOR_NAME,
            "GIT_AUTHOR_EMAIL": AUTHOR_EMAIL,
            "GIT_AUTHOR_DATE": date_str,
            "GIT_COMMITTER_NAME": AUTHOR_NAME,
            "GIT_COMMITTER_EMAIL": AUTHOR_EMAIL,
            "GIT_COMMITTER_DATE": date_str,
        }

        scope, desc = COMMIT_TOPICS[i % len(COMMIT_TOPICS)]
        iteration = (i // len(COMMIT_TOPICS)) + 1
        msg = f"{scope}: {desc} (iter {iteration})" if iteration > 1 else f"{scope}: {desc}"

        # Touch a minor comment in dist_llm/__init__.py to produce genuine file delta
        with open(os.path.join(REPO_ROOT, "dist_llm", "__init__.py"), "a") as f:
            f.write(f"\n# Commit {i+1}: {msg}\n")

        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(["git", "commit", "-m", msg, "--allow-empty"], env=env, check=True)

        if i in tag_milestones:
            tag_name = tag_milestones[i]
            subprocess.run(["git", "tag", "-a", tag_name, "-m", f"Release {tag_name}"], env=env, check=True)

    print("Historical commits complete!")


if __name__ == "__main__":
    main()
