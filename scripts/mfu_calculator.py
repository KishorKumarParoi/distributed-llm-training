#!/usr/bin/env python3
"""
Distributed LLM Model FLOPs Utilization (MFU) & Cluster Throughput Calculator
=============================================================================
Calculates exact theoretical FLOPs, step latencies, MFU, and HFU for LLaMA-3
architectures (8B, 70B, 405B) across NVIDIA H100 / A100 multi-node clusters.

Author: Kishor Kumar Paroi <1703053@student.ruet.ac.bd>
"""

import sys

LLAMA_CONFIGS = {
    "LLaMA-3 8B": {
        "params": 8.03e9,
        "layers": 32,
        "hidden_dim": 4096,
        "heads": 32,
        "kv_heads": 8,
        "vocab_size": 128256,
        "ffn_dim": 14336,
    },
    "LLaMA-3 70B": {
        "params": 70.6e9,
        "layers": 80,
        "hidden_dim": 8192,
        "heads": 64,
        "kv_heads": 8,
        "vocab_size": 128256,
        "ffn_dim": 28672,
    },
    "LLaMA-3 405B": {
        "params": 405.0e9,
        "layers": 126,
        "hidden_dim": 16384,
        "heads": 128,
        "kv_heads": 16,
        "vocab_size": 128256,
        "ffn_dim": 53248,
    },
}

GPU_PEAK_FLOPS = {
    "H100-SXM5-BF16": 989e12,
    "A100-SXM4-BF16": 312e12,
}

def compute_mfu(
    model_name: str,
    global_batch_size: int,
    seq_len: int,
    step_time_ms: float,
    num_gpus: int,
    gpu_type: str = "H100-SXM5-BF16",
    activation_checkpointing: bool = True
):
    cfg = LLAMA_CONFIGS[model_name]
    num_params = cfg["params"]
    
    tokens_per_step = global_batch_size * seq_len
    flops_per_token_model = 6.0 * num_params
    flops_per_token_hardware = 8.0 * num_params if activation_checkpointing else 6.0 * num_params
    
    total_model_flops = flops_per_token_model * tokens_per_step
    total_hardware_flops = flops_per_token_hardware * tokens_per_step
    
    step_time_s = step_time_ms / 1000.0
    actual_tflops_s = (total_model_flops / step_time_s) / 1e12
    hardware_tflops_s = (total_hardware_flops / step_time_s) / 1e12
    
    cluster_peak_flops = num_gpus * GPU_PEAK_FLOPS[gpu_type]
    cluster_peak_tflops = cluster_peak_flops / 1e12
    
    mfu = (total_model_flops / (step_time_s * cluster_peak_flops)) * 100.0
    hfu = (total_hardware_flops / (step_time_s * cluster_peak_flops)) * 100.0
    tokens_per_sec = tokens_per_step / step_time_s

    print(f"\n================================================================================")
    print(f"🚀 MFU BENCHMARK: {model_name} | {num_gpus}x {gpu_type}")
    print(f"================================================================================")
    print(f"Global Batch Size : {global_batch_size} sequences ({tokens_per_step:,} tokens/step)")
    print(f"Sequence Length   : {seq_len:,} tokens")
    print(f"Step Latency      : {step_time_ms:.1f} ms ({step_time_s:.3f} s)")
    print(f"Token Throughput  : {tokens_per_sec:,.0f} tokens/sec ({tokens_per_sec / num_gpus:,.0f} tok/s/GPU)")
    print(f"Attained Compute  : {actual_tflops_s:,.1f} TFLOPs/s (Hardware: {hardware_tflops_s:,.1f} TFLOPs/s)")
    print(f"Cluster Peak      : {cluster_peak_tflops:,.1f} TFLOPs/s")
    print(f"Model FLOPs Utilization (MFU)    : {mfu:.2f}% (Production Target: 42-52%)")
    print(f"Hardware FLOPs Utilization (HFU) : {hfu:.2f}%")

if __name__ == "__main__":
    # Benchmark 1: LLaMA-3 8B on 8x H100 (Single Node)
    compute_mfu("LLaMA-3 8B", global_batch_size=32, seq_len=4096, step_time_ms=1780.0, num_gpus=8)
    
    # Benchmark 2: LLaMA-3 70B on 64x H100 (8 Nodes, TP=8, PP=2, DP=4)
    compute_mfu("LLaMA-3 70B", global_batch_size=128, seq_len=4096, step_time_ms=7150.0, num_gpus=64)
    
    # Benchmark 3: LLaMA-3 405B on 512x H100 (64 Nodes, TP=8, PP=8, DP=8)
    compute_mfu("LLaMA-3 405B", global_batch_size=256, seq_len=8192, step_time_ms=19800.0, num_gpus=512)
