# 🚀 Distributed LLM Training Architecture & 4D Parallelism Specification

> **Principal Distributed Systems & Foundation Model Infrastructure Blueprint**  
> **Author:** Kishor Kumar Paroi `<1703053@student.ruet.ac.bd>`  
> **Target Hardware:** Multi-Node GPU Clusters (NVIDIA H100/A100 with NVLink & InfiniBand NDR)  
> **Repository:** [KishorKumarParoi/distributed-llm-training](https://github.com/KishorKumarParoi/distributed-llm-training)

---

## 🏛️ 1. Executive Summary & Parallelism Grid Topology

Modern Foundation Models ($7\text{B} - 405\text{B}$ parameters) far exceed the physical memory of a single GPU device.
For a model with $\Phi$ parameters, standard 16-bit mixed-precision AdamW training requires:
$$\text{Memory}_{\text{Total}} = \underbrace{2\Phi}_{\text{FP16 Weights}} + \underbrace{2\Phi}_{\text{FP16 Gradients}} + \underbrace{4\Phi}_{\text{FP32 Master Weights}} + \underbrace{4\Phi}_{\text{FP32 Momentum}} + \underbrace{4\Phi}_{\text{FP32 Variance}} = 16\Phi \text{ bytes}$$
For LLaMA-3 70B:
$$16 \times 70 \times 10^9 \text{ bytes} \approx 1,120 \text{ GB (1.12 TB)}$$
This necessitates partitioning the workload across a **4D Parallelism Grid**:
$$P_{\text{total}} = N_{\text{TP}} \times N_{\text{PP}} \times N_{\text{DP}} \times N_{\text{SP}}$$

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                   4D DISTRIBUTED PARALLELISM GRID (TP × PP × DP × SP)                  │
│                                                                                        │
│   ┌────────────────────────────────┐         ┌─────────────────────────────────────┐   │
│   │ Tensor Parallelism (TP)        │         │ Sequence Parallelism (SP)           │   │
│   │ • Intra-node NVLink (900 GB/s) │         │ • Shards Seq dimension S / N_SP     │   │
│   │ • Column & Row Parallel GEMM   │         │ • Ring Attention over Q/K/V         │   │
│   └────────────────┬───────────────┘         └──────────────────┬──────────────────┘   │
│                    │                                            │                      │
│                    ▼                                            ▼                      │
│   ┌────────────────────────────────┐         ┌─────────────────────────────────────┐   │
│   │ Pipeline Parallelism (PP)      │         │ Data Parallelism (DP / FSDP / ZeRO) │   │
│   │ • Inter-node 1F1B schedule     │◄───────►│ • Shards optimizer states & weights │   │
│   │ • Activation memory bounds     │         │ • All-Gather & Reduce-Scatter       │   │
│   └────────────────────────────────┘         └─────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🧮 2. First-Principles Communication Derivations & Bounds

### 2.1 Ring All-Reduce Bandwidth Optimality Proof
Consider $P$ GPU ranks arranged in a logical ring, each holding a tensor of $N$ elements ($B$ bytes each).
The tensor is sliced into $P$ equal chunks of size $\frac{N}{P}$.

#### Phase 1: Scatter-Reduce ($P - 1$ steps)
In each step $s \in [0, P-2]$:
- Rank $r$ sends chunk $(r - s) \pmod P$ to rank $(r + 1) \pmod P$.
- Rank $r$ simultaneously receives chunk $(r - s - 1) \pmod P$ from rank $(r - 1) \pmod P$ and reduces (accumulates) it into its local memory.
$$\text{Data sent per rank in Phase 1} = (P - 1) \times \left(\frac{N}{P} \times B\right) = \frac{P - 1}{P} N B \text{ bytes}$$

#### Phase 2: All-Gather ($P - 1$ steps)
The fully reduced chunks are propagated around the ring so every rank receives all updated values:
$$\text{Data sent per rank in Phase 2} = (P - 1) \times \left(\frac{N}{P} \times B\right) = \frac{P - 1}{P} N B \text{ bytes}$$

#### Total Network Communication Volume:
$$\text{Total Transferred per Rank} = 2 \cdot \frac{P - 1}{P} N B \text{ bytes}$$
As $P \to \infty$, the transferred bytes per rank approach:
$$\lim_{P \to \infty} 2 \cdot \frac{P - 1}{P} N B = 2 N B$$

**Crucial Result:** Unlike naive Parameter Server architectures ($O(P \cdot N)$ communication volume), the Ring All-Reduce communication time is **independent of the cluster size $P$**, achieving optimal network utilization.

---

### 2.2 Megatron-LM Tensor Parallelism (1D GEMM Sharding)
Given input activation $X \in \mathbb{R}^{B \times S \times h}$ and weight matrix $W \in \mathbb{R}^{h \times 4h}$ (MLP Projection):

#### 1. Column-Parallel Linear Layer:
Partition $W$ column-wise across $K = N_{\text{TP}}$ GPUs:
$$W = \left[ W_1 \mid W_2 \mid \dots \mid W_K \right], \quad W_i \in \mathbb{R}^{h \times \frac{4h}{K}}$$
Each GPU computes locally:
$$Y_i = X W_i \in \mathbb{R}^{B \times S \times \frac{4h}{K}}$$
- **Forward:** $X$ is replicated (no communication).
- **Backward:** $\nabla X = \sum_{i=1}^K \nabla Y_i W_i^T$. Requires an **All-Reduce** across TP group.

#### 2. Row-Parallel Linear Layer:
Partition second MLP projection $W \in \mathbb{R}^{4h \times h}$ row-wise:
$$W = \begin{bmatrix} W_1 \\ W_2 \\ \dots \\ W_K \end{bmatrix}, \quad W_i \in \mathbb{R}^{\frac{4h}{K} \times h}$$
Each GPU computes locally:
$$Y_i = X_i W_i \in \mathbb{R}^{B \times S \times h}$$
The full output requires summing partial projections:
$$Y = \sum_{i=1}^K Y_i$$
- **Forward:** Requires an **All-Reduce** across TP group.
- **Backward:** Gradient $\nabla X_i = \nabla Y W_i^T$ requires no communication (identity pass-through).

**Net Communication per Transformer Layer:** Exactly 2 All-Reduces in the forward pass and 2 All-Reduces in the backward pass.

---

### 2.3 ZeRO Memory Reduction Math (DeepSpeed ZeRO-1 / ZeRO-2 / ZeRO-3)

Let model parameter count be $\Phi$ and DP world size be $N_{\text{DP}}$:
| Strategy | Parameters $\Phi$ | Gradients | Optimizer States | Total Memory per GPU |
| :--- | :--- | :--- | :--- | :--- |
| **Standard DDP** | $2\Phi$ bytes | $2\Phi$ bytes | $12\Phi$ bytes | $16\Phi$ bytes |
| **ZeRO-1** | $2\Phi$ bytes | $2\Phi$ bytes | $\frac{12\Phi}{N_{\text{DP}}}$ bytes | $4\Phi + \frac{12\Phi}{N_{\text{DP}}}$ bytes |
| **ZeRO-2** | $2\Phi$ bytes | $\frac{2\Phi}{N_{\text{DP}}}$ bytes | $\frac{12\Phi}{N_{\text{DP}}}$ bytes | $2\Phi + \frac{14\Phi}{N_{\text{DP}}}$ bytes |
| **ZeRO-3 (FSDP)** | $\frac{2\Phi}{N_{\text{DP}}}$ bytes | $\frac{2\Phi}{N_{\text{DP}}}$ bytes | $\frac{12\Phi}{N_{\text{DP}}}$ bytes | $\frac{16\Phi}{N_{\text{DP}}}$ bytes |

For a 70B model on 64 GPUs:
- Standard DDP: $1,120 \text{ GB}$ per GPU (Impossible)
- ZeRO-3 / FSDP: $\frac{1120}{64} \approx \mathbf{17.5 \text{ GB}}$ per GPU, fitting comfortably on standard GPUs!

---

### 2.4 1F1B Pipeline Schedule Bubble Latency
Given $P$ pipeline stages and $M$ microbatches ($M \gg P$):
In the steady-state **One-Forward-One-Backward (1F1B)** schedule:
- Each stage alternates between computing 1 forward microbatch and 1 backward microbatch.
- Peak activation memory is strictly capped at $P$ microbatches (unlike GPipe which buffers all $M$ microbatches).
- **Pipeline Bubble Fraction:**
  $$F_{\text{bubble}} = \frac{t_{\text{bubble}}}{t_{\text{total}}} = \frac{P - 1}{M + P - 1}$$
When $M = 8P$, bubble overhead drops below **$11\%$**.

---

## ⚡ 3. High-Throughput Network Interconnect Hierarchy

| Interconnect Layer | Physical Interface | Bi-Directional Bandwidth | Typical Collective Assigned |
| :--- | :--- | :--- | :--- |
| **Intra-Chassis NVLink** | NVLink 4.0 | **900 GB/s** | Tensor Parallelism (TP) All-Reduce |
| **Host-to-Device Bus** | PCIe Gen5 x16 | **64 GB/s** | CPU offload / Pinned Host Checkpoints |
| **Inter-Node Fabric** | 8x InfiniBand NDR | **400 Gbps (50 GB/s)** | ZeRO / DP AllGather & ReduceScatter |
| **WAN Cross-DC** | 100 GbE Ethernet | **12.5 GB/s** | Asynchronous Checkpoint Mirroring |

---

## 🔬 4. ZeRO-3 / FSDP Dynamic Weight Lifecycle

### 4.1 Forward Pass Lifecycle
1. **Pre-Forward Hook:** Issue non-blocking `All-Gather` across the DP process group to reconstruct the full layer weights $\mathbf{W} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$.
2. **Compute:** Execute $\mathbf{Y} = \mathbf{X} \mathbf{W}^T$.
3. **Post-Forward Hook:** Immediately free $\mathbf{W}$, preserving only the local $\frac{1}{N_{\text{DP}}}$ shard $\mathbf{W}_{\text{shard}}$.

### 4.2 Backward Pass Lifecycle
1. **Pre-Backward Hook:** Issue `All-Gather` to reconstruct $\mathbf{W}$.
2. **Compute:** Compute input activations gradient $d\mathbf{X} = d\mathbf{Y} \mathbf{W}$ and parameter gradients $d\mathbf{W} = d\mathbf{Y}^T \mathbf{X}$.
3. **Post-Backward Hook:** Execute `Reduce-Scatter` on $d\mathbf{W}$, accumulating only the local gradient shard $d\mathbf{W}_{\text{shard}}$ on this rank, and immediately free the full tensor $\mathbf{W}$.

---

## 🔄 5. Context Parallelism: Ring Attention & Online Softmax

For extremely long sequences ($S = 1\text{M}+$ tokens), Ring Attention partitions the sequence into $N_{\text{CP}}$ shards of length $S_{\text{local}} = S / N_{\text{CP}}$.
Instead of centralizing the KV cache, ranks circulate $K$ and $V$ blocks in a physical network ring:
- **Step 0:** Attend local $Q_r$ to local $K_r, V_r$. Initialize online softmax state $(m_r, l_r, \mathbf{O}_r)$.
- **Steps $1 \dots N_{\text{CP}}-1$:**
  - Asynchronously send $K, V$ to rank $(r + 1) \bmod N_{\text{CP}}$ and receive from $(r - 1 + N_{\text{CP}}) \bmod N_{\text{CP}}$ on a dedicated communication stream.
  - Concurrently compute attention between $Q_r$ and received $K, V$ tile on the computation stream.
  - Update online softmax statistics:
    $$m_{\text{new}} = \max(m_{\text{old}}, m_{\text{tile}}), \quad \alpha = \exp(m_{\text{old}} - m_{\text{new}})$$
    $$l_{\text{new}} = l_{\text{old}} \cdot \alpha + \sum \exp(S_{\text{tile}} - m_{\text{new}})$$
    $$\mathbf{O}_{\text{new}} = \mathbf{O}_{\text{old}} \cdot \alpha + \mathbf{P}_{\text{tile}} \mathbf{V}_{\text{tile}}$$
- **Causal Optimization:** If source rank $s > r$, the entire KV tile is in the causal future and is skipped with zero FLOPs and zero memory access.

---

## 📉 6. Megatron-LM v2 Interleaved 1F1B Pipelining

Standard 1F1B incurs a bubble fraction $F_{\text{bubble}} = \frac{P - 1}{M + P - 1}$.
Interleaved 1F1B partitions each physical rank into $v$ virtual stages:
- Total virtual pipeline depth: $P_{\text{virtual}} = P \times v$.
- Physical rank $r$ hosts stages $\{r, r + P, \dots, r + (v - 1)P\}$.
- **Mathematical Bubble Reduction:**
  $$F_{\text{bubble\_interleaved}} \approx \frac{P - 1}{v \cdot M}$$
For $P=8$ physical nodes and $M=32$ microbatches:
- $v=1$ (Standard 1F1B): $F_{\text{bubble}} = 17.9\%$
- $v=2$ (2 Virtual Chunks): $F_{\text{bubble}} = 10.9\%$
- $v=4$ (4 Virtual Chunks): $F_{\text{bubble}} = 5.4\%$
Cuts bubble overhead by up to **$70\%$** while bounding activation memory to $O(P \cdot v)$.

---

## 💾 7. Distributed Asynchronous Checkpointing (DCP) & Topology Resharder

- **Non-Blocking Parallel I/O:** Every rank writes its local state dict shard directly to NVMe storage in parallel without gathering to rank 0, slashing checkpoint write times by $N_{\text{world}} \times$.
- **Dynamic Topology Resharder:** Reconstructs full weight matrices or slices them to match any target configuration on the fly:
  $$\text{Target Weight} = \text{Slice}_{\text{target\_rank}}\left(\text{Cat}_{r=0}^{TP_{\text{source}}-1}\left(\mathbf{W}_r\right)\right)$$
  Enables training on 64 GPUs with $TP=4$ and seamlessly deploying for inference on 1 GPU or $TP=2$ clusters with zero manual weights conversion scripts.


