# 🎯 Principal Distributed LLM Systems Interview Mastery Guide

> **Elite Foundation Model Infrastructure Interview Preparation ($100k+ to $350k+ Compensation Tier)**  
> **Target Roles:** Staff/Principal Distributed Systems Engineer, LLM Infrastructure Architect, Cluster Scaling Lead  
> **Companies:** Meta AI (FAIR), OpenAI, Anthropic, DeepSeek, Google DeepMind, ByteDance Seed  
> **Author:** Kishor Kumar Paroi `<1703053@student.ruet.ac.bd>`

---

## ⚡ Section 1: Parallelism Dimensions (TP, PP, DP, SP, FSDP)

### Q1: When scaling a 70B parameter model across an H100 cluster, how do you mathematically decide the exact grid configuration $(TP, PP, DP, SP)$?
**Answer:**
The design principles are governed by **Network Bandwidth Hierarchies**:
1. **Tensor Parallelism (TP)** has the highest communication frequency (2 All-Reduces per transformer layer in forward, 2 in backward).
   - *Rule:* TP must never cross node boundaries. Keep TP strictly within an 8-GPU chassis connected via **NVLink (900 GB/s)**. Set $TP \in \{2, 4, 8\}$.
2. **Pipeline Parallelism (PP)** communicates only boundary activation tensors between stages:
   $$\text{Transfer per microbatch} = B_{\text{micro}} \times S \times h \times 2 \text{ bytes}$$
   - *Rule:* PP has low communication volume and easily crosses inter-node **InfiniBand (400 Gbps)** boundaries. However, PP introduces a pipeline bubble fraction $F_{\text{bubble}} = \frac{P - 1}{M + P - 1}$. Bound $PP \le 8$ and ensure $M \ge 4 \times PP$.
3. **Sequence Parallelism (SP)**: For long context ($N \ge 16k$), Megatron-SP shards LayerNorm and Dropout along the sequence dimension across the TP group. For multi-node contexts, Ring Attention rotates $K, V$ blocks across ranks.
4. **Data Parallelism (DP) with FSDP / ZeRO-3**:
   - The remaining cluster ranks are assigned to DP.
   - For an 8-node cluster (64x H100s): An optimal setup is **$TP=8$ (intra-node NVLink), $PP=2$ (inter-node), $DP=4$ (with ZeRO-1 or ZeRO-2)**.

---

### Q2: Derive why ZeRO-3 / FSDP communication volume is $1.5\times$ that of standard DDP.
**Answer:**
Let $\Phi$ be the number of model parameters (in bytes).
1. **Standard DDP:**
   - Forward pass: Zero parameter communication (each GPU has all weights).
   - Backward pass: Ring All-Reduce on gradients:
     $$\text{Volume}_{\text{DDP}} = 2 \cdot \frac{P - 1}{P} \Phi \approx 2\Phi \text{ bytes}$$
2. **ZeRO-3 / FSDP:**
   - Every layer shards parameters across $P$ ranks ($\frac{\Phi}{P}$ bytes stored per rank).
   - **Forward pass:** Each layer performs an **All-Gather** to reconstruct the full weight matrix for GEMM computation, then immediately deletes it from SRAM/HBM:
     $$\text{Volume}_{\text{AllGather, Fwd}} = \frac{P - 1}{P} \Phi \approx 1\Phi \text{ bytes}$$
   - **Backward pass:** Each layer performs another **All-Gather** to re-materialize weights to compute input gradients $\nabla X$:
     $$\text{Volume}_{\text{AllGather, Bwd}} = \frac{P - 1}{P} \Phi \approx 1\Phi \text{ bytes}$$
   - **Gradient Reduction:** Rather than All-Reduce, gradients are reduced and scattered across the DP group via **Reduce-Scatter**:
     $$\text{Volume}_{\text{ReduceScatter}} = \frac{P - 1}{P} \Phi \approx 1\Phi \text{ bytes}$$
   - **Total ZeRO-3 Volume:**
     $$\text{Volume}_{\text{ZeRO-3}} = 1\Phi + 1\Phi + 1\Phi = 3\Phi \text{ bytes}$$
$$\frac{\text{Volume}_{\text{ZeRO-3}}}{\text{Volume}_{\text{DDP}}} = \frac{3\Phi}{2\Phi} = 1.5\times$$
**Trade-off:** ZeRO-3 increases communication traffic by $50\%$, but reduces per-GPU parameter memory by a factor of $P$.

---

### Q3: How do you achieve complete Communication Hiding between GEMM compute and TP All-Reduce?
**Answer:**
In naive execution, the GPU runs GEMM on Stream 0, waits, and then invokes NCCL All-Reduce on the default stream.
To achieve **zero-overhead communication hiding**:
1. Partition the matrix multiplication into $K$ chunks along the batch or sequence dimension (e.g. 2 chunks).
2. Use **Dual CUDA Streams**:
   - Stream 1: Launches GEMM for Chunk 0.
   - Stream 1 signals CUDA Event $E_0$.
   - Stream 2 waits on $E_0$, then launches NCCL All-Reduce for Chunk 0.
   - Concurrently, Stream 1 launches GEMM for Chunk 1.
3. Because Tensor Cores and the Copy/Network Engines (DMA & NVLink) operate independently, the All-Reduce transfer is completely hidden behind the GEMM computation of the subsequent chunk.

---

## 🌐 Section 2: Collective Communication & NCCL Algorithms

### Q4: Explain the difference between Ring All-Reduce and Tree All-Reduce in NCCL. When does NCCL switch between them?
**Answer:**
- **Ring All-Reduce:**
  - Network latency: $2(P - 1) \cdot \alpha$ where $\alpha$ is latency per message.
  - Transfer bandwidth term: $2 \frac{P - 1}{P} \frac{N}{\beta}$ where $\beta$ is network bandwidth.
  - *Characteristics:* High latency (linear in $P$), but maximum bandwidth efficiency. Best for **large message sizes** (>10 MB) such as whole-layer gradient tensors.
- **Tree All-Reduce (Double Binary Tree):**
  - Network latency: $2 \log_2(P) \cdot \alpha$.
  - Transfer bandwidth term: $2 \cdot \frac{N}{\beta}$ (slightly lower link utilization).
  - *Characteristics:* Logarithmic latency in cluster size $P$. Best for **small message sizes** (<1 MB) or very large clusters ($P > 1024$).
- **NCCL Decision Logic:** NCCL dynamically profiles the cluster at startup and switches from Tree to Ring when tensor size exceeds the empirical threshold:
  $$\text{Size}_{\text{threshold}} \approx \frac{\alpha \cdot (P - \log_2 P)}{\frac{1}{\beta_{\text{tree}}} - \frac{1}{\beta_{\text{ring}}}}$$

---

### Q5: What is Model FLOPs Utilization (MFU), and how do you calculate it for a decoder-only Transformer?
**Answer:**
MFU measures the fraction of the theoretical peak GPU hardware FLOPs that is directly dedicated to executing forward and backward transformer math (excluding recomputation and communication bubbles):
$$\text{MFU} = \frac{\text{Theoretical FLOPs per Step}}{\text{Step Time (s)} \times N_{\text{GPUs}} \times \text{Peak FLOPs per GPU}}$$

**Theoretical FLOPs per Token for a Decoder-Only Transformer:**
For each token in the forward pass:
- Self-Attention: $4 h^2 + 2 N_{\text{ctx}} h$ FLOPs
- SwiGLU MLP: $3 \times (2 \times h \times \frac{8}{3} h) = 16 h^2$ FLOPs
- Total Forward per layer $\approx 24 h^2$ FLOPs
- Total Backward is approximately $2\times$ forward: $48 h^2$ FLOPs
- Forward + Backward $\approx 72 h^2$ FLOPs per layer.
Across $L$ layers:
$$\text{FLOPs per token} \approx 6 \times \Phi_{\text{active}}$$
For a step with batch size $B$ and sequence length $S$:
$$\text{Theoretical FLOPs per Step} = 6 \cdot \Phi \cdot (B \times S)$$
A world-class distributed cluster achieves **$45\% - 55\%$ MFU** on NVIDIA H100s.

---

## 🛡️ Section 3: Fault Tolerance & Numerical Stability

### Q6: How do you detect and recover from Gradient Explosions, Loss Spikes, and Silent Data Corruption (SDC) in multi-node clusters?
**Answer:**
1. **Inf/NaN Skipping via Fast Bitmask Check:**
   Before optimizer step, launch a fused CUDA kernel computing the Frobenius norm $\|\mathbf{g}\|_2$. If $\text{isnan}(\|\mathbf{g}\|)$ or $\text{isinf}(\|\mathbf{g}\|)$, set a 1-bit flag on GPU, broadcast via 1-byte All-Reduce, and skip optimizer update without taking CPU-GPU sync stalls.
2. **Dynamic Loss Scaling (FP16 / FP8):**
   Maintain a scale factor $S$. If gradients are healthy for $N=2000$ steps, $S \leftarrow 2S$. If an overflow occurs, $S \leftarrow S / 2$ and discard step.
3. **Asynchronous Non-Blocking Checkpointing:**
   Do not pause GPUs during checkpoint writes. Copy weights from HBM to CPU Pinned Memory via high-speed PCIe Gen5 (taking $<200\text{ms}$), then let a background Python daemon stream the snapshot to NVMe/S3 asynchronously.
