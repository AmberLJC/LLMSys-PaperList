# NeurIPS 2025: Inference & Serving

A curated list of NeurIPS 2025 papers focused on efficient LLM inference, serving systems, and deployment optimization.

---

## Table of Contents

- [1. Systems](#1-systems)
  - [1.1 Scheduling & Resource Management](#11-scheduling--resource-management)
  - [1.2 Distributed Inference](#12-distributed-inference)
  - [1.3 KV Cache Systems](#13-kv-cache-systems)
  - [1.4 Energy Efficiency](#14-energy-efficiency)
  - [1.5 Multi-LoRA Serving](#15-multi-lora-serving)
  - [1.6 TPU Infrastructure](#16-tpu-infrastructure)
  - [1.7 Inference Reliability](#17-inference-reliability)
  - [1.8 Compression Systems](#18-compression-systems)
- [2. System-Algorithm Co-design](#2-system-algorithm-co-design)
  - [2.1 Speculative Decoding](#21-speculative-decoding)
  - [2.2 Long-Context Inference](#22-long-context-inference)
- [3. Algorithm](#3-algorithm)
  - [3.1 Inference-Time Optimization](#31-inference-time-optimization)
  - [3.2 KV Cache Algorithms](#32-kv-cache-algorithms)

---

## 1. Systems

### 1.1 Scheduling & Resource Management

#### Nexus: Proactive Intra-GPU Disaggregation of Prefill and Decode in LLM Serving

- **Link:** [arXiv:2507.06608](https://arxiv.org/abs/2507.06608)
- **Summary:** Nexus addresses throughput-latency tradeoffs through intra-engine prefill-decode disaggregation. The system dynamically partitions GPU resources within a single serving engine to avoid cross-GPU communication overhead—modern GPUs have sufficient memory for internal disaggregation, enabling better resource utilization without the complexity of multi-GPU coordination.

---

#### HyGen: Efficient LLM Serving via Elastic Online-Offline Request Co-location

- **Link:** [OpenReview PDF](https://openreview.net/pdf/40f47181fc0983cc4ba51054de34b5d6a4e75e14.pdf) | [arXiv:2501.14808](https://arxiv.org/abs/2501.14808)
- **Authors:** Ting Sun, Penghan Wang, Fan Lai (UIUC, Purdue)
- **Summary:** Production LLM clusters typically separate online serving (latency-sensitive) from offline batch processing (throughput-optimized), leading to underutilization during traffic valleys. HyGen enables interference-aware co-location of these workloads through a latency predictor for batch execution time estimation, SLO-aware profiler for interference quantification, and adaptive scheduler. The system achieves **3.87-5.84× throughput gains** over baselines while ensuring latency SLOs for online requests.

---

#### SmartCache: Context-aware Semantic Cache for Efficient Multi-turn LLM Inference

- **Link:** [NeurIPS](https://neurips.cc/virtual/2025/loc/san-diego/calendar)
- **Summary:** This paper addresses efficient multi-turn LLM inference through **semantic-aware caching**. In conversational AI applications, users often ask related questions or continue previous topics, creating opportunities for cache reuse beyond exact-match scenarios. SmartCache introduces a context-aware semantic cache that can identify when previous computation results can be partially reused based on semantic similarity, not just exact prefix matching. This reduces redundant computation in multi-turn conversations and improves overall serving throughput.

---

#### Oneiros: KV Cache Optimization through Parameter Remapping for Multi-tenant LLM Serving

- **Link:** [arXiv:2507.11507](https://arxiv.org/abs/2507.11507)
- **Summary:** Oneiros (previously MIRAGE) repurposes inactive model parameter memory for KV cache through dynamic allocation. The approach achieves **44.8-82.5% reduction in tail TBT latency**, **20.7-99.3% reduction in tail TTFT**, and **6.6-86.7% higher throughput** versus vLLM—particularly effective with high CPU-GPU bandwidth on Grace Hopper architectures.

---

### 1.2 Distributed Inference

#### ClusterFusion: Expanding Operator Fusion Scope for LLM Inference via Cluster-Level Collective Primitive

- **Link:** [NeurIPS](https://neurips.cc/Downloads/2025)
- **Summary:** This paper addresses LLM inference efficiency through **advanced operator fusion** at the cluster level. Traditional operator fusion optimizes computation within a single GPU by combining multiple operations to reduce memory bandwidth requirements. ClusterFusion expands this concept to multi-GPU and multi-node deployments by introducing cluster-level collective primitives that enable fusion across distributed compute resources. The system automatically identifies fusion opportunities that span multiple devices and implements efficient collective operations to realize these optimizations.

---

### 1.3 KV Cache Systems

#### ChunkKV: Semantic-Preserving KV Cache Compression for Efficient Long-Context LLM Inference

- **Link:** [NeurIPS](https://neurips.cc/Downloads/2025)
- **Summary:** This paper addresses the challenge of efficient long-context LLM inference by proposing a semantic-preserving approach to KV cache compression. ChunkKV focuses on preserving **semantic coherence** while compressing the cache at the system level, ensuring that the compressed representations maintain the essential information needed for accurate generation. The approach operates at the infrastructure level to enable processing of much longer contexts within fixed memory constraints.

---

#### R-KV: Redundancy-aware KV Cache Compression for Reasoning Models

- **Link:** [Microsoft Research](https://www.microsoft.com/en-us/research/event/neurips-2025/sessions/)
- **Authors:** Zefan Cai, Wen Xiao, Hanshi Sun, Cheng Luo, Yikai Zhang, Ke Wan, Yucheng Li, Yeyang Zhou, Li-Wen Chang, Jiuxiang Gu, Zhen Dong, Anima Anandkumar, Abedelkadir Asi, Junjie Hu
- **Summary:** This paper addresses KV cache compression specifically designed for **reasoning models**, which generate long chains of thought during inference. R-KV proposes a redundancy-aware compression strategy that identifies and removes redundant key-value pairs while preserving the information critical for maintaining reasoning quality. By understanding the patterns of redundancy in reasoning traces at the system level, the method achieves higher compression ratios without degrading the model's ability to perform complex multi-step reasoning.

---

### 1.4 Energy Efficiency

#### The ML.ENERGY Benchmark: Toward Automated Inference Energy Measurement and Optimization 🔦 **SPOTLIGHT**

- **Link:** [arXiv:2505.06371](https://arxiv.org/abs/2505.06371) | [Benchmark](https://github.com/ml-energy/benchmark) | [Leaderboard](https://ml.energy/leaderboard)
- **Authors:** Jae-Won Chung, Jeff J. Ma, Ruofan Wu, Jiachen Liu, Oh Jun Kweon, Yuxuan Xia, Zhiyu Wu, Mosharaf Chowdhury
- **Summary:** As generative AI becomes increasingly embedded in production services, energy has emerged as a critical bottleneck resource that remains overlooked and poorly understood in ML systems. The ML.ENERGY Benchmark addresses this gap by providing an open-source benchmarking suite and public leaderboard that measures inference energy consumption under realistic deployment conditions. The paper outlines four key design principles for effective energy benchmarking and demonstrates their implementation across 40 widely used model architectures spanning 6 different tasks, tested on NVIDIA A100 and H100 GPUs. Notably, automatic optimizations can cut energy consumption by **over 40%** without sacrificing output quality. The research also reveals that even with the model and inference parameters fixed, the software system used to serve inference requests—including batch size configurations and preemption mechanisms—significantly impacts energy consumption.

---

### 1.5 Multi-LoRA Serving

#### Loquetier: A Virtualized Multi-LoRA Framework for Unified Fine-Tuning and Inference

- **Link:** [NeurIPS](https://cs.nju.edu.cn/lm/en/post/2025-10-11-neurips-2025-accepted-papers/)
- **Summary:** This paper addresses the underexplored domain of unified fine-tuning and inference for LoRA-based models. While Low-Rank Adaptation (LoRA) has become a widely adopted parameter-efficient fine-tuning technique, most systems treat training and serving as separate processes, causing resource inefficiency. Loquetier proposes a virtualized multi-LoRA framework that seamlessly integrates LoRA fine-tuning and inference serving in a single runtime environment. The framework achieves **3.0× throughput** of top co-serving systems in inference-only tasks, and **46.4× higher SLO attainment** in mixed workloads.

---

### 1.6 TPU Infrastructure

#### Smarter LLM Serving: vLLM Meets Google Cloud TPUs

- **Link:** https://blog.vllm.ai/2025/10/16/vllm-tpu.html | https://cloud.google.com/blog/products/compute/in-q3-2025-ai-hypercomputer-adds-vllm-tpu-and-more
- **Authors:** Google & vLLM Team Collaboration
- **Summary:** A significant collaboration bringing native TPU support to the vLLM serving engine. Key innovations include:
  - **Pallas Kernels:** Rewrote key kernels (like PagedAttention) using Pallas, a grid-based kernel language for TPUs similar to Triton for GPUs
  - **Unified Backend:** The new tpu-inference backend unifies PyTorch and JAX, allowing models defined in PyTorch to be lowered to XLA efficiently
- **Impact:** This development breaks the NVIDIA monopoly on high-performance open-source LLM serving. Teams can now leverage the cost-efficiency of TPU v5e/v6 hardware for serving Llama and Mistral class models with the same ease of use as GPU deployments.

---

### 1.7 Inference Reliability

#### Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference 🎤 **ORAL**

- **Link:** [OpenReview](https://openreview.net/forum?id=Q3qAsZAEZw) | [arXiv:2506.09501](https://arxiv.org/abs/2506.09501)
- **Authors:** Jiayi Yuan, Hao Li, Xinheng Ding, and collaborators
- **Summary:** This paper presents the first systematic investigation of LLM inference reproducibility failures caused by numerical precision issues. The authors demonstrate that supposedly deterministic greedy decoding produces significantly different outputs when hardware configurations change. Under bfloat16 precision, DeepSeek-R1-Distill-Qwen-7B exhibits up to **9% accuracy variation** and **9,000 tokens difference** in response length merely from changing GPU count, GPU type, or batch size. The root cause is the non-associative nature of floating-point arithmetic. The solution, **LayerCast**, is a lightweight inference pipeline that stores weights in 16-bit precision (for memory efficiency) while performing all computations in FP32 (for numerical stability), achieving FP32-level determinism without the full memory overhead.

---

### 1.8 Compression Systems

#### DFloat11: Lossless LLM Compression for Efficient GPU Inference via Dynamic-Length Float

- **Link:** [NeurIPS 2025 Poster](https://neurips.cc/virtual/2025/poster/115225)
- **Summary:** Unlike lossy quantization methods that trade accuracy for size, DFloat11 achieves **~30% model size reduction while producing bit-for-bit identical outputs**. The approach uses entropy coding with dynamic-length floating-point representations and custom GPU kernels with hierarchical lookup tables for fast online decompression. Critically, this enables **lossless inference of Llama 3.1 405B (810GB) on 8×80GB GPUs** without any quality compromise. Compared to CPU offloading alternatives, DFloat11 achieves **2.3-46.2× higher throughput**.

---

## 2. System-Algorithm Co-design

### 2.1 Speculative Decoding

#### SuffixDecoding: Extreme Speculative Decoding for Emerging AI Applications 🔦 **SPOTLIGHT**

- **Link:** [arXiv:2411.04975](https://arxiv.org/abs/2411.04975) | [Project Page](https://suffix-decoding.github.io/) | [GitHub](https://github.com/snowflakedb/ArcticInference)
- **Authors:** Gabriele Oliaro, Zhihao Jia, Daniel Campos, Aurick Qiao (Carnegie Mellon University, Snowflake AI Research)
- **Summary:** Agentic AI applications present unique inference challenges—LLM-based agents submit repetitive requests in multi-agent pipelines and self-refinement loops, creating highly predictable token sequences that existing speculative decoding methods fail to exploit. SuffixDecoding addresses this by using efficient suffix trees to cache long token sequences from prompts and previous outputs, enabling adaptive speculation that extends beyond traditional draft model approaches. The method achieves up to **5.3× speedup** on agentic workloads like SWE-Bench and AgenticSQL, **outperforming model-based approaches like EAGLE-2/3 by 2.8×** and model-free approaches like Token Recycling by **1.9×**. Critically, SuffixDecoding requires only CPU memory—plentiful and underutilized on typical LLM serving nodes—making it highly practical for production deployments.

---

#### EasySpec: Layer-Parallel Speculative Decoding for Efficient Multi-GPU Utilization

- **Link:** [arXiv:2502.02493](https://arxiv.org/abs/2502.02493)
- **Authors:** Yize Wu et al.
- **Summary:** Multi-GPU inference typically achieves poor utilization during the sequential decode phase. EasySpec introduces layer-parallel speculation that enables multi-layer parallelization across devices, achieving **peak speedup of 4.17×** compared to vanilla decoding while preserving the original distribution (lossless). The drafting stage is accelerated by up to **1.62×** through better GPU utilization. This approach is particularly valuable for large models that require tensor parallelism but suffer from communication overhead during generation.

---

#### Accelerating Diffusion LLMs via Adaptive Parallel Decoding 🔦 **SPOTLIGHT**

- **Link:** [Paper PDF](https://starai.cs.ucla.edu/papers/IsraelNeurIPS25.pdf) | [OpenReview](https://openreview.net/forum?id=xwqTt26NJf)
- **Authors:** Daniel Israel, Justus Mattern, Jae Hyun Lim, Guy Van den Broeck, Yisong Yue (UCLA, Caltech)
- **Summary:** Diffusion-based language models offer an alternative to autoregressive generation but have struggled to match the speed of speculative decoding in AR models. This paper introduces a method that dynamically adjusts the number of tokens sampled in parallel for diffusion LLMs by defining a multiplicative mixture between dLLM marginal probabilities and joint probability of sequences under a small auxiliary autoregressive model. The approach substantially accelerates diffusion LLMs with minimal quality degradation, demonstrating that parallel decoding strategies can be effectively adapted to non-autoregressive architectures.

---

### 2.2 Long-Context Inference

#### Self-distilled Attention Gating for Efficient Long-context Prefilling

- **Link:** [Microsoft Research](https://www.microsoft.com/en-us/research/event/neurips-2025/sessions/)
- **Authors:** Yizhao Gao, Zhichen Zeng, DaYou Du, Shijie Cao, Peiyuan Zhou, Jiaxing Qi, Junjie Lai, Hayden Kwok-Hay So, Ting Cao, Fan Yang, Mao Yang
- **Summary:** This paper addresses the efficiency challenge in the **prefilling stage** of long-context LLM inference—a critical system bottleneck. The prefilling stage, which processes the entire input context before generation begins, becomes increasingly expensive as context length grows due to the quadratic complexity of attention computation. The proposed self-distilled attention gating mechanism learns to identify and skip less important attention computations during prefilling while maintaining output quality. This directly improves Time-To-First-Token (TTFT) for serving systems.

---

#### RetrievalAttention: Accelerating Long-Context LLM Inference via Vector Retrieval

- **Link:** [Microsoft Research](https://www.microsoft.com/en-us/research/event/neurips-2025/sessions/)
- **Authors:** Di Liu, Meng Chen, Baotong Lu, Huiqiang Jiang, Qianxi Zhang, Qi Chen, Chengruidong Zhang, Bailu Ding, Kai Zhang, Chen Chen, Fan Yang, Yuqing Yang, Lili Qiu
- **Summary:** This paper addresses the computational challenge of long-context LLM inference by introducing a **vector retrieval-based approach** to accelerate attention computation. The core insight is that for long contexts, most attention weights are negligible, and the model only needs to attend to a small subset of relevant tokens. Instead of computing full attention over the entire context, the system uses efficient vector retrieval techniques to quickly identify the most relevant tokens to attend to.

---

#### MonarchAttention: Zero-Shot Conversion to Fast, Hardware-Aware Structured Attention 🔦 **SPOTLIGHT**

- **Link:** [Paper](https://ece.engin.umich.edu/stories/fifteen-papers-by-ece-researchers-at-neurips-2025)
- **Authors:** Can Yaras, Alec Xu, Pierre Abillama, Changwoo Lee, Laura Balzano (University of Michigan)
- **Summary:** Attention's quadratic complexity remains a fundamental barrier to efficient long-context inference. MonarchAttention achieves sub-quadratic attention approximation via Monarch matrices, reducing computational complexity to **O(n√n)** and memory/IO complexity to **O(n)**. The method enables zero-shot conversion of existing models without retraining. Hardware-optimized kernels achieve **1.6× speedup** for shorter sequences, **2.0× for medium-length**, and **2.4× for longer sequences** over FlashAttention-2.

---

## 3. Algorithm

### 3.1 Inference-Time Optimization

#### Certaindex: Efficiently Scaling LLM Reasoning with Certaindex

- **Link:** [arXiv:2412.20993](https://arxiv.org/abs/2412.20993)
- **Authors:** UC Berkeley
- **Summary:** Certaindex introduces an algorithm-agnostic metric signaling when further computation won't change results, reducing computational waste on stabilized answers across different reasoning algorithms. This enables adaptive compute allocation for inference-time reasoning.

---

#### Thinkless: LLM Learns When to Think

- **Link:** [arXiv:2505.13379](https://arxiv.org/abs/2505.13379) | [OpenReview](https://openreview.net/forum?id=ariVQf0KZx)
- **Summary:** Thinkless teaches LLMs to adaptively decide when deep reasoning is necessary versus when quick responses suffice, optimizing inference-time compute allocation by avoiding unnecessary chain-of-thought reasoning for simple queries.

---

### 3.2 KV Cache Algorithms

#### KeyDiff: Key Similarity-Based KV Cache Eviction for Long-Context LLM Inference

- **Link:** [OpenReview](https://openreview.net/pdf/461c983f15c05127a5a4f31daa7628230411afbb.pdf)
- **Summary:** KeyDiff leverages the geometric properties of key vectors in the attention mechanism for KV cache eviction. The authors observe a strong correlation between the geometric distinctiveness of keys (low cosine similarity with other keys) and their attention scores. Unlike attention-based eviction which requires computing attention scores (expensive), KeyDiff evicts redundant keys based on **similarity measures**—an attention-free approach that can be computed efficiently. This method allows processing of arbitrarily long prompts on resource-constrained devices (edge processors, mobile phones) by maintaining a fixed cache budget while retaining semantic diversity. It outperforms previous methods like H2O by preserving "surprising" information that simple attention accumulation might miss.

---

#### PrefixKV: Adaptive Prefix KV Cache is What Vision Instruction-Following Models Need for Efficient Generation

- **Link:** [arXiv:2412.03409](https://arxiv.org/abs/2412.03409) | [GitHub](https://github.com/THU-MIG/PrefixKV)
- **Authors:** Tsinghua University
- **Summary:** PrefixKV reframes layer-wise KV cache allocation as a global prefix configuration search problem. Using binary search-based adaptive layer-wise retention, it achieves state-of-the-art performance on LLaVA vision-language models by optimizing cache allocation across layers rather than applying uniform compression.

---

#### Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation for Efficient LLM Inference

- **Link:** [NeurIPS](https://neurips.cc/virtual/2025/loc/san-diego/poster/115578) | [arXiv:2407.11550](https://arxiv.org/abs/2407.11550) | [GitHub](https://github.com/FFY0/AdaKV)
- **Authors:** Yuan Feng, Junlin Lv, Yukun Cao, Xike Xie, S Kevin Zhou
- **Summary:** This paper addresses the memory bottleneck in LLM inference caused by growing KV cache as batch size, context length, or model size increases. Existing KV cache compression methods allocate compression budgets uniformly across all attention heads, ignoring their varying importance patterns. Ada-KV proposes optimizing KV cache eviction by **adaptive budget allocation** that considers the heterogeneous behavior of different attention heads. By identifying critical KV cache entries from an output perturbation perspective, the method achieves significant memory reduction while preserving generation quality.

---
