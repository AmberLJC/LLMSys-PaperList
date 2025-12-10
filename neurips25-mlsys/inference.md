# Inference and Serving Papers 

### HyGen: Efficient LLM Serving via Elastic Online-Offline Request Co-location

**Authors:** Ting Sun, Penghan Wang, Fan Lai (UIUC, Purdue)

**Link:** [OpenReview PDF](https://openreview.net/pdf/40f47181fc0983cc4ba51054de34b5d6a4e75e14.pdf)

Production LLM clusters typically separate online serving (latency-sensitive) from offline batch processing (throughput-optimized), leading to underutilization during traffic valleys. HyGen enables interference-aware co-location of these workloads through a latency predictor for batch execution time estimation, SLO-aware profiler for interference quantification, and adaptive scheduler. The system achieves **3.87-5.84× throughput gains** over baselines while ensuring latency SLOs for online requests. This represents significant cost reduction for organizations operating large-scale LLM infrastructure.

### EasySpec: Layer-Parallel Speculative Decoding for Efficient Multi-GPU Utilization

**Authors:** Yize Wu et al.

**Link:** [arXiv:2502.02493](https://arxiv.org/abs/2502.02493)

Multi-GPU inference typically achieves poor utilization during the sequential decode phase. EasySpec introduces layer-parallel speculation that enables multi-layer parallelization across devices, achieving **peak speedup of 4.17×** compared to vanilla decoding while preserving the original distribution (lossless). The drafting stage is accelerated by up to **1.62×** through better GPU utilization. This approach is particularly valuable for large models that require tensor parallelism but suffer from communication overhead during generation.

### DFloat11: Lossless LLM Compression for Efficient GPU Inference via Dynamic-Length Float

**Title:** "70% Size, 100% Accuracy: Lossless LLM Compression for Efficient GPU Inference"

**Link:** [NeurIPS 2025 Poster](https://neurips.cc/virtual/2025/poster/115225)

Unlike lossy quantization methods that trade accuracy for size, DFloat11 achieves **~30% model size reduction while producing bit-for-bit identical outputs**. The approach uses entropy coding with dynamic-length floating-point representations and custom GPU kernels with hierarchical lookup tables for fast online decompression. Critically, this enables **lossless inference of Llama 3.1 405B (810GB) on 8×80GB GPUs** without any quality compromise. Compared to CPU offloading alternatives, DFloat11 achieves **2.3-46.2× higher throughput**, making it practical for organizations that cannot tolerate any quality degradation from quantization.

---

## Quantization research pushes to extreme compression ratios

### Q-Palette: Fractional-Bit Quantizers for Optimal Bit Allocation

**Authors:** Deokjae Lee, Hyun Oh Song (Seoul National University)

**Link:** [arXiv:2509.20214](https://arxiv.org/abs/2509.20214) | [GitHub](https://github.com/snu-mllab/Q-Palette)

Traditional quantization uses integer bit-widths, leaving optimization potential unexplored. Q-Palette introduces a collection of fractional-bit quantizers ranging from sophisticated trellis-coded to simpler vector/scalar variants, all optimized with custom CUDA kernels. A novel mixed-scheme quantization (MSQ) framework jointly optimizes quantizer selection and layer fusion, achieving **36% inference speed improvement** at equivalent accuracy. This represents a more nuanced approach to the accuracy-efficiency tradeoff in quantized inference.

### LittleBit: Ultra Low-Bit Quantization via Latent Factorization

**Authors:** Banseok Lee, Dongkyu Kim et al. (Samsung Research)

**Link:** [arXiv:2506.13771](https://arxiv.org/abs/2506.13771) | [NeurIPS 2025 Poster](https://neurips.cc/virtual/2025/poster/115061)

Pushing quantization to extreme limits, LittleBit achieves **0.1 bits per weight (BPW)**—approximately **31× memory reduction**. For example, Llama2-13B compresses to under 0.9GB while maintaining reasonable accuracy. The method uses latent matrix factorization with binarized factors and multi-scale compensation including a novel Dual-SVID technique for quantization-aware training initialization. While not suitable for all use cases, this research establishes new lower bounds on achievable compression for LLMs.

### Efficient Hybrid Language Model Compression through Group-Aware SSM Pruning

**Link:** [NeurIPS 2025 Poster](https://neurips.cc/virtual/2025/poster/116240)

As hybrid architectures combining Transformers with State Space Models (like Mamba) gain traction, efficient compression methods must evolve. This paper introduces group-aware pruning specifically for Mamba layers, compressing Nemotron-H 8B to 4B with **up to 40× fewer training tokens** while achieving **~2× faster inference throughput** and surpassing similarly-sized models in accuracy. This work anticipates the need for inference optimization techniques tailored to emerging architecture families.

---

## Additional notable inference efficiency papers

Several other papers merit attention for specific contributions to the inference efficiency landscape:

- **Reasoning Path Compression** ([GitHub](https://github.com/jiwonsong-dev/ReasoningPathCompression)): Training-free method exploiting semantic sparsity in reasoning paths, achieving **4× KV cache compression** with only 1.2% accuracy drop and **1.68× throughput improvement** for 32K token generation.

- **Polar Sparsity** ([arXiv:2505.14884](https://arxiv.org/abs/2505.14884)): First demonstration that contextual sparsity scales to large batch sizes, with sparsity-aware GPU kernels achieving **up to 2.2× end-to-end speedups** across OPT, LLaMA-2&3, Qwen, and Mistral families.

- **HoliTom: Holistic Token Merging for Video LLMs** ([arXiv:2505.21334](https://arxiv.org/html/2505.21334v1)): Training-free framework reducing computational costs to **6.9% of original FLOPs** while preserving **99.1% performance** with **2.28× acceleration** in decoding throughput.

- **Loquetier: Virtualized Multi-LoRA Framework**: Unified framework for LoRA fine-tuning and inference in a single runtime, achieving **3.0× throughput** of top co-serving systems and **46.4× higher SLO attainment**.

- **Learned Prefix Caching (LPC)** ([NeurIPS 2025 Poster](https://neurips.cc/virtual/2025/poster/117662)): First learned method for LLM prefix cache eviction using conversational content analysis, achieving **18-47% reduction in required cache sizes** and **11% prefilling throughput improvement**.

---
## Speculative decoding 

### SuffixDecoding: Extreme Speculative Decoding for Emerging AI Applications (SPOTLIGHT)

**Authors:** Gabriele Oliaro, Zhihao Jia, Daniel Campos, Aurick Qiao (Carnegie Mellon University, Snowflake AI Research)

**Link:** [arXiv:2411.04975](https://arxiv.org/abs/2411.04975) | [Project Page](https://suffix-decoding.github.io/) | [GitHub](https://github.com/snowflakedb/ArcticInference)

Agentic AI applications present unique inference challenges—LLM-based agents submit repetitive requests in multi-agent pipelines and self-refinement loops, creating highly predictable token sequences that existing speculative decoding methods fail to exploit. SuffixDecoding addresses this by using efficient suffix trees to cache long token sequences from prompts and previous outputs, enabling adaptive speculation that extends beyond traditional draft model approaches. The method achieves up to **5.3× speedup** on agentic workloads like SWE-Bench and AgenticSQL, **outperforming model-based approaches like EAGLE-2/3 by 2.8×** and model-free approaches like Token Recycling by **1.9×**. Critically, SuffixDecoding requires only CPU memory—plentiful and underutilized on typical LLM serving nodes—making it highly practical for production deployments. The approach is particularly effective when acceptance likelihood is high, adaptively speculating more tokens, while conserving computation when opportunities are limited.

### Accelerating Diffusion LLMs via Adaptive Parallel Decoding (SPOTLIGHT)

**Authors:** Daniel Israel, Justus Mattern, Jae Hyun Lim, Guy Van den Broeck, Yisong Yue (UCLA, Caltech)

**Link:** [Paper PDF](https://starai.cs.ucla.edu/papers/IsraelNeurIPS25.pdf) | [OpenReview](https://openreview.net/forum?id=xwqTt26NJf)

Diffusion-based language models offer an alternative to autoregressive generation but have struggled to match the speed of speculative decoding in AR models. This paper introduces a method that dynamically adjusts the number of tokens sampled in parallel for diffusion LLMs by defining a multiplicative mixture between dLLM marginal probabilities and joint probability of sequences under a small auxiliary autoregressive model. The approach substantially accelerates diffusion LLMs with minimal quality degradation, demonstrating that parallel decoding strategies can be effectively adapted to non-autoregressive architectures. This work opens new possibilities for inference optimization in the growing family of diffusion-based generative models.

### SpecEdge: Scalable Edge-Assisted Serving Framework for Interactive LLMs (SPOTLIGHT)

**Authors:** Jinwoo Park, Seunggeun Cho, Dongsu Han (KAIST)

**Link:** [arXiv:2505.17052](https://arxiv.org/abs/2505.17052) | [GitHub](https://github.com/kaist-ina/specedge)

The concentration of LLM serving in expensive cloud GPU clusters limits accessibility and increases costs. SpecEdge reimagines the serving architecture by leveraging consumer-grade edge GPUs for cost-effective serving at scale. The framework splits workloads between edge devices and servers using speculative decoding, with edge devices handling draft generation while servers verify outputs. Through proactive edge drafting and pipeline-aware scheduling, SpecEdge achieves **2.22× server throughput improvement** and **11.24% lower latency** compared to server-only baselines. This hybrid architecture enables organizations to dramatically reduce serving costs while maintaining quality, representing a significant step toward democratizing LLM deployment.

### MonarchAttention: Zero-Shot Conversion to Fast, Hardware-Aware Structured Attention (SPOTLIGHT)

**Authors:** Can Yaras, Alec Xu, Pierre Abillama, Changwoo Lee, Laura Balzano (University of Michigan)

**Link:** [Paper](https://ece.engin.umich.edu/stories/fifteen-papers-by-ece-researchers-at-neurips-2025)

Attention's quadratic complexity remains a fundamental barrier to efficient long-context inference. MonarchAttention achieves sub-quadratic attention approximation via Monarch matrices, reducing computational complexity to **O(n√n)** and memory/IO complexity to **O(n)**. The method enables zero-shot conversion of existing models without retraining. Hardware-optimized kernels achieve **1.6× speedup** for shorter sequences, **2.0× for medium-length**, and **2.4× for longer sequences** over FlashAttention-2. This represents a practical path to faster inference without the training overhead typically required for efficient attention variants.

---