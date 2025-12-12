# NeurIPS 2025: LLM Training Systems

A curated list of NeurIPS 2025 papers focused on improving the efficiency, stability, and scalability of Large Language Model (LLM) training.

---

## Table of Contents

- [1. Systems](#1-systems)
  - [1.1 Distributed & Communication-Efficient Training](#11-distributed--communication-efficient-training)
  - [1.2 Memory-Efficient Training](#12-memory-efficient-training)
  - [1.3 Long-Context Training](#13-long-context-training)
  - [1.4 Compiler & Hardware Optimization](#14-compiler--hardware-optimization)
  - [1.5 Energy Efficiency & Sustainable AI](#15-energy-efficiency--sustainable-ai)
- [2. System-Algorithm Co-design](#2-system-algorithm-co-design)
  - [2.1 Training Stability & Architecture](#21-training-stability--architecture)
  - [2.2 Attention Mechanisms & Architectural Efficiency](#22-attention-mechanisms--architectural-efficiency)
  - [2.3 Reinforcement Learning for LLM Training](#23-reinforcement-learning-for-llm-training)
  - [2.4 Fine-Tuning & Parameter-Efficient Training](#24-fine-tuning--parameter-efficient-training)
- [3. Algorithm](#3-algorithm)
  - [3.1 Hyperparameter Scaling Laws](#31-hyperparameter-scaling-laws)
  - [3.2 Data Mixture & Scaling](#32-data-mixture--scaling)
  - [3.3 Training Numerical Stability](#33-training-numerical-stability)

---

## 1. Systems

### 1.1 Distributed & Communication-Efficient Training

#### Synergistic Tensor and Pipeline Parallelism

- **Link:** [arXiv:2510.27257](https://arxiv.org/abs/2510.27257)
- **Summary:** This paper presents a novel hybrid strategy that simultaneously eliminates both tensor parallelism (TP) and pipeline parallelism (PP) bubbles. The approach decouples forward and backward passes into fine-grained computation units, using "braided" composite sequences for near-complete TP bubble elimination and a "V-shape" PP schedule achieving balanced memory. Results show **12% throughput improvement for LLMs** and **16% for multimodal LLMs**, with an offloading variant for memory-constrained scenarios.

---

#### Arnold: Efficient Pre-Training via Topology-Aware Communication Alignment on 9600+ GPUs

- **Link:** https://neurips.cc/virtual/2025/loc/san-diego/poster/115232
- **Authors:** Guoliang He, Youhe Jiang, Wencong Xiao, Jiang Kaihua, Shuguang Wang, Jun Wang, Du Zixian, Zhuo Jiang, Xinlei Zhang, Binhang Yuan, Eiko Yoneki
- **Summary:** At hyperscale, communication becomes the dominant bottleneck. Arnold is a scheduling system that aligns LLM communication patterns to datacenter topology, addressing bandwidth contention during sparse, high-volume communication bursts. The system reduces the maximum spread of communication groups by up to **1.67×** and improves end-to-end training performance by **10.6%** when training on more than **9,600 Hopper GPUs**.

---

#### Communication-Efficient Language Model Training Scales Reliably: Scaling Laws for DiLoCo

- **Link:** https://openreview.net/pdf?id=X4SCxcgb3O | https://arxiv.org/abs/2505.06371
- **Authors:** Zachary Charles, Gabriel Teston, Lucio Dery, Keith Rush, Nova Fallen, Zachary Garrett, Arthur Szlam, Arthur Douillard (Google)
- **Summary:** DiLoCo (Distributed Low-Communication) training fundamentally changes distributed training dynamics by reducing communication frequency by orders of magnitude. This paper develops scaling laws predicting (1) evaluation loss as a function of model size and (2) optimal hyperparameter choices—eliminating expensive hyperparameter sweeps. The work demonstrates that communication-efficient training scales reliably with model size, enabling practical training across geographically distributed clusters with limited interconnect bandwidth.

---

#### ACCO: Accumulate While You Communicate for Communication-Overlapped Sharded LLM Training

- **Link:** https://hal.science/hal-04592562v3/file/neurips_acco.pdf
- **Authors:** Adel Nabli, Louis Fournier, Pierre Erbacher, Louis Serrano, Eugene Belilovsky, et al.
- **Summary:** Modern LLM training requires sharded partitioning due to memory constraints, but synchronizing gradients and optimizer states often exceeds computation time. ACCO overlaps gradient computation and communication while partitioning optimizer states, introducing a two-stage compensation mechanism that corrects for delayed updates without warmup requirements.
  - **Key Results:**
    - Up to **87% reduction** in learning time compared to ZeRO
    - Successfully enables both sharding optimizer states and heterogeneous hardware usage
    - Memory-efficient while hiding communication latency
    - Provable convergence guarantees matching standard SGD

---

#### SDP4Bit: Toward 4-bit Communication Quantization in Sharded Data Parallelism

- **Link:** https://neurips.cc/virtual/2024/poster/95323 | https://arxiv.org/abs/2410.15526
- **Authors:** Jinda Jia et al.
- **Summary:** In sharded data parallelism, communication of weights and gradients consumes 30-50% of total training time. SDP4Bit reduces communication to approximately 4 bits via two novel techniques: (1) Quantization on weight differences—exploits temporal redundancy, (2) Two-level gradient smooth quantization—preserves gradient fidelity. An algorithm-system co-design with runtime optimization minimizes compression overhead.
  - **Key Results:**
    - Negligible impact on training loss for GPT models up to 6.7B parameters
    - Up to **4.08× speedup** in end-to-end throughput on 128 GPUs
    - Theoretical convergence guarantees provided

---

#### PaRO: Partial Redundancy Optimizer

- **Link:** https://neurips.cc/virtual/2024/poster/96664
- **Summary:** Existing distributed training strategies provide limited optimization options. PaRO (Partial Redundancy Optimizer) refines model state partitioning by considering communication topology with two variants: **PaRO-DP** accelerates training through refined state partitioning and tailored training procedures; **PaRO-CC** speeds up collective communications by rearranging network topology.
  - **Key Results:**
    - Up to **266% speedup** over ZeRO-3
    - PaRO-CC boosts Megatron training by **17%**

---

#### Communication Efficient Distributed Training with Distributed Lion

- **Link:** https://neurips.cc/proceedings
- **Summary:** Achieves performance comparable to standard Lion or AdamW optimizers on aggregated gradients but with significantly reduced communication bandwidth, enabling efficient distributed training.

---

#### Partial Parameter Updates for Efficient Distributed Training

- **Link:** https://arxiv.org/abs/2509.22418 | https://chatpaper.com/paper/192263
- **Authors:** Apple Research
- **Summary:** This paper challenges the fundamental assumption of Distributed Data Parallel (DDP) training that every parameter must be updated and synchronized at every step. The "Frozen Slice" technique partitions model parameters into K disjoint subsets (slices), with each compute node assigned a specific slice to update. During backward pass, gradients are only computed for the active slice, with synchronization being sparse concatenation rather than All-Reduce.
  - **Key Results:**
    - Enables effective training on low-bandwidth clusters (Ethernet-connected clouds)
    - **47% reduction** in peak memory usage (optimizer states only for active slice)
    - Perplexity parity with full-update baselines on 1.3B parameter models using **15% fewer FLOPs**

---

#### Revisiting 1-peer Exponential Graph for Enhancing Decentralized Learning Efficiency

- **Link:** https://group.ntt/en/topics/2025/12/02/neurips2025.html
- **Authors:** Kenta Niwa, Yuki Takezawa, Guoqiang Zhang, W. Bastiaan Kleijn (NTT Communications Science Laboratories)
- **Summary:** Novel communication patterns for decentralized learning allowing machines to flexibly change peers while keeping communication balanced. Enables faster and more accurate training under limited communication rounds—fundamental building blocks for efficient large-scale distributed training.

---

#### Exact and Linear Convergence for Federated Learning Under Arbitrary Client Participation

- **Link:** https://openreview.net/pdf?id=TeocEZCWnr
- **Authors:** Bicheng Ying, Zhe Li, HaiboYang (Google)
- **Summary:** Proves that exact and linear convergence is attainable in federated learning even under arbitrary client participation patterns, enabling more flexible and communication-efficient distributed training for generative models.

---

### 1.2 Memory-Efficient Training

#### TERAIO: Cost-Efficient LLM Training with Lifetime-Aware Tensor Offloading via GPUDirect Storage

- **Link:** https://neurips.cc/virtual/2025
- **Summary:** Active tensors take only ~**1.7%** of allocated GPU memory per training iteration. TERAIO accurately estimates tensor lifetime through profiling, generates optimized offloading/prefetching plans, and uses GPUDirect storage for direct tensor migration between GPUs and SSDs, maximizing bandwidth utilization for memory-constrained large model training.

---

#### Harmony in Divergence: Fast, Accurate, and Memory-efficient Zeroth-order LLM Fine-tuning

- **Link:** https://neurips.cc/Downloads/2025
- **Authors:** Qitao Tan, Jun Liu, Zheng Zhan, Caiwei Ding, Yanzhi Wang, Jin Lu, Geng Yuan
- **Summary:** Presents zeroth-order optimization for LLM fine-tuning achieving improvements across speed, accuracy, and memory efficiency. Zeroth-order methods avoid backpropagation entirely by estimating gradients through forward passes only, dramatically reducing memory requirements for fine-tuning on resource-constrained hardware.

---

#### Breaking the Frozen Subspace: Importance Sampling for Low-Rank Optimization in LLM Pretraining

- **Link:** https://neurips.cc/Downloads/2025
- **Summary:** Low-rank optimization methods for memory-efficient training constrain updates to a fixed subspace, limiting expressivity. This paper introduces importance sampling to enable more effective exploration of the optimization landscape while maintaining memory efficiency.

---

### 1.3 Long-Context Training

#### InfiniPipe: Elastic Pipeline Parallelism for Long-Context LLM Training

- **Link:** [arXiv:2509.21275](https://arxiv.org/abs/2509.21275)
- **Summary:** For long-context training, the distribution of sequence lengths can be highly skewed. Standard Pipeline Parallelism assumes uniform computation, leading to imbalances when training on varied lengths. InfiniPipe introduces **Elastic Pipeline Parallelism (EPP)**, which orchestrates token-level and batch-level pipeline parallelism simultaneously. It employs a resource-aware sequence processor that splits long sequences and packs short ones dynamically. The system jointly optimizes the pipeline schedule and gradient checkpointing strategy, adapting to the heterogeneity of the workload. This allows for efficient training on datasets with extreme length variations, typical of code repositories or book corpora.

---

#### Hierarchical Balance Packing (HBP): Towards Efficient Supervised Fine-tuning for Long-Context LLMs

- **Link:** https://arxiv.org/abs/2503.07680
- **Authors:** Yongqiang Yao et al.
- **Summary:** Training LLMs with hybrid long-context and short-context data leads to workload imbalances: excessive padding, unequal workload distribution, and unnecessary communication overhead. HBP introduces multi-level data packing with three key components:
  1. **Hierarchical group auto-selection:** Determines optimal packing-length groups with corresponding sequence parallelism degree and gradient checkpointing configuration
  2. **Optimal sample assignment:** Assigns training samples to their optimal groups
  3. **Dynamic training pipeline:** Includes curriculum learning, adaptive sequential parallelism, and stable loss normalization
  - **Key Results:**
    - Significant reduction in Data Balance Ratio (DBR), Padding Ratio (PR), and Attention Balance Ratio (ABR)
    - Substantial improvements in training speed for 128K sequence length with 32 GPUs
    - Optimal SP/GC configurations vary for different sequence lengths (32K, 64K, 128K)

---

### 1.4 Compiler & Hardware Optimization

#### XgenSilicon: Hardware-Aware Neural Network Compilation with Learned Optimization

- **Link:** https://www.arxiv.org/pdf/2512.00031
- **Summary:** Designing compilers for custom accelerators (like RISC-V ASICs) typically involves manually tuning heuristics for loop tiling, unrolling, and memory scheduling. XgenSilicon replaces these heuristics with a Multi-Algorithm Learned Optimization Framework employing Bayesian Optimization, Genetic Algorithms, and Simulated Annealing to search the optimization space. Crucially, it uses a learned cost model updated via feedback from hardware validation, ensuring adaptation to specific silicon quirks.
  - **Key Results:**
    - Generated assembly code is **2.5-4.5× faster** than hand-tuned baselines
    - **3-6× less power consumption**
    - Enables automated compilation for custom AI accelerators

---

#### Autocomp: LLM-Driven Code Optimization for Tensor Accelerators

- **Link:** [arXiv:2505.18574](https://arxiv.org/abs/2505.18574) | [OpenReview](https://openreview.net/forum?id=9AkW2jw3HA)
- **Summary:** Autocomp represents a paradigm shift from heuristic-based compilers to **agentic compilers**. Optimizing code for specific tensor accelerators (TPUs, custom NPUs, loose-coupled accelerators) is notoriously difficult due to specialized ISAs and opaque memory hierarchies. Autocomp utilizes a multi-agent LLM system to replace manual kernel tuning: a **Planner Agent** selects high-level optimizations (tiling, loop unrolling, fusion), a **Generator Agent** translates plans into low-level DSL code (e.g., using Exo language), and a **Feedback Loop** compiles, measures performance and correctness on actual hardware, and refines the plan. The system generates kernels for GEMM and Convolution that are **5.6× and 2.7× faster** than vendor-provided libraries, and remarkably **outperforms expert hand-tuned code by 1.4×**.

---

#### DCC: Data-Centric Compilation for Processing-In-Memory

- **Link:** [arXiv:2511.15503](https://arxiv.org/abs/2511.15503)
- **Summary:** Processing-In-Memory (PIM) architectures are promising for bandwidth-bound workloads (like LLM decoding) but suffer from a "programmability wall"—PIM cores can often only access data in their local memory bank, requiring complex data layout permutations that standard compilers ignore. DCC is the **first data-centric compiler** that co-optimizes data layout and compute code. Instead of treating memory as a flat address space, it abstracts the PIM memory hierarchy and generates a schedule that explicitly minimizes data rearrangement costs. DCC achieves **2.7× - 5.75× speedups** over GPU-only execution for bandwidth-heavy kernels on HBM-PIM and AttAcc architectures, essential for enabling PIM as a viable competitor to HBM-equipped GPUs for inference tasks.

---

#### Analog Foundation Models

- **Link:** [IBM Research](https://research.ibm.com/publications/analog-foundation-models) | [GitHub](https://github.com/IBM/analog-foundation-models)
- **Authors:** IBM Research
- **Summary:** This paper presents the first method enabling state-of-the-art LLMs on analog in-memory computing hardware. Phi-3-mini and Llama-3.2-1B retain performance comparable to 4-bit weight, 8-bit activation baselines with better test-time compute scaling—a pathway toward energy-efficient foundation models on specialized hardware.

---

#### Analog In-memory Training on General Non-ideal Resistive Elements

- **Link:** [IBM Research](https://research.ibm.com/publications/analog-in-memory-training-on-general-non-ideal-resistive-elements-the-impact-of-response-functions)
- **Authors:** IBM Research
- **Summary:** This paper proposes residual learning algorithms for training on non-ideal resistive memory devices (ReRAM, PCM), provably converging to critical points through bilevel optimization. Addresses practical challenges of analog hardware deployment.

---

#### MoE-CAP: Benchmarking Cost, Accuracy and Performance of Sparse Mixture-of-Experts Systems

- **Link:** [arXiv:2412.07067](https://arxiv.org/abs/2412.07067) | [OpenReview](https://openreview.net/forum?id=k2fWVhG0u5)
- **Authors:** Microsoft Research
- **Summary:** MoE-CAP provides comprehensive evaluation of cost, accuracy, and performance tradeoffs for Mixture-of-Experts systems—essential for MoE deployment decisions in production environments.

---

### 1.5 Energy Efficiency & Sustainable AI

#### Energy and Power as First-Class ML Design Metrics (Tutorial)

- **Link:** https://ml.energy/tutorials/neurips25/
- **Speakers:** Jae-Won Chung (University of Michigan), Ahmet Inci (NVIDIA), Ruofan Wu
- **Summary:** Comprehensive tutorial covering practical energy measurement techniques, power & energy as computing resources, and optimization methods from kernels to clusters. Collaboration between The ML.ENERGY Initiative and NVIDIA addresses energy as the ultimate bottleneck for scaling AI.

---

#### Carbon Literacy for Generative AI: Visualizing Training Emissions

- **Link:** https://openreview.net/forum?id=ZhosUbcpuJ
- **Authors:** 5th Muslims in ML Workshop
- **Summary:** Compiles reported and estimated carbon emissions for **13 state-of-the-art models** (2018-2024) during training. Translates emissions to human-friendly equivalences (trees required for absorption, per-capita footprints), advancing sustainable AI practice.

---

## 2. System-Algorithm Co-design

### 2.1 Training Stability & Architecture

#### Gated Attention for Large Language Models 🏆 **BEST PAPER AWARD**

- **Link:** https://arxiv.org/abs/2505.06708 | https://neurips.cc/virtual/2025/loc/san-diego/poster/120216
- **Authors:** Zihan Qiu, Zekun Wang, Bo Zheng, Zeyu Huang, Kaiyue Wen, Songlin Yang, Rui Men, Le Yu, Fei Huang, Suozhi Huang, Dayiheng Liu, Jingren Zhou, Junyang Lin (Qwen Team)
- **Summary:** Training large language models at scale suffers from loss spikes—sudden divergences requiring training restarts or checkpoint rollbacks. The authors introduce Gated Attention, applying a learnable, input-dependent sigmoid gate immediately after Scaled Dot-Product Attention (SDPA). The gate modulates attention output Y with σ(XWθ), introducing element-wise sparsity and non-linearity before the final projection.
  - **Key Results:**
    - Validated on 1.7B dense models and 15B MoE models trained on up to 3.5 trillion tokens
    - Eliminates loss spikes, enabling smooth convergence curves
    - Tolerates larger learning rates (4.0×10⁻³ → 4.5×10⁻³), accelerating convergence
    - Less than **2% wall-time latency overhead**
    - Eliminates attention sink phenomenon
    - Already integrated into Qwen3-Next production models

---

#### AlphaDecay: Module-wise Weight Decay for Heavy-Tailed Balancing in LLMs

- **Link:** https://neurips.cc/Downloads/2025
- **Summary:** Standard weight decay applies uniform regularization across all model parameters, ignoring that different modules in LLMs exhibit vastly different activation distributions. AlphaDecay introduces module-wise weight decay that accounts for heavy-tailed activation distributions in different network components, applying adaptive regularization strength based on local statistics.

---

#### Scaling Smart: Accelerating LLM Pre-training with Small Model Initialization

- **Link:** https://machinelearning.apple.com/research/scaling-smart
- **Authors:** Mohammad Samragh, Iman Mirzadeh, Keivan Alizadeh Vahid, Fartash Faghri, Minsik Cho, Moin Nabi, Devang Naik, Mehrdad Farajtabar (Apple)
- **Summary:** Current scaling trends make training large models from random initialization extremely costly. This paper demonstrates that initializing large language models using smaller pre-trained models significantly accelerates pre-training by transferring learned representations. The approach enables faster convergence to target performance levels, reducing the time and compute required to train larger models.

---

#### CompleteP: Depth-wise Hyperparameter Transfer for Deep Model Training

- **Link:** https://vectorinstitute.ai/
- **Authors:** Nolan Dey, Bin Zhang, Lorenzo Noci, Mufan Li, Blake Bordelon, Shane Bergsma, Cengiz Pehlevan, Boris Hanin, Joel Hestness
- **Summary:** Some parameterizations fail to transfer optimal hyperparameters (especially learning rate) across changes in model depth. CompleteP enables depth-wise HP transfer, providing FLOP savings when training deep models and expanding the range of compute-efficient width/depth ratios.

---

### 2.2 Attention Mechanisms & Architectural Efficiency

#### FlashBias: Fast Computation of Attention with Bias

- **Link:** https://github.com/thuml/FlashBias | https://arxiv.org/abs/2505.12044
- **Authors:** Haixu Wu, Minghao Guo, Yuezhou Ma, Yuanxu Sun, Jianmin Wang, Wojciech Matusik, Mingsheng Long (Tsinghua/MIT)
- **Summary:** Extends FlashAttention to efficiently handle attention with bias matrices (used for spatial/positional priors). Provides three implementations (Triton, PyTorch-SDPA, CuTE-based) achieving significant memory and runtime reduction. Applicable to GPT-2, Swin Transformer, Transformer PDE solvers, and AlphaFold 3.

---

#### DuSA: Dual-Stage Sparse Attention Accelerating Both Training and Inference

- **Link:** https://neurips.cc/virtual/2025/loc/san-diego/calendar
- **Summary:** Dual-stage sparse attention mechanism accelerating both training and inference. Identifies and focuses computation on important attention patterns while pruning less informative connections, achieving significant speedups without accuracy degradation.

---

#### MoBA: Mixture of Block Attention for Long-Context LLMs

- **Link:** [arXiv:2502.13189](https://arxiv.org/abs/2502.13189) | [GitHub](https://github.com/MoonshotAI/MoBA)
- **Authors:** Moonshot AI
- **Summary:** MoBA presents block-based attention routing without predefined biases, enabling efficient long contexts with mixture-of-experts style attention allocation. The approach dynamically routes queries to relevant key-value blocks, reducing computation while maintaining quality for extended context processing.

---

#### MonarchAttention: Sub-quadratic Attention via Monarch Matrices

- **Link:** https://ece.engin.umich.edu/stories/fifteen-papers-by-ece-researchers-at-neurips-2025
- **Authors:** University of Michigan ECE Researchers
- **Summary:** Novel sub-quadratic attention approximation using Monarch matrices—an expressive class of structured matrices. MonarchAttention is both transferable (minimal performance loss without additional training) and hardware-efficient (utilizing highest-throughput tensor core units), providing practical acceleration for LLM training.

---

### 2.3 Reinforcement Learning for LLM Training

#### DAPO: An Open-Source LLM Reinforcement Learning System at Scale

- **Link:** https://arxiv.org/abs/2503.14476 | https://neurips.cc/virtual/2025/loc/san-diego/calendar
- **Authors:** ByteDance Seed Team
- **Summary:** Key technical details of state-of-the-art reasoning LLMs (like OpenAI o1 and DeepSeek R1) are concealed, preventing the community from reproducing RL training results. DAPO (Decoupled Clip and Dynamic Sampling Policy Optimization) introduces four key techniques:
  1. **Clip-Higher:** Promotes diversity and avoids entropy collapse by adjusting the upper clip threshold
  2. **Dynamic Sampling:** Improves training efficiency and stability
  3. **Token-Level Policy Gradient Loss:** Critical for long chain-of-thought RL scenarios
  4. **Overlong Reward Shaping:** Reduces reward noise and stabilizes training
  - **Key Results:**
    - Achieves **50 points on AIME 2024** using Qwen2.5-32B base model
    - Outperforms DeepSeek-R1-Zero-Qwen-32B using only **50% of training steps**
    - Fully open-sourced: training code, verl framework implementation, and curated dataset

---

#### DAPO: Improving Multi-Step Reasoning Abilities with Direct Advantage-Based Policy Optimization

- **Link:** https://neurips.cc/virtual/2025/loc/san-diego/calendar
- **Summary:** Response-level RL methods (DPO, GRPO) apply uniform updates to all tokens, which is suboptimal for multi-step reasoning where individual steps contribute differently. DAPO introduces step-level offline RL with theoretical guarantees, providing fine-grained credit assignment for reasoning chains.

---

### 2.4 Fine-Tuning & Parameter-Efficient Training

#### Loquetier: Virtualized Multi-LoRA Framework

- **Link:** https://cs.nju.edu.cn/lm/en/post/2025-10-11-neurips-2025-accepted-papers/
- **Authors:** Nanjing University
- **Summary:** Existing systems fail to unify LoRA fine-tuning and inference serving efficiently. Loquetier provides: (1) a virtualization module that isolates PEFT-based model modifications, supporting multiple adapters on a shared base model; (2) fused computational kernels that integrate fine-tuning and inference paths in forward propagation.
  - **Key Results:**
    - **3.0× throughput improvement** in inference-only scenarios
    - **46.4× higher SLO attainment** in unified fine-tuning + inference workloads

---

#### GainLoRA: Gated Integration of Low-Rank Adaptation for Continual Learning

- **Link:** [arXiv:2505.15424](https://arxiv.org/abs/2505.15424)
- **Authors:** Nanjing University Large Model Center
- **Summary:** Existing LoRA-based continual learning methods expand new branches while freezing old ones, then use simple addition for integration. GainLoRA introduces gated integration that dynamically balances new and old LoRA branch contributions based on task requirements.

---

#### FVAE-LoRA: Latent Space Factorization in LoRA

- **Link:** [arXiv:2510.19640](https://arxiv.org/abs/2510.19640) | [NeurIPS Poster](https://neurips.cc/virtual/2025/loc/san-diego/poster/119339)
- **Summary:** FVAE-LoRA introduces latent space factorization techniques to improve LoRA fine-tuning. By learning factorized representations in the low-rank adaptation space, the method achieves better parameter efficiency and adaptation quality across various downstream tasks.

---

## 3. Algorithm

### 3.1 Hyperparameter Scaling Laws

#### Benchmarking Optimizers for Large Language Model Pretraining

- **Link:** https://openreview.net/pdf/973277f0cf8990c1a0f245f20103a01a8a9476a8.pdf
- **Summary:** The first large-scale controlled benchmark of **11 optimization methods** for LLM pretraining across various model sizes, batch sizes, and training iterations. Key findings: many methods can outperform AdamW when properly tuned; optimizer sensitivity changes with scale; methods like Lion and SOAP can match or exceed AdamW performance with proper tuning. The paper open-sources a benchmarking toolkit.

---

#### AdaLRS: Loss-Guided Adaptive Learning Rate Search for Efficient Foundation Model Pretraining

- **Link:** https://neurips.cc/virtual/2025/poster/118011
- **Summary:** Suboptimal learning rates waste substantial compute in LLM pretraining. AdaLRS adaptively searches for learning rates during training guided by loss signals, eliminating expensive offline hyperparameter sweeps. Enables training continuation without loss penalty and more flexible compute budget allocation.

---

#### Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training

- **Link:** https://neurips.cc/virtual/2025/poster/117191
- **Summary:** Establishes precise scaling laws for weight decay and batch size:
  - Optimal λ scales linearly with batch size B for fixed N,D
  - Optimal timescale B/(ηλD) follows a power law in tokens-per-parameter ratio D/N
  - Both optimal batch size (Bopt) and critical batch size (Bcrit) scale as power laws in D, independent of N
- Provides a method to predict λ_opt before large-scale training begins and enables Pareto-optimal selection of N and D under dual training time and compute objectives.

---

### 3.2 Data Mixture & Scaling

#### Scaling Laws for Optimal Data Mixtures (Apple Research)

- **Link:** https://machinelearning.apple.com/research/neurips-2025
- **Authors:** Apple Research
- **Summary:** Data mixture—the proportion of each domain used in training—critically impacts model performance, but the standard approach relies on costly trial-and-error. This paper presents a systematic method using scaling laws to determine optimal data mixture for any target domain:
  - Scaling laws predict loss as a function of model size N, training tokens D, and domain weights
  - Laws are universal across LLMs, native multimodal models (NMMs), and large vision models (LVMs)
  - Parameters can be estimated from small-scale runs and extrapolated to larger scales
  - **Key Results:**
    - Practitioners can derive optimal domain weights for any target domain under given training budget
    - Provides a principled alternative to trial-and-error methods
    - Validated on large-scale pretraining runs

---

#### GRAPE: The Best Instruction-Tuning Data are Those That Fit

- **Link:** [arXiv:2502.04194](https://arxiv.org/abs/2502.04194) | [OpenReview PDF](https://openreview.net/pdf?id=4jFSekBaDT)
- **Summary:** GRAPE presents a principled approach to instruction-tuning data selection. The paper argues that the best instruction-tuning data are those that fit the model's current capabilities, providing methods to identify and select optimal training examples for efficient fine-tuning.

---

### 3.3 Training Numerical Stability

#### Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference 🎤 **ORAL**

- **Link:** https://openreview.net/forum?id=Q3qAsZAEZw | https://arxiv.org/abs/2506.09501
- **Authors:** Jiayi Yuan, Hao Li, Xinheng Ding, and collaborators
- **Summary:** Floating-point non-associativity causes reproducibility issues in LLM training and inference. Changes in batch size, GPU count, or GPU type can cause up to **9% accuracy variation** in reasoning models. LayerCast stores weights in FP16 while computing in FP32, mitigating numerical nondeterminism without significant performance overhead.

---
