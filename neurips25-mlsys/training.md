# NeurIPS 2025 papers GenAI training efficiency
  
## LLM pretraining efficiency advances
 

### Arnold: Efficient Pre-Training via Topology-Aware Communication Alignment on 9600+ GPUs
**Link:** https://neurips.cc/virtual/2025/loc/san-diego/poster/115232  
**Authors:** Guoliang He, Youhe Jiang, Wencong Xiao, Jiang Kaihua, Shuguang Wang, Jun Wang, Du Zixian, Zhuo Jiang, Xinlei Zhang, Binhang Yuan, Eiko Yoneki

At hyperscale, communication becomes the dominant bottleneck. Arnold is a scheduling system that aligns LLM communication patterns to datacenter topology, addressing bandwidth contention during sparse, high-volume communication bursts. The system reduces the maximum spread of communication groups by up to **1.67×** and improves end-to-end training performance by **10.6%** when training on more than **9,600 Hopper GPUs**. This represents critical infrastructure for trillion-parameter model training.

### Communication-Efficient Language Model Training Scales Reliably: Scaling Laws for DiLoCo
**Link:** https://openreview.net/pdf?id=X4SCxcgb3O  
**Paper**: https://arxiv.org/abs/2505.06371  
**Authors:** Zachary Charles, Gabriel Teston, Lucio Dery, Keith Rush, Nova Fallen, Zachary Garrett, Arthur Szlam, Arthur Douillard (Google)

DiLoCo (Distributed Low-Communication) training fundamentally changes distributed training dynamics by reducing communication frequency by orders of magnitude. This paper develops scaling laws predicting (1) evaluation loss as a function of model size and (2) optimal hyperparameter choices—eliminating expensive hyperparameter sweeps. The work demonstrates that communication-efficient training scales reliably with model size, enabling practical training across geographically distributed clusters with limited interconnect bandwidth.

### ACCO: Accumulate While You Communicate for Communication-Overlapped Sharded LLM Training
**Link:** https://hal.science/hal-04592562v3/file/neurips_acco.pdf  
**Authors:** Adel Nabli, Louis Fournier, Pierre Erbacher, Louis Serrano, Eugene Belilovsky, et al.

Modern LLM training requires sharded partitioning due to memory constraints, but synchronizing gradients and optimizer states often exceeds computation time. ACCO overlaps gradient computation and communication while partitioning optimizer states, introducing a two-stage compensation mechanism that corrects for delayed updates without warmup requirements. The method provides provable convergence guarantees matching standard SGD while delivering significant speedups in bandwidth-constrained scenarios.

### Scaling Smart: Accelerating LLM Pre-training with Small Model Initialization
**Link:** https://machinelearning.apple.com/research/scaling-smart  
**Authors:** Mohammad Samragh, Iman Mirzadeh, Keivan Alizadeh Vahid, Fartash Faghri, Minsik Cho, Moin Nabi, Devang Naik, Mehrdad Farajtabar (Apple)

Current scaling trends make training large models from random initialization extremely costly. This paper demonstrates that initializing large language models using smaller pre-trained models significantly accelerates pre-training by transferring learned representations. The approach enables faster convergence to target performance levels, reducing the time and compute required to train larger models.

### CompleteP: Depth-wise Hyperparameter Transfer for Deep Model Training
**Link:** https://vectorinstitute.ai/  
**Authors:** Nolan Dey, Bin Zhang, Lorenzo Noci, Mufan Li, Blake Bordelon, Shane Bergsma, Cengiz Pehlevan, Boris Hanin, Joel Hestness

Some parameterizations fail to transfer optimal hyperparameters (especially learning rate) across changes in model depth. CompleteP enables depth-wise HP transfer, providing FLOP savings when training deep models and expanding the range of compute-efficient width/depth ratios. This addresses a key practical challenge in scaling up LLM training.

### Benchmarking Optimizers for Large Language Model Pretraining
**Link:** https://openreview.net/pdf/973277f0cf8990c1a0f245f20103a01a8a9476a8.pdf  
**Authors:** NeurIPS 2025 Accepted

The first large-scale controlled benchmark of **11 optimization methods** for LLM pretraining across various model sizes, batch sizes, and training iterations. Key findings: many methods can outperform AdamW when properly tuned; optimizer sensitivity changes with scale; methods like Lion and SOAP can match or exceed AdamW performance with proper tuning. The paper open-sources a benchmarking toolkit.

### AdaLRS: Loss-Guided Adaptive Learning Rate Search for Efficient Foundation Model Pretraining
**Link:** https://neurips.cc/virtual/2025/poster/118011  
**Authors:** NeurIPS 2025 Poster

Suboptimal learning rates waste substantial compute in LLM pretraining. AdaLRS adaptively searches for learning rates during training guided by loss signals, eliminating expensive offline hyperparameter sweeps. This is particularly important for large-scale pretraining where each training run costs millions of dollars.

 
---

## Memory-efficient training methods


### Harmony in Divergence: Fast, Accurate, and Memory-efficient Zeroth-order LLM Fine-tuning
**Link:** https://neurips.cc/Downloads/2025  
**Authors:** Qitao Tan, Jun Liu, Zheng Zhan, Caiwei Ding, Yanzhi Wang, Jin Lu, Geng Yuan

Presents zeroth-order optimization for LLM fine-tuning achieving improvements across speed, accuracy, and memory efficiency. Zeroth-order methods avoid backpropagation entirely by estimating gradients through forward passes only, dramatically reducing memory requirements for fine-tuning on resource-constrained hardware.

### TERAIO: Cost-Efficient LLM Training with Lifetime-Aware Tensor Offloading via GPUDirect Storage
**Link:** https://neurips.cc/virtual/2025   

Active tensors take only ~**1.7%** of allocated GPU memory per training iteration. TERAIO accurately estimates tensor lifetime through profiling, generates optimized offloading/prefetching plans, and uses GPUDirect storage for direct tensor migration between GPUs and SSDs, maximizing bandwidth utilization for memory-constrained large model training.

---
  
## Attention mechanisms and architectural efficiency

### FlashBias: Fast Computation of Attention with Bias
**Link:** https://github.com/thuml/FlashBias | arXiv: 2505.12044  
**Authors:** Haixu Wu, Minghao Guo, Yuezhou Ma, Yuanxu Sun, Jianmin Wang, Wojciech Matusik, Mingsheng Long (Tsinghua/MIT)

Extends FlashAttention to efficiently handle attention with bias matrices (used for spatial/positional priors). Provides three implementations (Triton, PyTorch-SDPA, CuTE-based) achieving significant memory and runtime reduction. Applicable to GPT-2, Swin Transformer, Transformer PDE solvers, and AlphaFold 3.

### DuSA: Dual-Stage Sparse Attention Accelerating Both Training and Inference
**Link:** https://neurips.cc/virtual/2025/loc/san-diego/calendar  
 
Dual-stage sparse attention mechanism accelerating both training and inference. Identifies and focuses computation on important attention patterns while pruning less informative connections, achieving significant speedups without accuracy degradation.

### MonarchAttention: Sub-quadratic Attention via Monarch Matrices
**Link:** https://ece.engin.umich.edu/stories/fifteen-papers-by-ece-researchers-at-neurips-2025  
**Authors:** University of Michigan ECE Researchers

Novel sub-quadratic attention approximation using Monarch matrices—an expressive class of structured matrices. MonarchAttention is both transferable (minimal performance loss without additional training) and hardware-efficient (utilizing highest-throughput tensor core units), providing practical acceleration for LLM training.
 
---

## Distributed and decentralized training

### Revisiting 1-peer Exponential Graph for Enhancing Decentralized Learning Efficiency
**Link:** https://group.ntt/en/topics/2025/12/02/neurips2025.html  
**Authors:** Kenta Niwa, Yuki Takezawa, Guoqiang Zhang, W. Bastiaan Kleijn (NTT Communications Science Laboratories)

Novel communication patterns for decentralized learning allowing machines to flexibly change peers while keeping communication balanced. Enables faster and more accurate training under limited communication rounds—fundamental building blocks for efficient large-scale distributed training.

### Communication Efficient Distributed Training with Distributed Lion
**Link:** https://neurips.cc/proceedings  
**Authors:** NeurIPS 2025 Accepted

Achieves performance comparable to standard Lion or AdamW optimizers on aggregated gradients but with significantly reduced communication bandwidth, enabling efficient distributed training.

### Exact and Linear Convergence for Federated Learning Under Arbitrary Client Participation
**Link:** https://openreview.net/pdf?id=TeocEZCWnr  
**Authors:** Bicheng Ying, Zhe Li, HaiboYang (Google)

Proves that exact and linear convergence is attainable in federated learning even under arbitrary client participation patterns, enabling more flexible and communication-efficient distributed training for generative models.

---

## Energy efficiency and sustainable AI

### Energy and Power as First-Class ML Design Metrics (Tutorial)
**Link:** https://ml.energy/tutorials/neurips25/  
**Speakers:** Jae-Won Chung (University of Michigan), Ahmet Inci (NVIDIA), Ruofan Wu

Comprehensive tutorial covering practical energy measurement techniques, power & energy as computing resources, and optimization methods from kernels to clusters. Collaboration between The ML.ENERGY Initiative and NVIDIA addresses energy as the ultimate bottleneck for scaling AI.

### Carbon Literacy for Generative AI: Visualizing Training Emissions
**Link:** https://openreview.net/forum?id=ZhosUbcpuJ  
**Authors:** 5th Muslims in ML Workshop

Compiles reported and estimated carbon emissions for **13 state-of-the-art models** (2018-2024) during training. Translates emissions to human-friendly equivalences (trees required for absorption, per-capita footprints), advancing sustainable AI practice.
 