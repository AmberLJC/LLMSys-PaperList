
# NeurIPS 2025: Efficient Reinforcement Learning for GenAI


## Policy Optimization Methods

### 1. Adaptive Batch-Wise Sample Scheduling for Direct Preference Optimization
- **Link:** https://neurips.cc/virtual/2025/poster/119641
- **Summary:** This paper addresses the efficiency of DPO training through intelligent data selection. The authors propose adaptive batch-wise sample scheduling that dynamically selects training samples based on their learning signal quality. This approach improves DPO convergence and final alignment quality while reducing the number of training iterations needed.
   
### 2. Flow-GRPO: Training Flow Matching Models via Online RL
- **Link:** https://github.com/yifan123/flow_grpo | arXiv: https://arxiv.org/abs/2505.05470
- **Authors:** Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, Wanli Ouyang
- **Summary:** Flow-GRPO extends GRPO to flow matching models for efficient text-to-image generation RL training. Key efficiency innovations include ODE-to-SDE conversion enabling stochastic sampling for RL, and Flow-GRPO-Fast which trains on only partial timesteps, reducing training costs dramatically.

### 3. Act Only When It Pays: Efficient RL for LLM Reasoning via Selective Rollouts
- **Link:** https://neurips.cc/Downloads/2025
- **Summary:** This paper proposes selective rollout strategies for RL training of LLMs, where training compute is focused on examples where it will have the most impact. By avoiding unnecessary rollouts on easy or already-mastered examples, the method improves training efficiency.

### 4. Hogwild! Inference: Parallel LLM Generation via Concurrent Attention
- **Link:** https://neurips.cc/virtual/2025/events/spotlights-2025
- **Summary:** This paper proposes a fundamentally new approach to parallel inference where multiple LLM "workers" run simultaneously with a **shared attention cache**, enabling instant access to each other's generated tokens. The problem addressed is that rollout generation is often the primary bottleneck in RL training pipelines (PPO, GRPO), where generating many samples per prompt consumes significant wall-clock time. By leveraging RoPE's mathematical properties to avoid recomputation while improving hardware utilization, Hogwild! Inference accelerates sampling without requiring additional fine-tuning or model modifications. The concurrent cache allows workers to develop implicit collaboration strategies, reducing total generation time for batch rollouts. This is directly applicable to RL training where faster rollout generation translates to faster training iterations and more efficient use of GPU resources.

---

## RL Training Systems and Infrastructure

### 5. AREAL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning
- **Link:** https://neurips.cc/virtual/2025/poster/117538
- **Summary:** AREAL addresses the systems challenge of scaling RL for LLM reasoning by introducing asynchronous training where samplers and trainers operate on different GPU pools, eliminating the synchronization overhead that causes GPU idle time. This is particularly important for reasoning tasks with heterogeneous response lengths. The asynchronous design achieves near-linear scaling to large GPU clusters.

### 6. Long-RL: Scaling RL to Long Sequences (LongVILA-R1)
- **Link:** https://github.com/NVlabs/Long-RL
- **Authors:** Yukang Chen, Wei Huang, Baifeng Shi, Qinghao Hu, et al. (NVIDIA, UC Berkeley, HKU)
- **Summary:** Long-RL addresses hour-level video RL training (3,600 frames = 256k tokens) on a single node. The key innovation is Multi-modal Reinforcement Sequence Parallelism (MR-SP), which incorporates sequence parallelism and a vLLM-based engine with cached video embeddings for efficient rollout and prefilling. The framework supports GRPO, DAPO, and REINFORCE for various modalities.
 