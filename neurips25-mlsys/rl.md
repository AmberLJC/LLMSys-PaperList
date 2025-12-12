# NeurIPS 2025: Reinforcement Learning for GenAI

A curated list of NeurIPS 2025 papers focused on efficient reinforcement learning systems and algorithms for large language models and generative AI.

---

## Table of Contents

- [1. Systems](#1-systems)
  - [1.1 RL Training Infrastructure](#11-rl-training-infrastructure)
  - [1.2 Communication-Efficient Training](#12-communication-efficient-training)
- [2. System-Algorithm Co-design](#2-system-algorithm-co-design)
  - [2.1 Efficient Rollout & Sampling](#21-efficient-rollout--sampling)
  - [2.2 Scalable Policy Optimization](#22-scalable-policy-optimization)
- [3. Algorithm](#3-algorithm)
  - [3.1 Policy Optimization Methods](#31-policy-optimization-methods)
  - [3.2 RL Scaling & Architecture](#32-rl-scaling--architecture)
  - [3.3 RL Theory & Analysis](#33-rl-theory--analysis)

---

## 1. Systems

### 1.1 RL Training Infrastructure

#### AREAL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning

- **Link:** https://neurips.cc/virtual/2025/poster/117538
- **Summary:** AREAL addresses the systems challenge of scaling RL for LLM reasoning by introducing asynchronous training where samplers and trainers operate on different GPU pools, eliminating the synchronization overhead that causes GPU idle time. This is particularly important for reasoning tasks with heterogeneous response lengths. The asynchronous design achieves near-linear scaling to large GPU clusters.

---

#### Long-RL: Scaling RL to Long Sequences (LongVILA-R1)

- **Link:** https://github.com/NVlabs/Long-RL
- **Authors:** Yukang Chen, Wei Huang, Baifeng Shi, Qinghao Hu, et al. (NVIDIA, UC Berkeley, HKU)
- **Summary:** Long-RL addresses hour-level video RL training (3,600 frames = 256k tokens) on a single node. The key innovation is Multi-modal Reinforcement Sequence Parallelism (MR-SP), which incorporates sequence parallelism and a vLLM-based engine with cached video embeddings for efficient rollout and prefilling. The framework supports GRPO, DAPO, and REINFORCE for various modalities.

---

#### PipelineRL: Faster On-policy Reinforcement Learning for Long Sequence Generation

- **Link:** https://openreview.net/forum?id=EqlmpJYapx
- **Authors:** Alexandre Piché, Ehsan Kamalloo, Rafael Pardinas, Xiaoyin Chen, Dzmitry Bahdanau (ServiceNow)
- **Summary:** PipelineRL tackles the fundamental trade-off between hardware utilization and on-policy data freshness that plagues large-scale RL for LLMs. The paper introduces **in-flight weight updates**, a mechanism allowing the LLM generation engine to receive updated model weights with minimal interruption during token sequence generation. Unlike conventional RL that alternates between generation and training phases, PipelineRL runs both concurrently, maintaining near on-policy data while maximizing GPU utilization. Experiments on long-form reasoning tasks using **32-128 H100 GPUs** demonstrate approximately **2× faster learning** compared to synchronous baselines while keeping training data highly on-policy. The open-source implementation provides a modular, Hydra-driven pipeline with separate actor, verifier, and trainer components—directly comparable to AReaL's decoupled architecture.

---

#### Trajectory Balance with Asynchrony (TBA): Decoupling Exploration and Learning for Fast, Scalable LLM Post-Training

- **Link:** https://arxiv.org/abs/2503.18929
- **Authors:** Brian R. Bartoldson, Siddarth Venkatraman, James Diffenderfer, Moksh Jain, Tal Ben-Nun, Seanie Lee, Minsu Kim, Johan Obando-Ceron, Yoshua Bengio, Bhavya Kailkhura (LLNL, Mila)
- **Summary:** TBA proposes a massively scalable LLM RL system that efficiently leverages experience replay buffers—something existing on-policy algorithms cannot do. The system uses a **dual-node architecture**: multiple SEARCHER nodes continuously generate off-policy data for a central replay buffer, while a TRAINER node simultaneously samples data based on reward or recency to update the policy. This decoupling achieves **4× or greater wall-clock speedup** over synchronous approaches. On mathematical reasoning (GSM8K), preference-tuning (TL;DR), and automated red-teaming tasks, TBA produces substantial speed and performance improvements. The framework supports models from Pythia 410M to Qwen 2.5 7B, demonstrating scalability across model sizes.

---

### 1.2 Communication-Efficient Training

#### ACCO: Accumulate While You Communicate for Communication-Overlapped Sharded LLM Training

- **Link:** https://openreview.net/forum?id=1qKUVyymXs
- **Authors:** Adel Nabli, Louis Fournier, Pierre Erbacher, Louis Serrano, Eugene Belilovsky, Edouard Oyallon
- **Summary:** ACCO addresses communication overhead in distributed LLM training by synchronizing delayed gradients *while* computing new ones between model updates. This accumulates local gradients until communication finishes, naturally reducing GPU idle time and enabling **heterogeneous hardware support**. The paper shows that one-step delay from parallel gradient computation and communication has drastic impacts on Transformer convergence, introducing a novel two-stage compensation mechanism to ensure training dynamics align with standard distributed optimization. Compared to ZeRO-1, ACCO reduces learning time by **up to 87%** and successfully enables both optimizer state sharding and heterogeneous hardware usage—addressing key infrastructure challenges similar to those AReaL tackles for RL workloads.

---

## 2. System-Algorithm Co-design

### 2.1 Efficient Rollout & Sampling

#### Hogwild! Inference: Parallel LLM Generation via Concurrent Attention

- **Link:** https://neurips.cc/virtual/2025/events/spotlights-2025
- **Summary:** This paper proposes a fundamentally new approach to parallel inference where multiple LLM "workers" run simultaneously with a **shared attention cache**, enabling instant access to each other's generated tokens. The problem addressed is that rollout generation is often the primary bottleneck in RL training pipelines (PPO, GRPO), where generating many samples per prompt consumes significant wall-clock time. By leveraging RoPE's mathematical properties to avoid recomputation while improving hardware utilization, Hogwild! Inference accelerates sampling without requiring additional fine-tuning or model modifications. The concurrent cache allows workers to develop implicit collaboration strategies, reducing total generation time for batch rollouts. This is directly applicable to RL training where faster rollout generation translates to faster training iterations and more efficient use of GPU resources.

---

#### Act Only When It Pays: Efficient RL for LLM Reasoning via Selective Rollouts

- **Link:** https://neurips.cc/Downloads/2025
- **Summary:** This paper proposes selective rollout strategies for RL training of LLMs, where training compute is focused on examples where it will have the most impact. By avoiding unnecessary rollouts on easy or already-mastered examples, the method improves training efficiency.

---

#### Flow-GRPO: Training Flow Matching Models via Online RL

- **Link:** https://github.com/yifan123/flow_grpo | https://arxiv.org/abs/2505.05470
- **Authors:** Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, Wanli Ouyang
- **Summary:** Flow-GRPO extends GRPO to flow matching models for efficient text-to-image generation RL training. Key efficiency innovations include ODE-to-SDE conversion enabling stochastic sampling for RL, and Flow-GRPO-Fast which trains on only partial timesteps, reducing training costs dramatically.

---

### 2.2 Scalable Policy Optimization

#### Adaptive Batch-Wise Sample Scheduling for Direct Preference Optimization

- **Link:** https://neurips.cc/virtual/2025/poster/119641
- **Summary:** This paper addresses the efficiency of DPO training through intelligent data selection. The authors propose adaptive batch-wise sample scheduling that dynamically selects training samples based on their learning signal quality. This approach improves DPO convergence and final alignment quality while reducing the number of training iterations needed.

---

## 3. Algorithm

### 3.1 Policy Optimization Methods

#### Reinforcement Learning with Verifiable Rewards: A Critical Analysis 🥈 **RUNNER-UP AWARD**

- **Link:** https://blog.neurips.cc/2025/11/26/announcing-the-neurips-2025-best-paper-awards/
- **Summary:** This paper delivers a critical negative result: current RLVR methods improve sampling efficiency toward correct paths but do NOT elicit fundamentally new reasoning patterns. Six popular RLVR algorithms perform similarly and remain far from optimal in leveraging base model potential.
  - **Implications:**
    - Distillation can introduce new reasoning patterns from teachers; RLVR cannot
    - Current RL paradigms need improvement—continual scaling and multi-turn agent-environment interaction may unlock this potential
    - Challenges foundational assumptions about RL's role in LLM reasoning

---

### 3.2 RL Scaling & Architecture

#### 1000 Layer Networks for Self-Supervised RL: Scaling Depth Can Enable New Goal-Reaching Capabilities 🏆 **BEST PAPER**

- **Link:** https://openreview.net/forum?id=s0JVsx3bx1 | [arXiv:2503.14858](https://arxiv.org/abs/2503.14858)
- **Summary:** Historically, Deep RL agents have relied on shallow MLPs (2-5 layers), with attempts to scale depth resulting in training instability. This paper hypothesizes that the scalar reward signal in RL is too sparse to effectively propagate gradients through deep networks. The solution decouples representation learning from policy learning: a massive backbone (ResNet/Transformer style, up to 1000 layers) is trained using **Self-Supervised Contrastive Learning** with dense supervisory signal predicting future states/goals, then a lightweight RL policy head is trained on frozen representations. This approach unlocks **50× improvement in goal-reaching capabilities** on complex robotic control tasks.

---

#### Compute-Optimal Scaling for Value-Based Deep RL

- **Link:** [arXiv:2508.14881](https://arxiv.org/abs/2508.14881) | [NeurIPS Poster](https://neurips.cc/virtual/2025/loc/san-diego/poster/119555)
- **Authors:** Preston Fu, Pieter Abbeel, Sergey Levine et al.
- **Summary:** This paper establishes the first systematic investigation of compute scaling for online, value-based deep RL. The authors derive compute-optimal configurations analogous to LLM scaling laws for allocating budget between model size, environment interactions, and replay ratio. This provides principled guidance for efficient resource allocation in deep RL training.

--- 