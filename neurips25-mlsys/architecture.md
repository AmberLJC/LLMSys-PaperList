 ## Best Paper Award 

**Title:** Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free

**Link:** https://openreview.net/forum?id=1b7whO4SfY

**Authors:** Zihan Qiu, Zekun Wang, Bo Zheng, Zeyu Huang, Kaiyue Wen, Songlin Yang, Rui Men, Le Yu, Fei Huang, Suozhi Huang, Dayiheng Liu, Jingren Zhou, Junyang Lin (Alibaba Qwen Team)

**Summary:** This Best Paper introduces a deceptively simple modification to standard Transformer attention—applying a head-specific sigmoid gate after Scaled Dot-Product Attention (Y′ = Y ⊙ σ(XWθ))—that yields substantial efficiency and quality improvements. The work addresses fundamental limitations of linear mappings between Value and Output projections in transformers, which cause the "attention sink" phenomenon (excessive attention to first tokens, reduced from ~46.7% to ~4.8%), training instability through massive activation outliers, and poor long-context extrapolation. By introducing non-linearity and query-dependent sparsity, the gating mechanism allows models to "reject" uninformative attention outputs. Validated across 30+ variants on 15B MoE and 1.7B dense models trained on 3.5 trillion tokens, the approach enables larger learning rates, better scaling, and improved context length extension—already integrated into Qwen3-Next production models.

---

## Efficient Attention Mechanisms

**Title:** Learning Linear Attention in Polynomial Time

**Link:** https://openreview.net/forum?id=QN0E0KX2LM

**Authors:** Morris Yau, Ekin Akyürek, Jiayuan Mao, Joshua B. Tenenbaum, Stefanie Jegelka, Jacob Andreas (MIT)

**Designation:** Oral

**Summary:** This oral paper provides the first polynomial-time learnability results—specifically strong, agnostic PAC learning—for single-layer Transformers with linear attention, bridging the gap between theoretical expressivity and practical learnability. Prior research extensively explored what Transformers can express but left efficient learnability from data as an open question. The key insight recasts learning optimal multi-head linear attention as finding the optimal kernel predictor in a suitably defined reproducing kernel Hilbert space (RKHS), enabling a polynomial-time algorithm that can also verify out-of-distribution generalization. Empirical validation spans learning random linear attention networks, key-value associations, and finite automata execution.

---

**Title:** The Emergence of Sparse Attention: Impact of Data Distribution and Benefits of Repetition

**Link:** https://openreview.net/forum?id=jMhRbV47pS

**Authors:** Nicolas Zucchet, Francesco D'Angelo, Andrew Kyle Lampinen, Stephanie C.Y. Chan

**Designation:** Oral

**Summary:** This oral paper investigates how sparse attention patterns—fundamental to LLM capabilities—emerge during training, revealing that emergence timing follows power laws based on task structure, architecture, and optimizer choices. The work addresses our limited understanding of when and how sparse attention capabilities develop during training, which has implications for training efficiency. Key findings show that data repetition can substantially accelerate sparse attention emergence, and the paper provides a theoretically grounded framework combining toy model analysis with empirical observations on small Transformers trained on linear regression variants and associative recall tasks.

---

**Title:** Twilight: Adaptive Sparse Attention via Top-p Sampling

**Link:** https://papers.cool/venue/NeurIPS.2025

**Authors:** (Multiple institutions)

**Designation:** Spotlight

**Summary:** Twilight applies the nucleus (top-p) sampling principle to sparse attention budget decisions, enabling **pruning up to 98% of tokens with near-zero accuracy loss** and achieving 1.4× inference speedup. The paper addresses the fundamental limitation that fixed-budget sparse attention fails for dynamic real-world scenarios where attention importance varies significantly across heads, layers, and inputs. By making adaptive budget decisions that enhance any existing sparse attention algorithm, the approach provides a universal wrapper that automatically determines appropriate sparsity levels based on the actual attention distribution rather than predetermined thresholds.

---

## Speculative Decoding Architectures

**Title:** SuffixDecoding: Extreme Speculative Decoding for Emerging AI Applications

**Link:** https://suffix-decoding.github.io/

**Authors:** (Multiple institutions)

**Designation:** Spotlight

**Summary:** SuffixDecoding uses suffix trees to cache long token sequences for speculative decoding, achieving up to **5.3× speedup** on agentic workloads (AgenticSQL) and **2.5× on SWE-Bench**. The paper addresses the limitation that existing speculative methods don't exploit the highly repetitive patterns common in agentic workflows—agents repeatedly generate similar code snippets, API calls, and structured outputs. Using global and per-request suffix trees with adaptive speculation length based on acceptance likelihood, the model-free approach requires no additional training.

---

**Title:** AutoJudge: Judge Decoding Without Manual Annotation

**Link:** NeurIPS 2025 Main Track

**Authors:** Roman Garipov, Fedor Velikonivtsev, Ivan Ermakov, Ruslan Svirschevski, Vage Egiazarian, Max Ryabinin (Yandex)

**Designation:** Spotlight

**Summary:** AutoJudge implements task-specific lossy speculative decoding that automatically identifies which token mismatches affect output quality, enabling acceleration by relaxing distribution-matching requirements for unimportant tokens. The paper addresses the fundamental question of which tokens actually matter for downstream quality, using semi-greedy search for correction decisions combined with a lightweight classifier that predicts skippable mismatches, enabling more aggressive speculation without human annotation of important tokens.

---

## KV-Cache Optimization Architectures

**Title:** Spotlight Attention: Towards Efficient LLM Generation via Non-linear Hashing-based KV Cache Retrieval

**Link:** https://neurips.cc/virtual/2025/poster/115727

**Authors:** (Multiple institutions)

**Designation:** Spotlight

**Summary:** Spotlight Attention employs non-linear hashing functions for efficient KV cache retrieval, achieving **5× shorter hash codes** than linear hashing with better precision. The paper identifies that linear hashing is inefficient for attention because queries and keys occupy orthogonal distributions in narrow cones. Using a Bradley-Terry ranking-based loss for training a lightweight non-linear hashing module (trainable on 16GB GPU in 8 hours), the approach enables efficient retrieval of relevant cached key-value pairs.

---

## Efficient Diffusion Model Architectures

**Title:** DiCo: Revitalizing ConvNets for Scalable and Efficient Diffusion Modeling

**Link:** https://openreview.net/forum?id=UnslcaZSnb | https://github.com/shallowdream204/DiCo

**Authors:** Yuang Ai, Qihang Fan, Xuefeng Hu, Zhenheng Yang, Ran He, Huaibo Huang

**Designation:** Spotlight

**Summary:** DiCo proposes a ConvNet-based architecture for diffusion modeling that consistently requires **fewer GFLOPs than Transformer counterparts** while achieving superior generative performance, challenging the dominance of Diffusion Transformers (DiTs). The paper addresses the high computational costs of DiTs due to quadratic attention complexity by demonstrating that properly designed ConvNets can match or exceed DiT quality with better efficiency.

---

**Title:** Exploring Diffusion Transformer Designs via Grafting

**Link:** https://github.com/keshik6/grafting | https://arxiv.org/abs/2506.05340

**Authors:** Keshigeyan Chandrasegaran, Michael Poli, Daniel Y. Fu, Dongjun Kim, Lea M. Hadzic, Manling Li, Agrim Gupta, Stefano Massaroli, Azalia Mirhoseini, Juan Carlos Niebles, Stefano Ermon, Fei-Fei Li

**Designation:** Oral

**Summary:** This oral paper proposes "grafting" to edit and restructure pretrained DiT architectures, achieving **2× model depth reduction** with better quality (FID: 2.77 on ImageNet 256×256). Rather than training new architectures from scratch, grafting modifies pretrained models through operator replacement and architecture restructuring. The work provides 22 grafted models exploring different design choices.

---

**Title:** STARFlow: Scaling Latent Normalizing Flows for High-resolution Image Synthesis

**Link:** NeurIPS 2025 Main Track

**Authors:** Apple Machine Learning Research

**Designation:** Spotlight

**Summary:** STARFlow provides the first successful demonstration of normalizing flows at large scale and high resolution, building on Transformer Autoregressive Flow (TARFlow) to rival diffusion and autoregressive methods while maintaining exact likelihood modeling and faster inference. Unlike diffusion models requiring many denoising steps, normalizing flows generate samples in a single pass through the network, offering fundamental efficiency advantages.

---

## State Space Models & Efficient Alternatives

**Title:** Memory Mosaics at Scale

**Link:** https://openreview.net/forum?id=IfD2MKTmWv

**Authors:** Jianyu Zhang, Leon Bottou (Meta AI)

**Designation:** Oral

**Summary:** This oral paper scales Memory Mosaics—networks of associative memories—to 10B parameters trained on 1 trillion tokens, presenting architectural modifications for LLM-scale deployment. The approach matches transformers on training knowledge while **significantly outperforming on new-task inference**—Memory Mosaics v2 on 1T tokens outperforms transformers on 8T tokens for new tasks. This demonstrates that associative memory architectures offer distinct advantages for in-context learning and knowledge incorporation.

---

## Best Paper - Datasets & Benchmarks Track

**Title:** Artificial Hivemind: The Open-Ended Homogeneity of Language Models (and Beyond)

**Link:** https://openreview.net/forum?id=saDOrrnNTz

**Authors:** Liwei Jiang, Yuanjun Chai, Margaret Li, Mickel Liu, Raymond Fok, Nouha Dziri, Yulia Tsvetkov, Maarten Sap, Yejin Choi

**Designation:** Best Paper (Datasets & Benchmarks Track)

**Summary:** This Best Paper introduces the Infinity-Chat benchmark (26K diverse open-ended queries + 31K human annotations) to study LLM diversity and mode collapse, revealing the "Artificial Hivemind effect"—pronounced intra-model repetition and inter-model homogeneity across 70+ models. The work creates the first comprehensive taxonomy of open-ended prompts enabling systematic evaluation of generative diversity, with implications for understanding the efficiency-diversity tradeoff in GenAI systems.
