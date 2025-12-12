# NeurIPS 2025: ML Systems for Large Models

A curated collection of **NeurIPS 2025** papers focused on efficient systems for generative AI models (Large Language Models (LLMs), multi-modal models, diffusion models, etc.) . This collection emphasizes systems research, system-algorithm co-design, and algorithmic innovations with practical efficiency implications.

---

## 📊 NeurIPS 2025 Statistics

![NeurIPS 2025 Summary](stats.png)
 
---

## 📑 Table of Contents

| Category | Description | Papers |
|----------|-------------|--------|
| [🏗️ Architecture](architecture.md) | Efficient attention, KV-cache systems, speculative decoding, sparse attention | ~15 |
| [📦 Compression](compression.md) | Quantization, pruning, KV cache compression | ~10 |
| [⚡ Inference](inference.md) | LLM serving, scheduling, distributed inference, long-context | ~25 |
| [🎨 Multi-Modality](multi-modality.md) | VLM efficiency, diffusion optimization, token pruning | ~20 |
| [🎮 RL](rl.md) | RL training infrastructure, policy optimization, scaling | ~15 |
| [🔧 Training](training.md) | Distributed training, memory efficiency, hyperparameter scaling | ~35 |

---
 
## 📂 Category Overview

### 🏗️ [Architecture & Efficient Mechanisms](architecture.md)

Innovations in model architecture and attention mechanisms:
- **Efficient Attention Kernels**: Tiled Flash Linear Attention for O(N) scaling
- **KV-Cache Systems**: Spotlight Attention with non-linear hashing
- **Speculative Decoding**: SuffixDecoding (5.3× on agentic), AutoJudge (lossy spec decoding)
- **Sparse Attention**: Gated Attention (Best Paper), MoBA, Twilight
- **State Space Models**: Memory Mosaics at 10B scale
- **Diffusion Architectures**: DiCo, STARFlow, Grafting

### 📦 [Model Compression & Quantization](compression.md)

Techniques for reducing model size and memory footprint:
- **Quantized Attention**: SageAttention3 (2-5× faster than FlashAttention)
- **KV Cache Compression**: R-KV (90% savings for reasoning), KVzip (query-agnostic)
- **Quantized Training**: FP4 All the Way (4-bit training on 1T tokens)
- **Quantized Fine-tuning**: FALQON (3× LoRA speedup with FP8)
- **Compression Theory**: Optimal sparsity→quantization ordering

### ⚡ [Inference & Serving](inference.md)

Systems for efficient LLM deployment:
- **Scheduling**: Nexus (intra-GPU disaggregation), HyGen (3.87-5.84× throughput)
- **Distributed Inference**: ClusterFusion (cluster-level operator fusion)
- **KV Cache Systems**: ChunkKV, R-KV, Oneiros (44-82% latency reduction)
- **Compression**: DFloat11 (lossless 30% reduction, 405B on 8×80GB)
- **Multi-LoRA**: Loquetier (3× throughput, 46.4× SLO attainment)
- **TPU Support**: vLLM + Google Cloud TPUs
- **Speculative Decoding**: SuffixDecoding, EasySpec, Diffusion LLM acceleration
- **Long-Context**: RetrievalAttention, MonarchAttention (2.4× over FA2)

### 🎨 [Multi-Modal & Diffusion Efficiency](multi-modality.md)

Efficient systems for vision-language and generative models:
- **Multi-Modal Serving**: ElasticMM (4.2× TTFT reduction)
- **Video Processing**: StreamForest (96.8% accuracy at 1024 tokens vs 8K)
- **Token Pruning**: CDPruner (95% FLOPs reduction), BTP (78% compression), HoliTom (6.9% FLOPs)
- **Diffusion Architectures**: DiCo (ConvNet-based), NiT (native resolution), Grafting
- **Training Efficiency**: μP for DiT (3% FLOPs of expert tuning)
- **Theory**: Why diffusion models don't memorize (Best Paper)

### 🎮 [Reinforcement Learning for GenAI](rl.md)

RL systems and algorithms for LLM training:
- **Training Infrastructure**: AREAL (asynchronous), Long-RL (hour-level video), PipelineRL (2× faster)
- **Communication**: ACCO (87% time reduction), TBA (4× speedup)
- **Efficient Rollout**: Hogwild! Inference (shared attention cache), Flow-GRPO
- **Policy Optimization**: DAPO, Adaptive batch-wise scheduling
- **Scaling**: 1000-layer networks (Best Paper), compute-optimal scaling laws

### 🔧 [LLM Training Systems](training.md)

Infrastructure for efficient model training:
- **Distributed Training**: Synergistic TP+PP (12-16% improvement), Arnold (9600+ GPUs), DiLoCo scaling laws
- **Communication**: ACCO (87% reduction), SDP4Bit (4.08× speedup), PaRO (266% over ZeRO-3)
- **Memory Efficiency**: TERAIO (GPUDirect storage), zeroth-order fine-tuning
- **Long-Context**: InfiniPipe (elastic PP), HBP (hierarchical packing)
- **Compiler/Hardware**: XgenSilicon (2.5-4.5× faster), Autocomp (LLM-driven), DCC (PIM)
- **Stability**: Gated Attention (Best Paper), AlphaDecay
- **Hyperparameter Scaling**: Optimizer benchmarks, AdaLRS, power laws for weight decay
  

## 🔗 Related Resources

- [NeurIPS 2025 Schedule](https://neurips.cc/virtual/2025)
- [Main LLMSys-PaperList Repository](../README.md)
 