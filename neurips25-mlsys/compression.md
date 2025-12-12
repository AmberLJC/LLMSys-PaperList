# NeurIPS 2025: Model Compression & Quantization

A curated list of NeurIPS 2025 papers focused on model compression, quantization, sparsification, and efficient representations.
Given the large volume of work in this area, I’ve highlighted only the oral and spotlight papers.

---

## Table of Contents

- [1. Systems](#1-systems)
  - [1.1 Efficient Attention Kernels](#11-efficient-attention-kernels)
  - [1.2 KV Cache Compression](#12-kv-cache-compression)
- [2. System-Algorithm Co-design](#2-system-algorithm-co-design)
  - [2.1 Quantized Training](#21-quantized-training)
  - [2.2 Quantized Fine-tuning](#22-quantized-fine-tuning)
- [3. Algorithm](#3-algorithm)
  - [3.1 Sparsification & Pruning](#31-sparsification--pruning)
  - [3.2 Compression Theory](#32-compression-theory)

---

## 1. Systems

### 1.1 Efficient Attention Kernels

#### SageAttention3: Accurate 8-Bit Attention with Plug-and-play Inference Acceleration 🔦 **SPOTLIGHT**

- **Link:** https://github.com/thu-ml/SageAttention
- **Authors:** Jintao Zhang, Jia Wei, Pengle Zhang, Jun Zhu, Jianfei Chen (Tsinghua University)
- **Summary:** SageAttention3 achieves **2-5× speedup** compared to FlashAttention through quantized attention without losing end-to-end metrics across language, image, and video models. The method introduces per-thread quantization for finer granularity while maintaining hardware efficiency, with Sparse SageAttention APIs computing attention with any block sparse pattern efficiently. Supports Hopper GPUs (H100, H800, H20), matching FlashAttention3-FP8 speed with better accuracy.

---

### 1.2 KV Cache Compression

#### R-KV: Redundancy-aware KV Cache Compression for Reasoning Models 🎤 **ORAL**

- **Link:** https://neurips.cc/virtual/2025/poster/120110 | https://github.com/Zefan-Cai/R-KV
- **Authors:** Zefan Cai, Wen Xiao, Hanshi Sun, Cheng Luo, Yikai Zhang, Ke Wan, Yucheng Li, Yeyang Zhou, Li-Wen Chang, Jiuxiang Gu, Zhen Dong, Anima Anandkumar, Abedelkadir Asi, Junjie Hu
- **Summary:** R-KV is the first KV cache compression method specifically designed for reasoning LLMs like DeepSeek-R1. Chain-of-thought generation explodes KV cache size—a single DeepSeek-R1-Distill-8B run can generate extremely long sequences. Existing compression tuned for long prompts fails on long generations, pruning wrong tokens because redundant self-checks attend heavily to themselves. R-KV ranks tokens on-the-fly for both importance AND non-redundancy, retaining only informative, diverse ones. At 16% cache budget, R-KV achieves **105% of full baseline accuracy**—evidence that trimming redundant tokens actually improves reasoning. Achieves **up to 90% KV-cache memory savings** with zero accuracy loss, training-free and plug-and-play.

---

#### KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction 🎤 **ORAL**

- **Link:** https://openreview.net/forum?id=JFygzwx8SJ | https://github.com/snu-mllab/KVzip
- **Authors:** Jang-Hyun Kim, Jinuk Kim, Sangwoo Kwon, Jae W. Lee, Sangdoo Yun, Hyun Oh Song (Seoul National University)
- **Summary:** KVzip provides query-agnostic KV cache eviction enabling **3-4× memory reduction** and **2× latency decrease**. Unlike existing methods requiring query-specific importance scoring, KVzip quantifies KV pair importance by using the LLM to reconstruct original contexts from cached pairs, enabling context-independent eviction with minimal compression ratio trade-off. Supports multi-turn inference by selectively evicting KV pairs while retaining interaction histories. Uses head-level importance-score optimization requiring only a few forward passes within one minute—**100× faster than alternatives**. Compatible with Qwen3/2.5, Gemma3, and LLaMA3.

---

## 2. System-Algorithm Co-design

### 2.1 Quantized Training

#### FP4 All the Way: Fully Quantized Training of LLMs

- **Link:** https://openreview.net/pdf?id=kuzye4EPLR
- **Authors:** Brian Chmiel, Maxim Fishman, Ron Banner (Nvidia/Intel), Daniel Soudry (Technion)
- **Summary:** First demonstration of fully quantized training using predominantly **4-bit floating-point** for weights, activations, AND gradients on datasets up to **1 trillion tokens**. Key innovations include a split rounding strategy and NVFP4 format with E4M3 scales. Successfully trains a 7B Llama2 model on 256 Intel Gaudi2 accelerators with performance matching BF16 baselines, expecting ~**85% time-to-train acceleration** versus BF16.

---

### 2.2 Quantized Fine-tuning

#### FALQON: Accelerating LoRA Fine-tuning with FP8 Arithmetic

- **Link:** https://neurips.cc/virtual/2025/poster/115144
- **Summary:** FP8 speedup diminishes for LoRA due to small-dimensional matrices. FALQON eliminates quantization overhead by directly merging LoRA adapters into an FP8-quantized backbone during fine-tuning, achieving approximately **3× training speedup** over existing quantized LoRA methods while maintaining accuracy.

---

## 3. Algorithm

### 3.1 Sparsification & Pruning

#### The Emergence of Sparse Attention: Impact of Data Distribution and Benefits of Repetition 🎤 **ORAL**

- **Link:** NeurIPS 2025 Schedule
- **Authors:** Nicolas Zucchet, Francesco D'Angelo, Andrew Lampinen, Stephanie Chan
- **Summary:** This Oral presentation investigates how sparse attention patterns emerge in transformers, examining the impact of data distribution and benefits of data repetition on attention sparsity development. The work provides fundamental understanding of mechanisms behind attention sparsification, with implications for training more efficient models by design.

---

#### STARFlow: Scaling Latent Normalizing Flows for High-resolution Image Synthesis 🔦 **SPOTLIGHT**

- **Link:** https://machinelearning.apple.com/research/neurips-2025
- **Authors:** Apple Machine Learning Research
- **Summary:** STARFlow presents scalable high-resolution generation avoiding diffusion computational costs through normalizing flows with autoregressive transformers. Produces images rivaling top diffusion and autoregressive methods while maintaining exact likelihood modeling and faster inference.
  

---

### Twilight: Adaptive Attention Sparsity with Hierarchical Top-p Pruning
 

**Authors:** Chaofan Lin, Jiaming Tang, Shuo Yang, Hanshuo Wang, Tian Tang, Boyu Tian, Ion Stoica (UC Berkeley), Song Han (MIT), Mingyu Gao (Tsinghua)

**Links:** https://arxiv.org/abs/2502.02770 

**Summary:** Existing sparse attention methods for long-context LLM inference rely on fixed token budgets (top-k selection), which fails to account for the varying computational needs across different layers, attention heads, and prompts. Twilight solves this by borrowing the concept of **top-p (nucleus) sampling** from text generation and applying it to attention sparsity—selecting tokens until cumulative attention weight reaches a threshold rather than selecting a fixed count. This enables truly adaptive budget decisions that adjust automatically to the information content of each attention computation. The framework implements hierarchical pruning that starts with rough filtering and progressively refines to key tokens. Results show Twilight can adaptively **prune up to 98% of tokens** with nearly no accuracy loss in mid- and long-context scenarios, achieving **15.4× acceleration** in self-attention operations, **3.9× acceleration** in end-to-end per-token latency, and **1.4× speedup** over prior state-of-the-art sparse attention mechanisms.

---

### 3.2 Compression Theory

#### Effective Interplay between Sparsity and Quantization: From Theory to Practice

- **Link:** https://openreview.net/forum?id=wJv4AIt4sK | https://arxiv.org/pdf/2405.20935
- **Summary:** A common practice in efficient ML is to prune a model (sparsity) and then quantize it. This paper mathematically proves that these operations are **non-orthogonal**. Applying quantization before sparsity (Q→S) disrupts the relative importance of tensor elements, leading to pruning of significant weights. The authors prove that applying **sparsity before quantization (S→Q)** is the optimal ordering to minimize error propagation, providing clear guidelines for compression pipelines.

---
