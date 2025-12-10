## Sparsification and pruning for transformers and attention
 

### SageAttention3: Accurate 8-Bit Attention with Plug-and-play Inference Acceleration 🔦 **Spotlight**
- **Link:** https://github.com/thu-ml/SageAttention
- **Authors:** Jintao Zhang, Jia Wei, Pengle Zhang, Jun Zhu, Jianfei Chen (Tsinghua University)
- **Summary:** SageAttention3 achieves **2-5× speedup** compared to FlashAttention through quantized attention without losing end-to-end metrics across language, image, and video models. The method introduces per-thread quantization for finer granularity while maintaining hardware efficiency, with Sparse SageAttention APIs computing attention with any block sparse pattern efficiently. Supports Hopper GPUs (H100, H800, H20), matching FlashAttention3-FP8 speed with better accuracy.

### The Emergence of Sparse Attention: Impact of Data Distribution and Benefits of Repetition 🎤 **Oral**
- **Link:** NeurIPS 2025 Schedule
- **Authors:** Nicolas Zucchet, Francesco D'Angelo, Andrew Lampinen, Stephanie Chan
- **Summary:** This Oral presentation investigates how sparse attention patterns emerge in transformers, examining the impact of data distribution and benefits of data repetition on attention sparsity development. The work provides fundamental understanding of mechanisms behind attention sparsification, with implications for training more efficient models by design.

---

## KV cache compression for efficient LLM inference

### R-KV: Redundancy-aware KV Cache Compression for Reasoning Models 🎤 **Oral**
- **Link:** https://neurips.cc/virtual/2025/poster/120110 | https://github.com/Zefan-Cai/R-KV
- **Authors:** Zefan Cai, Wen Xiao, Hanshi Sun, Cheng Luo, Yikai Zhang, Ke Wan, Yucheng Li, Yeyang Zhou, Li-Wen Chang, Jiuxiang Gu, Zhen Dong, Anima Anandkumar, Abedelkadir Asi, Junjie Hu
- **Summary:** R-KV is the first KV cache compression method specifically designed for reasoning LLMs like DeepSeek-R1. Chain-of-thought generation explodes KV cache size—a single DeepSeek-R1-Distill-8B run can generate extremely long sequences. Existing compression tuned for long prompts fails on long generations, pruning wrong tokens because redundant self-checks attend heavily to themselves. R-KV ranks tokens on-the-fly for both importance AND non-redundancy, retaining only informative, diverse ones. At 16% cache budget, R-KV achieves **105% of full baseline accuracy**—evidence that trimming redundant tokens actually improves reasoning. Achieves **up to 90% KV-cache memory savings** with zero accuracy loss, training-free and plug-and-play.

### KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction 🎤 **Oral**
- **Link:** https://openreview.net/forum?id=JFygzwx8SJ | https://github.com/snu-mllab/KVzip
- **Authors:** Jang-Hyun Kim, Jinuk Kim, Sangwoo Kwon, Jae W. Lee, Sangdoo Yun, Hyun Oh Song (Seoul National University)
- **Summary:** KVzip provides query-agnostic KV cache eviction enabling **3-4× memory reduction** and **2× latency decrease**. Unlike existing methods requiring query-specific importance scoring, KVzip quantifies KV pair importance by using the LLM to reconstruct original contexts from cached pairs, enabling context-independent eviction with minimal compression ratio trade-off. Supports multi-turn inference by selectively evicting KV pairs while retaining interaction histories. Uses head-level importance-score optimization requiring only a few forward passes within one minute—**100× faster than alternatives**. Compatible with Qwen3/2.5, Gemma3, and LLaMA3.
 

### STARFlow: Scaling Latent Normalizing Flows for High-resolution Image Synthesis 🔦 **Spotlight**
- **Link:** https://machinelearning.apple.com/research/neurips-2025
- **Authors:** Apple Machine Learning Research
- **Summary:** STARFlow presents scalable high-resolution generation avoiding diffusion computational costs through normalizing flows with autoregressive transformers. Produces images rivaling top diffusion and autoregressive methods while maintaining exact likelihood modeling and faster inference. 


### FP4 All the Way: Fully Quantized Training of LLMs
**Link:** https://openreview.net/pdf?id=kuzye4EPLR  
**Authors:** Brian Chmiel, Maxim Fishman, Ron Banner (Nvidia/Intel), Daniel Soudry (Technion)

First demonstration of fully quantized training using predominantly **4-bit floating-point** for weights, activations, AND gradients on datasets up to **1 trillion tokens**. Key innovations include a split rounding strategy and NVFP4 format with E4M3 scales. Successfully trains a 7B Llama2 model on 256 Intel Gaudi2 accelerators with performance matching BF16 baselines, expecting ~**85% time-to-train acceleration** versus BF16.

### FALQON: Accelerating LoRA Fine-tuning with FP8 Arithmetic
**Link:** https://neurips.cc/virtual/2025/poster/115144  
**Authors:** NeurIPS 2025 Poster

FP8 speedup diminishes for LoRA due to small-dimensional matrices. FALQON eliminates quantization overhead by directly merging LoRA adapters into an FP8-quantized backbone during fine-tuning, achieving approximately **3× training speedup** over existing quantized LoRA methods while maintaining accuracy.
 