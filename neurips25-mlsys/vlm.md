# NeurIPS 2025 Multi-Modal & Diffusion Efficiency Papers

## Multi-Modal Systems

### Efficient Serving

#### ElasticMM: Efficient Multimodal LLMs Serving with Elastic Multimodal Parallelism

**Link:** https://arxiv.org/abs/2507.10069 | https://openreview.net/forum?id=Zd6VyjmN1S

**Authors:** Zedong Liu, Shenggan Cheng, Guangming Tan, Yang You, Dingwen Tao

**Summary:** ElasticMM tackles the major challenge of efficiently serving multimodal LLMs, where additional components (feature extractors, projection modules) combined with complex inference pipelines create significant overhead. Current tightly coupled serving architectures struggle to distinguish between mixed request types or adapt parallelism strategies to different inference stages, leading to increased time-to-first-token (TTFT) latency and poor resource utilization. ElasticMM introduces Elastic Multimodal Parallelism (EMP), featuring: (1) modality-aware load balancing that separates requests into independent modality groups with dynamic resource allocation, (2) decoupled inference stages with parallelism adjustment via elastic partition scheduling, and (3) unified multimodal prefix caching with non-blocking encoding. Experiments show **up to 4.2× reduction in TTFT** and **3.2-4.5× higher throughput** compared to vLLM while meeting service-level objectives.

---
### Token Pruning & Compression

#### CDPruner: Maximizing Conditional Diversity for Token Pruning in MLLMs

**Link:** https://arxiv.org/abs/2506.10967 | https://github.com/Theia-4869/CDPruner

**Authors:** Qizhe Zhang, Mengzhen Liu, Lichen Li, Ming Lu, Yuan Zhang, Junwen Pan, Qi She, Shanghang Zhang

**Summary:** CDPruner addresses the high inference cost of MLLMs caused by visual tokens significantly outnumbering text tokens. Current approaches either use attention-based pruning (retaining duplicate tokens) or similarity-based pruning (ignoring instruction relevance). CDPruner goes beyond both by maximizing conditional diversity of retained tokens. The method defines conditional similarity between visual tokens conditioned on user instructions, then reformulates token pruning using Determinantal Point Process (DPP) to select maximally diverse, instruction-relevant subsets. This training-free, model-agnostic approach achieves new state-of-the-art across vision-language benchmarks. When applied to LLaVA, CDPruner **reduces FLOPs by 95%** and **CUDA latency by 78%** while maintaining **94% of original accuracy**—establishing a principled framework for efficient visual token selection in multimodal inference.

---

#### Balanced Token Pruning: Accelerating Vision Language Models Beyond Local Optimization

**Link:** https://arxiv.org/abs/2505.22038 | https://github.com/EmbodiedCity/NeurIPS2025-Balanced-Token-Pruning

**Authors:** Kaiyuan Li, Xiaoyue Chen, Chen Gao, Yong Li, Xinlei Chen

**Summary:** Balanced Token Pruning (BTP) addresses a fundamental limitation in existing LVLM token pruning methods: they overlook the joint impact of pruning on both current layer output (local) and subsequent layer outputs (global), leading to suboptimal decisions. Through empirical analysis, the paper reveals how shallow-layer pruning affects deeper layers—information critical for principled layer selection. BTP proposes a plug-and-play method using a calibration set to divide pruning into multiple stages, balancing local and global effects. The method supports FlashAttention for additional acceleration. BTP achieves **78% compression rate while preserving 96.7% of original model performance** on average across LLaVA models, demonstrating that systematic analysis of cross-layer pruning dynamics enables significantly better efficiency-accuracy tradeoffs than attention-score or diversity-based approaches alone.

---

#### HoliTom: Holistic Token Merging for Fast Video Large Language Models

**Link:** https://arxiv.org/abs/2505.21334 | https://cokeshao.github.io/HoliTom_Web/

**Authors:** Kele Shao, Keda Tao, Can Qin, Haoxuan You, Yang Sui, Huan Wang (Zhejiang University, Westlake University, Salesforce AI Research, Columbia University, Rice University)

**Summary:** HoliTom addresses the significant computational inefficiency in video LLMs caused by redundant visual tokens. Existing token pruning methods either operate within the LLM (incurring overhead in shallow layers) or before the LLM (addressing only spatial redundancy within frames). HoliTom introduces a training-free holistic token merging framework combining both strategies synergistically. Outer-LLM pruning uses global redundancy-aware temporal segmentation followed by spatial-temporal merging to reduce visual tokens by over 90%. Inner-LLM token similarity-based merging further streamlines processing. On LLaVA-OneVision-7B, HoliTom **reduces computational costs to 6.9% of FLOPs while maintaining 99.1% performance**, achieves **2.28× reduction in Time-To-First-Token**, and **1.32× acceleration in decoding throughput**—the practical benefits of integrated pruning for efficient video LLM inference.

---

#### Accelerating Multimodal Large Language Models via Dynamic Visual-Token Exit

**Link:** https://neurips.cc/virtual/2025/poster/118110

**Summary:** This paper introduces a dynamic visual token exit strategy for accelerating multimodal LLM inference. The core insight is that not all visual tokens need to be processed through all transformer layers—some can exit early when sufficient information has been extracted. The method learns to predict optimal exit points for different tokens based on their information content and relevance to the task, enabling adaptive computation allocation. Empirical findings reveal interesting patterns about which types of visual information require deep processing versus which can be resolved in shallow layers, providing guidance for future MLLM architecture design focused on efficiency.

---

#### Glance2Gaze: Efficient Vision-Language Models from Glance Fusion to Gaze Compression

**Link:** https://neurips.cc/virtual/2025/loc/san-diego/poster/116704

**Authors:** Juan Chen, Honglin Liu, Yingying Ao, Ting Zhang, Yan Huang, Xudong Liu, Biao Li, Jintao Fang

**Summary:** Glance2Gaze presents a cognitively-inspired framework for efficient VLMs that mimics human visual attention's two-stage process: glancing then gazing. The Glance Fusion module integrates multi-layer ViT features with text-aware attention for broad scene understanding, while the Gaze Compression module uses a query-guided mechanism to selectively compress visual tokens based on semantic relevance to the task. This approach concentrates computational resources on task-relevant visual regions while efficiently summarizing background context. The framework outperforms existing efficient VLM methods while achieving equal or lower computational cost, demonstrating that biologically-inspired attention mechanisms can guide more effective efficiency optimizations.

---

#### Visual Context Compression for Efficient Large Multi-modal Models

**Link:** https://openreview.net/pdf?id=5ujp72CiYB

**Summary:** Introduces a Visual Context Compressor reducing visual token redundancy in multimodal LLMs during training. The method enables training at various compression levels and demonstrates scalability to larger models. A two-stage training setup optimizes total training time while maintaining performance.

---


### Video Understanding

#### StreamForest: Efficient Online Video Understanding with Persistent Event Memory

**Link:** https://arxiv.org/abs/2509.24871 | https://github.com/MCG-NJU/StreamForest

**Authors:** Xiangyu Zeng, Kefan Qiu, Qingyu Zhang, Xinhao Li, Jing Wang, Jiaxin Li, Ziang Yan, Kun Tian, Meng Tian, Xinhai Zhao, Yi Wang, Limin Wang

**Summary:** StreamForest addresses the fundamental challenge of real-time streaming video understanding in multimodal LLMs, where historical visual feature storage limitations and insufficient real-time spatiotemporal reasoning constrain effectiveness. The paper introduces the Persistent Event Memory Forest (PEMF), a hierarchical event-level memory system that adaptively organizes video frames into tree structures guided by penalty functions based on temporal distance, content similarity, and merge frequency. This enables efficient long-term memory retention under limited computational resources. A Fine-grained Spatiotemporal Window enhances real-time perception. Remarkably, even under **extreme visual token compression (1024 tokens vs. default 8K)**, the model maintains **96.8% average accuracy** across eight benchmarks, achieving 77.3% on StreamingBench, 60.5% on OVBench, and **2.28× reduction in Time-To-First-Token latency**.

---

### Efficient Fine-tuning & Adaptation

#### MokA: Multimodal Low-Rank Adaptation for MLLMs

**Link:** https://arxiv.org/abs/2506.05191 | https://neurips.cc/virtual/2025/oral/116048

**Authors:** Yake Wei, Yu Miao, Dongzhan Zhou, Di Hu

**Summary:** MokA addresses a critical limitation in multimodal large language model (MLLM) fine-tuning: existing efficient adaptation methods are directly borrowed from LLMs and neglect the intrinsic differences of multimodal scenarios. This is problematic because multimodal models require both unimodal adaptation (processing each modality independently) and cross-modal adaptation (integrating information across modalities). MokA introduces a multimodal-aware low-rank adaptation strategy that uses modality-specific low-rank matrices to compress information independently per modality, followed by a cross-attention mechanism to strengthen text-visual interaction, and finally a shared matrix for unified projection. The method demonstrates consistent improvements across three multimodal scenarios (audio-visual-text, visual-text, speech-text) and multiple LLM backbones (LLaMA2/3, Qwen2, Qwen2.5-VL), providing more parameter-efficient fine-tuning while better utilizing all modalities.
 
---

## Diffusion Models

### Architecture Design

#### Representation Entanglement for Generation: Training Diffusion Transformers Is Much Easier Than You Think

**Link:** https://neurips.cc/virtual/2025/loc/san-diego/oral/116345

**Authors:** Ge Wu, Shen Zhang, Ruijing Shi, Shanghua Gao, Zhenyuan Chen, Lei Wang, Zhaowei Chen, Hongcheng Gao, Yao Tang, Jian Yang, Ming-Ming Cheng, Xiang Li

**Summary:** This oral paper challenges REPA-style methods that align noisy hidden projections with external clean image representations. The authors propose "representation entanglement"—a method that makes training diffusion transformers significantly easier by better integrating representations throughout training and inference processes, reducing the complexity of diffusion transformer training pipelines.

---

#### Grafting: Exploring Diffusion Transformer Designs via Efficient Architecture Editing

**Link:** https://arxiv.org/abs/2506.05340 | https://github.com/keshik6/grafting

**Authors:** Keshigeyan Chandrasegaran, Michael Poli, Daniel Y. Fu, Dongjun Kim, Lea M. Hadzic, Manling Li, Agrim Gupta, Stefano Massaroli, Azalia Mirhoseini, Juan Carlos Niebles, Stefano Ermon, Fei-Fei Li

**Summary:** Grafting enables editing pretrained diffusion transformers to materialize new architectures under small compute budgets—without costly pretraining. By analyzing activation behavior and attention locality in DiT-XL/2, the authors construct hybrid designs by grafting different components (replacing softmax attention with gated convolution, local attention, linear attention; replacing MLPs with variants). This reduces compute needed for diffusion model architecture research by orders of magnitude.

---

#### DiCo: Revitalizing ConvNets for Scalable and Efficient Diffusion Modeling

**Link:** https://arxiv.org/abs/2505.11196 | https://github.com/shallowdream204/DiCo

**Authors:** Yuang Ai, Qihang Fan, Xuefeng Hu, Zhenheng Yang, Ran He, Huaibo Huang

**Summary:** DiCo challenges the prevailing assumption that Diffusion Transformers (DiT) are the optimal architecture for visual generation. Analysis of pre-trained DiT models reveals that global self-attention is often redundant, predominantly capturing local patterns. This paper revisits convolution as an efficient alternative building block. The key insight is that naive replacement of self-attention with convolution degrades performance due to higher channel redundancy in ConvNets compared to Transformers. DiCo addresses this through architectural innovations that reduce channel redundancy while preserving expressiveness. The result is a Diffusion ConvNet that **requires fewer GFLOPs** than Transformer counterparts while achieving **superior generative performance** (FID **2.05** on ImageNet 256×256) on text-to-image generation tasks.

---

#### NiT: Native-Resolution Diffusion Transformer

**Link:** https://arxiv.org/abs/2506.03131 | https://github.com/WZDTHU/NiT

**Authors:** Zidong Wang, Lei Bai, Xiangyu Yue, Wanli Ouyang, Yiyuan Zhang

**Summary:** Rather than training separate models for different resolutions, NiT explicitly learns varying resolutions and aspect ratios within its denoising process. The model achieves SOTA results on both 256×256 (FID **2.08**) and 512×512 (FID **1.48**) ImageNet simultaneously, generalizing to arbitrary resolutions (FID 4.52 on 1024×1024). This multi-resolution approach eliminates resolution-specific training, dramatically improving overall training efficiency.

---

### Training Efficiency

#### Scaling Diffusion Transformers Efficiently via μP

**Link:** https://arxiv.org/abs/2505.15270 | https://github.com/ML-GSAI/Scaling-Diffusion-Transformers-muP

**Authors:** Chenyu Zheng, Xinyu Zhang, Rongzhen Wang, Wei Huang, Zhi Tian, Weilin Huang, Jun Zhu, Chongxuan Li

**Summary:** Hyperparameter tuning for large diffusion transformers is prohibitively expensive. This paper applies Maximal Update Parametrization (μP) to DiT, PixArt-α, and MMDiT, enabling stable hyperparameter transfer from small to large models. The approach achieves results with only **3% FLOPs** of human expert tuning costs for MMDiT-18B, dramatically reducing the cost of training large-scale diffusion transformers.

---

#### E2D2: Encoder-Decoder Block Diffusion for Efficient Training and Inference

**Link:** https://arxiv.org/abs/2510.22852 | https://neurips.cc/virtual/2025/poster/119836

**Authors:** Marianne Arriola, Yair Schiff, Hao Phung, Aaron Gokaslan, Volodymyr Kuleshov

**Summary:** Standard decoder-only block diffusion architectures double computational costs. E2D2 proposes an encoder-decoder architecture that separates computation: the encoder represents clean tokens while a lightweight decoder iteratively refines corrupted sequences. This enables training larger, higher-quality block diffusion models for the same compute budget, achieving superior trade-offs on summarization, translation, and mathematical reasoning tasks.

---

#### Ambient Diffusion Omni: Training Good Models with Bad Data

**Link:** https://neurips.cc/Downloads/2025

**Summary:** Training on high-quality curated data is expensive. Ambient Diffusion Omni presents a framework to train diffusion models that extract signal from all available images, including low-quality, synthetic, and out-of-distribution data typically discarded. By exploiting spectral power law decay and locality properties of natural images, the method achieves state-of-the-art ImageNet FID while training on corrupted data (Gaussian blur, JPEG compression, motion blur).
 
### Autoregressive & Visual Tokenization

#### InfinityStar: Unified Spacetime AutoRegressive Modeling for Visual Generation

**Link:** https://github.com/FoundationVision/InfinityStar

**Authors:** Jinlai Liu, Jian Han, Bin Yan, Hui Wu, et al.

**Summary:** First discrete autoregressive video generator capable of producing industrial-level 720p videos. Uses FlexAttention for training speedup, achieving 83.74 on VBench—outperforming all autoregressive models by large margins while supporting text-to-image, text-to-video, image-to-video, and long interactive video generation.

---

#### VFMTok: Vision Foundation Models as Effective Visual Tokenizers

**Link:** https://github.com/CVMI-Lab/VFMTok

**Summary:** Uses pre-trained vision foundation models (DINOv2) as visual tokenizers for autoregressive generation. Key innovations include region-adaptive quantization and semantic reconstruction objectives. The approach **accelerates convergence by 3×** when integrated into AR generative models while enabling high-fidelity synthesis without classifier-free guidance.
