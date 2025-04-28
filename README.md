# Awesome LLM Systems Papers

A curated list of Large Language Model systems related academic papers, articles, tutorials, slides and projects. Star this repository, and then you can keep abreast of the latest developments of this booming research field.
## Table of Contents

- [LLM Systems](#llm-systems)
  - [Training](#Training)
    - [Post Training - RLHF](#Systems-for-Post-training--RLHF)
    - [Fault Tolerance / Straggler Mitigation](#fault-tolerance--straggler-mitigation)
  - [Serving](#serving)
    - [Compound AI Systems](#compound-ai-systems)
    - [Serving at the edge](#serving-at-the-edge)
    - [System Efficiency Optimization](#system-efficiency-optimization)
  - [Fine-tuning Systems](#fine-tuning-systems)
  - [Multi-Modal Systems](#multi-modal-systems)
- [LLM for Systems](#llm-for-systems)
- [ML Systems](mlsystems.md)
- [Survey Paper](#survey-paper)
- [LLM Benchmark / Leaderboard / Traces](#llm-benchmark--leaderboard--traces)
- [LLM Frameworks](#llm-frameworks)
- [Related ML Readings](#related-ml-readings) 

## LLM Systems
### Training
- [Megatron-LM](https://arxiv.org/pdf/1909.08053.pdf): Training Multi-Billion Parameter Language Models Using Model Parallelism
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/pdf/2104.04473.pdf)
- [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/pdf/2205.05198.pdf)
- [Optimized Network Architectures for Large Language Model Training with Billions of Parameters](https://arxiv.org/pdf/2307.12169.pdf) | MIT
- [Carbon Emissions and Large Neural Network Training](https://arxiv.org/pdf/2104.10350.pdf?fbclid=IwAR2o0_3HCtTnMxKbXka0OPrHzl8sCzQSSOYp0AOav76-zVWl_pYek2jX8Pk) | Google, UCB
- [Perseus](https://arxiv.org/abs/2312.06902v1): Removing Energy Bloat from Large Model Training | SOSP' 24
- [MegaScale](https://arxiv.org/abs/2402.15627): Scaling Large Language Model Training to More Than 10,000 GPUs | ByteDance
- [DISTMM](https://www.amazon.science/publications/distmm-accelerating-distributed-multimodal-model-training): Accelerating distributed multimodal model training | NSDI' 24
- [A Codesign of Scheduling and Parallelization for Large Model Training in Heterogeneous Clusters](https://arxiv.org/pdf/2403.16125)
- [Pipeline Parallelism with Controllable Memory](https://arxiv.org/abs/2405.15362) | Sea AI Lab
- [Boosting Large-scale Parallel Training Efficiency with C4](https://arxiv.org/abs/2406.04594): A Communication-Driven Approach
- [Scaling Beyond the GPU Memory Limit for Large Mixture-of-Experts Model Training](https://openreview.net/pdf?id=uLpyWQPyF9) | ICML' 24
- [Alibaba HPN:](https://ennanzhai.github.io/pub/sigcomm24-hpn.pdf) A Data Center Network for Large Language ModelTraining
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) (Section 3)
- [FALCON](https://arxiv.org/abs/2410.12588): Pinpointing and Mitigating Stragglers for Large-Scale Hybrid-Parallel Training
- Enabling Parallelism Hot Switching for Efficient Training of Large Language Models | SOSP' 24
- [Revisiting Reliability in Large-Scale Machine Learning Research Clusters](https://arxiv.org/abs/2410.21680)
- [ScheMoE](https://dl.acm.org/doi/10.1145/3627703.3650083): An Extensible Mixture-of-Experts Distributed Training System with Tasks Scheduling | EuroSys '24
- [DynaPipe](https://arxiv.org/abs/2311.10418) : Optimizing Multi-task Training through Dynamic Pipelines | EuroSys '24
- [HAP](https://dl.acm.org/doi/10.1145/3627703.3650074): SPMD DNN Training on Heterogeneous GPU Clusters with Automated Program Synthesis | EuroSys'24
- [Demystifying Workload Imbalances in Large Transformer Model Training over Variable-length Sequences](https://arxiv.org/abs/2412.07894) | PKU
- [Improving training time and GPU utilization in geo-distributed language model training](https://arxiv.org/abs/2411.14458)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [Comet](https://arxiv.org/pdf/2502.19811): Fine-grained Computation-communication Overlapping for Mixture-of-Experts | ByteDance
- [ByteScale](https://arxiv.org/pdf/2502.21231) : Efficient Scaling of LLM Training with a 2048K Context Length on More Than 12,000 GPUs | ByteDance
- [Megalodon](https://arxiv.org/abs/2404.08801): Efficient LLM Pretraining and Inference with Unlimited Context Length
- [SPPO](https://arxiv.org/abs/2503.10377):Efficient Long-sequence LLM Training via Adaptive Sequence Pipeline Parallel Offloading
- [TileLink: Generating Efficient Compute-Communication Overlapping Kernels using Tile-Centric Primitives](https://arxiv.org/abs/2503.20313) | MLSys' 25
- [Every FLOP Counts](https://arxiv.org/abs/2503.05139): Scaling a 300B Mixture-of-Experts LING LLM without Premium GPUs| Ant Group
- [FlexSP](https://dl.acm.org/doi/abs/10.1145/3676641.3715998): Accelerating Large Language Model Training via Flexible Sequence Parallelism | ASPLOS '25
- [WeiPipe](https://dl.acm.org/doi/pdf/10.1145/3710848.3710869): Weight Pipeline Parallelism for Communication-Effective Long-Context Large Model Training | PPoPP ’25
- [WLB-LLM: Workload-Balanced 4D Parallelism for Large Language Model TraininG](https://arxiv.org/pdf/2503.17924) | UCSD
- [Mixtera](https://mboether.com/assets/pdf/bother2025mixtera.pdf): A Data Plane for Foundation Model Training | ETH
- [Flex Attention](https://arxiv.org/abs/2412.05496): A Programming Model for Generating Optimized Attention Kernels | MLSys' 25
- [Balancing Pipeline Parallelism with Vocabulary Parallelism](https://arxiv.org/abs/2411.05288) | MLSys' 25
- [SlimPipe](https://arxiv.org/abs/2504.14519): Memory-Thrifty and Efficient Pipeline Parallelism for Long-Context LLM Training | Kuaishou

  
#### Systems for Post-training / RLHF 
- [RLHFuse](https://arxiv.org/abs/2409.13221): Efficient RLHF Training for Large Language Models with Inter- and Intra-Stage Fusion | NSDI'25
- [HybridFlow](https://arxiv.org/pdf/2409.19256): A Flexible and Efficient RLHF Framework
- [ReaLHF](https://arxiv.org/html/2406.14088v1): Optimized RLHF Training for Large Language Models through Parameter Reallocation
- [NeMo-Aligner](https://arxiv.org/pdf/2405.01481): Scalable Toolkit for Efficient Model Alignment | Nvidia
- [An Adaptive Placement and Parallelism Framework for Accelerating RLHF Training](https://arxiv.org/pdf/2312.11819) | Ant
- [Systems Opportunities for LLM Fine-Tuning using Reinforcement Learning](https://dl.acm.org/doi/pdf/10.1145/3721146.3721944)

#### Fault Tolerance / Straggler Mitigation
- [Oobleck:](https://arxiv.org/abs/2309.08125) Resilient Distributed Training of Large Models Using Pipeline Templates | SOSP' 23
- [Malleus](https://arxiv.org/abs/2410.13333): Straggler-Resilient Hybrid Parallel Training of Large-scale Models via Malleable Data and Model Parallelization
- [Fire-Flyer AI-HPC: A Cost-Effective Software-Hardware Co-Design for Deep Learning](https://arxiv.org/abs/2408.14158) | DeepSeek SC' 24
- [Lazarus: Resilient and Elastic Training of Mixture-of-Experts Models with Adaptive Expert Placement](https://arxiv.org/pdf/2407.04656)
- [GEMINI:](https://dl.acm.org/doi/10.1145/3600006.3613145) Fast Failure Recovery in Distributed Training with In-Memory Checkpoints
- [ByteCheckpoint:](https://arxiv.org/abs/2407.20143) A Unified Checkpointing System for LLM Development
- [ReCycle](https://arxiv.org/pdf/2405.14009): Resilient Training of Large DNNs using Pipeline Adaptation | SOSP' 24
- [Minder](https://arxiv.org/pdf/2411.01791): Faulty Machine Detection for Large-scale Distributed Model Training | THU
- [The Streaming Batch Model for Efficient and Fault-Tolerant Heterogeneous Execution](https://arxiv.org/pdf/2501.12407)  
- [TrainMover](https://arxiv.org/pdf/2412.12636): Efficient ML Training Live Migration with No Memory Overhead | Alibaba
- [Characterizing GPU Resilience and Impact on AI/HPC Systems](https://arxiv.org/abs/2503.11901) | UIUC

### Serving
- [Orca](https://www.usenix.org/conference/osdi22/presentation/yu): A Distributed Serving System for Transformer-Based Generative Models | OSDI'22
- [Response Length Perception and Sequence Scheduling: An LLM-Empowered LLM Inference Pipeline](https://arxiv.org/abs/2305.13144) | NUS
- [Efficiently Scaling Transformer Inference](https://arxiv.org/pdf/2211.05102.pdf) | MLSys' 23
- [Flover](https://arxiv.org/pdf/2305.13484.pdf): A Temporal Fusion Framework for Efficient Autoregressive Model Parallel Inference 
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/pdf/2205.14135.pdf)
- [FlashAttention-3:](https://tridao.me/blog/2024/flash3/) Fast and Accurate Attention with Asynchrony and Low-precision
- [DeepSpeed Inference](https://arxiv.org/abs/2207.00032) : Enabling Efficient Inference of Transformer Models at Unprecedented Scale.  
- [TurboTransformers](https://arxiv.org/pdf/2010.05680.pdf): An Efficient GPU Serving System For Transformer Models
- [FlexGen](https://arxiv.org/abs/2303.06865): High-throughput Generative Inference of Large Language Models with a Single GPU | ICML' 23
- [MPCFormer](https://arxiv.org/pdf/2211.01452.pdf) : fast, performant, and private transformer inference with MPC | ICLR'23
- [POLCA](https://arxiv.org/abs/2308.12908): Power Oversubscription in LLM Cloud Providers | Microsoft
- [SARATHI](https://arxiv.org/abs/2308.16369): Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills | Microsoft 
- [AttMemo](https://arxiv.org/pdf/2301.09262.pdf): Accelerating Self-Attention with Memoization on Big Memory Systems
- [vLLM](https://vllm.ai/): Easy, Fast, and Cheap LLM Serving with PagedAttention | SOSP' 23
- [Tabi](https://dl.acm.org/doi/pdf/10.1145/3552326.3587438): An Efficient Multi-Level Inference System for Large Language Models | EuroSys' 23 
- [Flash-LLM](https://arxiv.org/pdf/2309.10285v1.pdf): Enabling Cost-Effective and Highly-Efficient Large Generative Model Inference with Unstructured Sparsity | VLDB' 24
- [AutoGen](https://arxiv.org/abs/2308.08155): Enabling Next-Gen LLM Applications via Multi-Agent Conversation | Microsoft
- [FlashDecoding++](https://arxiv.org/pdf/2311.01282.pdf): Faster Large Language Model Inference on GPUs | Tsinghua
- [DeepSpeed-MII](https://github.com/microsoft/DeepSpeed-MII): Model Implementations for Inference (MII) ｜ Microsoft
- [Punica](https://arxiv.org/abs/2310.18547): Multi-Tenant LoRA Serving | MLSys' 24
- [S-LoRA](https://arxiv.org/abs/2311.03285): Serving Thousands of Concurrent LoRA Adapters | MLSys' 24
- [SpotServe](https://arxiv.org/abs/2311.15566): Serving Generative Large Language Models on Preemptible Instances | CMU
- [SuperServe:](https://arxiv.org/pdf/2312.16733.pdf) Fine-Grained Inference Serving for Unpredictable Workloads
- [Fairness in Serving Large Language Models](https://arxiv.org/abs/2401.00588) | OSDI' 24
- [Infinite-LLM](https://arxiv.org/abs/2401.02669): Efficient LLM Service for Long Context with DistAttention and Distributed KVCache
- [CaraServe](https://arxiv.org/abs/2401.11240): CPU-Assisted and Rank-Aware LoRA Serving for Generative LLM Inference
- [DistServe](https://arxiv.org/abs/2401.09670): Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving| OSDI' 24
- [Inference without Interference](https://arxiv.org/abs/2401.11181): Disaggregate LLM Inference for Mixed Downstream Workloads
- [APIServe](https://arxiv.org/pdf/2402.01869.pdf): Efficient API Support for Large-Language Model Inferencing
- [FlexLLM](https://arxiv.org/abs/2402.18789): A System for Co-Serving Large Language Model Inference and Parameter-Efficient Finetuning
- [DéjàVu](https://arxiv.org/abs/2403.01876): KV-cache Streaming for Fast, Fault-tolerant Generative LLM Serving
- [Optimizing LLM Queries in Relational Workloads](https://arxiv.org/abs/2403.05821) | UCB
- [AttentionStore:](https://arxiv.org/pdf/2403.19708.pdf) Cost-effective Attention Reuse across Multi-turn Conversations in Large Language Model Serving | NUS
- [MuxServe:](https://arxiv.org/abs/2404.02015) Flexible Multiplexing for Efficient Multiple LLM Serving
- [LoongServe:](https://arxiv.org/pdf/2404.09526.pdf) Efficiently Serving Long-context Large Language Models with Elastic Sequence Parallelism | SOSP' 24
- [RAGCache:](https://arxiv.org/abs/2404.12457) Efficient Knowledge Caching for Retrieval-Augmented Generation | PKU
- [Andes:](https://arxiv.org/abs/2404.16283) Defining and Enhancing Quality-of-Experience in LLM-Based Text Streaming Services | Umich
- [BlockLLM:](https://arxiv.org/abs/2404.18322) Multi-tenant Finer-grained Serving for Large Language Models
- [vAttention:](https://arxiv.org/abs/2405.04437) Dynamic Memory Management for Serving LLMs without PagedAttention
- [Helix](https://arxiv.org/abs/2406.01566): Distributed Serving of Large Language Models via Max-Flow on Heterogeneous GPUs | CMU
- [Eloquent](https://arxiv.org/pdf/2401.12961v2): A More Robust Transmission Scheme for LLM Token Streaming | NAIC' 24
- [Optimizing Speculative Decoding for Serving Large Language Models Using Goodput](https://arxiv.org/abs/2406.14066v1) | UCB
- [Enabling Elastic Model Serving with MultiWorld](https://arxiv.org/html/2407.08980v1) | Cisco Research
- [Prepacking](https://arxiv.org/abs/2404.09529): A Simple Method for Fast Prefilling and Increased Throughput in Large Language Models
- [NanoFlow](https://arxiv.org/abs/2408.12757): Towards Optimal Large Language Model Serving Throughput
- [Responsive ML inference in multi-tenanted environments using AQUA](https://arxiv.org/abs/2407.21255)
- [One Queue Is All You Need](https://arxiv.org/abs/2407.00047): Resolving Head-of-Line Blocking in Large Language Model Serving
- [MemServe](https://arxiv.org/abs/2406.17565): Context Caching for Disaggregated LLM Serving with Elastic Memory Pool
- [dLoRA: Dynamically Orchestrating Requests and Adapters for LoRA LLM Serving](https://www.usenix.org/conference/osdi24/presentation/wu-bingyang) | OSDI' 24
- [Llumnix](https://www.usenix.org/conference/osdi24/presentation/sun-biao): Dynamic Scheduling for Large Language Model Serving | OSDI' 24
- [Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve](https://www.usenix.org/conference/osdi24/presentation/agrawal) | OSDI' 24
- [InfiniGen](https://www.usenix.org/conference/osdi24/presentation/lee): Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management
- [ServerlessLLM: Low-Latency Serverless Inference for Large Language Models](https://www.usenix.org/conference/osdi24/presentation/fu) | OSDI' 24
- [CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving](https://dl.acm.org/doi/10.1145/3651890.3672274) | SIGCOMM' 24
- [Preble](https://arxiv.org/abs/2407.00023): Efficient Distributed Prompt Scheduling for LLM Serving
- [Mnemosyne](https://arxiv.org/abs/2409.17264): Parallelization Strategies for Efficiently Serving Multi-Million Context Length LLM Inference Requests Without Approximations
- [ConServe](https://arxiv.org/html/2410.01228v1): Harvesting GPUs for Low-Latency and High-Throughput Large Language Model Serving
- [BlockLLM](https://arxiv.org/abs/2404.18322): Multi-tenant Finer-grained Serving for Large Language Models
- [Context Parallelism for Scalable Million-Token Inference](https://arxiv.org/abs/2411.01783)
- [xDiT](https://arxiv.org/abs/2411.01738): an Inference Engine for Diffusion Transformers (DiTs) with Massive Parallelism
- [Pie](https://arxiv.org/abs/2411.09317): Pooling CPU Memory for LLM Inference
- [NEO](https://arxiv.org/abs/2411.01142): Saving GPU Memory Crisis with CPU Offloading for Online LLM Inference
- [FastSwitch](https://arxiv.org/abs/2411.18424): Optimizing Context Switching Efficiency in Fairness-aware Large Language Model Serving
- [Flash Communication](https://arxiv.org/abs/2412.04964): Reducing Tensor Parallelization Bottleneck for Fast Large Language Model Inference
- [FlashInfer](https://arxiv.org/abs/2501.01005): Efficient and Customizable Attention Engine for LLM Inference Serving
- [Fast Inference for Augmented Large Language Models](https://arxiv.org/abs/2410.18248)
- [A System for Microserving of LLMs](https://arxiv.org/abs/2412.12488) | CMU
- [iServe](https://arxiv.org/abs/2501.13111) : An Intent-based Serving System for LLMs| UT Austin
- [Locality-aware Fair Scheduling in LLM Serving](https://arxiv.org/abs/2501.14312) | UCB
- [Towards Efficient Large Multimodal Model Serving](https://arxiv.org/abs/2502.00937) | MSFT
- [DeltaZip: Efficient Serving of Multiple Full-Model-Tuned LLMs](https://anakli.inf.ethz.ch/papers/deltazip.pdf)
- [PIM Is All You Need](https://arxiv.org/abs/2502.07578): A CXL-Enabled GPU-Free System for Large Language Model Inference | ASPLOS' 25
- [λScale](https://arxiv.org/abs/2502.09922): Enabling Fast Scaling for Serverless Large Language Model Inference
- [AIBrix: Towards Scalable and Cost-Effective LLM Inference Infrastructure](https://github.com/vllm-project/aibrix/blob/main/docs/paper/AIBrix_White_Paper_0219_2025.pdf) | vLLM
- [Serving Models, Fast and Slow:Optimizing Heterogeneous LLM Inferencing Workloads at Scale](https://arxiv.org/abs/2502.14617) 
- [Make LLM Inference Affordable to Everyone: Augmenting GPU Memory with NDP-DIMM](https://arxiv.org/abs/2502.16963)
- [Jenga](https://arxiv.org/abs/2503.18292): Effective Memory Management for Serving LLM with Heterogeneity
- [AQUA](https://arxiv.org/abs/2407.21255) : Network-Accelerated Memory Offloading for LLMs in Scale-Up GPU Domains | ASPLOS 2025
- [MegaScale-Infer](https://arxiv.org/pdf/2504.02263): Serving Mixture-of-Experts at Scale with Disaggregated Expert Parallelism | Bytedance
- [Towards End-to-End Optimization of LLM-based Applications with Ayo](https://dl.acm.org/doi/10.1145/3676641.3716278) | ASPLOS '25
- [CacheBlend](https://dl.acm.org/doi/10.1145/3689031.3696098) : Fast Large Language Model Serving for RAG with Cached Knowledge Fusion | EuroSys' 25 (Best Paper)
- [ThunderServe](https://arxiv.org/pdf/2502.09334): High-performance and Cost-efficient LLM Serving in Cloud Environments | MLSys' 25
- [SLOs-Serve](https://arxiv.org/abs/2504.08784): Optimized Serving of Multi-SLO LLMs
  
#### Compound AI Systems  
- [ALTO](https://arxiv.org/abs/2403.04311): An Efficient Network Orchestrator for Compound AI Systems | Stanford & UCB
- [Parrot](https://www.usenix.org/conference/osdi24/presentation/lin-chaofan): Efficient Serving of LLM-based Applications with Semantic Variable | OSDI' 24
- [Efficiently Serving LLM Reasoning Programs with Certaindex](https://arxiv.org/pdf/2412.20993) | UCSD
- [Autellix](https://arxiv.org/pdf/2502.13965): An Efficient Serving Engine for LLM Agents as General Programs | UCB
- [RAGO](https://arxiv.org/abs/2503.14649v2): Systematic Performance Optimization for Retrieval-Augmented Generation Serving | ISCA'25

#### Serving at the edge
- [LLM in a flash: Efficient Large Language Model Inference with Limited Memory](https://arxiv.org/abs/2312.11514) | Apple
- [STI](https://arxiv.org/abs/2207.05022): Turbocharge NLP Inference at the Edge via Elastic Pipelining | ASPLOS 23 
- [PowerInfer](https://arxiv.org/abs/2312.12456): Fast Large Language Model Serving with a Consumer-grade GPU | SOSP' 24
- [MoE-Lightning: High-Throughput MoE Inference on Memory-constrained GPUs](https://arxiv.org/abs/2411.11217)
- [InfiniteHiP](https://arxiv.org/abs/2502.08910): Extending Language Model Context Up to 3 Million Tokens on a Single GPU
- [prima.cpp](https://arxiv.org/pdf/2504.08791): PRIMA.CPP: Speeding Up 70B-Scale LLM Inference on Low-Resource Everyday Home Clusters


#### System Efficiency Optimization - Model Co-design
- [Fast Distributed Inference Serving for Large Language Models](https://arxiv.org/abs/2305.05920) | PKU
- [FrugalGPT](https://arxiv.org/pdf/2305.05176.pdf): How to Use Large Language Models While Reducing Cost and Improving Performance |  Stanford
- [H2O](https://arxiv.org/abs/2306.14048): Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models | ICML ES-FoMo Workshop 2023
- [Inference with Reference](https://arxiv.org/abs/2304.04487): Lossless Acceleration of Large Language Models
- [SkipDecode](https://arxiv.org/abs/2307.02628): Autoregressive Skip Decoding with Batching and Caching for Efficient LLM Inferencex
- [Scissorhands](https://arxiv.org/abs/2305.17118): Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time
- [Knowledge-preserving Pruning for Pre-trained Language Models without Retraining](https://arxiv.org/pdf/2308.03449.pdf) | SNU
- [Accelerating LLM Inference with Staged Speculative Decoding](https://arxiv.org/pdf/2308.04623.pdf) | ICML' 23
- [SpecInfer](https://www.cs.cmu.edu/~zhihaoj2/papers/specinfer.pdf): Accelerating Generative LLM Serving with Speculative Inference and Token Tree Verification | CMU
- [Deja Vu](https://proceedings.mlr.press/v202/liu23am.html): Contextual Sparsity for Efficient LLMs at Inference Time | ICML' 23
- [S3](https://arxiv.org/pdf/2306.06000.pdf): Increasing GPU Utilization during Generative Inference for Higher Throughput | Havard
- [LLMCad](https://arxiv.org/abs/2309.04255): Fast and Scalable On-device Large Language Model Inference
- [Skeleton-of-Thought](https://arxiv.org/abs/2307.15337): Large Language Models Can Do Parallel Decoding | THU
- [LoRAShear](https://arxiv.org/abs/2310.18356): Efficient Large Language Model Structured Pruning and Knowledge Recovery ｜ Microsoft
- [Ring Attention](https://arxiv.org/pdf/2310.01889.pdf) with Blockwise Transformers for Near-Infinite Context | UCB
- [Learned Best-Effort LLM Serving](https://arxiv.org/abs/2401.07886) | UCB
- [Star Attention](https://arxiv.org/pdf/2411.17116) : Efficient LLM Inference over Long Sequences| NVIDIA
- [FFN Fusion](https://arxiv.org/abs/2503.18908): Rethinking Sequential Computation in Large Language Models
- [SageAttention](https://arxiv.org/pdf/2410.02367): Accurate 8-Bit Attention for Plug-and-play Inference Acceleration
- [SageAttention2](https://arxiv.org/pdf/2411.10958): Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization
- [SpargeAttn](https://arxiv.org/pdf/2502.18137): Accurate Sparse Attention Accelerating Any Model Inference
- [Training Transformers with 4-bit Integers](https://arxiv.org/abs/2306.11987) | NeurIPS' 23
- [Jetfire: Efficient and Accurate Transformer Pretraining with INT8 Data Flow and Per-Block Quantization](https://arxiv.org/abs/2403.12422) | ICML' 24
- [COAT: Compressing Optimizer states and Activation for Memory-Efficient FP8 Training](https://arxiv.org/abs/2410.19313) | ICLR'25
  
### Fine-tuning Systems
- [Ymir:](https://tianweiz07.github.io/Papers/24-ics-2.pdf) A Scheduler for Foundation Model Fine-tuning Workloads in Datacenters | ICS' 24
  
### Multi-Modal Systems
- [MOSEL](https://arxiv.org/pdf/2310.18481.pdf): Inference Serving Using Dynamic Modality Selection
- [DISTMM](https://www.amazon.science/publications/distmm-accelerating-distributed-multimodal-model-training): Accelerating distributed multimodal model training | NSDI' 24
- [Approximate Caching for Efficiently Serving Diffusion Models](https://arxiv.org/abs/2312.04429) | Adobe Research
- [Generative AI Beyond LLMs](https://arxiv.org/pdf/2312.14385): System Implications of Multi-Modal Generation | Meta
- [Characterizing and Efficiently Accelerating Multimodal Generation Model Inference](https://arxiv.org/abs/2410.00215) | Meta
- [DistriFusion:](https://arxiv.org/abs/2402.19481) Distributed Parallel Inference for High-Resolution Diffusion Models |  MIT
- [Optimus:](https://www.arxiv.org/abs/2408.03505) Accelerating Large-Scale Multi-Modal LLM Training by Bubble Exploitation
- [Addressing Model and Data Heterogeneity in Multimodal Large Language Model Training](https://arxiv.org/pdf/2408.04275v1) | PKU
- [LongVILA: Scaling Long-Context Visual Language Models for Long Videos](https://arxiv.org/abs/2408.10188) | NVIDIA
- [FlexCache: Flexible Approximate Cache System for Video Diffusion](https://arxiv.org/abs/2501.04012) | University of Waterloo
- [Cornstarch](https://arxiv.org/abs/2503.11367): Distributed Multimodal Training Must Be Multimodality-Aware | UMich
- [PipeWeaver](https://arxiv.org/abs/2504.14145): Addressing Data Dynamicity in Large Multimodal Model Training with Dynamic Interleaved Pipeline | SJTU
 
## LLM for Systems
- [Large Language Models for Compiler Optimization](https://arxiv.org/abs/2309.07062)
- [The Hitchhiker's Guide to Program Analysis](https://arxiv.org/abs/2308.00245): A Journey with Large Language Models
- [LLM-Assisted Code Cleaning For Training Accurate Code Generators](https://arxiv.org/abs/2311.14904) | UCB
- [Efficient Multi-Task Large Model Training via Data Heterogeneity-aware Model Management](https://arxiv.org/abs/2409.03365)
- [If At First You Don’t Succeed, Try, Try, Again...?](https://www.microsoft.com/en-us/research/publication/if-at-first-you-dont-succeed-try-try-again-insights-and-llm-informed-tooling-for-detecting-retry-bugs-in-software-systems/) | SOSP' 24
- [Aceso](https://dl.acm.org/doi/pdf/10.1145/3627703.3629554): Efficient Parallel DNN Training through Iterative Bottleneck Alleviation | EuroSys '24
- [GMorph](https://dl.acm.org/doi/10.1145/3627703.3650074): Accelerating Multi-DNN Inference via Model Fusion | EuroSys '24
- [Automatic Root Cause Analysis via Large Language Models for Cloud Incidents](https://dl.acm.org/doi/10.1145/3627703.3629553) | EuroSys '24


## [ML Systems](mlsystems.md)


## Survey Paper
- [Efficient Large Language Models: A Survey](https://arxiv.org/abs/2312.03863)
- [Challenges and Applications of Large Language Models](https://arxiv.org/pdf/2307.10169.pdf)
- [Beyond Efficiency: A Systematic Survey of Resource-Efficient Large Language Models](https://arxiv.org/abs/2401.00625)
- [Towards Efficient Generative Large Language Model Serving: A Survey from Algorithms to Systems](https://arxiv.org/abs/2312.15234)


## LLM Benchmark / Leaderboard ? Traces
-  [LLM Energy Leaderboard](https://ml.energy/leaderboard) | Umich
-  [LLM-Perf Leaderboard](https://huggingface.co/spaces/optimum/llm-perf-leaderboard) | HuggingFace
-  [Aviary Explorer](https://aviary.anyscale.com/) | Anyscale
-  [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) | HuggingFace
-  [HELM](https://crfm.stanford.edu/helm/latest/) | Stanford
-  [LMSYS](https://chat.lmsys.org) | UCB
-  [Towards Efficient and Reliable LLM Serving: A Real-World Workload Study](https://arxiv.org/abs/2401.17644)
 
## LLM Frameworks
- [DeepSpeed](https://github.com/microsoft/DeepSpeed): a deep learning optimization library that makes distributed training and inference easy, efficient, and effective | Microsoft
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | Nvidia
- [Accelerate](https://huggingface.co/docs/accelerate/index) | Hugging Face
- [Ray-LLM](https://github.com/ray-project/ray-llm) | Ray
- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [Megatron](https://github.com/NVIDIA/Megatron-LM) | Nvidia
- [NeMo](https://github.com/NVIDIA/NeMo) | Nvidia
- [torchtitan](https://github.com/pytorch/torchtitan) | PyTorch
- [vLLM](https://github.com/vllm-project/vllm) | UCB
- [SGLang](https://github.com/sgl-project/sglang) | UCB
- [TGI](https://huggingface.co/docs/text-generation-inference/en/index) | Hugging Face
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)
- [veRL](https://github.com/volcengine/verl) | ByteDance
- [veScale](https://github.com/volcengine/vescale) | ByteDance
- [KV Transformers](https://github.com/kvcache-ai/ktransformers)
- [DeepSeek Open Infra](https://github.com/deepseek-ai/open-infra-index)
- [Dynamo](https://github.com/ai-dynamo/dynamo) | NVIDA
- [Cornstarch](https://github.com/cornstarch-org/Cornstarch): Build, Train, Run Your Own Multimodal Model | UMich
- [verl](https://github.com/volcengine/verl): Volcano Engine Reinforcement Learning for LLM
  
## Related ML Readings
- [Large Transformer Model Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/)
- [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/)
- [The Transformer Family Version 2.0](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/)
- [Full Stack Optimization of Transformer Inference](https://arxiv.org/pdf/2302.14017.pdf): a Survey | UCB

## MLSys Courses
- Systems for Machine Learning | (Stanford)[https://cs229s.stanford.edu/fall2023/]
- Systems for Generative AI | (Umich)[https://github.com/mosharaf/eecs598/tree/w24-genai]
- Systems for AI - LLMs | (GT)[https://cs8803-sp24.anand-iyer.com/]


## Other Reading
- [A curated list of Large Language Model](https://github.com/Hannibal046/Awesome-LLM)
- [AI systems paper list](https://github.com/lambda7xx/awesome-AI-system)
- [A baseline repository of Auto-Parallelism in Training Neural Networks](https://github.com/ConnollyLeon/awesome-Auto-Parallelism)
- [Numbers every LLM Developer should know](https://github.com/ray-project/llm-numbers)
- [100,000 H100 Clusters:](https://www.semianalysis.com/p/100000-h100-clusters-power-network) Power, Network Topology, Ethernet vs InfiniBand, Reliability, Failures, Checkpointing
- [OpenAI Keynote on Building Scalable AI Infrastructure](https://www.servethehome.com/openai-keynote-on-building-scalable-ai-infrastructure/)
- [Awesome ML SYS Tutorial](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/tree/main)
