# Awesome LLM Systems Papers

A curated list of Large Language Model systems related academic papers, articles, tutorials, slides and projects. Star this repository, and then you can keep abreast of the latest developments of this booming research field.
## Table of Contents

- [LLM Systems](#llm-systems)
  - [Training](#training)
    - [Pre-training](#pre-training)
    - [Post Training](#systems-for-post-training--rlhf)
    - [Fault Tolerance / Straggler Mitigation](#fault-tolerance--straggler-mitigation)
  - [Serving](#serving)
    - [LLM serving](#llm-serving)
    - [Agent Systems](#agent-systems)
    - [Serving at the edge](#serving-at-the-edge)
    - [System Efficiency Optimization - Model Co-design](#system-efficiency-optimization---model-co-design)
  - [Multi-Modal Training Systems](#multi-modal-training-systems)
  - [Multi-Modal Serving Systems](#multi-modal-serving-systems)
- [LLM for Systems](#llm-for-systems)
- [Industrial LLM Technical Report](#industrial-llm-technical-report)
- [LLM Frameworks](#llm-frameworks)
  - [Training](#training-1)
  - [Post-Training](#post-training)
  - [Serving](#serving-1)
- [ML Systems](#ml-systems)
- [Survey Paper](#survey-paper)
- [LLM Benchmark / Leaderboard / Traces](#llm-benchmark--leaderboard--traces)
- [Related ML Readings](#related-ml-readings)
- [MLSys Courses](#mlsys-courses)
- [Other Reading](#other-reading)


## LLM Systems
### Training
#### Pre-training
- [Megatron-LM](https://arxiv.org/pdf/1909.08053.pdf): Training Multi-Billion Parameter Language Models Using Model Parallelism
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/pdf/2104.04473.pdf)
- [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/pdf/2205.05198.pdf)
- [Optimized Network Architectures for Large Language Model Training with Billions of Parameters](https://arxiv.org/pdf/2307.12169.pdf) | MIT
- [Carbon Emissions and Large Neural Network Training](https://arxiv.org/pdf/2104.10350.pdf?fbclid=IwAR2o0_3HCtTnMxKbXka0OPrHzl8sCzQSSOYp0AOav76-zVWl_pYek2jX8Pk) | Google, UCB
- [Perseus](https://arxiv.org/abs/2312.06902v1): Removing Energy Bloat from Large Model Training | SOSP' 24
- [MegaScale](https://arxiv.org/abs/2402.15627): Scaling Large Language Model Training to More Than 10,000 GPUs | ByteDance
- [DISTMM](https://www.usenix.org/conference/nsdi24/presentation/huang): Accelerating distributed multimodal model training | NSDI' 24
- [A Codesign of Scheduling and Parallelization for Large Model Training in Heterogeneous Clusters](https://arxiv.org/pdf/2403.16125)
- [Pipeline Parallelism with Controllable Memory](https://arxiv.org/abs/2405.15362) | Sea AI Lab
- [Boosting Large-scale Parallel Training Efficiency with C4](https://arxiv.org/abs/2406.04594): A Communication-Driven Approach
- [Scaling Beyond the GPU Memory Limit for Large Mixture-of-Experts Model Training](https://openreview.net/pdf?id=uLpyWQPyF9) | ICML' 24
- [Alibaba HPN:](https://ennanzhai.github.io/pub/sigcomm24-hpn.pdf) A Data Center Network for Large Language ModelTraining
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) (Section 3)
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
- [WLB-LLM: Workload-Balanced 4D Parallelism for Large Language Model TraininG](https://arxiv.org/pdf/2503.17924) | OSDI' 25
- [Mixtera](https://mboether.com/assets/pdf/bother2025mixtera.pdf): A Data Plane for Foundation Model Training | ETH
- [Flex Attention](https://arxiv.org/abs/2412.05496): A Programming Model for Generating Optimized Attention Kernels | MLSys' 25
- [Balancing Pipeline Parallelism with Vocabulary Parallelism](https://arxiv.org/abs/2411.05288) | MLSys' 25
- [SlimPipe](https://arxiv.org/abs/2504.14519): Memory-Thrifty and Efficient Pipeline Parallelism for Long-Context LLM Training | Kuaishou
- [Scaling Llama 3 Training with Efficient Parallelism Strategies](https://aisystemcodesign.github.io/papers/Llama3-ISCA25.pdf) | ISCA' 25
- [Lumos](https://arxiv.org/abs/2504.09307) : Efficient Performance Modeling and Estimation for Large-scale LLM Training| MLSys' 25
- [BurstEngine](https://arxiv.org/abs/2509.19836): an Efficient Distributed Framework for Training Transformers on Extremely Long Sequences of over 1M Tokens
- [Zeppelin](https://arxiv.org/abs/2509.21841): Balancing Variable-length Workloads in Data Parallel Large Model Training
- [Robust LLM Training Infrastructure at ByteDance](https://sigops.org/s/conferences/sosp/2025/accepted.html) | SOSP' 25
- [Sailor: Automating Distributed Training over Dynamic, Heterogeneous, and Geo-distributed Clusters](https://sigops.org/s/conferences/sosp/2025/accepted.html) | SOSP' 25
- [Tempo: Compiled Dynamic Deep Learning with Symbolic Dependence Graphs](https://sigops.org/s/conferences/sosp/2025/accepted.html) | SOSP' 25
- [Mycroft: Tracing Dependencies in Collective Communication Towards Reliable LLM Training](https://sigops.org/s/conferences/sosp/2025/accepted.html) | SOSP' 25
- [DCP: Addressing Input Dynamism In Long-Context Training via Dynamic Context Parallelism](https://sigops.org/s/conferences/sosp/2025/accepted.html) | SOSP' 25
- [TrainVerify: Equivalence-Based Verification for Distributed LLM Training](https://sigops.org/s/conferences/sosp/2025/accepted.html) | SOSP' 25
- [Collective Communication for 100k+ GPUs](https://arxiv.org/abs/2510.20171): Large-scale collective communication optimization for massive GPU clusters


#### Systems for Post-training / RLHF 
- [Ymir:](https://tianweiz07.github.io/Papers/24-ics-2.pdf) A Scheduler for Foundation Model Fine-tuning Workloads in Datacenters | ICS' 24
- [RLHFuse](https://arxiv.org/abs/2409.13221): Efficient RLHF Training for Large Language Models with Inter- and Intra-Stage Fusion | NSDI'25
- [HybridFlow](https://arxiv.org/pdf/2409.19256): A Flexible and Efficient RLHF Framework
- [ReaLHF](https://arxiv.org/html/2406.14088v1): Optimized RLHF Training for Large Language Models through Parameter Reallocation
- [NeMo-Aligner](https://arxiv.org/pdf/2405.01481): Scalable Toolkit for Efficient Model Alignment | Nvidia
- [An Adaptive Placement and Parallelism Framework for Accelerating RLHF Training](https://arxiv.org/pdf/2312.11819) | Ant
- [Systems Opportunities for LLM Fine-Tuning using Reinforcement Learning](https://dl.acm.org/doi/pdf/10.1145/3721146.3721944)
- [AReaL](https://arxiv.org/pdf/2505.24298): A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning | [Code](https://github.com/inclusionAI/AReaL) | Ant
- [StreamRL](https://arxiv.org/abs/2504.15930): Scalable, Heterogeneous, and Elastic RL for LLMs with Disaggregated Stream Generation
- [RL-Factory](https://github.com/Simple-Efficient/RL-Factory): Train your Agent model via our easy and efficient framework
- [PLoRA](https://arxiv.org/pdf/2508.02932): Efficient LoRA Hyperparameter Tuning for Large Models
- [History Rhymes](https://arxiv.org/abs/2508.18588): Accelerating LLM Reinforcement Learning with RhymeRL
- [APRIL](https://arxiv.org/abs/2509.18521): Active Partial Rollouts in Reinforcement Learning to tame long-tail generation

#### Fault Tolerance / Straggler Mitigation
- [Oobleck:](https://arxiv.org/abs/2309.08125) Resilient Distributed Training of Large Models Using Pipeline Templates | SOSP' 23
- [FALCON](https://arxiv.org/abs/2410.12588): Pinpointing and Mitigating Stragglers for Large-Scale Hybrid-Parallel Training
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
- [Understanding Stragglers in Large Model Training Using What-if Analysis](https://arxiv.org/abs/2505.05713) | OSDI' 25


### Serving
#### LLM serving
- [Orca](https://www.usenix.org/conference/osdi22/presentation/yu): A Distributed Serving System for Transformer-Based Generative Models | OSDI'22
- [Response Length Perception and Sequence Scheduling: An LLM-Empowered LLM Inference Pipeline](https://arxiv.org/abs/2305.13144) | NUS
- [Efficiently Scaling Transformer Inference](https://arxiv.org/pdf/2211.05102.pdf) | MLSys' 23
- [Flover](https://arxiv.org/pdf/2305.13484.pdf): A Temporal Fusion Framework for Efficient Autoregressive Model Parallel Inference 
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/pdf/2205.14135.pdf)
- [FlashAttention-3:](https://tridao.me/blog/2024/flash3/) Fast and Accurate Attention with Asynchrony and Low-precision
- [SageAttention](https://arxiv.org/pdf/2410.02367): Accurate 8-Bit Attention for Plug-and-play Inference Acceleration | ICLR 2025
- [SageAttention2](https://arxiv.org/pdf/2411.10958): Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization | ICML 2025
- [SageAttention3](https://arxiv.org/pdf/2505.11594): SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-Bit Training | NeurIPS 2025 spotlight
- [SageAttention2++](https://arxiv.org/abs/2505.21136): SageAttention2++: A More Efficient Implementation of SageAttention2 | ICML ES-FoMo Workshop 2025
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
- [Tempo](https://arxiv.org/abs/2504.20068): Application-aware LLM Serving with Mixed SLO Requirements
- [Prism](https://arxiv.org/pdf/2505.04021): Unleashing GPU Sharing for Cost-Efficient Multi-LLM Serving | UCLA
- [RetroInfer](https://arxiv.org/abs/2505.02922): A Vector-Storage Approach for Scalable Long-Context LLM Inference
- [Efficient Serving of LLM Applications with Probabilistic Demand Modeling](https://arxiv.org/abs/2506.14851)
- [eLLM](https://arxiv.org/abs/2506.14851) : Elastic Memory Management Framework for Efficient LLM Serving
- [DiSCo](https://arxiv.org/abs/2502.11417v2): Device-Server Collaborative LLM-Based Text Streaming Services
- [DynaServe](https://arxiv.org/abs/2504.09285): Unified and Elastic Execution for Dynamic Disaggregated LLM Serving
- [HyGen](https://arxiv.org/pdf/2501.14808): Efficient LLM Serving via Elastic Online-Offline Request Co-location
- [WaferLLM: A Wafer‑Scale LLM Inference System](https://arxiv.org/abs/2502.04563) | OSDI 25
- [BlitzScale: Fast and Live Large Model Autoscaling with O(1) Host Caching](https://arxiv.org/abs/2412.17246) | OSDI 25
- [TokenWeave: Efficient Compute-Communication Overlap for Distributed LLM Inference](https://arxiv.org/abs/2505.11329) | [Code](https://github.com/microsoft/tokenweave) | ArXiv'25
- [Nexus](https://arxiv.org/abs/2507.06608v2): Taming Throughput-Latency Tradeoff in LLM Serving via Efficient GPU Sharing
- [Taming the Chaos](https://arxiv.org/abs/2508.19559): Coordinated Autoscaling for Heterogeneous and Disaggregated LLM Inference | Seed
- [TokenLake](https://arxiv.org/abs/2508.17219): A Unified Segment-level Prefix Cache Pool for Fine-grained Elastic Long-Context LLM Serving
- [Expert-as-a-Service](https://arxiv.org/abs/2509.17863): Towards Efficient, Scalable, and Robust Large-scale MoE Serving
- [Shift Parallelism](https://arxiv.org/pdf/2509.16495): Low-Latency, High-Throughput LLM Inference for Dynamic Workloads
- [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
- [The Cost of Dynamic Reasoning: Demystifying AI Agents and Test-Time Scaling from an AI Infrastructure Perspective](https://arxiv.org/abs/2506.04301)
- [Barbarians at the Gate: How AI is Upending Systems Research](https://arxiv.org/pdf/2510.06189)
- [Mercury: Unlocking Multi-GPU Operator Optimization for LLMs via Remote Memory Scheduling](https://sigops.org/s/conferences/sosp/2025/accepted.html) | SOSP' 25
- [DiffKV: Differentiated Memory Management for Large Language Models with Parallel KV Compaction](https://sigops.org/s/conferences/sosp/2025/accepted.html) | SOSP' 25
- [Pie: A Programmable Serving System for Emerging LLM Applications](https://sigops.org/s/conferences/sosp/2025/accepted.html) | SOSP' 25
- [Aegaeon: Effective GPU Pooling for Concurrent LLM Serving on the Market](https://sigops.org/s/conferences/sosp/2025/accepted.html) | SOSP' 25
- [Jenga: Effective Memory Management for Serving LLM with Heterogeneity](https://sigops.org/s/conferences/sosp/2025/accepted.html) | SOSP' 25
- [IC-Cache: Efficient Large Language Model Serving via In-context Caching](https://sigops.org/s/conferences/sosp/2025/accepted.html) | SOSP' 25
- [PrefillOnly: An Inference Engine for Prefill-only Workloads in Large Language Model Applications](https://sigops.org/s/conferences/sosp/2025/accepted.html) | SOSP' 25
- [KTransformers: Unleashing the Full Potential of CPU/GPU Hybrid Inference for MoE Models](https://sigops.org/s/conferences/sosp/2025/accepted.html) | SOSP' 25
- [The ML.ENERGY Benchmark](https://arxiv.org/abs/2505.06371): Toward Automated Inference Energy Measurement and Optimization | NeurIPS' 25
- [Serve Programs, Not Prompts](https://arxiv.org/abs/2510.25412): Efficient LLM serving system for structured program execution


#### Agent Systems
- [Supporting Our AI Overlords](https://arxiv.org/pdf/2509.00997): Redesigning Data Systems to be Agent-First | UCB
- [ALTO](https://arxiv.org/abs/2403.04311): An Efficient Network Orchestrator for Compound AI Systems | Stanford & UCB
- [Parrot](https://www.usenix.org/conference/osdi24/presentation/lin-chaofan): Efficient Serving of LLM-based Applications with Semantic Variable | OSDI' 24
- [Efficiently Serving LLM Reasoning Programs with Certaindex](https://arxiv.org/pdf/2412.20993) | UCSD
- [Autellix](https://arxiv.org/pdf/2502.13965): An Efficient Serving Engine for LLM Agents as General Programs | UCB
- [RAGO](https://arxiv.org/abs/2503.14649v2): Systematic Performance Optimization for Retrieval-Augmented Generation Serving | ISCA'25
- [Circinus](https://arxiv.org/abs/2504.16397): Efficient Query Planner for Compound ML Serving | UIUC
- [Patchwork: A Unified Framework for RAG Serving](https://arxiv.org/abs/2505.07833)
- [KVFlow](https://arxiv.org/abs/2507.07400): Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows
- [DroidSpeak](https://arxiv.org/abs/2411.02820): KV Cache Sharing for Cross-LLM Communication and Multi-LLM Serving
- [Murakkab](https://arxiv.org/abs/2508.18298): Resource-Efficient Agentic Workflow Orchestration in Cloud Platforms
- [HedraRAG: Co-Optimizing Generation and Retrieval for Heterogeneous RAG Workflows](https://sigops.org/s/conferences/sosp/2025/accepted.html) | SOSP' 25
- [METIS: Fast Quality-Aware RAG Systems with Configuration Adaptation](https://sigops.org/s/conferences/sosp/2025/accepted.html) | SOSP' 25

#### Serving at the edge
- [LLM in a flash: Efficient Large Language Model Inference with Limited Memory](https://arxiv.org/abs/2312.11514) | Apple
- [STI](https://arxiv.org/abs/2207.05022): Turbocharge NLP Inference at the Edge via Elastic Pipelining | ASPLOS 23 
- [PowerInfer](https://arxiv.org/abs/2312.12456): Fast Large Language Model Serving with a Consumer-grade GPU | SOSP' 24
- [MoE-Lightning: High-Throughput MoE Inference on Memory-constrained GPUs](https://arxiv.org/abs/2411.11217)
- [InfiniteHiP](https://arxiv.org/abs/2502.08910): Extending Language Model Context Up to 3 Million Tokens on a Single GPU
- [prima.cpp](https://arxiv.org/pdf/2504.08791): PRIMA.CPP: Speeding Up 70B-Scale LLM Inference on Low-Resource Everyday Home Clusters
- [Characterizing Mobile SoC for Accelerating Heterogeneous LLM Inference](https://sigops.org/s/conferences/sosp/2025/accepted.html) | SOSP' 25


#### System Efficiency Optimization - Model Co-design
- [Sparse-Linear Attention](https://www.arxiv.org/pdf/2509.24006): SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse–Linear Attention | Tsinghua
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
- [SpargeAttention](https://arxiv.org/pdf/2502.18137): SpargeAttention: Accurate and Training-free Sparse Attention Accelerating Any Model Inference | ICML' 25
- [Training Transformers with 4-bit Integers](https://arxiv.org/abs/2306.11987) | NeurIPS' 23
- [Jetfire: Efficient and Accurate Transformer Pretraining with INT8 Data Flow and Per-Block Quantization](https://arxiv.org/abs/2403.12422) | ICML' 24
- [COAT: Compressing Optimizer states and Activation for Memory-Efficient FP8 Training](https://arxiv.org/abs/2410.19313) | ICLR'25
- [Efficient Mixed-Precision Large Language Model Inference with TurboMind](https://arxiv.org/pdf/2508.15601v1) | Shanghai AI Lab 
  
### Multi-Modal Training Systems
- [DISTMM](https://www.usenix.org/conference/nsdi24/presentation/huang): Accelerating distributed multimodal model training | NSDI' 24
- [Optimus:](https://www.arxiv.org/abs/2408.03505) Accelerating Large-Scale Multi-Modal LLM Training by Bubble Exploitation
- [Addressing Model and Data Heterogeneity in Multimodal Large Language Model Training](https://arxiv.org/pdf/2408.04275v1) | PKU
- [Cornstarch](https://arxiv.org/abs/2503.11367): Distributed Multimodal Training Must Be Multimodality-Aware | UMich
- [PipeWeaver](https://arxiv.org/abs/2504.14145): Addressing Data Dynamicity in Large Multimodal Model Training with Dynamic Interleaved Pipeline | SJTU

### Multi-Modal Serving Systems
- [xDiT](https://arxiv.org/abs/2411.01738): an Inference Engine for Diffusion Transformers (DiTs) with Massive Parallelism
- [MOSEL](https://arxiv.org/pdf/2310.18481.pdf): Inference Serving Using Dynamic Modality Selection
- [Approximate Caching for Efficiently Serving Diffusion Models](https://arxiv.org/abs/2312.04429) | Adobe Research
- [Generative AI Beyond LLMs](https://arxiv.org/pdf/2312.14385): System Implications of Multi-Modal Generation | Meta
- [Characterizing and Efficiently Accelerating Multimodal Generation Model Inference](https://arxiv.org/abs/2410.00215) | Meta
- [DistriFusion:](https://arxiv.org/abs/2402.19481) Distributed Parallel Inference for High-Resolution Diffusion Models |  MIT
- [LongVILA: Scaling Long-Context Visual Language Models for Long Videos](https://arxiv.org/abs/2408.10188) | NVIDIA
- [FlexCache: Flexible Approximate Cache System for Video Diffusion](https://arxiv.org/abs/2501.04012) | University of Waterloo
- [DDiT](https://arxiv.org/abs/2506.13497v1): Dynamic Resource Allocation for Diffusion Transformer Model Serving
- [PATCHEDSERVE](https://arxiv.org/pdf/2501.09253): A Patch Management Framework for SLO-Optimized Hybrid Resolution Diffusion Serving
- [ElasticMM](https://arxiv.org/abs/2507.10069): Efficient Multimodal LLMs Serving with Elastic Multimodal Parallelism
- [TetriServe](https://arxiv.org/abs/2510.01565): Efficient DiT Serving for Heterogeneous Image Generation
- [dInfer](https://arxiv.org/abs/2510.08666): An Efficient Inference Framework for Diffusion Language Models
- [Fast-dLLM v2](https://arxiv.org/abs/2509.26328): Efficient Block-Diffusion LLM


## LLM for Systems
- [Large Language Models for Compiler Optimization](https://arxiv.org/abs/2309.07062)
- [The Hitchhiker's Guide to Program Analysis](https://arxiv.org/abs/2308.00245): A Journey with Large Language Models
- [LLM-Assisted Code Cleaning For Training Accurate Code Generators](https://arxiv.org/abs/2311.14904) | UCB
- [Efficient Multi-Task Large Model Training via Data Heterogeneity-aware Model Management](https://arxiv.org/abs/2409.03365)
- [If At First You Don't Succeed, Try, Try, Again...?](https://www.microsoft.com/en-us/research/publication/if-at-first-you-dont-succeed-try-try-again-insights-and-llm-informed-tooling-for-detecting-retry-bugs-in-software-systems/) | SOSP' 24
- [Aceso](https://dl.acm.org/doi/pdf/10.1145/3627703.3629554): Efficient Parallel DNN Training through Iterative Bottleneck Alleviation | EuroSys '24
- [GMorph](https://dl.acm.org/doi/10.1145/3627703.3650074): Accelerating Multi-DNN Inference via Model Fusion | EuroSys '24
- [Automatic Root Cause Analysis via Large Language Models for Cloud Incidents](https://dl.acm.org/doi/10.1145/3627703.3629553) | EuroSys '24
- [KNighter: Transforming Static Analysis with LLM-Synthesized Checkers](https://sigops.org/s/conferences/sosp/2025/accepted.html) | SOSP' 25

## Industrial LLM Technical Report   
 
- [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115) - (Dec 2024)   
- [Qwen 3 Technical Report](https://arxiv.org/abs/2505.09388) – (May 2025)
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) - (Feb 2023)
- [Llama 2: Open Foundation and Fine‑Tuned Chat Models](https://arxiv.org/abs/2307.09288) - (Jul 2023)
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2404.11083) - (Aug 2024)
- [Gemini: A Family of Highly Capable Multimodal Models](https://assets.bwbx.io/documents/users/iqjWHBFdfxIU/r7G7RrtT6rnM/v0) - (Dec 2023)
- [Gemini 1.5: Unlocking multimodal understanding across millions of tokens](https://storage.googleapis.com/deepmind-media/gemini/gemini_v1_5_report.pdf) - (Feb 2024)
- [Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next‑Generation Agentic Capabilities](https://storage.googleapis.com/deepmind-media/gemini/gemini_v2_5_report.pdf) - (Jun 2025)
- [Phi‑4‑reasoning Technical Report](https://arxiv.org/abs/2504.21318) – (Apr 2025)
- [Phi‑4 Technical Report](https://arxiv.org/abs/2412.08905) – (Dec 2024)
- [Kimi‑VL Technical Report](https://arxiv.org/abs/2504.07491) – (Apr 2025)
- [Kimi k1.5: Scaling Reinforcement Learning with LLMs](https://arxiv.org/abs/2501.12599) – (Jan 2025)
- [DeepSeek-LLM Technical Report](https://nairl.kr/wp-content/uploads/2025/02/deepseek_r1_techreport.pdf) - (Jan 2024)
- [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434) - (05/2024)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) - (12/2024)
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://www.boozallen.com/content/dam/home/docs/ai/a-technical-primer-on-deepseek.pdf) - (012025)
- [Kimi-VL: Multimodal LLM with Vision, Language, and Long Context](https://arxiv.org/abs/2504.07491) – (Apr 2025)
- [Kimi k1.5: Reinforcement Learning with Multimodal LLMs](https://arxiv.org/abs/2501.12599) – (Jan 2025)
- [Kimi-K2: Open Agentic Intelligence](https://arxiv.org/abs/2507.20534) – (Jul 2025)
- [GPT-oss-120b & GPT-oss-20b](https://cdn.openai.com/pdf/419b6906-9da6-406c-a19d-1bb078ac7637/oai_gpt-oss_model_card.pdf) – (Aug 2025)

## LLM Frameworks
### Training
- [DeepSpeed](https://github.com/microsoft/DeepSpeed): a deep learning optimization library that makes distributed training and inference easy, efficient, and effective | Microsoft
- [Accelerate](https://huggingface.co/docs/accelerate/index) | Hugging Face
- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [Megatron](https://github.com/NVIDIA/Megatron-LM) | Nvidia
- [NeMo](https://github.com/NVIDIA/NeMo) | Nvidia
- [torchtitan](https://github.com/pytorch/torchtitan) | PyTorch
- [veScale](https://github.com/volcengine/vescale) | ByteDance
- [DeepSeek Open Infra](https://github.com/deepseek-ai/open-infra-index)
- [VeOmni](https://github.com/ByteDance-Seed/VeOmni): Scaling any Modality Model Training  
- [Cornstarch](https://github.com/cornstarch-org/Cornstarch): Distributed Multimodal Training Must Be Multimodality-Aware | UMich


- **Post-Training**
  - [TRL](https://github.com/huggingface/trl): Transformers Reinforcement Learning
  - [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF): An Easy-to-use, Scalable and High-performance RLHF Framework based on Ray
  - [VeRL](https://github.com/volcengine/verl): Volcano Engine Reinforcement Learning for LLMs
  - [rLLM](https://github.com/agentica-project/rllm): Reinforcement Learning for Language Agents
  - [SkyRL](https://github.com/NovaSky-AI/SkyRL): A Modular Full-stack RL Library for LLMs
  - [AReal](https://github.com/inclusionAI/AReaL): Distributed RL System for LLM Reasoning
  - [ROLL](https://github.com/alibaba/ROLL): Reinforcement Learning Optimization for Large-Scale Learning
  - [slime](https://github.com/THUDM/slime): a LLM post-training framework aiming for RL Scaling
  - [RAGEN](https://github.com/RAGEN-AI/RAGEN): Training Agents by Reinforcing Reasoning
  - [Agent Lightning](https://arxiv.org/pdf/2508.03680): Train ANY AI Agents with Reinforcement Learning
 
  
### Serving
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | Nvidia
- [Ray-LLM](https://github.com/ray-project/ray-llm) | Ray
- [TGI](https://huggingface.co/docs/text-generation-inference/en/index) | Hugging Face
- [vLLM](https://github.com/vllm-project/vllm) | UCB
- [SGLang](https://github.com/sgl-project/sglang) | UCB
- [KV Transformers](https://github.com/kvcache-ai/ktransformers)
- [Dynamo](https://github.com/ai-dynamo/dynamo): A Datacenter Scale Distributed Inference Serving Framework | NVIDA
- [LMCache](https://github.com/LMCache/LMCache): Supercharge Your LLM with the Fastest KV Cache Layer
 

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
 




## Related ML Readings
- [Large Transformer Model Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/)
- [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/)
- [The Transformer Family Version 2.0](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/)
- [Full Stack Optimization of Transformer Inference](https://arxiv.org/pdf/2302.14017.pdf): a Survey | UCB
- [The Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook): The Secrets to Building World-Class LLMs | Hugging Face
- [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook): Training LLMs on GPU Clusters | Hugging Face

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
