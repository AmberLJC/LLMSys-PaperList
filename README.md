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
- [ML Conferences](#ml-conferences)
  - [NeurIPS 2025](#neurips-2025)
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
- [Arena](https://arxiv.org/abs/2403.16125): Efficiently Training Large Models via Dynamic Scheduling and Adaptive Parallelism Co-Design | EuroSys' 26
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
- [Zeppelin](https://arxiv.org/abs/2509.21841): Balancing Variable-length Workloads in Data Parallel Large Model Training | EuroSys' 26
- [Robust LLM Training Infrastructure at ByteDance](https://sigops.org/s/conferences/sosp/2025/accepted.html) | SOSP' 25
- [Sailor: Automating Distributed Training over Dynamic, Heterogeneous, and Geo-distributed Clusters](https://sigops.org/s/conferences/sosp/2025/accepted.html) | SOSP' 25
- [Tempo: Compiled Dynamic Deep Learning with Symbolic Dependence Graphs](https://sigops.org/s/conferences/sosp/2025/accepted.html) | SOSP' 25
- [Mycroft: Tracing Dependencies in Collective Communication Towards Reliable LLM Training](https://sigops.org/s/conferences/sosp/2025/accepted.html) | SOSP' 25
- [DCP: Addressing Input Dynamism In Long-Context Training via Dynamic Context Parallelism](https://sigops.org/s/conferences/sosp/2025/accepted.html) | SOSP' 25
- [TrainVerify: Equivalence-Based Verification for Distributed LLM Training](https://sigops.org/s/conferences/sosp/2025/accepted.html) | SOSP' 25
- [Collective Communication for 100k+ GPUs](https://arxiv.org/abs/2510.20171): Large-scale collective communication optimization for massive GPU clusters
- [RDMA Point-to-Point Communication for LLM Systems](https://arxiv.org/abs/2510.27656): RDMA-based point-to-point communication optimization for distributed LLM systems | MLSys' 26
- [MoEBlaze](https://arxiv.org/abs/2601.05296): Breaking the Memory Wall for Efficient MoE Training on Modern GPUs | MLSys' 26
- [Kareus](https://arxiv.org/abs/2601.17654): Joint Reduction of Dynamic and Static Energy in Large Model Training
- [AXLearn](https://arxiv.org/abs/2507.05411): Modular Large Model Training on Heterogeneous Infrastructure | MLSys' 26
- [MoSE](https://arxiv.org/abs/2602.06154): Mixture of Slimmable Experts for Efficient and Adaptive Language Models
- [MegaScale-MoE](https://arxiv.org/abs/2505.11432): Large-Scale Communication-Efficient Training of Mixture-of-Experts Models in Production | EuroSys' 26
- [MegaScale-Data](https://arxiv.org/abs/2504.09844): Scaling DataLoader for Multisource Large Foundation Model Training | EuroSys' 26
- [HetAuto](https://dl.acm.org/doi/10.1145/3767295.3803590): Cross-Cluster Auto-Parallelism for Heterogeneous Distributed Training | EuroSys' 26
- [HARP](https://dl.acm.org/doi/10.1145/3767295.3803603): Orchestrating Automated Parallel Training on Heterogeneous GPU Clusters | EuroSys' 26
- [Crimson](https://dl.acm.org/doi/10.1145/3767295.3803606): Collaborative Parameter Updates for Efficient Pipeline Training of Large Language Models | EuroSys' 26
- [Suika](https://dl.acm.org/doi/10.1145/3767295.3803623): Efficient and High-quality Re-scheduling of 3D-parallelized LLM Training Jobs in Shared Clusters | EuroSys' 26
- [Efficient and Adaptable Overlapping for Computation and Communication via Signaling and Reordering](https://dl.acm.org/doi/10.1145/3767295.3769370) | EuroSys' 26
- [BOOST](https://arxiv.org/abs/2512.12131): BOttleneck-Optimized Scalable Training Framework for Low-Rank Large Language Models | MLSys' 26
- [MTraining](https://arxiv.org/abs/2510.18830): Distributed Dynamic Sparse Attention for Efficient Ultra-Long Context Training | MLSys' 26
- [ProTrain](https://arxiv.org/abs/2406.08334): Efficient LLM Training via Automatic Memory Management | MLSys' 26
- [DreamDDP](https://arxiv.org/abs/2502.11058): Accelerating Data Parallel Distributed LLM Training with Layer-wise Scheduled Partial Synchronization | MLSys' 26
- [Multipath Collective Communication Beyond Scale-up Networks in GPU Clouds](https://dl.acm.org/doi/10.1145/3767295.3769330) | EuroSys' 26
- [STAlloc: Enhancing Memory Efficiency in Large-Scale Model Training through Spatio-Temporal Allocation Planning](https://dl.acm.org/doi/10.1145/3767295.3769335) | EuroSys' 26
- [Maya: Optimizing Deep Learning Training Workloads using GPU Runtime Emulation](https://dl.acm.org/doi/10.1145/3767295.3769366) | EuroSys' 26
- [Bridging the GPU Utilization Gap: Predictive Multi-Dimensional Resource Scheduling for AI Workloads](https://dl.acm.org/doi/10.1145/3767295.3803579) | EuroSys' 26
- [Reducing the GPU Memory Bottleneck with Lossless Compression for ML](https://dl.acm.org/doi/10.1145/3767295.3803595) | EuroSys' 26
- [Efficient Long-Context LM Training by Core Attention Disaggregation](https://mlsys.org/virtual/2026/oral/3754) | MLSys' 26
- [Zorse: Optimizing LLM Training Efficiency on Heterogeneous GPU Clusters](https://mlsys.org/virtual/2026/poster/3636) | MLSys' 26
- [Unleashing Scalable Context Parallelism via Fully Connected Pipeline](https://mlsys.org/virtual/2026/oral/3822) | MLSys' 26
- [FlexTrain: Scalable Hybrid-Parallel Training for Long-Context LLMs](https://mlsys.org/virtual/2026/poster/3553) | MLSys' 26
- [veScale-FSDP: Flexible and High-Performance FSDP at Scale](https://mlsys.org/virtual/2026/poster/3637) | MLSys' 26
- [HexiScale: LLM Training over Heterogeneous Hardware](https://mlsys.org/virtual/2026/poster/3605) | MLSys' 26
- [FP8-Flow-MoE: Casting-Free FP8 Recipe for MoE without Double Quantization Error](https://mlsys.org/virtual/2026/oral/3737) | MLSys' 26


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
- [Laminar](https://arxiv.org/abs/2510.12633): A Scalable Asynchronous RL Post-Training Framework | EuroSys' 26
- [Seer](https://arxiv.org/abs/2511.14617): Online Context Learning for Fast Synchronous LLM Reinforcement Learning
- [SkyRL-Agent](https://arxiv.org/abs/2511.16108): Efficient RL Training for Multi-turn LLM Agent
- [LoRAFusion](https://dl.acm.org/doi/10.1145/3767295.3769331): Efficient LoRA Fine-Tuning for LLMs | EuroSys' 26
- [HetRL](https://arxiv.org/abs/2512.12476): Efficient Reinforcement Learning for LLMs in Heterogeneous Environments | MLSys' 26
- [ReSpec](https://arxiv.org/abs/2510.26475): Towards Optimizing Speculative Decoding in Reinforcement Learning Systems | MLSys' 26
- [Beat the Long Tail: Distribution-Aware Speculative Decoding for Reinforcement Learning](https://mlsys.org/virtual/2026/oral/3766) | MLSys' 26
- [FLoRIST: Federated Low-Rank Adaptation with Random Subspaces for LLMs](https://mlsys.org/virtual/2026/poster/3617) | MLSys' 26

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
- [GoCkpt](https://arxiv.org/abs/2511.07035): Gradient-Assisted Multi-Step Overlapped Checkpointing for Efficient LLM Training | PPoPP' 26
- [BitSnap](https://arxiv.org/abs/2511.12376): Checkpoint Sparsification and Quantization in LLM Training
- [Handling Network Faults in Distributed AI Training: Failover is Now an Option](https://dl.acm.org/doi/10.1145/3767295.3769322) | EuroSys' 26
- [GUARD: Scalable Straggler Detection and Mitigation in LLM Training](https://mlsys.org/virtual/2026/poster/3608) | MLSys' 26


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
- [Hogwild! Inference](https://arxiv.org/abs/2504.06261): Parallel LLM Generation via Concurrent Attention
- [Prism](https://arxiv.org/pdf/2505.04021): Unleashing GPU Sharing for Cost-Efficient Multi-LLM Serving | UCLA
- [RetroInfer](https://arxiv.org/abs/2505.02922): A Vector-Storage Approach for Scalable Long-Context LLM Inference
- [Efficient Serving of LLM Applications with Probabilistic Demand Modeling](https://arxiv.org/abs/2506.14851)
- [eLLM](https://arxiv.org/abs/2506.14851) : Elastic Memory Management Framework for Efficient LLM Serving
- [DiSCo](https://arxiv.org/abs/2502.11417v2): Device-Server Collaborative LLM-Based Text Streaming Services
- [DynaServe](https://arxiv.org/abs/2504.09285): Unified and Elastic Execution for Dynamic Disaggregated LLM Serving
- [HyGen](https://arxiv.org/pdf/2501.14808): Efficient LLM Serving via Elastic Online-Offline Request Co-location
- [WaferLLM: A Wafer‑Scale LLM Inference System](https://arxiv.org/abs/2502.04563) | OSDI 25
- [BlitzScale: Fast and Live Large Model Autoscaling with O(1) Host Caching](https://arxiv.org/abs/2412.17246) | OSDI 25
- [TokenWeave: Efficient Compute-Communication Overlap for Distributed LLM Inference](https://arxiv.org/abs/2505.11329) | [Code](https://github.com/microsoft/tokenweave) | MLSys' 26
- [Nexus](https://arxiv.org/abs/2507.06608v2): Taming Throughput-Latency Tradeoff in LLM Serving via Efficient GPU Sharing
- [Taming the Chaos](https://arxiv.org/abs/2508.19559): Coordinated Autoscaling for Heterogeneous and Disaggregated LLM Inference | Seed
- [TokenLake](https://arxiv.org/abs/2508.17219): A Unified Segment-level Prefix Cache Pool for Fine-grained Elastic Long-Context LLM Serving
- [Expert-as-a-Service](https://arxiv.org/abs/2509.17863): Towards Efficient, Scalable, and Robust Large-scale MoE Serving
- [Shift Parallelism](https://arxiv.org/pdf/2509.16495): Low-Latency, High-Throughput LLM Inference for Dynamic Workloads
- [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
- [Deterministic Inference across Tensor Parallel Sizes That Eliminates Training-Inference Mismatch](https://arxiv.org/abs/2511.17826): Ensuring deterministic inference across different tensor parallelism configurations
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
- [Continuum](https://arxiv.org/abs/2511.02230): Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live
- [AIConfigurator](https://arxiv.org/abs/2601.06288): Lightning-Fast Configuration Optimization for Multi-Framework LLM Serving
- [SuperInfer](https://arxiv.org/abs/2601.20309): SLO-Aware Rotary Scheduling and Memory Management for LLM Inference on Superchips | MLSys' 26
- [Scaling Up Efficient Small Language Models Serving](https://arxiv.org/abs/2510.22101): Serving and Deployment for Semantic Job Search | MLSys' 26
- [BestServe](https://arxiv.org/abs/2506.05871): Serving Strategies with Optimal Goodput in Collocation and Disaggregation Architectures
- [OptiKIT](https://arxiv.org/abs/2601.20408): Meeting SLOs, Slashing Hours - Automated Enterprise LLM Optimization | MLSys' 26
- [BlendServe](https://dl.acm.org/doi/abs/10.1145/3779212.3790133): Optimizing Offline Inference for Auto-regressive Large Models with Resource-aware Batching | ASPLOS' 26
- [SwiftSpec](https://dl.acm.org/doi/abs/10.1145/3779212.3790246): Ultra-Low Latency LLM Decoding by Scaling Asynchronous Speculative Decoding with Disaggregated Pipeline and Fused Kernels | ASPLOS' 26
- [MuxWise](https://dl.acm.org/doi/abs/10.1145/3779212.3790236): Towards High-Goodput LLM Serving with Prefill-decode Multiplexing | ASPLOS' 26
- [MoEless](https://arxiv.org/abs/2603.06350): Efficient MoE LLM Serving via Serverless Computing
- [Online Scheduling for LLM Inference with KV Cache Constraints](https://arxiv.org/abs/2502.07115): Optimal Batching and Scheduling for KV Cache-Constrained Inference
- [BiScale](https://arxiv.org/abs/2602.18755): Energy-Efficient Disaggregated LLM Serving via Phase-Aware Placement and DVFS
- [Harvest](https://arxiv.org/abs/2602.00328): Opportunistic Peer-to-Peer GPU Caching for LLM Inference
- [TokenFlow](file/report.pdf): Responsive LLM Text Streaming Serving under Request Burst via Preemptive Scheduling | Plagiarism
- [MineDraft](https://arxiv.org/abs/2603.18016): A Framework for Batch Parallel Speculative Decoding — overlaps drafting and verification across two batches, hiding draft latency. Up to +75% throughput, -39% latency. Integrated into vLLM. | NUS & MIT
- [Foundry](https://arxiv.org/abs/2604.06664): Template-Based CUDA Graph Context Materialization for Fast LLM Serving Cold Start
- [AdaServe](https://arxiv.org/abs/2501.12162): Accelerating Multi-SLO LLM Serving with SLO-Customized Speculative Decoding | EuroSys' 26
- [FlexPipe](https://arxiv.org/abs/2510.11938): Adapting Dynamic LLM Serving Through Inflight Pipeline Refactoring in Fragmented Serverless Clusters | EuroSys' 26
- [Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via Fine-Grained Expert Offloading](https://dl.acm.org/doi/10.1145/3767295.3769319) | EuroSys' 26
- [KunServe](https://arxiv.org/abs/2412.18169): Parameter-centric Memory Management for Efficient Memory Overloading Handling in LLM Serving | EuroSys' 26
- [AdaGen](https://dl.acm.org/doi/10.1145/3767295.3769345): Workload-Adaptive Cluster Scheduler for Latency-Optimal LLM Inference Serving | EuroSys' 26
- [SkyWalker](https://arxiv.org/abs/2505.24095): A Locality-Aware Cross-Region Load Balancer for LLM Inference | EuroSys' 26
- [High Throughput and Low Latency LLM Serving via Adaptive KV Caching](https://dl.acm.org/doi/10.1145/3767295.3803570) | EuroSys' 26
- [PARD](https://dl.acm.org/doi/10.1145/3767295.3803581): Enhancing Goodput for Inference Pipeline via Proactive Request Dropping | EuroSys' 26
- [PiLLM](https://dl.acm.org/doi/10.1145/3767295.3769393): Resource-Efficient LLM Inference Using Workload Prediction | EuroSys' 26
- [Automated End-to-End Model Serving with Cooperative Compilation and Scheduling](https://dl.acm.org/doi/10.1145/3767295.3769392) | EuroSys' 26
- [MFS](https://dl.acm.org/doi/10.1145/3767295.3769355): An Efficient Model Family Serving System for LLMs | EuroSys' 26
- [CRAFT](https://arxiv.org/abs/2603.28768): Cost-aware Expert Replica Allocation with Fine-Grained Layerwise Estimations for Efficient MoE Serving | MLSys' 26
- [MorphServe](https://arxiv.org/abs/2506.02006): Efficient and Workload-Aware LLM Serving via Runtime Quantized Layer Swapping and KV Cache Resizing | MLSys' 26
- [FlexiCache](https://arxiv.org/abs/2511.00868): Leveraging Temporal Stability of Attention Heads for Efficient KV Cache Management | MLSys' 26
- [Kitty](https://arxiv.org/abs/2511.18643): Accurate and Efficient 2-bit KV Cache Quantization with Dynamic Channel-wise Precision Boost | MLSys' 26
- [SkipKV](https://arxiv.org/abs/2512.07993): Selective Skipping of KV Generation and Storage for Efficient Inference with Large Reasoning Models | MLSys' 26
- [BOute](https://arxiv.org/abs/2602.10729): Cost-Efficient LLM Serving with Heterogeneous LLMs and GPUs via Multi-Objective Bayesian Optimization | MLSys' 26
- [From Tokens to Layers](https://arxiv.org/abs/2510.08055): Redefining Stall-Free Scheduling for LLM Serving with Layered Prefill | MLSys' 26
- [HELIOS](https://arxiv.org/abs/2504.10724): Adaptive Model And Early-Exit Selection for Efficient LLM Inference Serving | MLSys' 26
- [BatchLLM](https://arxiv.org/abs/2412.03594): Optimizing Large Batched LLM Inference with Global Prefix Sharing and Throughput-oriented Token Batching | MLSys' 26
- [GhostServe](https://arxiv.org/abs/2605.00831): A Lightweight Checkpointing System in the Shadow for Fault-Tolerant LLM Serving | MLSys' 26
- [PRISM](https://arxiv.org/abs/2602.01762): Parametrically Refactoring Inference for Speculative Decoding Draft Models | MLSys' 26
- [FarSkip-Collective](https://arxiv.org/abs/2511.11505): Unhobbling Blocking Communication in Mixture of Experts Models | MLSys' 26
- [Efficient Data Passing for Serverless Inference Workflows: A GPU-Centric Approach](https://dl.acm.org/doi/10.1145/3767295.3769336) | EuroSys' 26
- [TrustWeave: Integrity Measurement and Attestation for Multi-Cloud LLMs](https://dl.acm.org/doi/10.1145/3767295.3803586) | EuroSys' 26
- [Stream2LLM: Overlapping Context Streaming and Prefill for Low-Latency LLM Serving](https://mlsys.org/virtual/2026/oral/3842) | MLSys' 26
- [Locality-Aware Beam Scheduling for Efficient Test-Time Compute](https://mlsys.org/virtual/2026/oral/3788) | MLSys' 26
- [Optimizing Deployment Configurations for LLM Inference](https://mlsys.org/virtual/2026/oral/3780) | MLSys' 26
- [ContextPilot: Fast Long-Context Inference via Context Reuse](https://mlsys.org/virtual/2026/oral/3810) | MLSys' 26
- [Speculative Decoding: Performance or Illusion?](https://mlsys.org/virtual/2026/oral/3782) | MLSys' 26
- [SHIP: SRAM-Based Huge Inference Pipelines for Fast LLM Serving](https://mlsys.org/virtual/2026/oral/3834) | MLSys' 26
- [BEAM: Joint Resource-Power Optimization for LLM Inference](https://mlsys.org/virtual/2026/oral/3849) | MLSys' 26
- [Beyond the Buzz: A Pragmatic Take on Inference Disaggregation](https://mlsys.org/virtual/2026/oral/3819) | MLSys' 26
- [PLA-Serve: Prefill-Length-Aware LLM Serving System](https://mlsys.org/virtual/2026/oral/3787) | MLSys' 26
- [Accelerating Reasoning Model Inference with Sparse Self-Speculative Decoding](https://mlsys.org/virtual/2026/oral/3733) | MLSys' 26
- [FaaScale: Unlocking Fast LLM Scaling for Serverless Inference](https://mlsys.org/virtual/2026/oral/3769) | MLSys' 26
- [Breaking the Ice: Analyzing Cold Start Latency in vLLM](https://mlsys.org/virtual/2026/oral/3784) | MLSys' 26
- [Demystifying the Mixture of Experts Serving Tax](https://mlsys.org/virtual/2026/oral/3764) | MLSys' 26
- [RaidServe: High-Performance Resilient LLM Serving](https://mlsys.org/virtual/2026/oral/3856) | MLSys' 26
- [Toward Principled LLM Safety Testing: Solving the Jailbreak Oracle Problem](https://mlsys.org/virtual/2026/oral/3739) | MLSys' 26

#### Agent Systems
- [Supporting Our AI Overlords](https://arxiv.org/pdf/2509.00997): Redesigning Data Systems to be Agent-First | UCB
- [ALTO](https://arxiv.org/abs/2403.04311): An Efficient Network Orchestrator for Compound AI Systems | Stanford & UCB
- [Parrot](https://www.usenix.org/conference/osdi24/presentation/lin-chaofan): Efficient Serving of LLM-based Applications with Semantic Variable | OSDI' 24
- [Efficiently Serving LLM Reasoning Programs with Certaindex](https://arxiv.org/pdf/2412.20993) | UCSD
- [Autellix](https://arxiv.org/pdf/2502.13965): An Efficient Serving Engine for LLM Agents as General Programs | UCB
- [RAGO](https://arxiv.org/abs/2503.14649v2): Systematic Performance Optimization for Retrieval-Augmented Generation Serving | ISCA'25
- [Circinus](https://arxiv.org/abs/2504.16397): Efficient Query Planner for Compound ML Serving | UIUC
- [Patchwork: A Unified Framework for RAG Serving](https://arxiv.org/abs/2505.07833)
- [DS SERVE](https://berkeley-large-rag.github.io/RAG-DS-Serve/): A Framework for Efficient and Scalable Neural Retrieval | UCB
- [KVFlow](https://arxiv.org/abs/2507.07400): Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows
- [DroidSpeak](https://arxiv.org/abs/2411.02820): KV Cache Sharing for Cross-LLM Communication and Multi-LLM Serving
- [Murakkab](https://arxiv.org/abs/2508.18298): Resource-Efficient Agentic Workflow Orchestration in Cloud Platforms
- [HedraRAG: Co-Optimizing Generation and Retrieval for Heterogeneous RAG Workflows](https://sigops.org/s/conferences/sosp/2025/accepted.html) | SOSP' 25
- [METIS: Fast Quality-Aware RAG Systems with Configuration Adaptation](https://sigops.org/s/conferences/sosp/2025/accepted.html) | SOSP' 25
- [Aragog: Just-in-Time Model Routing for Scalable Serving of Agentic Workflows](https://arxiv.org/pdf/2511.20975)
- [DualPath](https://arxiv.org/abs/2602.21548): Breaking the Storage Bandwidth Bottleneck in Agentic LLM Inference | DeepSeek
- [AIMS](https://dl.acm.org/doi/10.1145/3767295.3803622): Cost-Efficient LLM-Based Agent Deployment in Hybrid Cloud-Edge Environments | EuroSys' 26
- [From Imperative to Declarative](https://dl.acm.org/doi/10.1145/3767295.3803576): Towards LLM-friendly OS Interfaces for Boosted Computer-Use Agents | EuroSys' 26
- [Hippocampus](https://arxiv.org/abs/2602.13594): An Efficient and Scalable Memory Module for Agentic AI | MLSys' 26
- [PROMPTS: Performance Optimization via Multi-Agent Planning for Test-time Compute Scaling](https://mlsys.org/virtual/2026/oral/3843) | MLSys' 26
- [TeleRAG: Efficient Retrieval-Augmented Generation Inference with Lookahead Retrieval](https://mlsys.org/virtual/2026/poster/3573) | MLSys' 26
- [OpenHands Software Agent SDK](https://mlsys.org/virtual/2026/poster/3526) | MLSys' 26
- [FlashAgents: Accelerating Multi-Agent LLM Systems via Streaming Prefill Overlap](https://mlsys.org/virtual/2026/poster/3537) | MLSys' 26
- [AgenticCache: Cache-Driven Asynchronous Planning for Agentic LLM Systems](https://mlsys.org/virtual/2026/oral/3806) | MLSys' 26
- [Matrix: Peer-to-Peer Multi-Agent Synthetic Data Generation](https://mlsys.org/virtual/2026/oral/3753) | MLSys' 26
- [Ontology-Guided Long-Term Agent Memory for Conversational RAG](https://mlsys.org/virtual/2026/oral/3738) | MLSys' 26
- [OSWorld-Human: Benchmarking Efficiency of Computer-Use Agents](https://mlsys.org/virtual/2026/oral/3865) | MLSys' 26

#### Serving at the edge
- [LLM in a flash: Efficient Large Language Model Inference with Limited Memory](https://arxiv.org/abs/2312.11514) | Apple
- [STI](https://arxiv.org/abs/2207.05022): Turbocharge NLP Inference at the Edge via Elastic Pipelining | ASPLOS 23 
- [PowerInfer](https://arxiv.org/abs/2312.12456): Fast Large Language Model Serving with a Consumer-grade GPU | SOSP' 24
- [MoE-Lightning: High-Throughput MoE Inference on Memory-constrained GPUs](https://arxiv.org/abs/2411.11217)
- [InfiniteHiP](https://arxiv.org/abs/2502.08910): Extending Language Model Context Up to 3 Million Tokens on a Single GPU
- [prima.cpp](https://arxiv.org/pdf/2504.08791): PRIMA.CPP: Speeding Up 70B-Scale LLM Inference on Low-Resource Everyday Home Clusters
- [Characterizing Mobile SoC for Accelerating Heterogeneous LLM Inference](https://sigops.org/s/conferences/sosp/2025/accepted.html) | SOSP' 25
- [TZ-LLM](https://dl.acm.org/doi/10.1145/3767295.3769334): Protecting On-Device Large Language Models with Arm TrustZone | EuroSys' 26
- [TailorLLM](https://dl.acm.org/doi/10.1145/3767295.3769346): Collaborative End-Cloud Inference of Large and Small Language Models Based on Low-Rank Adaptation | EuroSys' 26
- [Federated Fine-Tuning of Sparsely-Activated Large Language Models on Resource-Constrained Devices](https://dl.acm.org/doi/10.1145/3767295.3769329) | EuroSys' 26
- [Scaling LLM Test-Time Compute with Mobile NPU on Smartphones](https://dl.acm.org/doi/10.1145/3767295.3769382) | EuroSys' 26
- [On-device Semantic Selection Made Low Latency and Memory Efficient with Monolithic Forwarding](https://dl.acm.org/doi/10.1145/3767295.3803572) | EuroSys' 26
- [SwiftFL: Enabling Speculative Training for On-Device Federated Deep Learning](https://dl.acm.org/doi/10.1145/3767295.3803605) | EuroSys' 26
- [viNPU: Optimizing Vision Transformer Inference on Mobile NPUs](https://dl.acm.org/doi/10.1145/3767295.3803619) | EuroSys' 26
- [Efficient, VRAM-Constrained Cross-Lingual Model Inference on Client Devices](https://mlsys.org/virtual/2026/oral/3802) | MLSys' 26
- [Rethinking DVFS for Mobile LLMs: CORE for Energy-Efficient On-Device Inference](https://mlsys.org/virtual/2026/oral/3814) | MLSys' 26
- [IntAttention: Fully Integer Attention Pipeline for Edge LLM Inference](https://mlsys.org/virtual/2026/oral/3848) | MLSys' 26


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
- [Reducing GPU Memory Fragmentation via Spatio-Temporal Allocation Planning](https://arxiv.org/abs/2507.16274) | EuroSys' 26
- [SAS](https://dl.acm.org/doi/10.1145/3767295.3769364): Sparse Attention Synthesizer for Efficient Language Model Inference | EuroSys' 26
- [LLMFolder](https://dl.acm.org/doi/10.1145/3767295.3769339): Revisiting Constant Folding in Large Language Models | EuroSys' 26
- [FlashAttention-4](https://arxiv.org/abs/2603.05451): Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling (Blackwell) | MLSys' 26
- [BLASST: Dynamic Blocked Attention Sparsity for Scalable Transformer Inference](https://mlsys.org/virtual/2026/poster/3631) | MLSys' 26
- [Attribution-based Sparse Activation in Large Language Models](https://mlsys.org/virtual/2026/poster/3556) | MLSys' 26
- [MixLLM: LLM Quantization with Global Mixed-Precision between Output and Embeddings](https://mlsys.org/virtual/2026/oral/3805) | MLSys' 26
- [MAC-Attention: Match-Amend-Complete Attention for Efficient Long-Context Inference](https://mlsys.org/virtual/2026/oral/3794) | MLSys' 26
- [Flashlight: PyTorch Compiler Extensions for Attention Variants](https://mlsys.org/virtual/2026/poster/3540) | MLSys' 26
- [CAGE: Curvature-Aware Gradient Estimation for Quantization-Aware Training](https://mlsys.org/virtual/2026/oral/3841) | MLSys' 26
- [OPKV: Recallable Sparsity in Paged KV Cache for Efficient LLM Inference](https://mlsys.org/virtual/2026/poster/3621) | MLSys' 26
- [Using Span Queries to Optimize Cache and Attention Locality](https://mlsys.org/virtual/2026/oral/3747) | MLSys' 26

### Multi-Modal Training Systems
- [DISTMM](https://www.usenix.org/conference/nsdi24/presentation/huang): Accelerating distributed multimodal model training | NSDI' 24
- [Optimus:](https://www.arxiv.org/abs/2408.03505) Accelerating Large-Scale Multi-Modal LLM Training by Bubble Exploitation
- [Addressing Model and Data Heterogeneity in Multimodal Large Language Model Training](https://arxiv.org/pdf/2408.04275v1) | PKU
- [Cornstarch](https://arxiv.org/abs/2503.11367): Distributed Multimodal Training Must Be Multimodality-Aware | UMich
- [PipeWeaver](https://arxiv.org/abs/2504.14145): Addressing Data Dynamicity in Large Multimodal Model Training with Dynamic Interleaved Pipeline | SJTU
- [MegaScale-Omni](https://dl.acm.org/doi/10.1145/3767295.3803587): A Hyper-Scale, Workload-Resilient System for MultiModal LLM Training in Production | EuroSys' 26

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
- [Argus](https://arxiv.org/abs/2511.06724): Quality-Aware High-Throughput Text-to-Image Inference Serving System
- [Cornserve](https://arxiv.org/abs/2512.14098): Efficiently Serving Any-to-Any Multimodal Models
- [HydraInfer](https://arxiv.org/abs/2505.12658): Hybrid Disaggregated Scheduling for Multimodal Large Language Model Serving
- [Enabling Disaggregated Multi-Stage MLLM Inference via GPU-Internal Scheduling and Resource Sharing](https://arxiv.org/abs/2512.17574)
- [VoxServe](https://arxiv.org/abs/2602.00269): Streaming-Centric Serving System for Speech Language Models
- [dLLM-Serve](https://arxiv.org/abs/2512.17077): Taming the Memory Footprint Crisis for Efficient Diffusion LLM Serving
- [HADIS](https://arxiv.org/abs/2509.00642): Hybrid Adaptive Diffusion Model Serving for Efficient Text-to-Image Generation
- [Efficient Multimodal Serving via Module Multiplexing](https://dl.acm.org/doi/10.1145/3767295.3769389) | EuroSys' 26
- [FlashPS](https://dl.acm.org/doi/10.1145/3767295.3769379): Efficient Generative Image Editing with Mask-aware Caching and Scheduling | EuroSys' 26
- [StreamDiffusionV2](https://arxiv.org/abs/2511.07399): A Streaming System for Dynamic and Interactive Video Generation | MLSys' 26
- [SpecDiff-2](https://arxiv.org/abs/2511.00606): Scaling Diffusion Drafter Alignment For Faster Speculative Decoding | MLSys' 26
- [Million-Scale Text-to-Video Retrieval with Hyperdimensional Computing](https://dl.acm.org/doi/10.1145/3767295.3803610) | EuroSys' 26
- [TriInfer: Hybrid Encode-Prefill-Decode Disaggregation for Multimodal LLM Inference](https://mlsys.org/virtual/2026/oral/3756) | MLSys' 26
- [CDLM: Consistency Diffusion Language Models for Faster Text Generation Sampling](https://mlsys.org/virtual/2026/oral/3785) | MLSys' 26
- [db-SP: Accelerating Sparse Attention for Visual Generative Models](https://mlsys.org/virtual/2026/poster/3575) | MLSys' 26
- [TiDAR: Think in Diffusion, Talk in Autoregression for Multimodal Generation](https://mlsys.org/virtual/2026/poster/3528) | MLSys' 26


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
- [Barbarians at the Gate: How AI is Upending Systems Research](https://arxiv.org/abs/2510.06189)
- [Let the Barbarians In](https://arxiv.org/abs/2512.14806): How AI Can Accelerate Systems Performance Research
- [AI Research Engineering Skills Library](https://github.com/zechenzhangAGI/claude-ai-research-skills): A collection of AI research engineering skills and best practices
- [K-Search](https://arxiv.org/abs/2602.19128): LLM Kernel Generation via Co-Evolving Intrinsic World Model
- [AI-Driven Research for Databases](https://arxiv.org/abs/2604.06566): Automated database optimization via co-evolving evaluators and AI-generated solutions
- [No More Translation at Runtime](https://dl.acm.org/doi/10.1145/3767295.3803600): LLM-Empowered Static Binary Translation | EuroSys' 26
- [Unified LLM Model for PPA Prediction from Hardware Code](https://mlsys.org/virtual/2026/poster/3538) | MLSys' 26
- [Optimizing PyTorch Inference with LLM-Based Multi-Agent Systems](https://mlsys.org/virtual/2026/oral/3823) | MLSys' 26
- [AccelOpt: Self-Improving LLM Agentic System for Kernel Optimization](https://mlsys.org/virtual/2026/oral/3808) | MLSys' 26
- [VeriMoA: Mixture-of-Agents for Spec-to-HDL Verification and Generation](https://mlsys.org/virtual/2026/poster/3632) | MLSys' 26

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

## ML Conferences
### NeurIPS 2025

A curated collection of **[NeurIPS 2025 papers](neurips25-mlsys/)** focused on efficient systems for generative AI models. The collection includes papers on:
- [Architecture & Efficient Mechanisms](neurips25-mlsys/architecture.md) - Efficient attention, KV-cache systems, speculative decoding
- [Model Compression & Quantization](neurips25-mlsys/compression.md) - Quantization, pruning, KV cache compression
- [Inference & Serving](neurips25-mlsys/inference.md) - LLM serving, scheduling, distributed inference
- [Multi-Modal & Diffusion](neurips25-mlsys/multi-modality.md) - VLM efficiency, diffusion optimization
- [Reinforcement Learning](neurips25-mlsys/rl.md) - RL training infrastructure, policy optimization
- [Training Systems](neurips25-mlsys/training.md) - Distributed training, memory efficiency

See the **[full NeurIPS 2025 collection](neurips25-mlsys/)** for detailed categorization and paper summaries.

## LLM Frameworks
### Training
- [DeepSpeed](https://github.com/microsoft/DeepSpeed): a deep learning optimization library that makes distributed training and inference easy, efficient, and effective | Microsoft
- [Accelerate](https://huggingface.co/docs/accelerate/index) | Hugging Face
- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [Megatron](https://github.com/NVIDIA/Megatron-LM) | Nvidia
- [NeMo](https://github.com/NVIDIA/NeMo) | Nvidia
- [torchtitan](https://github.com/pytorch/torchtitan) | PyTorch
- [torchtune](https://github.com/pytorch/torchtune): PyTorch-native fine-tuning library for LLMs with minimal dependencies | PyTorch
- [veScale](https://github.com/volcengine/vescale) | ByteDance
- [DeepSeek Open Infra](https://github.com/deepseek-ai/open-infra-index)
- [VeOmni](https://github.com/ByteDance-Seed/VeOmni): Scaling any Modality Model Training  
- [Cornstarch](https://github.com/cornstarch-org/Cornstarch): Distributed Multimodal Training Must Be Multimodality-Aware | UMich
- [GPT-NeoX](https://github.com/EleutherAI/gpt-neox): Model-parallel autoregressive LLM training combining Megatron and DeepSpeed | EleutherAI
- [nanotron](https://github.com/huggingface/nanotron): Minimalistic 3D-parallel (tensor/pipeline/data) LLM training framework | Hugging Face
- [litgpt](https://github.com/lightning-ai/litgpt): 20+ LLM implementations with pre-training and fine-tuning recipes | Lightning AI
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory): Unified efficient fine-tuning of 100+ LLMs and VLMs via LoRA, full fine-tuning, and RL methods | ACL' 24
- [Unsloth](https://github.com/unslothai/unsloth): 2-5x faster LLM fine-tuning with ~80% less memory via custom Triton/CUDA kernels


- **Post-Training**
  - [PEFT](https://github.com/huggingface/peft): Parameter-efficient fine-tuning library (LoRA, QLoRA, Prompt Tuning, IA3, etc.) | Hugging Face
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
  - [LMFlow](https://github.com/OptimalScale/LMFlow): Extensible toolkit for fine-tuning and inference of large foundation models
  - [NeMo-Aligner](https://github.com/NVIDIA/NeMo-Aligner): Scalable alignment toolkit for SFT, PPO, DPO, and SteerLM on NeMo | Nvidia
 
  
### Serving
- [llama.cpp](https://github.com/ggml-org/llama.cpp): LLM inference in C/C++ with GGUF quantization; supports CPU, Metal, CUDA, and wide hardware
- [Ollama](https://github.com/ollama/ollama): Local LLM serving with model management and OpenAI-compatible API
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | Nvidia
- [Triton Inference Server](https://github.com/triton-inference-server/server): Production multi-framework model serving platform with dynamic batching | Nvidia
- [Ray-LLM](https://github.com/ray-project/ray-llm) | Ray
- [TGI](https://huggingface.co/docs/text-generation-inference/en/index) | Hugging Face
- [vLLM](https://github.com/vllm-project/vllm) | UCB
- [SGLang](https://github.com/sgl-project/sglang) | UCB
- [LMDeploy](https://github.com/InternLM/lmdeploy): LLM compression, deployment, and serving toolkit with TurboMind persistent batching engine | InternLM
- [LightLLM](https://github.com/ModelTC/lightllm): Lightweight Python LLM serving with tri-process architecture decoupling prefill and decode
- [DeepSpeed-MII](https://github.com/microsoft/DeepSpeed-MII): Low-latency, high-throughput LLM inference powered by DeepSpeed | Microsoft
- [CTranslate2](https://github.com/OpenNMT/CTranslate2): Fast C++/Python inference engine for Transformer models with int8/int16 quantization | OpenNMT
- [Petals](https://github.com/bigscience-workshop/petals): Distributed LLM inference and fine-tuning across volunteer GPUs in a BitTorrent-like fashion | ACL' 23
- [KV Transformers](https://github.com/kvcache-ai/ktransformers)
- [Dynamo](https://github.com/ai-dynamo/dynamo): A Datacenter Scale Distributed Inference Serving Framework | Nvidia
- [LMCache](https://github.com/LMCache/LMCache): Supercharge Your LLM with the Fastest KV Cache Layer
- [aibrix](https://github.com/vllm-project/aibrix): Cost-efficient pluggable infrastructure for GenAI inference (KV cache routing, autoscaling, disaggregated prefill) | vLLM Project
 

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
-  [FlashInfer-Bench / LLMInfer-Bench: Benchmarking LLM Inference Kernels and Systems](https://mlsys.org/virtual/2026/poster/3609) | MLSys' 26
-  [DriftBench: Measuring and Predicting Infrastructure Drift in LLM Serving Systems](https://mlsys.org/virtual/2026/oral/3799) | MLSys' 26
-  [Charon: A Unified Simulator for LLM Training and Inference](https://mlsys.org/virtual/2026/poster/3638) | MLSys' 26
-  [ProfInfer: eBPF-based Fine-Grained LLM Inference Profiler](https://mlsys.org/virtual/2026/oral/3740) | MLSys' 26
 




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
