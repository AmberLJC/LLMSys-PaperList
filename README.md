# Awesome LLM Systems Papers

A curated list of Large Language Model systems related academic papers, articles, tutorials, slides and projects. Star this repository, and then you can keep abreast of the latest developments of this booming research field.

## LLM Systems
### Pre-Training
- [Megatron-LM](https://arxiv.org/pdf/1909.08053.pdf): Training Multi-Billion Parameter Language Models Using Model Parallelism
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/pdf/2104.04473.pdf)
- [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/pdf/2205.05198.pdf)
- [Optimized Network Architectures for Large Language Model Training with Billions of Parameters](https://arxiv.org/pdf/2307.12169.pdf) | MIT
- [Carbon Emissions and Large Neural Network Training](https://arxiv.org/pdf/2104.10350.pdf?fbclid=IwAR2o0_3HCtTnMxKbXka0OPrHzl8sCzQSSOYp0AOav76-zVWl_pYek2jX8Pk) | Google, UCB
- [MegaScale](https://arxiv.org/abs/2402.15627): Scaling Large Language Model Training to More Than 10,000 GPUs | ByteDance
- [DISTMM](https://www.amazon.science/publications/distmm-accelerating-distributed-multimodal-model-training): Accelerating distributed multimodal model training | NSDI' 24
- [A Codesign of Scheduling and Parallelization for Large Model Training in Heterogeneous Clusters](https://arxiv.org/pdf/2403.16125)

### Serving
- [Orca](https://www.usenix.org/conference/osdi22/presentation/yu): A Distributed Serving System for Transformer-Based Generative Models | OSDI 22
- [Response Length Perception and Sequence Scheduling: An LLM-Empowered LLM Inference Pipeline](https://arxiv.org/abs/2305.13144) | NUS
- [Efficiently Scaling Transformer Inference](https://arxiv.org/pdf/2211.05102.pdf) | MLSys' 23
- [Flover](https://arxiv.org/pdf/2305.13484.pdf): A Temporal Fusion Framework for Efficient Autoregressive Model Parallel Inference 
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/pdf/2205.14135.pdf)
- [DeepSpeed Inference](https://arxiv.org/abs/2207.00032) : Enabling Efficient Inference of Transformer Models at Unprecedented Scale.  
- [TurboTransformers](https://arxiv.org/pdf/2010.05680.pdf): An Efficient GPU Serving System For Transformer Models
- [MPCFormer](https://arxiv.org/pdf/2211.01452.pdf) : fast, performant, and private transformer inference with MPC | ICLR'23
- [POLCA](https://arxiv.org/abs/2308.12908): Power Oversubscription in LLM Cloud Providers | Microsoft
- [SARATHI](https://arxiv.org/abs/2308.16369): Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills | Microsoft 
- [FlexGen](https://arxiv.org/abs/2303.06865): High-throughput Generative Inference of Large Language Models with a Single GPU | ICML' 23
- [AttMemo](https://arxiv.org/pdf/2301.09262.pdf): Accelerating Self-Attention with Memoization on Big Memory Systems
- [vLLM](https://vllm.ai/): Easy, Fast, and Cheap LLM Serving with PagedAttention | SOSP' 23
- [Tabi](https://dl.acm.org/doi/pdf/10.1145/3552326.3587438): An Efficient Multi-Level Inference System for Large Language Models | EuroSys' 23 
- [Flash-LLM](https://arxiv.org/pdf/2309.10285v1.pdf): Enabling Cost-Effective and Highly-Efficient Large Generative Model Inference with Unstructured Sparsity | VLDB' 24
- [AutoGen](https://arxiv.org/abs/2308.08155): Enabling Next-Gen LLM Applications via Multi-Agent Conversation | Microsoft
- [FlashDecoding++](https://arxiv.org/pdf/2311.01282.pdf): Faster Large Language Model Inference on GPUs | Tsinghua
- [DeepSpeed-MII](https://github.com/microsoft/DeepSpeed-MII): Model Implementations for Inference (MII) ｜ Microsoft
- [Punica](https://arxiv.org/abs/2310.18547): Multi-Tenant LoRA Serving
- [S-LoRA](https://arxiv.org/abs/2311.03285): Serving Thousands of Concurrent LoRA Adapters
- [STI](https://arxiv.org/abs/2207.05022): Turbocharge NLP Inference at the Edge via Elastic Pipelining | ASPLOS 23 
- [SpotServe](https://arxiv.org/abs/2311.15566): Serving Generative Large Language Models on Preemptible Instances | CMU
- [LLM in a flash: Efficient Large Language Model Inference with Limited Memory](https://arxiv.org/abs/2312.11514) | Apple
- [SuperServe:](https://arxiv.org/pdf/2312.16733.pdf) Fine-Grained Inference Serving for Unpredictable Workloads
- [Fairness in Serving Large Language Models](https://arxiv.org/abs/2401.00588) | OSDI' 24
- [Infinite-LLM](https://arxiv.org/abs/2401.02669): Efficient LLM Service for Long Context with DistAttention and Distributed KVCache
- [CaraServe](https://arxiv.org/abs/2401.11240): CPU-Assisted and Rank-Aware LoRA Serving for Generative LLM Inference
- [DistServe](https://arxiv.org/abs/2401.09670): Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving
- [Inference without Interference](https://arxiv.org/abs/2401.11181): Disaggregate LLM Inference for Mixed Downstream Workloads
- [APIServe](https://arxiv.org/pdf/2402.01869.pdf): Efficient API Support for Large-Language Model Inferencing
- [FlexLLM](https://arxiv.org/abs/2402.18789): A System for Co-Serving Large Language Model Inference and Parameter-Efficient Finetuning
- [DéjàVu](https://arxiv.org/abs/2403.01876): KV-cache Streaming for Fast, Fault-tolerant Generative LLM Serving
- [Optimizing LLM Queries in Relational Workloads](https://arxiv.org/abs/2403.05821) | UCB
- [AttentionStore:](https://arxiv.org/pdf/2403.19708.pdf) Cost-effective Attention Reuse across Multi-turn Conversations in Large Language Model Serving | NUS
- [MuxServe:](https://arxiv.org/abs/2404.02015) Flexible Multiplexing for Efficient Multiple LLM Serving
- [LoongServe:](https://arxiv.org/pdf/2404.09526.pdf) Efficiently Serving Long-context Large Language Models with Elastic Sequence Parallelism | PKU
- [RAGCache:](https://arxiv.org/abs/2404.12457) Efficient Knowledge Caching for Retrieval-Augmented Generation | PKU
- [Andes:](https://arxiv.org/abs/2404.16283) Defining and Enhancing Quality-of-Experience in LLM-Based Text Streaming Services | Umich

### Fine-tuning Systems
- [Ymir:](https://tianweiz07.github.io/Papers/24-ics-2.pdf) A Scheduler for Foundation Model Fine-tuning Workloads in Datacenters | ICS' 24
  
### Multi-Model Systems
- [MOSEL](https://arxiv.org/pdf/2310.18481.pdf): Inference Serving Using Dynamic Modality Selection
- [DISTMM](https://www.amazon.science/publications/distmm-accelerating-distributed-multimodal-model-training): Accelerating distributed multimodal model training | NSDI' 24

### Image Generation Systems
- [Approximate Caching for Efficiently Serving Diffusion Models](https://arxiv.org/abs/2312.04429) | Adobe Research
- [DistriFusion:](https://arxiv.org/abs/2402.19481) Distributed Parallel Inference for High-Resolution Diffusion Models |  MIT

## LLM for Systems
- [Large Language Models for Compiler Optimization](https://arxiv.org/abs/2309.07062)
- [The Hitchhiker's Guide to Program Analysis](https://arxiv.org/abs/2308.00245): A Journey with Large Language Models
- [LLM-Assisted Code Cleaning For Training Accurate Code Generators](https://arxiv.org/abs/2311.14904) | UCB


### System Efficiency Optimization
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


## ML Systems
- [INFaaS](https://www.usenix.org/conference/atc21/presentation/romero): Automated Model-less Inference Serving | ATC’ 21
- [Alpa](https://arxiv.org/abs/2201.12023) : Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning | OSDI' 22
- [Pathways](https://proceedings.mlsys.org/paper/2022/hash/98dce83da57b0395e163467c9dae521b-Abstract.html) : Asynchronous Distributed Dataflow for ML | MLSys' 22
- [AlpaServe](https://arxiv.org/pdf/2302.11665.pdf): Statistical Multiplexing with Model Parallelism for Deep Learning Serving
- [DeepSpeed-MoE](https://arxiv.org/abs/2201.05596): Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale ICML' 2022.
- [ZeRO-Offload](https://www.usenix.org/conference/atc21/presentation/ren-jie) : Democratizing Billion-Scale Model Training. 
- [ZeRO-Infinity](https://arxiv.org/pdf/2104.07857) : Breaking the GPU Memory Wall for Extreme Scale Deep Learning  
- [ZeRO](https://arxiv.org/abs/1910.02054) : memory optimizations toward training trillion parameter models.  
- [Band](https://dl.acm.org/doi/pdf/10.1145/3498361.3538948): Coordinated Multi-DNN Inference on Heterogeneous Mobile Processors | MobiSys ’22
- [Serving Heterogeneous Machine Learning Models on Multi-GPU Servers with Spatio-Temporal Sharing](https://www.usenix.org/conference/atc22/presentation/choi-seungbeom) | ATC'22
- [Fast and Efficient Model Serving Using Multi-GPUs with Direct-Host-Access](https://dl.acm.org/doi/pdf/10.1145/3552326.3567508) | Eurosys'23
- [Cocktail](https://www.usenix.org/system/files/nsdi22-paper-gunasekaran.pdf): A Multidimensional Optimization for Model Serving in Cloud | NSDI'22
- [Merak](https://arxiv.org/abs/2206.04959): An Efficient Distributed DNN Training Framework with Automated 3D Parallelism for Giant Foundation Models
- [SHEPHERD](https://www.usenix.org/system/files/nsdi23-zhang-hong.pdf) : Serving DNNs in the Wild
- [Efficient GPU Kernels for N:M-Sparse Weights in Deep Learning](https://proceedings.mlsys.org/paper_files/paper/2023/file/4552cedd396a308320209f75f56a5ad5-Paper-mlsys2023.pdf)
- [AutoScratch](https://proceedings.mlsys.org/paper_files/paper/2023/file/627b5f83ffa130fb33cb03dafb47a630-Paper-mlsys2023.pdf): ML-Optimized Cache Management for Inference-Oriented GPUs
- [ZeRO++](https://arxiv.org/abs/2306.10209): Extremely Efficient Collective Communication for Giant Model Training
- [Channel Permutations for N:M Sparsity](https://proceedings.neurips.cc/paper/2021/hash/6e8404c3b93a9527c8db241a1846599a-Abstract.html) | MLSys' 23
- [Welder](https://www.usenix.org/conference/osdi23/presentation/shi) : Scheduling Deep Learning Memory Access via Tile-graph | OSDI' 23
- [Optimizing Dynamic Neural Networks with Brainstorm](https://www.usenix.org/conference/osdi23/presentation/cui) | OSDI'23
- [ModelKeeper](https://www.usenix.org/conference/nsdi23/presentation/lai-fan): Accelerating DNN Training via Automated Training Warmup | NSDI'23
- [Breadth-First Pipeline Parallelism](https://proceedings.mlsys.org/paper_files/paper/2023/file/14bc46029b7ac590f56a203e0a3ef586-Paper-mlsys2023.pdf) | MLSys' 23
- [MGG](https://www.usenix.org/system/files/osdi23-wang-yuke.pdf) : Accelerating Graph Neural Networks with Fine-Grained Intra-Kernel Communication-Computation Pipelining on Multi-GPU Platforms | OSDI' 23
- [Hydro](https://www.usenix.org/system/files/osdi23-hu.pdf): Surrogate-Based Hyperparameter Tuning Service in Datacenters | OSDI' 23
- [Cocktailer](https://www.usenix.org/system/files/osdi23-zhang-chen.pdf): Analyzing and Optimizing Dynamic Control Flow in Deep Learning | OSDI' 23
- [BPipe](https://proceedings.mlr.press/v202/kim23l/kim23l.pdf): Memory-Balanced Pipeline Parallelism for TrainingLarge Language Models

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
 
## Other list
- [A curated list of Large Language Model](https://github.com/Hannibal046/Awesome-LLM)
- [AI systems paper list](https://github.com/lambda7xx/awesome-AI-system)
- [A baseline repository of Auto-Parallelism in Training Neural Networks](https://github.com/ConnollyLeon/awesome-Auto-Parallelism)
- [Numbers every LLM Developer should know](https://github.com/ray-project/llm-numbers)

## LLM Frameworks
- [AutoGen](https://github.com/microsoft/autogen#autogen) :Enable Next-Gen Large Language Model Applications | Microsoft
- [DeepSpeed](https://github.com/microsoft/DeepSpeed): a deep learning optimization library that makes distributed training and inference easy, efficient, and effective | Microsoft
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | Nvidia
- [Accelerate](https://huggingface.co/docs/accelerate/index) | Hugging Face
- [vLLM](https://github.com/vllm-project/vllm) | UCB
- [Ray-LLM](https://github.com/ray-project/ray-llm) | Ray
  
## Related Readings
- [Large Transformer Model Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/)
- [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/)
- [The Transformer Family Version 2.0](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/)
- [Full Stack Optimization of Transformer Inference](https://arxiv.org/pdf/2302.14017.pdf): a Survey | UCB
