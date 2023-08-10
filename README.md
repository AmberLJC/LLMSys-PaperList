# Awesome LLM Systems Papers

A curated list of FL systems-related academic papers, articles, tutorials, slides and projects. Star this repository, and then you can keep abreast of the latest developments of this booming research field.

## LLM Systems
- [Orca](https://www.usenix.org/conference/osdi22/presentation/yu): A Distributed Serving System for Transformer-Based Generative Models | OSDI 22
- [FrugalGPT](https://arxiv.org/pdf/2305.05176.pdf): How to Use Large Language Models While Reducing Cost and Improving Performance |  Stanford
- [Fast Distributed Inference Serving for Large Language Models](https://arxiv.org/abs/2305.05920) | Peking University
- [Response Length Perception and Sequence Scheduling: An LLM-Empowered LLM Inference Pipeline](https://arxiv.org/abs/2305.13144) | NUS
- [Efficiently Scaling Transformer Inference](https://arxiv.org/pdf/2211.05102.pdf) | MLSys' 23
- [Flover](https://arxiv.org/pdf/2305.13484.pdf): A Temporal Fusion Framework for Efficient Autoregressive Model Parallel Inference 
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/pdf/2205.14135.pdf)
- [Megatron-LM](https://arxiv.org/pdf/1909.08053.pdf): Training Multi-Billion Parameter Language Models Using Model Parallelism
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/pdf/2104.04473.pdf)
- [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/pdf/2205.05198.pdf)
- [DeepSpeed Inference](https://arxiv.org/abs/2207.00032) : Enabling Efficient Inference of Transformer Models at Unprecedented Scale.  
- [FlexGen](https://arxiv.org/abs/2303.06865): High-throughput Generative Inference of Large Language Models with a Single GPU | UCB
- [S3](https://arxiv.org/pdf/2306.06000.pdf): Increasing GPU Utilization during Generative Inference for Higher Throughput
- [Scissorhands](https://arxiv.org/abs/2305.17118): Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time
- [AttMemo](https://arxiv.org/pdf/2301.09262.pdf): Accelerating Self-Attention with Memoization on Big Memory Systems
- [vLLM](https://vllm.ai/): Easy, Fast, and Cheap LLM Serving with PagedAttention | SOSP' 23
- [Tabi](https://dl.acm.org/doi/pdf/10.1145/3552326.3587438): An Efficient Multi-Level Inference System for Large Language Models | EuroSys' 23 
- [TurboTransformers](https://arxiv.org/pdf/2010.05680.pdf): An Efficient GPU Serving System For Transformer Models
- [Inference with Reference](https://arxiv.org/abs/2304.04487): Lossless Acceleration of Large Language Models
- [H2O](https://arxiv.org/abs/2306.14048): Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models
- [SkipDecode](https://arxiv.org/abs/2307.02628): Autoregressive Skip Decoding with Batching and Caching for Efficient LLM Inferencex
- [Full Stack Optimization of Transformer Inference](https://arxiv.org/pdf/2302.14017.pdf): a Survey
- [Optimized Network Architectures for Large Language Model Training with Billions of Parameters](https://arxiv.org/pdf/2307.12169.pdf) | UCB
- [MPCFormer](https://arxiv.org/pdf/2211.01452.pdf) : fast, performant, and private transformer inference with MPC | ICLR'23 



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


## Other list
- [A curated list of Large Language Model](https://github.com/Hannibal046/Awesome-LLM)
- [AI systems paper list](https://github.com/lambda7xx/awesome-AI-system)
- [A baseline repository of Auto-Parallelism in Training Neural Networks](https://github.com/ConnollyLeon/awesome-Auto-Parallelism)

## Related Readings
- [Large Transformer Model Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/)
- [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/)
- [The Transformer Family Version 2.0](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/)
- [Challenges and Applications of Large Language Models](https://arxiv.org/pdf/2307.10169.pdf)
