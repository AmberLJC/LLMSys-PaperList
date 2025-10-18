# Claude Code Guidelines for LLMSys-PaperList

This document provides guidance for Claude Code when working with the LLMSys-PaperList repository.

## Repository Overview

This is a curated list of Large Language Model (LLM) systems-related academic papers, articles, tutorials, slides, and projects. The repository serves as a comprehensive resource for researchers and practitioners to stay updated on the latest developments in LLM systems research.

## Repository Structure

The repository consists primarily of a single `README.md` file organized into the following main sections:

### 1. **LLM Systems**
The core section containing system-level research papers organized by:

#### Training
- **Pre-training**: Papers focused on initial model training (parallel training, optimization, infrastructure)
- **Post Training / RLHF**: Papers on fine-tuning and reinforcement learning from human feedback
- **Fault Tolerance / Straggler Mitigation**: Papers on reliability and handling failures

#### Serving
- **LLM serving**: Papers on efficient LLM inference and serving
- **Agent Systems**: Papers on LLM-based agent frameworks and orchestration
- **Serving at the edge**: Papers on edge deployment and resource-constrained inference
- **System Efficiency Optimization - Model Co-design**: Papers on co-designing systems and models for efficiency

#### Multi-Modal Systems
- **Multi-Modal Training Systems**: Papers on training multimodal models
- **Multi-Modal Serving Systems**: Papers on serving multimodal models (including diffusion models)

### 2. **LLM for Systems**
Papers where LLMs are used to optimize or improve traditional systems (compilers, debugging, etc.)

### 3. **Industrial LLM Technical Reports**
Official technical reports from major AI companies (OpenAI, Meta, Google, DeepSeek, etc.)

### 4. **LLM Frameworks**
Open-source frameworks organized by:
- **Training**: DeepSpeed, Megatron, NeMo, etc.
- **Post-Training**: TRL, OpenRLHF, VeRL, etc.
- **Serving**: vLLM, SGLang, TensorRT-LLM, etc.

### 5. **ML Systems**
General machine learning systems papers (separate file: `mlsystems.md`)

### 6. **Survey Papers**
Comprehensive survey papers on LLM efficiency and serving

### 7. **LLM Benchmark / Leaderboard / Traces**
Benchmarks, leaderboards, and workload traces

### 8. **Related ML Readings**
Blog posts and articles on LLM inference and transformers

### 9. **MLSys Courses**
University courses on ML systems

### 10. **Other Reading**
Additional curated lists and resources

## Formatting Guidelines

When adding new papers to this repository, follow these conventions:

### Paper Entry Format
```markdown
- [Paper Title](https://arxiv.org/abs/XXXX.XXXXX): Brief description | Venue/Organization
```

**Key formatting rules:**
1. **Links**: Use arXiv links in format `https://arxiv.org/abs/XXXX.XXXXX` (without `www.` prefix)
2. **Conference links**: Use official conference URLs (e.g., USENIX, ACM) when available
3. **Titles**: Use exact paper titles with proper capitalization
4. **Descriptions**: After the colon, provide a brief description of the paper's contribution
5. **Metadata**: After the pipe `|`, include venue (e.g., `OSDI' 24`) and/or organization (e.g., `Microsoft`)
6. **Spacing**: Use consistent spacing with other entries in the section

### Section Headers
- Main sections: `##` (h2)
- Subsections: `###` (h3)
- Sub-subsections: `####` (h4)

### Examples

**Good:**
```markdown
- [The ML.ENERGY Benchmark](https://arxiv.org/abs/2505.06371): Toward Automated Inference Energy Measurement and Optimization | NeurIPS' 25
- [DISTMM](https://www.usenix.org/conference/nsdi24/presentation/huang): Accelerating distributed multimodal model training | NSDI' 24
```

**Avoid:**
```markdown
- [Paper](https://www.arxiv.org/abs/2505.06371) - description (venue)  # Wrong: has www., wrong separators
```

## Content Organization

### Where to Add Papers

When adding new papers, consider the primary focus:

1. **Training-focused papers** → `### Training` section
   - Initial training → `#### Pre-training`
   - Fine-tuning/RLHF → `#### Post Training`
   - Fault tolerance → `#### Fault Tolerance / Straggler Mitigation`

2. **Inference/serving papers** → `### Serving` section
   - General LLM serving → `#### LLM serving`
   - Agent systems → `#### Agent Systems`
   - Edge deployment → `#### Serving at the edge`
   - Model-system co-design → `#### System Efficiency Optimization - Model Co-design`

3. **Multimodal papers**:
   - Training → `### Multi-Modal Training Systems`
   - Inference/serving → `### Multi-Modal Serving Systems`

4. **Benchmarks and measurement tools** → `## LLM Benchmark / Leaderboard / Traces`

5. **Framework implementations** → `## LLM Frameworks`

## Best Practices for Updates

1. **Consistency**: Always match the existing formatting style
2. **Verification**: Verify URLs work and point to the correct papers
3. **Completeness**: Include venue/conference information when available
4. **Chronological order**: Papers are generally added in chronological order within sections
5. **Avoid duplicates**: Check if a paper already exists before adding
6. **Subsections**: Use existing subsections when appropriate, create new ones sparingly

## Table of Contents

When adding new subsections, remember to update the Table of Contents at the top of README.md to maintain navigation consistency.

## Common Tasks

### Adding a new paper
1. Identify the appropriate section based on the paper's primary focus
2. Format the entry following the guidelines above
3. Add it to the appropriate location (usually at the end of the subsection or in chronological order)
4. Verify the link works

### Reorganizing sections
1. When creating new subsections, use `####` for subsections under `###`
2. Update the Table of Contents if adding new major sections
3. Maintain alphabetical or logical ordering within sections

### Updating links
1. Prefer official conference/journal URLs over arXiv when available
2. Always remove `www.` from arXiv URLs
3. Ensure consistency across similar entries

## Notes

- This is a living document that tracks the rapidly evolving field of LLM systems
- Papers are typically from top-tier venues (OSDI, SOSP, MLSys, NeurIPS, etc.) or well-cited arXiv preprints
- The repository focuses on **systems research**, not pure ML or algorithm papers
- Both academic papers and industrial technical reports are included
