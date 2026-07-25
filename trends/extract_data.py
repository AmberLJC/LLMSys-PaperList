#!/usr/bin/env python3
"""Extract paper counts from README.md into a JSON for figure scripts."""
import re, json
from collections import Counter, defaultdict
from pathlib import Path

README = Path(__file__).resolve().parent.parent / "README.md"
text = README.read_text()
lines = text.split("\n")

section, subsection, year_bucket = None, None, None
papers = []
for line in lines:
    s = line.strip()
    if s.startswith("## ") and not s.startswith("### "):
        section, subsection, year_bucket = s[3:].strip(), None, None
        continue
    if s.startswith("### "):
        subsection, year_bucket = s[4:].strip(), None
        continue
    if s.startswith("#### "):
        subsection, year_bucket = s[5:].strip(), None
        continue
    m = re.search(r"<summary><b>(.*?)</b></summary>", s)
    if m:
        year_bucket = m.group(1).strip(); continue
    m2 = re.match(r"\*\*(\d{4})\*\*", s)
    if m2:
        year_bucket = m2.group(1); continue
    if s.startswith("- "):
        papers.append((section, subsection, year_bucket, s))

def norm_year(yb):
    if yb is None: return None
    if "Before" in yb: return "<=2023"
    return yb

def year_from_line(line):
    m = re.search(r"arxiv\.org/abs/(\d{2})(\d{2})\.", line)
    if m: return 2000 + int(m.group(1))
    m2 = re.search(r"(MLSys|EuroSys|OSDI|SOSP|NSDI|ASPLOS|ICML|NeurIPS|ICLR|ACL|SC|VLDB|SIGCOMM|ISCA|PPoPP|ICS)['\s]+(\d{2})", line)
    if m2: return 2000 + int(m2.group(2))
    return None

# Build counts per subarea per year
subarea_year = defaultdict(Counter)
SECTIONS_OF_INTEREST = {
    ("LLM Systems", "Pre-training"): "Pre-training",
    ("LLM Systems", "Systems for Post-training / RLHF"): "Post-training/RLHF",
    ("LLM Systems", "Fault Tolerance / Straggler Mitigation"): "Fault Tolerance",
    ("LLM Systems", "LLM serving"): "LLM Serving",
    ("LLM Systems", "Agent Systems"): "Agent Systems",
    ("LLM Systems", "Serving at the edge"): "Edge Serving",
    ("LLM Systems", "System Efficiency Optimization - Model Co-design"): "Model Co-design",
    ("LLM Systems", "Multi-Modal Training Systems"): "Multi-Modal Training",
    ("LLM Systems", "Multi-Modal Serving Systems"): "Multi-Modal Serving",
    ("LLM for Systems", None): "LLM-for-Systems",
}

for sec, sub, yb, line in papers:
    key = SECTIONS_OF_INTEREST.get((sec, sub))
    if not key: continue
    y = norm_year(yb)
    if y is None:  # infer for sections without explicit year buckets
        yi = year_from_line(line)
        if yi:
            y = str(yi) if yi >= 2024 else "<=2023"
        else:
            y = "unspecified"
    subarea_year[key][y] += 1

# Industrial reports by (company, year)
ind = re.search(r"## Industrial LLM Technical Report\n(.*?)\n## ", text, re.DOTALL).group(1)
COMPANY_NORMALIZE = {
    "Google / DeepMind": "Google",
    "Moonshot AI": "Moonshot",
    "Mistral AI": "Mistral",
    "Zhipu AI": "Zhipu",
    "01.AI": "01.AI",
    "InternLM (Shanghai AI Lab)": "InternLM",
    "BigScience": "BigScience",
    "BigCode": "BigCode",
}
company_year = Counter()
for ln in ind.split("\n"):
    m = re.search(r"–\s*([^(]+?)\s*\((\w+\s+(\d{4}))\)", ln)
    if not m: continue
    c = m.group(1).strip()
    c = COMPANY_NORMALIZE.get(c, c)
    y = int(m.group(3))
    company_year[(c, y)] += 1

# Hot theme keyword counts per year (look at every "- " line with a year tag)
THEMES = {
    "MoE / Experts": r"\b(MoE|Mixture[- ]of[- ]Experts|expert[- ]parallel|expert serving)\b",
    "KV Cache": r"\bKV[- ]?cache\b|\bKVCache\b",
    "Disaggregation": r"\b(disaggregat|prefill[- ]decode|prefill\-only|PD[- ]disaggreg)",
    "Speculative Decoding": r"\b(speculative|specInfer|spec\-decoding|draft model)",
    "RAG": r"\bRAG\b|\bretrieval[- ]augmented\b",
    "Agent / Agentic": r"\bagent\b|\bagentic\b|\bmulti[- ]agent\b",
    "Long Context": r"\blong[- ]context\b|\bmillion[- ]token\b|\bultra[- ]long\b|\bcontext parallel",
    "Quantization": r"\b(quantiz|FP8|INT4|INT8|AWQ|GPTQ)\b",
}

theme_year = defaultdict(Counter)
for sec, sub, yb, line in papers:
    y = norm_year(yb)
    if y is None:
        yi = year_from_line(line)
        if yi and yi >= 2024:
            y = str(yi)
        else:
            continue
    if y not in {"2024", "2025", "2026"}:
        continue
    for theme, pat in THEMES.items():
        if re.search(pat, line, re.IGNORECASE):
            theme_year[theme][y] += 1

out = {
    "subarea_year": {k: dict(v) for k, v in subarea_year.items()},
    "company_year": {f"{c}|{y}": n for (c, y), n in company_year.items()},
    "theme_year": {k: dict(v) for k, v in theme_year.items()},
}
out_path = Path(__file__).resolve().parent / "data.json"
out_path.write_text(json.dumps(out, indent=2))
print(f"Wrote {out_path}")
print(json.dumps(out, indent=2))
