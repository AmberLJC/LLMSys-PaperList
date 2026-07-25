#!/usr/bin/env python3
"""Extract per-year paper / technique statistics from README.md into trends.json.

Year attribution priority:
  1. Explicit year bucket  (<details><summary><b>2025</b></summary>  or  **2026**)
  2. Venue tag in the line (| MLSys' 26, | OSDI' 26, ...)
  3. arXiv id YYMM         (arxiv.org/abs/2601.xxxxx -> 2026)
"""
import re, json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
text = README.read_text()

# ----------------------------------------------------------------- parse tree
AREA = {
    ("LLM Systems", "Pre-training"): "Pre-training",
    ("LLM Systems", "Systems for Post-training / RLHF"): "Post-training / RL",
    ("LLM Systems", "Fault Tolerance / Straggler Mitigation"): "Fault Tolerance",
    ("LLM Systems", "LLM serving"): "LLM Serving",
    ("LLM Systems", "Agent Systems"): "Agent Systems",
    ("LLM Systems", "Serving at the edge"): "Edge / On-device",
    ("LLM Systems", "System Efficiency Optimization - Model Co-design"): "Model Co-design",
    ("LLM Systems", "Multi-Modal Training Systems"): "Multi-Modal Training",
    ("LLM Systems", "Multi-Modal Serving Systems"): "Multi-Modal Serving",
    ("LLM for Systems", None): "LLM-for-Systems",
}

VENUE = (r"(MLSys|EuroSys|OSDI|SOSP|NSDI|ASPLOS|ICML|NeurIPS|ICLR|ACL|ISCA|PPoPP|"
         r"VLDB|SIGCOMM|SIGMOD|ICS|CGO|SC|ISC|MICRO|HPCA|ATC|USENIX)['\s]*'?\s*(\d{2})\b")


def venue_year(line):
    ys = [2000 + int(m.group(2)) for m in re.finditer(VENUE, line)]
    return max(ys) if ys else None


def arxiv_year(line):
    m = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{2})(\d{2})\.", line)
    if not m:
        return None
    yy, mm = int(m.group(1)), int(m.group(2))
    return 2000 + yy if 1 <= mm <= 12 else None


def mlsys_virtual(line):
    return 2026 if "mlsys.org/virtual/2026" in line else None


def sosp25(line):
    return 2025 if "conferences/sosp/2025" in line else None


papers = []          # (area, year, raw_line)
section = subsection = bucket = None
for raw in text.split("\n"):
    s = raw.strip()
    if s.startswith("## ") and not s.startswith("###"):
        section, subsection, bucket = s[3:].strip(), None, None
        continue
    if s.startswith("#### "):
        subsection, bucket = s[5:].strip(), None
        continue
    if s.startswith("### "):
        subsection, bucket = s[4:].strip(), None
        continue
    m = re.search(r"<summary><b>(.*?)</b></summary>", s)
    if m:
        bucket = m.group(1).strip()
        continue
    m = re.match(r"\*\*(\d{4})\*\*\s*$", s)
    if m:
        bucket = m.group(1)
        continue
    if not raw.startswith("- "):        # skip nested "  - " framework entries
        continue

    area = AREA.get((section, subsection))
    if area is None:
        continue

    if bucket and "Before" in bucket:
        year = "<=2023"
    elif bucket:
        year = bucket
    else:
        y = (venue_year(s) or mlsys_virtual(s) or sosp25(s) or arxiv_year(s))
        year = str(y) if y and y >= 2024 else ("<=2023" if y else "unknown")
    papers.append((area, year, s))

# ------------------------------------------------------------ area x year
area_year = defaultdict(Counter)
for area, year, _ in papers:
    area_year[area][year] += 1

YEARS = ["2024", "2025", "2026"]
year_total = Counter(y for _, y, _ in papers)

# ------------------------------------------------------------ techniques
THEMES = {
    "MoE / expert parallelism":
        r"\bMoE\b|mixture[- ]of[- ]experts|expert[- ]parallel|\bexperts?\b",
    "Long-context / seq+ctx parallelism":
        r"long[- ]context|context parallel|sequence parallel|long[- ]sequence|"
        r"million[- ]token|1M token|ultra[- ]?long|long video",
    "Sparse attention":
        r"sparse attention|sparse[- ]linear|attention sparsit|\bsparsity\b|"
        r"sparse self|blocked attention|top[- ]k",
    "KV cache / context reuse":
        r"KV[- ]?cache|KVCache|prefix cach|context cach|paged ?attention|"
        r"context reuse|prefix sharing",
    "Prefill-decode disaggregation":
        r"disaggregat|prefill[- ]?decode|prefill[- ]only|\bPD\b|\bEPD\b|"
        r"prefill[- ]length|layered prefill",
    "Speculative decoding":
        r"speculative|\bdrafter?\b|draft model|speculate",
    "Quantization / low precision":
        r"quantiz|\bFP8\b|\bFP4\b|\bINT8\b|\bINT4\b|\d-bit|low[- ]bit|"
        r"low[- ]precision|mixed[- ]precision|microscaling|bit[- ]pack",
    "Heterogeneous / cross-cluster / spot":
        r"heterogene|cross[- ]cluster|cross[- ]region|geo[- ]distributed|"
        r"multi[- ]region|commodity GPU|superchip|\bspot\b|preemptible|"
        r"decentraliz|multi[- ]cloud|cloud[- ]edge",
    "Agentic / multi-agent":
        r"\bagent|agentic|tool[- ]call|computer[- ]use|tree[- ]of[- ]thought|"
        r"multi[- ]turn|test[- ]time compute|\bworkflow",
    "RAG / retrieval / vector search":
        r"\bRAG\b|retrieval|vector search|\bANNS\b|vector[- ]storage|reranker|embedding",
    "RL post-training / rollout":
        r"reinforcement learning|\bRLHF\b|\bRL\b|rollout|post[- ]training|\bPPO\b|\bGRPO\b",
    "Energy / power / thermal":
        r"\benergy\b|\bpower\b|\bDVFS\b|joule|carbon|watt",
    "Reliability / fault tolerance / SDC":
        r"fault|failure|straggler|checkpoint|\bSDC\b|resilien|reliab|recovery|"
        r"diagnos|debug|verif|integrity|robust|determinis",
    "Compiler / kernel / megakernel":
        r"compil|\bkernel|\bCUDA\b|megakernel|mega[- ]kernel|\btile|dataflow|"
        r"operator|Triton|warp|superoptimiz|\bcode ?gen|\bPTX\b|layout synthesis",
    "Serverless / elastic / cold start":
        r"serverless|autoscal|\belastic|cold start|ballooning|\bFaaS|scale[- ]up|"
        r"live migration|multiplex",
    "Diffusion / video / any-to-any":
        r"diffusion|\bDiT\b|\bvideo\b|text[- ]to[- ]image|image generation|"
        r"\bspeech\b|\bVLA\b|\bomni\b|any[- ]to[- ]any|image editing",
    "Scheduling / SLO / goodput":
        r"schedul|\bSLO\b|goodput|fairness|head[- ]of[- ]line|\bqueue|admission|"
        r"load balanc|routing|placement",
    "Memory offload / hierarchy (CPU,SSD,PIM,CXL)":
        r"offload|CPU memory|memory management|memory pool|\bNDP\b|\bPIM\b|\bCXL\b|"
        r"\bSSD\b|storage bandwidth|memory wall|fragmentation|\bSRAM\b|zero[- ]copy",
    "Communication / collectives / network":
        r"communication|collective|\bRDMA\b|\bnetwork|\bNCCL\b|overlap|all[- ]reduce|"
        r"multipath|transport|interconnect|point[- ]to[- ]point",
    "Benchmark / profiling / characterization":
        r"benchmark|profil|character|demystif|measur|\btrace|simulat|"
        r"pragmatic|illusion|revisit|rethink|\bstudy\b|insights|what[- ]if",
}

theme_year = defaultdict(Counter)
for area, year, line in papers:
    if year not in YEARS:
        continue
    for th, pat in THEMES.items():
        if re.search(pat, line, re.IGNORECASE):
            theme_year[th][year] += 1

theme_share = {
    th: {y: 100.0 * c[y] / year_total[y] for y in YEARS}
    for th, c in theme_year.items()
}

# ------------------------------------------------------------ 2026 venues
venue_2026 = Counter()
for area, year, line in papers:
    if year != "2026":
        continue
    tags = {m.group(1) for m in re.finditer(VENUE, line) if 2000 + int(m.group(2)) == 2026}
    if "mlsys.org/virtual/2026" in line:
        tags.add("MLSys")
    if not tags:
        venue_2026["arXiv / preprint"] += 1
    else:
        for t in tags:
            venue_2026["USENIX ATC" if t == "ATC" else t] += 1

# ------------------------------------------------------------ 2026 affiliations
ORGS = ["CMU", "UCB", "UMich", "Stanford", "UT Austin", "PKU", "THU", "SJTU", "UW",
        "MIT", "NUS", "ETH", "UCSD", "UIUC", "UChicago", "UCLA", "MBZUAI", "Purdue",
        "ByteDance", "DeepSeek", "Alibaba", "Microsoft", "NVIDIA", "Meta", "Google",
        "Ant", "Huawei", "Moonshot", "Kuaishou", "Tencent", "Apple", "Seed", "vLLM"]
aff_2026 = Counter()
for area, year, line in papers:
    if year != "2026":
        continue
    tail = line.split("|")[-1]
    for o in ORGS:
        if re.search(rf"\b{re.escape(o)}\b", tail) or re.search(rf"\|\s*[^|]*\b{re.escape(o)}\b", line):
            aff_2026[o] += 1

# ------------------------------------------------------------ industrial reports
ind = re.search(r"## Industrial LLM Technical Report\n(.*?)\n## ", text, re.DOTALL).group(1)
NORM = {"Google / DeepMind": "Google", "Moonshot AI": "Moonshot", "Mistral AI": "Mistral",
        "Zhipu AI": "Zhipu", "InternLM (Shanghai AI Lab)": "InternLM"}
company_year = Counter()
for ln in ind.split("\n"):
    m = re.search(r"[–-]\s*([^(|]+?)\s*\(\w+\s+(\d{4})\)", ln)
    if m:
        company_year[(NORM.get(m.group(1).strip(), m.group(1).strip()), int(m.group(2)))] += 1

out = {
    "years": YEARS,
    "year_total": dict(year_total),
    "area_year": {k: dict(v) for k, v in sorted(area_year.items())},
    "theme_count": {k: dict(v) for k, v in theme_year.items()},
    "theme_share": theme_share,
    "venue_2026": dict(venue_2026.most_common()),
    "aff_2026": dict(aff_2026.most_common(14)),
    "company_year": {f"{c}|{y}": n for (c, y), n in company_year.items()},
}
(Path(__file__).parent / "trends.json").write_text(json.dumps(out, indent=2))
print(json.dumps({k: out[k] for k in
                  ["year_total", "area_year", "venue_2026", "aff_2026"]}, indent=2))
print("\n--- theme share (% of that year's papers) ---")
for th, sh in sorted(theme_share.items(), key=lambda kv: -kv[1]["2026"]):
    c = theme_year[th]
    print(f"{th:46s} 24:{c['2024']:3d} ({sh['2024']:4.1f}%)  "
          f"25:{c['2025']:3d} ({sh['2025']:4.1f}%)  26:{c['2026']:3d} ({sh['2026']:4.1f}%)")
print("\nunknown-year lines:", sum(1 for _, y, _ in papers if y == "unknown"))
for a, y, l in papers:
    if y == "unknown":
        print("   ", l[:110])
