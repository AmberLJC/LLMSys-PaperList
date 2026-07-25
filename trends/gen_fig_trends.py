#!/usr/bin/env python3
"""Generate the 2024->2026 trend charts for LLMSys-PaperList.

Reads figures/trends.json (produced by extract_trends.py) and emits:
  fig_area_shift      - per-area counts + composition share per year
  fig_theme_heatmap   - technique x year share heatmap
  fig_theme_momentum  - share change 2024 -> 2026 (percentage points)
  fig_trajectory      - rising vs receding technique trajectories
  fig_2026_venues     - where the 2026 evidence comes from
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
D = json.loads((HERE / "trends.json").read_text())
YEARS = D["years"]
TOT = {y: D["year_total"][y] for y in YEARS}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.labelsize": 9, "legend.fontsize": 8, "legend.frameon": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.15, "grid.linestyle": "-",
    "lines.linewidth": 1.8, "lines.markersize": 5,
})

# "Ocean Dusk"
C_DEEP, C_TEAL, C_GOLD, C_SAND, C_CORAL = "#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51"
C_GRAY, C_BLUE = "#B0BEC5", "#0072B2"
YEAR_COLOR = {"2024": C_GRAY, "2025": C_TEAL, "2026": C_CORAL}
OCEAN = LinearSegmentedColormap.from_list("ocean", ["#FBFCFC", "#CFE6E2", C_TEAL, C_SAND, C_CORAL])


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"fig_{name}.{ext}")
    plt.close(fig)
    print(f"  wrote fig_{name}.pdf / .png")


# --------------------------------------------------------------- Figure 1
MACRO = {
    "Pre-training": ("Training infra", C_DEEP),
    "Multi-Modal Training": ("Training infra", C_DEEP),
    "Post-training / RL": ("Post-training / RL", C_TEAL),
    "Fault Tolerance": ("Reliability", "#7B8794"),
    "LLM Serving": ("Serving runtime", C_GOLD),
    "Multi-Modal Serving": ("Multi-modal / diffusion", C_SAND),
    "Edge / On-device": ("Edge / on-device", "#9C6644"),
    "Model Co-design": ("Kernel / model co-design", C_CORAL),
    "Agent Systems": ("Agentic systems", "#5E60CE"),
    "LLM-for-Systems": ("AI-for-systems", "#118AB2"),
}


SHORT = {
    "LLM Serving": "LLM serving", "Pre-training": "Pre-training",
    "Model Co-design": "Model co-design", "Agent Systems": "Agent systems",
    "LLM-for-Systems": "AI-for-systems", "Multi-Modal Serving": "MM serving",
    "Edge / On-device": "Edge/on-device", "Post-training / RL": "Post-train/RL",
    "Fault Tolerance": "Fault tolerance", "Multi-Modal Training": "MM training",
}


def fig_area_shift():
    areas = sorted(D["area_year"], key=lambda a: -D["area_year"][a].get("2026", 0))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.3),
                                  gridspec_kw={"width_ratios": [2.05, 1.0]})
    x = np.arange(len(areas))
    w = 0.26
    for i, y in enumerate(YEARS):
        v = [D["area_year"][a].get(y, 0) for a in areas]
        b = ax.bar(x + (i - 1) * w, v, w * 0.92, label=f"{y}  (n={TOT[y]})",
                   color=YEAR_COLOR[y], edgecolor="white", linewidth=0.4)
        if y == "2026":
            for bar, s in zip(b, v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                        str(s), ha="center", va="bottom", fontsize=6, color=C_CORAL)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[a] for a in areas], fontsize=7, rotation=32,
                       ha="right", rotation_mode="anchor")
    ax.set_ylabel("papers in list")
    ax.set_title("(a) Papers per sub-area", loc="left")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 95)

    # (b) composition share, macro buckets
    order = ["Training infra", "Post-training / RL", "Reliability", "Serving runtime",
             "Multi-modal / diffusion", "Edge / on-device", "Kernel / model co-design",
             "Agentic systems", "AI-for-systems"]
    color = {MACRO[a][0]: MACRO[a][1] for a in MACRO}
    share = {b: [] for b in order}
    for y in YEARS:
        agg = {b: 0 for b in order}
        for a, yc in D["area_year"].items():
            agg[MACRO[a][0]] += yc.get(y, 0)
        tot = sum(agg.values())
        for b in order:
            share[b].append(100 * agg[b] / tot)
    left = np.zeros(len(YEARS))
    ypos = np.arange(len(YEARS))[::-1]
    for b in order:
        v = np.array(share[b])
        ax2.barh(ypos, v, left=left, height=0.62, color=color[b], label=b,
                 edgecolor="white", linewidth=0.6)
        for yy, vv, ll in zip(ypos, v, left):
            if vv >= 7:
                ax2.text(ll + vv / 2, yy, f"{vv:.0f}", ha="center", va="center",
                         fontsize=6.5, color="white", weight="bold")
        left += v
    ax2.set_yticks(ypos)
    ax2.set_yticklabels(YEARS)
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("share of that year's papers (%)")
    ax2.set_title("(b) Composition shift", loc="left")
    ax2.grid(False)
    h, l = ax2.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=5,
               fontsize=6.6, handlelength=1.1, columnspacing=1.0, labelspacing=0.3)
    fig.subplots_adjust(wspace=0.26)
    save(fig, "area_shift")


# --------------------------------------------------------------- Figure 2
def fig_theme_heatmap():
    sh = D["theme_share"]
    cnt = D["theme_count"]
    themes = sorted(sh, key=lambda t: -sh[t]["2026"])
    M = np.array([[sh[t][y] for y in YEARS] for t in themes])
    fig, ax = plt.subplots(figsize=(4.3, 6.4))
    im = ax.imshow(M, cmap=OCEAN, aspect="auto", vmin=0, vmax=M.max())
    ax.set_xticks(range(len(YEARS)))
    ax.set_xticklabels([f"{y}\nn={TOT[y]}" for y in YEARS], fontsize=8.5)
    ax.set_yticks(range(len(themes)))
    ax.set_yticklabels(themes, fontsize=7.4)
    for i, t in enumerate(themes):
        for j, y in enumerate(YEARS):
            v = M[i, j]
            ax.text(j, i, f"{v:.1f}%\n({cnt[t].get(y, 0)})", ha="center", va="center",
                    fontsize=6.2, color="white" if v > 0.80 * M.max() else "#2C2C2C",
                    linespacing=0.95)
    ax.set_title("Technique share of each year's papers\n"
                 "(keyword match on title + blurb; papers may match several)",
                 loc="left", fontsize=8.6)
    ax.grid(False)
    ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(themes), 1), minor=True)
    ax.tick_params(which="minor", length=0)
    ax.grid(which="minor", color="white", linewidth=1.4)
    cb = fig.colorbar(im, ax=ax, shrink=0.35, aspect=14, pad=0.03)
    cb.set_label("% of year", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    save(fig, "theme_heatmap")


# --------------------------------------------------------------- Figure 3
def fig_theme_momentum():
    sh = D["theme_share"]
    delta = {t: sh[t]["2026"] - sh[t]["2024"] for t in sh}
    themes = sorted(delta, key=lambda t: delta[t])
    v = [delta[t] for t in themes]
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    yp = np.arange(len(themes))
    colors = [C_CORAL if d > 0.5 else (C_TEAL if d < -0.5 else C_GRAY) for d in v]
    ax.barh(yp, v, height=0.6, color=colors, edgecolor="white", linewidth=0.5)
    for i, t in enumerate(themes):
        d = delta[t]
        lab = f"{sh[t]['2024']:.0f}% → {sh[t]['2025']:.0f}% → {sh[t]['2026']:.0f}%"
        ax.text(d + (0.28 if d >= 0 else -0.28), i, lab, va="center",
                ha="left" if d >= 0 else "right", fontsize=6.3, color="#555")
    ax.set_yticks(yp)
    ax.set_yticklabels(themes, fontsize=7.6)
    ax.axvline(0, color="#2C2C2C", linewidth=0.9)
    ax.set_xlabel("change in share of the year's papers, 2024 → 2026 (percentage points)")
    ax.set_title("What the field moved toward — and away from", loc="left")
    ax.set_xlim(-7.8, 16.5)
    ax.set_ylim(-0.9, len(themes) - 0.1)
    ax.text(12.6, len(themes) - 1.6, "rising", color=C_CORAL, fontsize=10,
            weight="bold", ha="center")
    ax.text(-5.4, 2.4, "receding", color=C_TEAL, fontsize=10, weight="bold", ha="center")
    save(fig, "theme_momentum")


# --------------------------------------------------------------- Figure 4
RISING = ["Agentic / multi-agent", "Compiler / kernel / megakernel",
          "Heterogeneous / cross-cluster / spot", "MoE / expert parallelism",
          "Speculative decoding", "Energy / power / thermal"]
RECEDING = ["Scheduling / SLO / goodput", "Reliability / fault tolerance / SDC",
            "KV cache / context reuse", "Serverless / elastic / cold start",
            "Long-context / seq+ctx parallelism", "RAG / retrieval / vector search"]
TLABEL = {
    "Agentic / multi-agent": "Agentic / multi-agent",
    "Compiler / kernel / megakernel": "Compiler / kernel",
    "Heterogeneous / cross-cluster / spot": "Heterogeneous fleet",
    "MoE / expert parallelism": "MoE / expert par.",
    "Speculative decoding": "Speculative decoding",
    "Energy / power / thermal": "Energy / power",
    "Scheduling / SLO / goodput": "Scheduling / SLO",
    "Reliability / fault tolerance / SDC": "Reliability / SDC",
    "KV cache / context reuse": "KV cache reuse",
    "Serverless / elastic / cold start": "Serverless / elastic",
    "Long-context / seq+ctx parallelism": "Long-context par.",
    "RAG / retrieval / vector search": "RAG / retrieval",
}


def fig_trajectory():
    sh = D["theme_share"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), sharey=True)
    xs = np.arange(3)
    pal = [C_CORAL, C_DEEP, C_TEAL, C_SAND, "#5E60CE", "#9C6644"]
    mks = ["o", "s", "^", "D", "v", "P"]
    for ax, group, title, style in (
            (axes[0], RISING, "(a) Rising", "-"),
            (axes[1], RECEDING, "(b) Plateauing / receding", "--")):
        for i, t in enumerate(group):
            v = [sh[t][y] for y in YEARS]
            ax.plot(xs, v, style, color=pal[i], marker=mks[i], markersize=4.2,
                    markeredgecolor="white", markeredgewidth=0.6)
            ax.annotate(TLABEL[t], (xs[-1] + 0.05, v[-1]), fontsize=6.6, color=pal[i],
                        va="center", ha="left")
        ax.set_xticks(xs)
        ax.set_xticklabels(YEARS)
        ax.set_xlim(-0.12, 3.05)
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("share of the year's papers (%)")
    axes[0].set_ylim(-0.4, 14.5)
    fig.suptitle("Technique trajectories in the LLM-systems literature", x=0.062,
                 ha="left", fontsize=10, fontweight="bold", y=1.02)
    fig.subplots_adjust(wspace=0.08)
    save(fig, "trajectory")


# --------------------------------------------------------------- Figure 5
def fig_2026_venues():
    v = D["venue_2026"]
    items = [(k, n) for k, n in v.items() if n >= 4]
    other = sum(n for k, n in v.items() if n < 4)
    items.append(("other venues", other))
    items.sort(key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    yp = np.arange(len(items))
    cols = [C_CORAL if k != "arXiv / preprint" else C_GRAY for k, _ in items]
    ax.barh(yp, [n for _, n in items], height=0.6, color=cols,
            edgecolor="white", linewidth=0.5)
    for i, (_, n) in enumerate(items):
        ax.text(n + 1.2, i, str(n), va="center", fontsize=7, color="#555")
    ax.set_yticks(yp)
    ax.set_yticklabels([k for k, _ in items], fontsize=7.5)
    ax.set_xlabel("2026 entries")
    ax.set_xlim(0, max(n for _, n in items) * 1.16)
    ax.set_title("Where the 2026 signal comes from", loc="left", fontsize=9)
    fig.text(0.0, -0.10, "2026 is a partial year, and it is dominated by three full\n"
             "accepted-paper lists (MLSys / EuroSys / OSDI). Read the\n"
             "share-of-year numbers, not the absolute counts.",
             fontsize=6.2, color="#6B7280", va="top")
    save(fig, "2026_venues")


# --------------------------------------------------------------- Figure 6
CN = {"Zhipu", "Alibaba", "DeepSeek", "Moonshot", "MiniMax", "01.AI", "Baichuan",
      "Skywork", "InternLM", "Tencent", "Huawei", "MiniCPM"}
US = {"Google", "OpenAI", "Meta", "Anthropic", "Microsoft", "NVIDIA", "Apple", "AI2",
      "Databricks", "Snowflake", "IBM", "xAI", "Reka", "Bloomberg"}


def fig_industrial_shift():
    years = list(range(2022, 2027))
    buckets = {"US labs": [], "Chinese labs": [], "EU / other": []}
    for y in years:
        agg = {k: 0 for k in buckets}
        for key, n in D["company_year"].items():
            c, yy = key.rsplit("|", 1)
            if int(yy) != y:
                continue
            agg["US labs" if c in US else
                "Chinese labs" if c in CN else "EU / other"] += n
        for k in buckets:
            buckets[k].append(agg[k])
    fig, ax = plt.subplots(figsize=(4.1, 2.9))
    x = np.arange(len(years))
    bottom = np.zeros(len(years))
    for (k, v), col in zip(buckets.items(), [C_DEEP, C_CORAL, C_GOLD]):
        v = np.array(v)
        ax.bar(x, v, 0.62, bottom=bottom, label=k, color=col,
               edgecolor="white", linewidth=0.6)
        bottom += v
    for xi, tot in zip(x, bottom):
        ax.text(xi, tot + 0.7, f"{int(tot)}", ha="center", fontsize=7, color="#555")
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years])
    ax.set_ylabel("technical reports in list")
    ax.set_ylim(0, max(bottom) * 1.16)
    ax.legend(loc="upper right")
    ax.set_title("Industrial technical reports: fewer, and\nno longer US-led",
                 loc="left", fontsize=9)
    fig.text(0.0, -0.06, "2026 is a partial year (through July).",
             fontsize=6.2, color="#6B7280")
    save(fig, "industrial_shift")


if __name__ == "__main__":
    print("generating trend charts ...")
    fig_industrial_shift()
    fig_area_shift()
    fig_theme_heatmap()
    fig_theme_momentum()
    fig_trajectory()
    fig_2026_venues()
    print("done.")
