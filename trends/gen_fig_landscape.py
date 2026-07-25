#!/usr/bin/env python3
"""Modern-minimal grouped bar chart: paper counts per subarea per year (2024-2026).

Reads trends.json (produced by extract_trends.py).
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
data = json.loads((HERE / "trends.json").read_text())
area = data["area_year"]
TOT = data["year_total"]

ORDER = [
    "LLM Serving", "Pre-training", "Model Co-design", "Agent Systems",
    "LLM-for-Systems", "Multi-Modal Serving", "Edge / On-device",
    "Post-training / RL", "Fault Tolerance", "Multi-Modal Training",
]
SHORT = {
    "LLM Serving": "LLM Serving", "Pre-training": "Pre-training",
    "Model Co-design": "Kernel / Model\nCo-design", "Agent Systems": "Agent Systems",
    "LLM-for-Systems": "AI-for-Systems", "Multi-Modal Serving": "Multi-Modal\nServing",
    "Edge / On-device": "Edge / On-device", "Post-training / RL": "Post-training / RL",
    "Fault Tolerance": "Fault Tolerance", "Multi-Modal Training": "Multi-Modal\nTraining",
}
YEARS = ["2024", "2025", "2026"]
COLORS = {"2024": "#94A3B8", "2025": "#5E81AC", "2026": "#E11D48"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "semibold",
    "axes.labelsize": 10, "legend.fontsize": 9, "legend.frameon": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.grid": True, "grid.alpha": 0.18, "grid.linestyle": "-",
    "axes.axisbelow": True,
})

fig, ax = plt.subplots(figsize=(9.6, 4.8))
fig.subplots_adjust(bottom=0.30, left=0.07, right=0.98, top=0.86)

x = np.arange(len(ORDER))
width = 0.27
for i, year in enumerate(YEARS):
    vals = [area.get(name, {}).get(year, 0) for name in ORDER]
    offset = (i - 1) * width
    bars = ax.bar(x + offset, vals, width * 0.92, label=f"{year}  (n={TOT[year]})",
                  color=COLORS[year], edgecolor="white", linewidth=0.6)
    for bar, v in zip(bars, vals):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.9,
                    str(v), ha="center", va="bottom",
                    fontsize=7.5, color="#374151")

ax.set_xticks(x)
ax.set_xticklabels([SHORT[a] for a in ORDER], rotation=28, ha="right", fontsize=8.5)
ax.set_ylabel("Papers in this list")
ax.set_title("Serving still dominates, but agents, kernels and AI-for-systems are where the growth is",
             loc="left", pad=12, color="#111827")
ax.tick_params(axis="x", length=0)
ax.tick_params(axis="y", length=0, colors="#6B7280")
ax.set_ylim(0, 92)
ax.xaxis.grid(False)
ax.yaxis.grid(True)

leg = ax.legend(loc="upper right", title="Year", title_fontsize=9,
                ncol=3, handlelength=1.2, columnspacing=1.2)
leg.get_title().set_color("#374151")

# Annotation callouts highlighting the two structural shifts
ax.annotate("Kernel / model co-design:\n3 -> 29 papers",
            xy=(2 + width, 31), xytext=(2.5, 58),
            fontsize=8.5, color="#E11D48", weight="semibold",
            arrowprops=dict(arrowstyle="->", color="#E11D48", lw=0.9,
                            connectionstyle="arc3,rad=-0.25"))
ax.annotate("Agents 5x, AI-for-systems 3x\nsince 2024",
            xy=(4 + width, 24), xytext=(5.0, 46),
            fontsize=8.5, color="#E11D48", weight="semibold",
            arrowprops=dict(arrowstyle="->", color="#E11D48", lw=0.9,
                            connectionstyle="arc3,rad=0.25"))

fig.text(0.07, 0.015,
         "Source: LLMSys-PaperList README. 2026 is a partial year and includes the full accepted "
         "lists of MLSys'26, EuroSys'26 and OSDI'26 -- compare shares, not raw counts.",
         fontsize=7.5, color="#6B7280")

fig.savefig(HERE / "fig_landscape.pdf")
fig.savefig(HERE / "fig_landscape.png", dpi=300)
print(f"Saved {HERE / 'fig_landscape.pdf'}")
print(f"Saved {HERE / 'fig_landscape.png'}")
