#!/usr/bin/env python3
"""Modern-minimal line chart: hot themes mentioned in paper titles/descriptions, 2024-2026."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
data = json.loads((HERE / "data.json").read_text())

YEARS = ["2024", "2025", "2026"]
ORDER = [
    "Agent / Agentic",
    "Speculative Decoding",
    "Disaggregation",
    "MoE / Experts",
    "Long Context",
    "KV Cache",
    "RAG",
]
COLORS = {
    "Agent / Agentic":      "#E11D48",  # rose, fastest grower
    "Speculative Decoding": "#D97706",  # amber, surprise spike
    "Disaggregation":       "#2563EB",  # blue
    "MoE / Experts":        "#059669",  # emerald
    "Long Context":         "#7C3AED",  # violet
    "KV Cache":             "#0891B2",  # cyan
    "RAG":                  "#9CA3AF",  # muted
}
MARKERS = {
    "Agent / Agentic": "o", "Speculative Decoding": "s",
    "Disaggregation": "D", "MoE / Experts": "^",
    "Long Context": "v", "KV Cache": "P", "RAG": "X",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "semibold",
    "axes.labelsize": 10, "legend.frameon": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.grid": True, "grid.alpha": 0.18, "axes.axisbelow": True,
    "lines.linewidth": 2.2, "lines.markersize": 7,
})

fig, ax = plt.subplots(figsize=(7.8, 4.4))

x = np.arange(len(YEARS))
end_labels = []
for theme in ORDER:
    vals = [data["theme_year"].get(theme, {}).get(y, 0) for y in YEARS]
    ax.plot(x, vals, color=COLORS[theme], marker=MARKERS[theme],
            label=theme, markeredgecolor="white", markeredgewidth=1.0, zorder=3)
    end_labels.append((vals[-1], theme))

ax.set_xticks(x)
ax.set_xticklabels(YEARS, fontsize=11)
ax.set_xlim(-0.15, len(YEARS) - 1 + 0.6)
ax.set_ylabel("Papers mentioning the theme")
ax.set_title("Three themes broke out in 2026: agents, speculative decoding, and disaggregation",
             loc="left", pad=12, color="#111827")
ax.tick_params(axis="x", length=0)
ax.tick_params(axis="y", length=0, colors="#6B7280")
ax.xaxis.grid(False)

# Side labels at the right edge (instead of legend)
end_labels.sort(key=lambda t: -t[0])
used_y = []
for v, theme in end_labels:
    y = v
    # nudge to avoid overlap
    while any(abs(y - uy) < 0.7 for uy in used_y):
        y += 0.7
    used_y.append(y)
    ax.text(len(YEARS) - 1 + 0.12, y, theme,
            color=COLORS[theme], fontsize=9.2, va="center",
            weight="semibold")

ax.set_ylim(-0.5, 14)
# Annotate dramatic shifts: speculative decoding 1 -> 9
ax.annotate("9x in 2026",
            xy=(2, 9), xytext=(1.35, 11.7),
            fontsize=8.8, color="#D97706", weight="semibold",
            arrowprops=dict(arrowstyle="->", color="#D97706", lw=0.9,
                            connectionstyle="arc3,rad=-0.2"))

fig.text(0.01, -0.02,
         "Theme frequency = number of paper entries (2024-2026) whose title/description matches the theme keywords. "
         "Categories overlap.",
         fontsize=7, color="#6B7280")

out_pdf = HERE / "fig_themes.pdf"
out_png = HERE / "fig_themes.png"
fig.savefig(out_pdf)
fig.savefig(out_png, dpi=300)
print(f"Saved {out_pdf}\nSaved {out_png}")
