#!/usr/bin/env python3
"""Modern-minimal heatmap: industrial LLM technical reports by company × year."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
data = json.loads((HERE / "data.json").read_text())

raw = {}
for k, v in data["company_year"].items():
    c, y = k.rsplit("|", 1)
    raw[(c, int(y))] = v

YEARS = [2022, 2023, 2024, 2025, 2026]
companies = {}
for (c, y), n in raw.items():
    companies[c] = companies.get(c, 0) + n
top = sorted(companies.items(), key=lambda x: -x[1])[:14]
LABELS = [c for c, _ in top]

mat = np.zeros((len(LABELS), len(YEARS)), dtype=int)
for i, c in enumerate(LABELS):
    for j, y in enumerate(YEARS):
        mat[i, j] = raw.get((c, y), 0)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "semibold",
    "axes.labelsize": 10,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})

# Custom Modern Minimal colormap: warm sand -> emerald
cmap = LinearSegmentedColormap.from_list(
    "modern_minimal",
    ["#F8FAFC", "#E0EAF2", "#A7C5DD", "#2563EB", "#1E3A8A"]
)

fig, ax = plt.subplots(figsize=(6.4, 4.6))
im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=0, vmax=max(mat.max(), 6))

# Annotations
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        v = mat[i, j]
        if v == 0: continue
        color = "white" if v >= 4 else "#111827"
        ax.text(j, i, str(v), ha="center", va="center",
                fontsize=9.5, color=color, weight="medium")

ax.set_xticks(np.arange(len(YEARS)))
ax.set_xticklabels(YEARS)
ax.set_yticks(np.arange(len(LABELS)))
ax.set_yticklabels(LABELS)
ax.tick_params(top=False, bottom=False, left=False, right=False)
ax.set_xticks(np.arange(len(YEARS) + 1) - 0.5, minor=True)
ax.set_yticks(np.arange(len(LABELS) + 1) - 0.5, minor=True)
ax.grid(which="minor", color="white", linewidth=2)
for s in ax.spines.values():
    s.set_visible(False)

# Year totals at top
totals = mat.sum(axis=0)
for j, t in enumerate(totals):
    ax.text(j, -0.85, f"{t}", ha="center", va="center",
            fontsize=9, color="#1E3A8A", weight="semibold")
ax.text(-0.6, -0.85, "Yr Σ", ha="right", va="center",
        fontsize=8, color="#6B7280", style="italic")

ax.set_title("Industrial LLM technical reports: 2024 was peak—Chinese labs dominate 2025–2026",
             loc="left", pad=24, color="#111827")

# Colorbar
cb = fig.colorbar(im, ax=ax, shrink=0.65, pad=0.02)
cb.outline.set_visible(False)
cb.set_label("Reports per cell", fontsize=8.5, color="#6B7280")
cb.ax.tick_params(labelsize=8, length=0)

fig.text(0.01, -0.02,
         "Top 14 labs by total report count. Annotations show release year per lab. "
         "2026 data through May 2026.",
         fontsize=7, color="#6B7280")

out_pdf = HERE / "fig_industrial.pdf"
out_png = HERE / "fig_industrial.png"
fig.savefig(out_pdf)
fig.savefig(out_png, dpi=300)
print(f"Saved {out_pdf}\nSaved {out_png}")
