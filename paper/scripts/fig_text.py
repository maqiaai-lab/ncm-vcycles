"""Figure: generative perplexity after one V-cycle, relative to the equal-compute control, for every released
unconditional model. Read from results/summary.json (key text_ratios, written by experiments/analyze.py)."""
import numpy as np
from matplotlib.lines import Line2D

import style
from style import GOLD, GOLD_EDGE, GOLD_LIGHT, INK, MUTED, RING, SPINE, plt

ratios = style.load("summary.json")["text_ratios"]
rows = [("small", "LM1B-small (97M)"), ("medium", "LM1B-medium (317M)"), ("large", "LM1B-large (725M)"),
        ("tinystories", "TinyStories")]
ARMS = [(6, "o", GOLD, 6.0, 0.15), (8, "D", GOLD_LIGHT, 5.0, -0.15)]   # restart level, marker, fill, size, row offset

# drawn at the width it is printed (0.8 of the text width), so the text is not scaled
fig, ax = plt.subplots(figsize=(5.2, 2.2))
ys = np.arange(len(rows))[::-1]
for y, (key, name) in zip(ys, rows):
    for m, marker, fill, ms, dy in ARMS:
        r = ratios[key].get(f"vcycle|m={m} / no-restrict|m={m}")
        if not r:
            continue
        ax.errorbar([r[0]], [y + dy], xerr=[[r[0] - r[1]], [r[2] - r[0]]], fmt=marker, color=fill, ms=ms, mec=GOLD_EDGE,
                    mew=RING, ecolor=SPINE, elinewidth=1.1, capsize=2.2, capthick=1.1, zorder=4)
        style.label(ax, f"{r[0]:.3f}", (r[2], y + dy), dx=5, dy=-2.6, color=GOLD_EDGE)
ax.axvline(1.0, color="0.55", lw=1.1, zorder=1)
ax.annotate("no effect", (1.0, ys[0] + 0.44), xytext=(-3, 0), textcoords="offset points", ha="right", va="center",
            fontsize=style.FS_LAB, color=MUTED)
ax.set_yticks(ys)
ax.set_yticklabels([n for _, n in rows], color=INK)
ax.tick_params(axis="y", length=0)
ax.set_ylim(ys[-1] - 0.55, ys[0] + 0.62)
ax.set_xlim(0.74, 1.015)
ax.grid(axis="y", visible=False)
ax.set_title("Gen. PPL after one V-cycle, relative to the equal-compute control")
ax.set_xlabel("Gen. PPL ratio (95% CI) $\\downarrow$")
ax.legend(handles=[Line2D([], [], marker="o", color=GOLD, ls="", ms=6.0, mec=GOLD_EDGE, mew=RING,
                          label="restart level $m=6$ (+6 NFE)"),
                   Line2D([], [], marker="D", color=GOLD_LIGHT, ls="", ms=5.0, mec=GOLD_EDGE, mew=RING,
                          label="restart level $m=8$ (+4 NFE)")],
          loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=2, fontsize=7.4, labelcolor=GOLD_EDGE)
fig.tight_layout(pad=0.3)
style.save(fig, "fig_text")
