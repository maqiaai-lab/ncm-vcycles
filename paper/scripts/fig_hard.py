"""Figure: hard Sudoku, released checkpoint, boards solved against mean generator evaluations per board
(3 seeds x 2,000 boards). Read from results/sudoku_stack_hard_nosingles.json and results/sudoku_stack_hard.json."""
import numpy as np

import style
from style import AQUA, BLUE, GOLD, GOLD_EDGE, INK, LW, MS, MUTED, NEUTRAL, VIOLET, plt

SERIES = [("N", "Best-of-$N$ (fresh restarts)", NEUTRAL, INK, "o"),
          ("E", "V-cycle retries (level 5)", BLUE, INK, "o"),
          ("F", "Fresh restarts in random symmetry frames", VIOLET, INK, "s"),
          ("A", "Symmetry-twisted V-cycle", AQUA, INK, "D"),
          ("D", "Soft decisions + symmetry-twisted V-cycle", GOLD, GOLD_EDGE, "^")]
PANELS = [("sudoku_stack_hard_nosingles.json", "Hard Sudoku, NCM alone"),
          ("sudoku_stack_hard.json", "Hard Sudoku, with singles propagation")]


def mean_curve(curves, xmax=20.0):
    x = np.mean([c["nfe"] for c in curves], 0)
    y = 100 * np.mean([c["solved"] for c in curves], 0)
    keep = x <= xmax + 1e-9
    return x[keep], y[keep]


fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.85), sharey=True)
for ax, (fname, title) in zip(axes, PANELS):
    d = style.load(fname)
    for i, (key, lab, color, edge, marker) in enumerate(SERIES):
        x, y = mean_curve(d["curves"][key])
        ax.plot(x, y, "-", color=color, lw=LW, marker=marker, ms=MS * 0.85, markevery=list(range(min(len(x), 9))),
                mec=edge, mew=0.6, label=lab, solid_capstyle="round", zorder=3 + 0.1 * i)
    ax.axhline(86.1, color="0.55", lw=1.1, ls=(0, (3, 2)), zorder=1)
    ax.annotate("single pass, NCM paper (86.1%)", (20, 86.1), xytext=(0, -3.5), textcoords="offset points", ha="right",
                va="top", fontsize=6.6, color=MUTED)
    ax.set_xlim(11.6, 20.2)
    ax.set_xticks(range(12, 21, 2))
    ax.set_xlabel("Mean generator evaluations per board")
    ax.set_title(title)
axes[0].set_ylabel("Hard boards solved (%) $\\uparrow$")
axes[0].set_ylim(84.3, 100)
# every curve of the second panel starts above 92%, which leaves its lower half free for the legend of both panels
leg = axes[1].legend(loc="lower right", bbox_to_anchor=(1.0, 0.13), handlelength=2.0, labelspacing=0.32,
                     borderaxespad=0.0)
for t, (_, _, color, _, _) in zip(leg.get_texts(), SERIES):
    t.set_color(style.text_color(color))
fig.tight_layout(pad=0.3, w_pad=1.0, rect=(0, 0.085, 1, 1))
style.sublabels(fig, axes, ["(a) NCM alone", "(b) With singles propagation"])
style.save(fig, "fig_hard")
