"""Figure: where soft decisions act. Gain over the single pass on the report half of the first run, for soft decisions
on the coarse levels only, on all levels but the last, and on all levels, at three temperatures.

Read from results/cond_guide_soft_{wmt14,xsum}.json (half B; paired bootstrap over documents).
"""
import numpy as np

import style
from style import GOLD_RAMP, INK, RING, SPINE, plt

# an ordered series (more and more levels are soft), so one hue from light to dark
SETS = [("coarse", "coarse levels only", GOLD_RAMP[0]), ("not final", "all levels but the last", GOLD_RAMP[1]),
        ("all", "all levels", GOLD_RAMP[2])]
TS = [0.25, 0.5, 1]
PANELS = [("wmt14", "bleu", "WMT14 German–English", "BLEU gain over single pass $\\uparrow$", "levels 0–4 of 11"),
          ("xsum", "rouge_mean", "XSum, mean of ROUGE-1/2/L", "ROUGE gain over single pass $\\uparrow$",
           "levels 0–3 of 9")]

fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.85))
for ax, (task, metric, title, ylabel, coarse) in zip(axes, PANELS):
    arms = style.load(f"cond_guide_soft_{task}.json")["arms"]
    w = 0.25
    for i, (key, name, color) in enumerate(SETS):
        d = [arms[f"soft {key} Ts={t:g}"]["B"]["delta"][metric] for t in TS]
        ci = [arms[f"soft {key} Ts={t:g}"]["B"]["ci"][metric] for t in TS]
        x = np.arange(len(TS)) + (i - 1) * w
        err = [[v - c[0] for v, c in zip(d, ci)], [c[1] - v for v, c in zip(d, ci)]]
        ax.bar(x, d, w * 0.9, color=color, edgecolor=INK, linewidth=RING * 0.7,
               label=name if key != "coarse" else f"coarse levels only ({coarse})",
               error_kw=dict(ecolor=SPINE, lw=0.9, capsize=1.8), yerr=err, zorder=3)
        if key == "all":   # label the main series only; every value is in the appendix table
            for xi, v, c in zip(x, d, ci):
                ax.text(xi, c[1] + (0.07 if task == "wmt14" else 0.03), f"{v:+.1f}" if task == "wmt14" else f"{v:+.2f}",
                        ha="center", va="bottom", fontsize=style.FS_LAB, color=style.text_color(color))
    if ax.get_ylim()[0] < 0 or task == "xsum":
        ax.axhline(0, color=SPINE, lw=1.1, zorder=2)
    ax.set_xticks(np.arange(len(TS)))
    ax.set_xticklabels([f"$T_s={t:g}$" for t in TS])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(*((0, 4.7) if task == "wmt14" else (-0.3, 1.9)))
    leg = ax.legend(loc="upper left", handlelength=1.0, handletextpad=0.5, labelspacing=0.28, borderaxespad=0.2)
    for t, (_, _, color) in zip(leg.get_texts(), SETS):
        t.set_color(style.text_color(color))
fig.tight_layout(pad=0.3, w_pad=1.6, rect=(0, 0.085, 1, 1))
style.sublabels(fig, axes, ["(a) Translation", "(b) Summarization"])
style.save(fig, "fig_levels")
