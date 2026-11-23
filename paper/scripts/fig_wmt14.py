"""Figure: WMT14 German-English. (a) Choice of the soft-decision temperature on newstest2013 (cased BLEU).
(b) Lowercased BLEU on newstest2014 against generator evaluations, next to the decoding methods of the NCM paper.

Our numbers are read from results/cond_soft_confirm_wmt14.json and, for our own minimum-Bayes-risk run, from
results/cond_mbr_wmt14{,_dev}.json. The three points quoted from the NCM paper (arXiv 2610.14162; lowercased BLEU) are
typed in below.
"""
import style
from style import BLUE, GOLD, GOLD_DARK, GOLD_DARK_EDGE, GOLD_EDGE, GOLD_LIGHT, INK, MS, MUTED, NEUTRAL, RING, SPINE, VIOLET, plt

j = style.load("cond_soft_confirm_wmt14.json")
dev, test = j["dev"], j["arms"]

fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 2.8), gridspec_kw={"width_ratios": [1.0, 1.12]})

# ---- (a) temperature on the development set ---------------------------------------------------------------------
ts = [0.25, 0.5, 0.75, 1, 1.25, 1.5]
soft = [dev[f"soft Ts={t:g}"]["bleu"] for t in ts]
single, cfg = dev["single pass"]["bleu"], dev["cfg ramp s=1"]["bleu"]
a.axhline(single, color=NEUTRAL, lw=1.1, ls=(0, (4, 3)), zorder=2)
a.axhline(cfg, color=BLUE, lw=1.1, ls=(0, (1, 2)), zorder=2)
style.curve(a, ts, soft, GOLD_DARK, GOLD_DARK_EDGE)
a.plot([1], [dev["soft Ts=1"]["bleu"]], "o", color=GOLD, ms=MS * 1.6, mec=GOLD_EDGE, mew=RING, zorder=5)
style.label(a, f"chosen: $T_s=1$, {dev['soft Ts=1']['bleu']:.2f}", (1, dev["soft Ts=1"]["bleu"]), -86, 7, color=GOLD_EDGE)
style.label(a, f"{dev['soft Ts=1.5']['bleu']:.1f}", (1.5, dev["soft Ts=1.5"]["bleu"]), -24, -2, color=GOLD_DARK_EDGE)
a.text(0.24, single - 0.25, f"single pass {single:.2f}", color=style.text_color(NEUTRAL), fontsize=style.FS_LAB, va="top")
a.text(0.56, cfg + 0.2, f"guidance, 21 NFE: {cfg:.2f}", color=style.text_color(BLUE), fontsize=style.FS_LAB, va="bottom")
a.set_xlim(0.15, 1.6)
a.set_ylim(15.5, 27.6)
a.set_xticks(ts)
a.set_xticklabels(["0.25", "0.5", "0.75", "1", "1.25", "1.5"])
a.set_xlabel("Soft-decision temperature $T_s$")
a.set_ylabel("BLEU $\\uparrow$")
a.set_title("newstest2013 (development)")

# ---- (b) quality against generator evaluations -------------------------------------------------------------------
NCM_GUIDANCE = 23.9   # NCM paper, 22 evaluations: beside our own guidance run, so the two share one label
ours = [("single pass {:.1f}", test["single pass"], NEUTRAL, INK, "o", 1.25, (7, -3)),
        ("guidance: our run {:.1f}, NCM paper " + f"{NCM_GUIDANCE:.1f}", test["cfg ramp s=1"], BLUE, INK, "s", 1.15,
         (-4, -13)),
        ("soft decisions {:.1f}", test["soft Ts=1"], GOLD, GOLD_EDGE, "o", 1.6, (8, -9)),
        ("soft decisions + guidance {:.1f}", test["soft Ts=1 + cfg ramp s=0.5"], GOLD_LIGHT, GOLD_EDGE, "s", 1.45, (8, 2))]
for name, r, color, edge, marker, f, (dx, dy) in ours:
    b.plot([r["nfe"]], [r["bleu_lc"]], marker, color=color, ms=MS * f, mec=edge, mew=RING, zorder=4)
    style.label(b, name.format(r["bleu_lc"]), (r["nfe"], r["bleu_lc"]), dx, dy, color=style.text_color(color))
# our own minimum-Bayes-risk run (12 sampled candidates): temperature and utility chosen on newstest2013
mbr_dev, mbr = style.load("cond_mbr_wmt14_dev.json"), style.load("cond_mbr_wmt14.json")
cfg_, util_ = max(((c, u) for c in mbr_dev["configs"] if c.startswith("sampled|") for u in mbr_dev["configs"][c]["mbr"]),
                  key=lambda cu: mbr_dev["configs"][cu[0]]["mbr"][cu[1]]["k=12"]["bleu"])
m_ = mbr["configs"][cfg_]["mbr"][util_]["k=12"]
b.plot([mbr["configs"][cfg_]["nfe"]], [m_["bleu_lc"]], "^", color=VIOLET, ms=MS * 1.35, mec=INK, mew=RING, zorder=4)
style.label(b, f"MBR, our run {m_['bleu_lc']:.1f}", (mbr["configs"][cfg_]["nfe"], m_["bleu_lc"]), 9, -3,
            color=style.text_color(VIOLET), ha="left")
PAPER = [(None, 22, NCM_GUIDANCE, (0, 0), "left"),
         ("NCM paper:\nMBR 25.2", 132, 25.2, (0, 7), "center"),
         ("NCM paper:\nautoregressive 25.2", 64, 25.2, (6, -22), "right")]
for name, nfe, bleu, (dx, dy), ha in PAPER:
    b.plot([nfe], [bleu], "D", color="white", ms=MS * 1.25, mec=SPINE, mew=RING, zorder=3)
    if name:
        style.label(b, name, (nfe, bleu), dx, dy, size=6.6, color=MUTED, ha=ha, linespacing=1.05)
b.set_xscale("log")
b.set_xlim(8.5, 700)
b.set_ylim(21.8, 27.4)
b.set_xticks([11, 21, 64, 132])
b.set_xticklabels(["11", "21", "64", "132"])
b.minorticks_off()
b.set_xlabel("Evaluations per sentence (log scale)")
b.set_ylabel("Lowercased BLEU $\\uparrow$")
b.set_title("newstest2014 (test)")

fig.tight_layout(pad=0.3, w_pad=1.6, rect=(0, 0.085, 1, 1))
style.sublabels(fig, (a, b), ["(a) Choosing $T_s$", "(b) Quality against compute"])
style.save(fig, "fig_wmt14")
