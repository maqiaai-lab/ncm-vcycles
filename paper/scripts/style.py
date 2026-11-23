"""One visual style for every data figure of the paper. It follows the figures of the NCM paper (arXiv 2610.14162):
sans-serif labels, heavy left and bottom axes, a dashed grid, plain centred panel titles, thick curves with dark-edged
markers, labels in the colour of their series, and the same colour roles. The gold family marks the proposed decoding
(the highlighted rows of the tables), neutral grey the released decoding, and the other hues of the same reference
palette the comparison arms.

Colour check (dataviz palette validator, all pairs of grey, blue, violet, aqua and gold on white): colour-vision-
deficiency separation >= 9.1 and normal-vision separation >= 16.3 (OKLab dE x100). Gold and aqua are below 3:1 contrast
with the page, so every mark has a dark edge and every series a label."""
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.interpolate import PchipInterpolator  # noqa: E402

PAPER = Path(__file__).resolve().parents[1]
FIG = PAPER / "figures"
# result files of the study: <repo>/results, or the directory named by NCM_SOFT_RESULTS
RESULTS = Path(os.environ.get("NCM_SOFT_RESULTS") or PAPER.parent / "results")

# gold family: the proposed decoding
GOLD, GOLD_EDGE, GOLD_LIGHT = "#EDA100", "#7A5200", "#FFE3A0"
GOLD_DARK, GOLD_DARK_EDGE = "#A86F00", "#5E3D00"
GOLD_RAMP = ["#F2C14E", "#E09A0E", "#A86F00"]   # ordered series, light to dark
# comparison arms, and the neutral grey of the released decoding
BLUE, VIOLET, AQUA = "#2A78D6", "#4A3AA7", "#1BAF7A"
NEUTRAL = "#6F6F6F"
INK, SPINE, MUTED = "#111111", "#4A4A4A", "#595959"
LW, MS, RING = 1.5, 4.0, 0.9          # curve width, marker size, marker edge
FS, FS_LAB = 8.2, 6.8                 # titles and axis labels; labels inside the panels

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "mathtext.fontset": "dejavusans",
    "font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS, "xtick.labelsize": 8.0, "ytick.labelsize": 8.0,
    "legend.fontsize": FS_LAB, "legend.frameon": False,
    "axes.linewidth": 1.4, "axes.edgecolor": SPINE, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "axes.titleweight": "normal", "axes.titlelocation": "center", "axes.titlepad": 3.5, "axes.labelpad": 3.0,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "axes.grid": True, "grid.color": "0.75", "grid.alpha": 0.42, "grid.linewidth": 1.0, "grid.linestyle": (0, (4, 4)),
    "xtick.color": SPINE, "ytick.color": SPINE, "xtick.major.width": 1.1, "ytick.major.width": 1.1,
    "xtick.major.size": 3.2, "ytick.major.size": 3.2,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})

_GOLD_TEXT = {GOLD: GOLD_EDGE, GOLD_LIGHT: GOLD_EDGE, GOLD_DARK: GOLD_DARK_EDGE}


def text_color(color, target=4.5):
    """The colour of a series for its text: the same chromaticity, darkened to `target`:1 contrast on white."""
    if color in _GOLD_TEXT:
        return _GOLD_TEXT[color]
    rgb = np.array(matplotlib.colors.to_rgb(color))
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    lum = float(lin @ np.array([0.2126, 0.7152, 0.0722]))
    lin = lin * min(1.0, (1.05 / target - 0.05) / lum)   # contrast with white is 1.05 / (luminance + 0.05)
    return matplotlib.colors.to_hex(np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055))


def load(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIG / f"{name}.png", dpi=320, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", FIG / f"{name}.pdf")


def curve(ax, x, y, color, edge=INK, marker="o", ms=MS):
    """A thick curve through the measured points (shape-preserving interpolation) and the points themselves."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    xs = np.linspace(x[0], x[-1], 320)
    ax.plot(xs, PchipInterpolator(x, y)(xs), "-", color=color, lw=LW, solid_capstyle="round", zorder=3)
    ax.plot(x, y, marker, color=color, ms=ms, mec=edge, mew=RING, ls="none", zorder=4)


def label(ax, text, xy, dx=0, dy=0, **kw):
    ax.annotate(text, xy, xytext=(dx, dy), textcoords="offset points", fontsize=kw.pop("size", FS_LAB),
                color=kw.pop("color", INK), zorder=6, bbox=dict(facecolor="white", edgecolor="none", pad=0.4, alpha=0.9),
                **kw)


def sublabels(fig, axes, texts, y=0.012):
    """Panel labels "(a) ...", centred under each panel. Call after the layout is final."""
    for ax, text in zip(axes, texts):
        b = ax.get_position()
        fig.text((b.x0 + b.x1) / 2, y, text, ha="center", va="bottom", fontsize=FS, color=INK)
