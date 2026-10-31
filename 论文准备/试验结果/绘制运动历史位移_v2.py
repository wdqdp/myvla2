"""Compact qpos-history figure. Style: AnyTouch 2, Fig.4; RTC, Figs.5--6.

No-stop annotations have no numerical ordinate.
H100 is a model reference, not a ground-truth displacement target.
"""

import os
from pathlib import Path
import tempfile

_cache = Path(tempfile.gettempdir()) / "tafir_plot_cache"
_cache.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


OUT = Path(__file__).resolve().parent
COLORS = ["#899EB5", "#D8AB72", "#AB94C0", "#41988B"]
LABELS = ["H0", "H30", "H60", "H100 (ref.)"]

# Outer dimension: direction; inner dimension: history window.
PANELS = [
    {
        "title": "(a) Moderately",
        "means": [[None, 93.1, 108.5, 48.4], [19.2, 61.4, 174.4, 53.4]],
        "stds": [[None, None, None, 4.3], [None, None, None, 8.6]],
        "limit": 210,
        "ticks": [0, 50, 100, 150, 200],
    },
    {
        "title": "(b) Slightly",
        "means": [[None, 2.1, 11.9, 17.1], [15.1, 7.4, 25.7, 22.6]],
        "stds": [[None, 1.2, 3.1, 2.5], [13.7, 3.3, 17.1, 3.1]],
        "limit": 55,
        "ticks": [0, 10, 20, 30, 40, 50],
    },
]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.7,
    "axes.edgecolor": "#555555",
    "text.color": "#242424",
    "xtick.color": "#424242",
    "ytick.color": "#424242",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "hatch.linewidth": 0.45,
    "savefig.facecolor": "white",
})


def main():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.65))
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.17, top=0.80, wspace=0.23)
    centers = np.array([0.0, 1.15])
    width = 0.16
    offsets = (np.arange(4) - 1.5) * 0.19

    for ax, spec in zip(axes, PANELS):
        limit = spec["limit"]
        ax.set_axisbelow(True)
        ax.grid(axis="y", linewidth=0.55, color="#E5E5E5")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="both", length=2.5, width=0.6)
        ax.tick_params(axis="x", length=0, pad=6)
        ax.set_xlim(-0.5, 1.65)
        ax.set_ylim(0, limit)
        ax.set_yticks(spec["ticks"])
        ax.set_xticks(centers, ["Left", "Right"])
        ax.set_title(spec["title"], fontsize=10, fontweight="medium", loc="left", pad=9)

        for direction, center in enumerate(centers):
            for history, offset in enumerate(offsets):
                x = center + offset
                mean = spec["means"][direction][history]
                std = spec["stds"][direction][history]
                if mean is None:
                    ax.annotate(
                        "", xy=(x, limit * 0.94), xytext=(x, limit * 0.79),
                        arrowprops={"arrowstyle": "-|>", "lw": 1.05, "color": "#637B97"},
                    )
                    ax.text(x, limit * 0.75, "No stop", ha="center", va="top",
                            fontsize=7, color="#637B97")
                    continue
                ref = history == 3
                ax.bar(
                    x, mean, width=width, color=COLORS[history],
                    edgecolor="#286E64" if ref else "white",
                    linewidth=0.65 if ref else 0.35,
                    hatch="///" if ref else None, zorder=3,
                )
                if std is not None:
                    ax.errorbar(
                        x, mean, yerr=std, fmt="none", color="#363636",
                        elinewidth=0.75, capsize=2.0, capthick=0.75, zorder=4,
                    )
                ax.text(
                    x, mean + (std or 0) + limit * 0.023, f"{mean:g}",
                    ha="center", va="bottom", fontsize=7.1, color="#343434",
                )

    axes[0].set_ylabel("Recovery displacement (mm)", labelpad=5)
    handles = [
        Patch(facecolor=c, edgecolor="#286E64" if i == 3 else "white",
              hatch="///" if i == 3 else None, linewidth=0.6, label=LABELS[i])
        for i, c in enumerate(COLORS)
    ]
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.54, 1.00),
        ncol=4, frameon=False, fontsize=8.5, handlelength=1.4,
        handleheight=0.85, columnspacing=1.7, handletextpad=0.5,
    )
    for extension in ["png", "pdf", "svg"]:
        fig.savefig(OUT / f"运动历史与恢复位移_v2.{extension}", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()
