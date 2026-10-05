"""The paper's frontier figure, drawn from ``results/`` only.

Two curves per cell: the bond cap (``per_bond`` ladder) and the DP allocation (``dense`` sweep).
The dashed line is the cell's infidelity floor.

    python dp_figure.py            # figures/budget_frontiers.{pdf,png}
    python dp_figure.py --install  # also copy it into paper/Truncation/figures/
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DENSE = RESULTS / "dense"
OUT = HERE / "figures"
#: Where ``--install`` puts the figure. Nothing writes here without that flag.
PAPER = Path(r"C:\Users\edpou\Desktop\BUG\paper\Truncation\figures")

#: Two consecutive rows within this error ratio mean the curve has reached its floor.
FLOOR_BAND = 1.05

#: Panel order.
CELLS = ["four", "six_mid", "six_hard", "six_wide", "pyrazine", "holstein"]
TITLE = {"four": "spin-boson\n$d=(32,16,16,16)$",
         "six_mid": "spin-boson\n$d=(24,16,12,8,6,4)$",
         "six_hard": "spin-boson\n$d=(8,8,8,8,8,8)$",
         "six_wide": "spin-boson\n$d=(64,32,16,8,4,2)$",
         "pyrazine": "pyrazine", "holstein": "Holstein chain"}

STYLE = {"cap": (r"maximum bond dimension $\chi$", "#4c72b0", "o"),
         "dp": (r"bound $P_{\max}$", "#55a868", "D")}


def load(name):
    """``cap`` from the chi ladder, ``dp`` from the dense 12-target sweep."""
    store = json.loads((RESULTS / f"{name}.json").read_text())
    dense = json.loads((DENSE / f"{name}.json").read_text())["arms"]
    return {"cap": store["arms"]["per_bond"], "dp": dense["dp"]}


def floor_of(arms):
    flat = []
    for rows in arms.values():
        ordered = sorted(rows, key=lambda r: r["params"])
        if len(ordered) >= 2 and ordered[-2]["err"] <= FLOOR_BAND * ordered[-1]["err"]:
            flat.append(ordered[-1]["err"])
    return min(flat) if flat else 0.0


def figure(all_arms):
    fig, axes = plt.subplots(2, 3, figsize=(8.5, 5.4))
    for ax, name in zip(axes.ravel(), CELLS):
        arms = all_arms[name]
        for key in ("cap", "dp"):
            label, colour, marker = STYLE[key]
            rows = sorted(arms[key], key=lambda r: r["params"])
            ax.plot([r["params"] for r in rows], [r["err"] for r in rows],
                    marker=marker, color=colour, ms=3.6, lw=1.0, label=label)
        floor = floor_of(arms)
        if floor > 0:
            ax.axhline(floor, ls="--", lw=0.8, color="0.5")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(TITLE[name], fontsize=9)
        ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 3.0), numticks=6))
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda v, _: f"{int(v)}" if v < 1000 else f"{v / 1000:g}k"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(labelsize=7.5)
        ax.grid(True, which="major", lw=0.3, alpha=0.5)
    for ax in axes[1]:
        ax.set_xlabel("stored parameters", fontsize=8.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("infidelity", fontsize=8.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "budget_frontiers.pdf")
    fig.savefig(OUT / "budget_frontiers.png", dpi=160)


def install() -> None:
    """Copy the drawn figure over the paper's."""
    for suffix in (".pdf", ".png"):
        shutil.copyfile(OUT / f"budget_frontiers{suffix}", PAPER / f"budget_frontiers{suffix}")
    print(f"  installed into {PAPER}")


if __name__ == "__main__":
    all_arms = {name: load(name) for name in CELLS}
    figure(all_arms)
    if "--install" in sys.argv[1:]:
        install()
    print(f"  wrote {OUT / 'budget_frontiers.pdf'} and .png")
