#!/usr/bin/env python
"""The device-layout and many-seam figures, both single-column. 0 QPU.

  fig_regions.pdf  the two disjoint 17-qubit lines with their data and ancilla roles
  fig_seams.pdf    what the seam count does to the flagged fraction, the certified bound and the
                   charged work, from data/multi_seam.json

Both were originally drawn at full width and carried too little information for the space. The
palette matches tools/make_figures.py and stays legible in greyscale.

Usage:  python tools/make_figures_extra.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper", "figures")

NAVY, CRIM, TEAL, AMBER = "#12395B", "#B03A2E", "#1E8A8A", "#E8A33D"
COL = 3.5

R1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 19]
R2 = [16, 23, 22, 21, 36, 41, 42, 43, 44, 45, 37, 25, 26, 27, 28, 29, 30]

plt.rcParams.update({
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "lines.linewidth": 1.1, "lines.markersize": 3.0,
    "font.family": "serif",
})


def regions():
    fig, ax = plt.subplots(figsize=(COL, 1.25))
    for row, (name, qs, col) in enumerate([("$R_1$", R1, NAVY), ("$R_2$", R2, TEAL)]):
        y = 1 - row
        ax.plot([0, len(qs) - 1], [y, y], color="0.75", lw=1.0, zorder=1)
        for i, q in enumerate(qs):
            data = (i % 2 == 0)
            ax.scatter([i], [y], s=108, zorder=2,
                       facecolor="white" if data else col,
                       edgecolor=col, linewidths=0.9,
                       marker="o" if data else "s")
            ax.text(i, y, str(q), ha="center", va="center", fontsize=4.4,
                    color=col if data else "white", zorder=3)
        ax.text(-1.5, y, name, ha="center", va="center", fontsize=7.5)
    ax.scatter([], [], s=44, facecolor="white", edgecolor="0.3", marker="o", label="data")
    ax.scatter([], [], s=44, facecolor="0.3", edgecolor="0.3", marker="s", label="ancilla")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=2, frameon=False,
              handletextpad=0.2, columnspacing=1.2)
    ax.set_xlim(-2.4, len(R1) - 0.5)
    ax.set_ylim(-0.5, 1.5)
    ax.axis("off")
    fig.savefig(os.path.join(OUT, "fig_regions.pdf"))
    plt.close(fig)
    return "fig_regions.pdf"


def seams():
    with open(os.path.join(ROOT, "data", "multi_seam.json")) as fh:
        d = json.load(fh)
    rows = sorted((r for r in d["rows"] if r["regime"] == "device SEAM rate"),
                  key=lambda r: r["S"])
    S = [r["S"] for r in rows]
    flagged = [100 * r["any_seam_flag_rate"] for r in rows]
    cert = [100 * r["cert_ub"] for r in rows]
    local = [r["cost_total_layers"] for r in rows]
    whole = [r["cost_whole_layers"] + r["cost_windows_layers"] for r in rows]
    joint = rows[0]["joint_layers"]

    fig, (a, b) = plt.subplots(1, 2, figsize=(COL, 1.55))
    a.plot(S, flagged, "o-", color=NAVY, label="any seam flagged")
    a.plot(S, cert, "s--", color=AMBER, label=r"bound on $\Pr[\Delta_1]$")
    a.set_ylabel("per cent of shots", labelpad=1)
    a.legend(frameon=False, loc="upper left", handlelength=1.4, handletextpad=0.4,
             borderpad=0.1)

    b.plot(S, local, "o-", color=NAVY, label="local repair")
    b.plot(S, whole, "^--", color=CRIM, label="whole-record")
    b.axhline(joint, color=TEAL, lw=1.0, ls=":", label="joint")
    b.set_ylabel("charged layers/shot", labelpad=1)
    b.legend(frameon=False, loc="upper left", handlelength=1.4, handletextpad=0.4,
             borderpad=0.1)

    for ax in (a, b):
        # These are simulator measurements on a device-calibrated substrate, not hardware. The tint
        # and the tag say so without relying on a reader reaching the caption.
        ax.set_facecolor("#FBF6EC")
        ax.set_xscale("log", base=2)
        ax.set_xticks(S)
        ax.set_xticklabels([str(s) for s in S])
        ax.set_xlabel("seams $S$", labelpad=1)
        ax.grid(alpha=0.22, lw=0.4)
    fig.tight_layout(pad=0.2, w_pad=1.0)
    fig.text(0.5, 1.005, "SIMULATION, device-calibrated substrate", ha="center", va="bottom",
             fontsize=5.6, color=AMBER, weight="bold")
    fig.savefig(os.path.join(OUT, "fig_seams.pdf"))
    plt.close(fig)
    return "fig_seams.pdf"


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for f in (regions, seams):
        print("wrote", f())
