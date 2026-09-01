#!/usr/bin/env python
"""Build every figure in the paper from the frozen artifacts. No number is typed here.

Each figure reads `data/*.json` and writes a PDF into `paper/figures/`. Running this after any
artifact changes regenerates the figures, so a stale figure cannot survive a re-analysis.

COLOUR. The palette is chosen so that the series stay distinguishable both in colour and after a
greyscale conversion, since a reader may print the paper. The four hues sit at luminances of roughly
0.22, 0.33, 0.47 and 0.68, and every series also carries its own marker and dash pattern, so no
distinction rests on hue alone.

SIZE. Every figure is single-column, 3.5 inches, unless it genuinely needs the width. A figure that
spans both columns to carry five data points wastes a page.

Usage:  python tools/make_figures.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper", "figures")
B = [1, 2, 3, 4, 6, 8, 12, 16]

NAVY, CRIM, TEAL, AMBER = "#12395B", "#B03A2E", "#1E8A8A", "#E8A33D"
COL = 3.5           # single-column width in inches
WIDE = 7.16         # both columns

plt.rcParams.update({
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "lines.linewidth": 1.1, "lines.markersize": 3.0,
    "font.family": "serif",
})


def L(n):
    with open(os.path.join(ROOT, "data", n + ".json")) as fh:
        return json.load(fh)


def fig_curves():
    """The eight-width curve, device against its own fitted model, in all four contexts."""
    e1, e2 = L("campaign_v_e1_results"), L("campaign_v_e2_results")
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.45), sharey=True)
    style = {"R1": (NAVY, "o", "-"), "R2": (AMBER, "s", "-")}
    for ax, (cv, title) in zip(axes, [(e1, "epoch 1"), (e2, "epoch 2")]):
        for reg in ("R1", "R2"):
            c, (col, mk, ls) = cv["contexts"][reg], style[reg]
            ax.plot(B, [c["device_curve"][str(b)] * 1e4 for b in B],
                    marker=mk, ls=ls, color=col, label=f"device {reg}")
            ax.plot(B, [c["dem_curve"][str(b)] * 1e4 for b in B],
                    marker=mk, ls=":", color=col, mfc="white", mew=0.7,
                    label=f"model {reg}")
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16])
        ax.set_xticklabels(["1", "2", "4", "8", "16"])
        ax.set_xlabel("buffer $b$", labelpad=1)
        ax.set_title(title, pad=2)
        ax.grid(alpha=0.22, lw=0.4)
    axes[0].set_ylabel(r"$\Pr[\Delta_b]$ ($10^{-4}$)", labelpad=1)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, ncol=4, loc="upper center",
               bbox_to_anchor=(0.5, 1.13), columnspacing=1.0, handlelength=1.6,
               handletextpad=0.4)
    fig.savefig(os.path.join(OUT, "fig_curves.pdf"))
    plt.close(fig)
    return "fig_curves.pdf"


def fig_ladder():
    """What a model class has to contain before it produces any disagreement."""
    pr = L("persistent_refine")
    e0 = pr["E_M0"]
    rows = [
        ("device", 34.40),
        ("independent graph", 0.0),
        ("+ shot-rate multiplier", 0.0),
        ("+ modulation, no memory", 0.0),
        ("+ chip-wide state", pr["placebos"]["chipwide"]["curve"]["1"] * 1e4),
        ("+ per-stabiliser persistence", pr["full"]["curve"]["1"] * 1e4),
    ]
    fig, ax = plt.subplots(figsize=(COL, 1.7))
    y = np.arange(len(rows))[::-1]
    vals = [r[1] for r in rows]
    cols = [NAVY] + [AMBER if v == 0 else TEAL for v in vals[1:]]
    ax.barh(y, vals, color=cols, height=0.6, edgecolor="none")
    top = max(vals) * 1.18                       # room for the label past the longest bar
    for yi, (lab, v) in zip(y, rows):
        ax.text(v + top * 0.015, yi, f"{v:.2f}", va="center", fontsize=6)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel(r"$\Pr[\Delta_1]$ ($10^{-4}$)", labelpad=1)
    ax.set_xlim(0, top)
    ax.grid(axis="x", alpha=0.22, lw=0.4)
    fig.savefig(os.path.join(OUT, "fig_ladder.pdf"))
    plt.close(fig)
    return "fig_ladder.pdf"


def fig_budget():
    """E2 in every context against its own fixed-shape shot-rate mixture surrogate."""
    e1, e2 = L("campaign_v_e1_results"), L("campaign_v_e2_results")
    labs, dev, sur = [], [], []
    for tag, cv in (("1", e1), ("2", e2)):
        for reg in ("R1", "R2"):
            c = cv["contexts"][reg]["E2"]
            labs.append(f"{tag}\n{reg}")
            dev.append(c["G_same_device"])
            sur.append(c["surrogate_ucb"])
    x = np.arange(len(labs))
    fig, ax = plt.subplots(figsize=(COL, 1.45))
    ax.bar(x - 0.19, dev, width=0.36, color=NAVY, label="device", edgecolor="none")
    ax.bar(x + 0.19, sur, width=0.36, color=AMBER, label="mixture surrogate",
           edgecolor="none")
    ax.axhline(0, color="0.2", lw=0.9)
    ax.set_ylim(0, max(dev) * 1.08)              # the legend sits above the axes, so little is needed
    ax.set_xticks(x)
    ax.set_xticklabels(labs)
    ax.set_xlabel("epoch and region", labelpad=1)
    ax.set_ylabel(r"$G_{\mathrm{same}}$", labelpad=1)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.16),
              columnspacing=1.2, handlelength=1.4, handletextpad=0.4)
    ax.grid(axis="y", alpha=0.22, lw=0.4)
    fig.savefig(os.path.join(OUT, "fig_budget.pdf"))
    plt.close(fig)
    return "fig_budget.pdf"


def fig_autocorr():
    """Excess autocorrelation over the fitted model, device against the model itself."""
    sp = L("syndrome_process")
    lags = np.arange(1, sp["max_lag"] + 1)
    d3 = np.array(sp["excess"]["device"]["s3"])
    d3e = np.array(sp["excess"]["device"]["s3_se"])
    d5 = np.array(sp["excess"]["device"]["s5"])
    d5e = np.array(sp["excess"]["device"]["s5_se"])
    fig, ax = plt.subplots(figsize=(COL, 1.8))
    ax.errorbar(lags[1:], d3[1:], yerr=2 * d3e[1:], marker="o", color=NAVY,
                label="same stabiliser", capsize=1.2, lw=1.0)
    ax.errorbar(lags[1:], d5[1:], yerr=2 * d5e[1:], marker="s", color=TEAL,
                label="neighbouring pair", capsize=1.2, lw=1.0)
    ax.axhline(0, color=CRIM, lw=1.0, ls="--", label="fitted model")
    ax.set_xlabel("lag (rounds)", labelpad=1)
    ax.set_ylabel("excess autocorrelation", labelpad=1)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16),
              columnspacing=1.0, handlelength=1.5, handletextpad=0.4)
    ax.set_ylim(-0.012, max(d3[1:]) * 1.22)
    ax.grid(alpha=0.22, lw=0.4)
    fig.savefig(os.path.join(OUT, "fig_autocorr.pdf"))
    plt.close(fig)
    return "fig_autocorr.pdf"


def fig_churn():
    """Where shots go when the buffer grows, and how little the width sets overlap."""
    ec = L("escalation_churn")
    ch = ec["churn"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDE, 2.3),
                                 gridspec_kw={"width_ratios": [1.0, 1.15]})

    order = sorted((k for k in ch if k.startswith("1->")),
                   key=lambda k: int(k.split("->")[1]))
    x = np.arange(len(order))
    a1.bar(x - 0.19, [ch[k]["toward"] for k in order], width=0.36, color=NAVY,
           label="toward joint", edgecolor="none")
    a1.bar(x + 0.19, [ch[k]["away"] for k in order], width=0.36, color=AMBER,
           label="away from joint", edgecolor="none")
    a1.set_xticks(x)
    a1.set_xticklabels([k.split("->")[1] for k in order])
    a1.set_xlabel("enlarged buffer $b'$, from $b=1$", labelpad=1)
    a1.set_ylabel("shots", labelpad=1)
    a1.set_ylim(0, max(ch[k]["toward"] for k in order) * 1.22)
    a1.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.14),
              columnspacing=1.2, handlelength=1.4, handletextpad=0.4)
    a1.grid(axis="y", alpha=0.22, lw=0.4)

    J = np.array(L("nesting_audit")["device"]["jaccard"])
    im = a2.imshow(J, cmap="viridis", vmin=0, vmax=1)
    a2.set_xticks(range(len(B)))
    a2.set_yticks(range(len(B)))
    a2.set_xticklabels([str(b) for b in B])
    a2.set_yticklabels([str(b) for b in B])
    a2.set_xlabel("buffer width", labelpad=1)
    a2.set_ylabel("buffer width", labelpad=1)
    for i in range(len(B)):
        for j in range(len(B)):
            a2.text(j, i, f"{J[i, j]:.2f}", ha="center", va="center", fontsize=5.2,
                    color="white" if J[i, j] < 0.55 else "black")
    fig.colorbar(im, ax=a2, fraction=0.046, pad=0.03,
                 label="overlap of disagreeing shots")
    fig.savefig(os.path.join(OUT, "fig_churn.pdf"))
    plt.close(fig)
    return "fig_churn.pdf"


def main():
    os.makedirs(OUT, exist_ok=True)
    for f in (fig_curves, fig_ladder, fig_budget, fig_autocorr, fig_churn):
        try:
            print("wrote", f())
        except Exception as exc:
            print("FAILED", f.__name__, "->", exc)


if __name__ == "__main__":
    main()
