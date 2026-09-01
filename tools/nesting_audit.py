#!/usr/bin/env python
"""Round 14 part C: how much information do eight marginal widths actually carry? 0 QPU.

Pre-registered at `a8dc617`, `docs/expected.md` Round 14, falsifier 17.

`ELIMINATION_LADDER.md` section 3 argued that a single amplitude factor explains the residual,
because all eight per-width device/model ratios have overlapping confidence intervals. The eight
widths are computed on the SAME shots. If disagreement is nested -- a shot that disagrees at
`b = 16` also disagrees at every smaller buffer -- then the eight rates are one survival curve, and
"eight intervals agree" is close to one agreement, not eight.

This measures the per-shot pattern directly, on the device and on `M2`.

Usage:  python tools/nesting_audit.py
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_campaign_r import load_arm  # noqa: E402
from detectors import build_detectors  # noqa: E402
from parallel_window import two_window  # noqa: E402
from persistent_noise_model import B_GRID, N_HALF, W1, sample_persistent  # noqa: E402
from weight_model_robustness import fit_weights_v2  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_MODEL = 100_000


def patterns(D, wtab, ttab):
    """The per-shot boolean matrix (shots, len(B_GRID)) of disagreement at each buffer width."""
    cols = []
    for b in B_GRID:
        r = two_window(D, W1, b, logical_j=0, eps=1e-3, seed=0, wtab=wtab, ttab=ttab)
        cols.append(np.asarray(r["diverged_repaired"], dtype=bool))
    return np.stack(cols, axis=1)


def analyse(M, tag):
    n_b = M.shape[1]
    any_dis = M.any(axis=1)
    k = int(any_dis.sum())
    rows = M[any_dis]
    # nested == the pattern is a prefix of Trues followed by Falses, i.e. no re-entry with b
    dec = np.all(rows[:, :-1] >= rows[:, 1:], axis=1)
    n_nested = int(dec.sum())
    frac = n_nested / k if k else float("nan")
    # last width at which a shot still disagrees
    last = np.where(rows.any(axis=1), n_b - 1 - np.argmax(rows[:, ::-1], axis=1), -1)
    hist = {str(B_GRID[i]): int((last == i).sum()) for i in range(n_b)}
    uniq = {tuple(r) for r in rows.tolist()}
    # Jaccard between width columns, over the disagreeing shots
    J = np.zeros((n_b, n_b))
    for i in range(n_b):
        for j in range(n_b):
            inter = int((M[:, i] & M[:, j]).sum())
            union = int((M[:, i] | M[:, j]).sum())
            J[i, j] = inter / union if union else float("nan")
    print(f"\n  {tag}:  {k} shots disagree at some width, of {M.shape[0]:,}")
    print(f"    monotone (no re-entry as b grows): {n_nested}/{k} = {frac:.4f}")
    print(f"    distinct patterns observed: {len(uniq)} of {2 ** n_b} possible")
    print("    largest width at which the shot still disagrees:")
    print("      " + "  ".join(f"b{b}:{hist[str(b)]}" for b in B_GRID))
    print("    Jaccard between width columns:")
    print("        " + " ".join(f"{b:>5}" for b in B_GRID))
    for i, b in enumerate(B_GRID):
        print(f"    {b:>3} " + " ".join(f"{J[i, j]:5.2f}" for j in range(n_b)))
    # A DECOMPOSITION of the pre-registered overlap matrix, not a new test, and it carries no
    # falsifier. Escalating from buffer b to a larger b' is the decision ADaPT and STCG both make.
    # Against the joint decoder as the reference, that move HELPS on shots where b disagrees and
    # b' agrees, and HURTS on shots where b agreed and b' does not. Nesting would make `hurt` zero.
    help_hurt = {}
    for i in range(n_b):
        for j in range(i + 1, n_b):
            small, large = M[:, i], M[:, j]
            help_hurt[f"{B_GRID[i]}->{B_GRID[j]}"] = dict(
                helps=int((small & ~large).sum()), hurts=int((~small & large).sum()),
                both=int((small & large).sum()), n_small=int(small.sum()),
                n_large=int(large.sum()))
    print("    ESCALATION b -> b' against the joint reference (helps / hurts / large-set size):")
    for i in range(n_b):
        row = []
        for j in range(i + 1, n_b):
            e = help_hurt[f"{B_GRID[i]}->{B_GRID[j]}"]
            row.append(f"{B_GRID[j]}:{e['helps']}/{e['hurts']}")
        if row:
            print(f"      from b={B_GRID[i]:>2}  " + "  ".join(row))
    return dict(n_disagree=k, n_monotone=n_nested, frac_monotone=frac,
                n_distinct_patterns=len(uniq), last_width_hist=hist,
                per_width_counts={str(B_GRID[i]): int(M[:, i].sum()) for i in range(n_b)},
                escalation=help_hurt,
                jaccard=[[float(v) for v in row] for row in J])


def main():
    t0 = time.time()
    syn, fin, _ = load_arm(0)
    Ddev = build_detectors(syn, fin)
    w, t, ps, pt = fit_weights_v2(Ddev[:N_HALF], n_fit=N_HALF)

    print("ROUND 14 PART C -- ARE THE EIGHT WIDTHS ONE SURVIVAL CURVE?  0 QPU")
    print("  falsifier 17 fires at >= 95% monotone: the eight marginals are then near-redundant\n")

    dev = analyse(patterns(Ddev[N_HALF:2 * N_HALF], w, t), "device (held-out 50,000)")

    with open(os.path.join(ROOT, "data", "persistent_refine.json")) as fh:
        sel = json.load(fh)["selected"]
    print(f"\n  M2 at the round-12 selection: pi1={sel['pi1']}, L={sel['L']}, rho={sel['rho']}, "
          f"gamma={sel['gamma']:.3f}", flush=True)
    Dm = sample_persistent(ps, pt, N_MODEL, np.random.default_rng(11), pi1=sel["pi1"],
                           L=sel["L"], rho=sel["rho"], gamma=sel["gamma"])
    mod = analyse(patterns(Dm, w, t), f"M2 ({N_MODEL:,} shots)")

    fired = dev["frac_monotone"] >= 0.95
    print("\n" + "=" * 96)
    print(f"  FALSIFIER 17: {'FIRES' if fired else 'does NOT fire'} -- device monotone fraction "
          f"{dev['frac_monotone']:.4f} against the 0.95 bar")
    if fired:
        print("  => the eight widths are one survival curve; ELIMINATION_LADDER.md section 3 must")
        print("     say that its eight agreeing intervals are near-redundant, not eight tests.")
    else:
        print("  => the widths carry independent structure; the section-3 statement stands as")
        print("     written, with the dependence caveat it already carries.")

    out = os.path.join(ROOT, "data", "nesting_audit.json")
    tmp = out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(dict(b_grid=B_GRID, device=dev, model_M2=mod, n_model=N_MODEL,
                       monotone_bar=0.95, falsifier17_fired=bool(fired)), fh, indent=1)
    os.replace(tmp, out)
    print(f"\nwrote {out}   ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
