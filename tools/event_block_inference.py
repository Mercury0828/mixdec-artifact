#!/usr/bin/env python
"""Gate-5 fatal 3: dependence-robust inference for `P9`'s and `P12`'s OWN event sequences. 0 QPU.

🔴 WHY. `block_inference.py` established that the *overall* split-vs-joint disagreement indicator
shows no detectable short-range dependence on the evaluation half. The fifth gate pointed out that
this **cannot be transferred** to the much rarer processes `P9` and `P12` actually rest on:

    P9   the UNFLAGGED-disagreement indicator: the shot disagrees with Joint and no seam fired.
         `3 / 50,000` and its "certified bound" 2.24e-4.
    P12  the DISCORDANCE indicator behind the paired McNemar: the gap trigger is wrong where the
         seam residual is right, and vice versa. `46 / 1` and `p = 6.82e-13`.

Those are different, much sparser point processes, and a Clopper-Pearson bound or an exact McNemar
`p` on them assumes i.i.d. shots that this project has never established for them.

WHAT THIS FILE DOES, on exactly the sequences those claims use:

  1. per-pub counts, so the reader can see the dispersion directly;
  2. a CIRCULAR within-pub moving-block bootstrap of each rate, and the design effect;
  3. for `P12`, a **block permutation test**: the sign of each discordant pair is flipped in whole
     blocks rather than shot by shot, which is the exchangeability a dependent sequence supports.
     That p-value replaces the exact McNemar one wherever dependence is a concern.

Usage:  python tools/event_block_inference.py
"""
import json
import os
import sys
import time

import numpy as np
from scipy.stats import beta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_campaign_r import load_arm  # noqa: E402
from block_inference import moving_block_bootstrap  # noqa: E402
from detectors import build_detectors  # noqa: E402
from parallel_window import TBND, WindowGraph, two_window  # noqa: E402
from weight_model_robustness import fit_weights_v2  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W1, B = 25, 1
N_HALF = 50_000
N_EVAL = 20_000
PUB = 25_000
BLOCK_SWEEP = [100, 250, 500, 1_000, 2_000, 5_000]
BLOCK = 2_000
N_BOOT = 40_000
N_PERM = 40_000
DELTA = 0.05


def cp_upper(k, n, a):
    return 1.0 if k >= n else float(beta.ppf(1.0 - a, k + 1, n - k))


def describe(x, name, alpha, pub, rng):
    """Per-pub counts, circular within-pub bootstrap interval, and the design effect sweep."""
    n = len(x)
    k = int(x.sum())
    rate = k / n
    pubs = [int(x[o:o + pub].sum()) for o in range(0, n, pub)]
    var_bin = rate * (1 - rate) / n if 0 < rate < 1 else 0.0
    bm = moving_block_bootstrap(x, BLOCK, N_BOOT, rng, pub=pub)
    lo, hi = np.percentile(bm, [100 * alpha, 100 * (1 - alpha)])
    sweep = {}
    for bl in BLOCK_SWEEP:
        b2 = moving_block_bootstrap(x, bl, 4_000, np.random.default_rng(3), pub=pub)
        sweep[str(bl)] = (float(b2.var(ddof=1)) / var_bin) if var_bin > 0 else float("nan")
    deff = (float(bm.var(ddof=1)) / var_bin) if var_bin > 0 else float("nan")
    return dict(name=name, k=k, n=n, rate=rate, pub_counts=pubs,
                cp_upper=cp_upper(k, n, alpha), block_lo=float(lo), block_hi=float(hi),
                design_effect=float(deff), design_effect_by_block=sweep,
                design_effect_max=max(v for v in sweep.values() if v == v))


def exact_block_permutation_p(a, b, block, origin=0):
    """EXACT two-sided p by enumerating all `2^nb` block sign assignments, when `nb` is small.

    🔴 The Monte-Carlo version returned 1.87e-3, BELOW its own exact floor of 2/1024 = 1.953e-3.
    With ten blocks the permutation distribution has 1,024 points, so enumerate it. `origin` shifts
    the block grid, because the sixth gate showed the p moves from 0.001953 to 0.003906 when it does.
    """
    import itertools
    d = a.astype(np.int64) - b.astype(np.int64)
    n = len(d)
    idx = ((np.arange(n) + origin) // block).astype(int)
    idx -= idx.min()
    nb = int(idx.max()) + 1
    sums = np.array([d[idx == k].sum() for k in range(nb)], dtype=np.int64)
    obs = abs(int(sums.sum()))
    if nb > 22:
        return None, obs, nb
    hits = 0
    for signs in itertools.product((-1, 1), repeat=nb):
        if abs(int((sums * np.array(signs)).sum())) >= obs:
            hits += 1
    return hits / (2 ** nb), obs, nb


def block_permutation_p(a, b, block, n_perm, rng):
    """Two-sided p for `sum(a) vs sum(b)` on discordant pairs, flipping signs in whole BLOCKS.

    The exact McNemar test flips each discordant pair independently. If the sequence is dependent at
    the block scale that is too generous, so here a single fair coin is drawn per block and applied
    to every discordant pair inside it. Strictly more conservative than the exact test.
    """
    d = a.astype(np.int8) - b.astype(np.int8)          # +1 a-only, -1 b-only, 0 concordant
    n = len(d)
    nb = int(np.ceil(n / block))
    obs = abs(int(d.sum()))
    idx = np.repeat(np.arange(nb), block)[:n]
    cnt = 0
    for _ in range(n_perm):
        sgn = rng.integers(0, 2, size=nb) * 2 - 1
        if abs(int((d * sgn[idx]).sum())) >= obs:
            cnt += 1
    return (cnt + 1) / (n_perm + 1), obs


def main():
    t0 = time.time()
    syn, fin, _ = load_arm(0)
    D = build_detectors(syn, fin)
    n_layers, n_anc = D.shape[1], D.shape[2]
    w, t, _, _ = fit_weights_v2(D[:N_HALF], n_fit=N_HALF)
    Dev = D[N_HALF:N_HALF + N_HALF]
    rng = np.random.default_rng(0)
    alpha = DELTA / 4                    # four event sequences reported here

    print("GATE-5 FATAL 3 -- DEPENDENCE-ROBUST INFERENCE FOR P9 AND P12'S OWN SEQUENCES. 0 QPU")
    print(f"  campaign R Y=0 evaluation half, pubs 4 and 6, alpha = {alpha:.5f}\n")

    # ---------------------------------------------------------------- P9's sequence
    r = two_window(Dev, W1, B, logical_j=0, eps=1e-3, seed=0, wtab=w, ttab=t)
    dv = r["diverged_repaired"].astype(bool)
    flag = r["seam_weight"] > 0
    unflagged = (dv & ~flag).astype(np.float64)
    rows = [describe(unflagged, "P9 unflagged disagreement (b=1)", alpha, PUB, rng),
            describe(dv.astype(np.float64), "all disagreement (b=1)", alpha, PUB, rng)]

    # ---------------------------------------------------------------- P12's sequences
    sub = Dev[:N_EVAL]
    joint = WindowGraph(0, n_layers, n_anc, 0, False, False, wtab=w, ttab=t)
    e1, s2 = min(W1 + B, n_layers), max(W1 - B, 0)
    g1 = WindowGraph(0, e1, n_anc, 0, False, e1 < n_layers, wtab=w, ttab=t)
    g2 = WindowGraph(s2, n_layers, n_anc, 0, s2 > 0, False, wtab=w, ttab=t)
    q1 = WindowGraph(0, e1, n_anc, 0, False, e1 < n_layers, wtab=w, ttab=t, obs_as_detector=True)
    q2 = WindowGraph(s2, n_layers, n_anc, 0, s2 > 0, False, wtab=w, ttab=t, obs_as_detector=True)
    c1 = (g1.kind != TBND) & (g1.glayer < W1)
    c2 = (g2.kind != TBND) & (g2.glayer >= W1)

    print("  recomputing P12's per-shot sequences ...", flush=True)
    par_j = np.empty(N_EVAL, np.uint8)
    par_s = np.empty(N_EVAL, np.uint8)
    seam = np.empty(N_EVAL, np.int32)
    gap = np.empty(N_EVAL)
    for s in range(N_EVAL):
        par_j[s] = joint.logical[joint.decode(sub[s]).astype(bool)].sum() & 1
        k1 = g1.decode(sub[s, :e1]) & c1
        k2 = g2.decode(sub[s, s2:]) & c2
        par_s[s] = (g1.logical[k1.astype(bool)].sum() + g2.logical[k2.astype(bool)].sum()) & 1
        st = np.zeros((n_layers, n_anc), dtype=np.uint8)
        st[:e1] ^= g1.boundary_of(k1)
        st[s2:] ^= g2.boundary_of(k2)
        sm = st ^ sub[s]
        seam[s] = int(sm.sum())
        if seam[s]:
            fix = joint.decode(sm)
            par_s[s] ^= int(joint.logical[fix.astype(bool)].sum() & 1)
        gap[s] = min(q1.complementary_gap(sub[s, :e1])[1], q2.complementary_gap(sub[s, s2:])[1])

    with open(os.path.join(ROOT, "data", "resource_frontier.json")) as fh:
        tau = json.load(fh)["gap_tau"]
    fs, fg = seam > 0, gap <= tau
    wrong_seam = np.where(fs, par_j, par_s) != par_j
    wrong_gap = np.where(fg, par_j, par_s) != par_j
    a_only = (wrong_gap & ~wrong_seam)
    b_only = (wrong_seam & ~wrong_gap)
    rows.append(describe(a_only.astype(np.float64), "P12 gap-only-wrong", alpha, PUB, rng))
    rows.append(describe(b_only.astype(np.float64), "P12 seam-only-wrong", alpha, PUB, rng))

    print(f"{'sequence':>34} {'k':>4} {'rate':>9} {'per-pub':>12} {'CP UB':>10} "
          f"{'block hi':>10} {'deff':>6} {'deff max':>9}")
    for x in rows:
        print(f"{x['name']:>34} {x['k']:>4} {x['rate']:>9.5f} {str(x['pub_counts']):>12} "
              f"{x['cp_upper']:>10.2e} {x['block_hi']:>10.2e} {x['design_effect']:>6.2f} "
              f"{x['design_effect_max']:>9.2f}")

    pperm, obs = block_permutation_p(a_only, b_only, BLOCK, N_PERM, np.random.default_rng(5))
    # 🔴 THE BLOCK LENGTH IS AN UNARGUED CHOICE AND IT DECIDES THE ANSWER. Gate-6 finding 3.
    perm_sweep = {}
    for bl in BLOCK_SWEEP:
        pe, _, nb = exact_block_permutation_p(a_only, b_only, bl)
        if pe is None:
            pe, _ = block_permutation_p(a_only, b_only, bl, N_PERM, np.random.default_rng(9))
            perm_sweep[str(bl)] = dict(p=pe, blocks=nb, method="monte-carlo")
        else:
            perm_sweep[str(bl)] = dict(p=pe, blocks=nb, method="exact")
    origin_sens = {}
    for off in (0, BLOCK // 4, BLOCK // 2, 3 * BLOCK // 4):
        pe, _, _ = exact_block_permutation_p(a_only, b_only, BLOCK, origin=off)
        origin_sens[str(off)] = pe
    ps = [e["p"] for e in perm_sweep.values()]
    print("")
    print("  ** BLOCK-LENGTH SENSITIVITY -- the choice decides the answer:")
    print(f"    {'block':>7} {'blocks':>7} {'p':>12}  method")
    for bl in BLOCK_SWEEP:
        e = perm_sweep[str(bl)]
        print(f"    {bl:>7} {e['blocks']:>7} {e['p']:>12.3e}  {e['method']}")
    print(f"    range {min(ps):.1e} to {max(ps):.1e}; the MOST CONSERVATIVE block length gives "
          f"p = {max(ps):.3f}")
    print("    shifting the block grid: "
          + ", ".join(f"offset {k} -> {v:.4f}" for k, v in origin_sens.items()))
    print(f"\n  P12 BLOCK PERMUTATION TEST, {BLOCK}-shot blocks, {N_PERM:,} permutations")
    print(f"    discordant: gap-only {int(a_only.sum())}, seam-only {int(b_only.sum())}, "
          f"observed |difference| {obs}")
    nb_used = int(np.ceil(N_EVAL / BLOCK))
    floor = 2.0 / (2 ** nb_used)
    print(f"    p = {pperm:.3e}   (exact McNemar gave 6.82e-13; this is strictly more "
          f"conservative)")
    print(f"    RESOLUTION FLOOR: {nb_used} blocks give 2^{nb_used} sign assignments, so the "
          f"smallest attainable two-sided p is ~{floor:.1e}.")
    if pperm <= 4 * floor:
        print("    The observed p is AT that floor: the contrast is as significant as a block")
        print("    permutation on this many blocks can show.")

    worst = max(x["design_effect_max"] for x in rows)
    f12 = worst > 1.5
    print("\n" + "=" * 104)
    print(f"  F12  design effect above 1.5 on any of these sequences at any block length?  "
          f"max {worst:.2f} -> "
          f"{'FIRES -- P9/P12 bounds must be replaced by the block versions' if f12 else 'does NOT fire'}")

    out = os.path.join(ROOT, "data", "event_block_inference.json")
    tmp = out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(dict(alpha=alpha, block=BLOCK, block_sweep=BLOCK_SWEEP, n_boot=N_BOOT,
                       n_perm=N_PERM, pub=PUB, gap_tau=tau,
                       falsifier_design_effect_fired=bool(f12),
                       p12_block_permutation_p=pperm,
                       p12_permutation_by_block=perm_sweep,
                       p12_permutation_most_conservative=max(e["p"] for e in perm_sweep.values()),
                       p12_permutation_origin_sensitivity=origin_sens,
                       p12_permutation_blocks=int(np.ceil(N_EVAL / BLOCK)),
                       p12_permutation_p_floor=2.0 / (2 ** int(np.ceil(N_EVAL / BLOCK))),
                       p12_discordant=dict(gap_only=int(a_only.sum()),
                                           seam_only=int(b_only.sum())),
                       scope=("P9 rows: evaluation pubs 4 and 6, 50,000 shots, two "
                              "clusters. P12 rows: the FIRST 20,000 evaluation shots, "
                              "which lie inside pub 4 alone -- ONE partial cluster, "
                              "not two. Gate-6 finding 3."),
                       rows=rows), fh, indent=1)
    os.replace(tmp, out)
    print(f"\nwrote {out}   ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
