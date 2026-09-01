#!/usr/bin/env python
"""Round 9C: `P5` without the i.i.d. assumption. Device shots, 0 QPU.

🔴 WHY. The pre-writing audit's finding 9. Fisher's exact test and Clopper-Pearson coverage both
require i.i.d. evaluation shots. The evaluation half of campaign R's `Y = 0` arm is **pubs 4 and 6**
in collection order, and this project has already measured drift between pubs. So the advertised
"exact 5% familywise" statements and every CP bar on hardware data are not exact over device time.
The tool's own docstring says so; the prose does not.

WHAT THIS FILE DOES, on the same frozen shots and the same frozen decoder:

  1. PER-BLOCK COUNTS. Divergence counts over contiguous blocks in collection order -- pub-level
     (25,000) and 2,000-shot sub-blocks -- with a chi-square homogeneity test. This test is
     legitimate here and `B10` does not forbid it: `B10` forbids reusing ONE shot set across a buffer
     grid, whereas these blocks are DISJOINT shot sets at a fixed `b`.
  2. MOVING-BLOCK BOOTSTRAP. Resample contiguous blocks with replacement to rebuild a record of the
     same length, and take the percentile interval. This assumes stationarity at the block scale and
     nothing at the shot scale, so serial dependence within a block is preserved rather than
     destroyed.
  3. DESIGN EFFECT. `Var(block bootstrap) / Var(binomial)` and the implied EFFECTIVE SAMPLE SIZE
     `n_eff = n / deff`. That is the number that should be quoted next to a rate, not 50,000.
  4. `P5` RESTATED. The separation from each surrogate, using the block interval on the device side
     in place of Clopper-Pearson.

Pre-registered at `c5157fe`, `docs/expected.md` Round 9C, falsifier 5:
*if the block-bootstrap 95% interval for the device rate overlaps the surrogate's upper bound at any
width where `P5` currently claims a quantified separation (`b <= 6`), the separation at that width
does not survive drift and is dropped.*

Usage:  python tools/block_inference.py
"""
import json
import os
import sys
import time

import numpy as np
from scipy.stats import beta, chi2_contingency

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_campaign_r import load_arm  # noqa: E402
from detectors import build_detectors  # noqa: E402
from parallel_window import two_window  # noqa: E402
from weight_model_robustness import fit_weights_v2  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B_GRID = [1, 2, 3, 4, 6, 8, 12, 16]
W1 = 25
N_HALF = 50_000
BLOCK = 2_000                 # sub-block length in shots, ~1/12 of a pub
BLOCK_SWEEP = [100, 250, 500, 1_000, 2_000, 5_000]   # a single length proves nothing
N_BOOT_SWEEP = 1_500
PUB = 25_000
# 40,000 replicates so the alpha = 0.00104 tail has ~42 draws, not ~4. Gate-4
# finding 3: the all-eight-width claim rested on an extreme percentile that was not
# stably estimated.
N_BOOT = 40_000
DELTA = 0.05


def cp_upper(k, n, a):
    return 1.0 if k >= n else float(beta.ppf(1.0 - a, k + 1, n - k))


def moving_block_bootstrap(x, block, n_boot, rng, pub=None):
    """Percentile interval for the mean of a 0/1 series, resampling CONTIGUOUS blocks.

    Serial dependence inside a block is carried through untouched; only the block ORDER is
    resampled.

    🔴 `pub` confines every block WITHIN one pub. The evaluation half is pubs 4 and 6 of campaign R
    and the campaign ALTERNATED logical states, so pub 5 sits between them in device time -- a block
    spanning the boundary splices two non-adjacent stretches and calls the join "contiguous".
    Re-gate finding 5. With `pub` set, blocks are drawn inside a pub and the resample keeps the
    original per-pub shot counts, so the estimator stays unbiased for the pooled mean.
    """
    n = len(x)
    if pub is None:
        nb = int(np.ceil(n / block))
        starts_max = n - block
        means = np.empty(n_boot)
        for i in range(n_boot):
            st = rng.integers(0, starts_max + 1, size=nb)
            idx = (st[:, None] + np.arange(block)[None, :]).ravel()[:n]
            means[i] = x[idx].mean()
        return means

    # 🔴 CIRCULAR blocks within each pub. Non-circular overlapping blocks with uniformly sampled
    # starts underweight the pub's endpoints -- materially at block 5,000 against a 25,000-shot pub
    # -- so the resampled mean is NOT unbiased, though the previous version's docstring said it was.
    # Wrapping the block index modulo the pub length makes every position equally likely and
    # restores unbiasedness exactly. Gate-4 finding 3.
    segs = [(k * pub, min((k + 1) * pub, n)) for k in range(int(np.ceil(n / pub)))]
    means = np.empty(n_boot)
    for i in range(n_boot):
        tot, cnt = 0.0, 0
        for lo, hi in segs:
            m = hi - lo
            blk = min(block, m)
            nb = int(np.ceil(m / blk))
            st = rng.integers(0, m, size=nb)
            idx = lo + ((st[:, None] + np.arange(blk)[None, :]) % m).ravel()[:m]
            tot += x[idx].sum()
            cnt += m
        means[i] = tot / cnt
    return means


def main():
    t0 = time.time()
    syn, fin, _ = load_arm(0)
    D = build_detectors(syn, fin)
    fitsl, evsl = slice(0, N_HALF), slice(N_HALF, 2 * N_HALF)
    w, t, _, _ = fit_weights_v2(D[fitsl], n_fit=N_HALF)
    rng = np.random.default_rng(0)

    with open(os.path.join(ROOT, "data", "heterogeneity_control.json")) as fh:
        het = json.load(fh)
    sur = {r["b"]: r for r in het["rows"]}
    n_sur = het["n_surrogate"]

    # 🔴 ONE 5% simultaneous family over all THREE series (device + two surrogates) at every
    # width, not delta spent separately on each. Gate-3 finding 8.
    alpha = DELTA / (2 * 3 * len(B_GRID))
    print("ROUND 9C -- P5 WITHOUT THE I.I.D. ASSUMPTION.  device, 0 QPU")
    print(f"  campaign R Y=0 evaluation half = pubs 4 and 6, {N_HALF} shots in collection order")
    print(f"  moving-block bootstrap, block = {BLOCK} shots, {N_BOOT} resamples, "
          f"alpha = {alpha:.5f}\n")

    rows = []
    print(f"{'b':>3} {'k':>5} {'rate':>9} | {'CP lo':>9} {'CP hi':>9} | {'block lo':>9} "
          f"{'block hi':>9} | {'deff':>6} {'n_eff':>8} | {'pub chi2 p':>11} {'blk chi2 p':>11}")
    for b in B_GRID:
        r = two_window(D[evsl], W1, b, logical_j=0, eps=1e-3, seed=0, wtab=w, ttab=t)
        x = r["diverged_repaired"].astype(np.float64)
        k, n = int(x.sum()), len(x)
        rate = k / n

        # --- per-block counts
        pubs = x.reshape(-1, PUB).sum(axis=1).astype(int)
        # 🔴 built WITHIN each pub. `reshape(-1, BLOCK)` crossed the boundary because 25,000
        # is not divisible by 2,000, so one block spanned shots 24,000-25,999. Gate-3
        # finding 8.
        blks = np.concatenate([x[o:o + PUB][: (PUB // BLOCK) * BLOCK]
                               .reshape(-1, BLOCK).sum(axis=1)
                               for o in range(0, n, PUB)]).astype(int)
        def homog_p(counts, size):
            tab = np.array([counts, size - np.array(counts)])
            if (tab <= 0).all(axis=0).any() or tab.sum() == 0:
                return float("nan")
            try:
                return float(chi2_contingency(tab)[1])
            except ValueError:
                return float("nan")
        p_pub = homog_p(pubs, PUB)
        p_blk = homog_p(blks, BLOCK)

        # --- moving-block bootstrap, blocks confined WITHIN a pub
        bm = moving_block_bootstrap(x, BLOCK, N_BOOT, rng, pub=PUB)
        lo_b, hi_b = np.percentile(bm, [100 * alpha, 100 * (1 - alpha)])
        var_boot = float(bm.var(ddof=1))
        var_bin = rate * (1 - rate) / n
        deff = var_boot / var_bin if var_bin > 0 else float("nan")
        n_eff = n / deff if deff and deff == deff else float("nan")


        lo_cp = 0.0 if k == 0 else float(beta.ppf(alpha, k, n - k + 1))
        hi_cp = cp_upper(k, n, alpha)

        # --- block-length sensitivity: a single block length proves nothing about dependence
        sweep = {}
        for bl in BLOCK_SWEEP:
            bms = moving_block_bootstrap(x, bl, N_BOOT_SWEEP, np.random.default_rng(7), pub=PUB)
            vb = float(bms.var(ddof=1))
            sweep[str(bl)] = vb / var_bin if var_bin > 0 else float("nan")

        row = dict(b=b, k=k, n=n, rate=rate, cp_lo=lo_cp, cp_hi=hi_cp,
                   design_effect_by_block=sweep,
                   design_effect_max_over_blocks=max(sweep.values()),
                   block_lo=float(lo_b), block_hi=float(hi_b), var_boot=var_boot,
                   var_binomial=var_bin, design_effect=float(deff), n_eff=float(n_eff),
                   pub_counts=pubs.tolist(), pub_homogeneity_p=p_pub,
                   block_homogeneity_p=p_blk)
        rows.append(row)
        print(f"{b:>3} {k:>5} {rate:>9.5f} | {lo_cp:>9.2e} {hi_cp:>9.2e} | {lo_b:>9.2e} "
              f"{hi_b:>9.2e} | {deff:>6.2f} {n_eff:>8.0f} | {p_pub:>11.4f} {p_blk:>11.4f}")

    # ---------------------------------------------------------------- P5 restated
    print("\nP5 RESTATED: device block-bootstrap LOWER bound vs surrogate CP UPPER bound")
    print(f"{'b':>3} {'device block lo':>16} {'flat UB':>10} {'het UB':>10} "
          f"{'separated?':>11}")
    sep = []
    for r in rows:
        b = r["b"]
        ub_f = cp_upper(sur[b]["k_flat"], n_sur, alpha)
        ub_h = cp_upper(sur[b]["k_heterogeneous"], n_sur, alpha)
        ok = r["block_lo"] > max(ub_f, ub_h)
        sep.append(ok)
        r["surrogate_flat_ub"] = ub_f
        r["surrogate_het_ub"] = ub_h
        r["separated_block"] = bool(ok)
        print(f"{b:>3} {r['block_lo']:>16.2e} {ub_f:>10.2e} {ub_h:>10.2e} "
              f"{'YES' if ok else 'no':>11}")

    prev = [1, 2, 3, 4, 6]                     # widths P5 currently claims a quantified separation
    lost = [r["b"] for r in rows if r["b"] in prev and not r["separated_block"]]
    f5 = len(lost) > 0
    print("\n" + "=" * 104)
    print(f"  F5  widths where the previously claimed separation does NOT survive the block "
          f"interval: {lost if lost else 'none'}")
    print(f"      -> {'FIRES -- those widths are dropped' if f5 else 'does NOT fire'}")
    small_b = [r for r in rows if r["b"] <= 6]
    f9 = any(max(r["design_effect_by_block"].values()) > 1.5 for r in small_b)
    print(f"  F9  design effect over block lengths {BLOCK_SWEEP} at b <= 6: max "
          f"{max(max(r['design_effect_by_block'].values()) for r in small_b):.2f}")
    print(f"      -> {'FIRES -- CP is materially optimistic, replace it' if f9 else 'does NOT fire'}")
    print("\n  DESIGN EFFECT BY BLOCK LENGTH (blocks confined within a pub)")
    print("    b  " + "  ".join(f"{bl:>7}" for bl in BLOCK_SWEEP))
    for r in rows:
        print(f"{r['b']:>5}  " + "  ".join(f"{r['design_effect_by_block'][str(bl)]:>7.2f}"
                                           for bl in BLOCK_SWEEP))
    print("\n  PER-PUB COUNTS (the evaluation half is TWO pubs; this test has little power)")
    for r in rows:
        print(f"{r['b']:>5}  {r['pub_counts']}  chi2 p = {r['pub_homogeneity_p']:.4f}")
    print("\n  ** CONDITIONAL ON THESE TWO PUBS. Two clusters cannot establish drift-robustness in")
    print("     general; what is shown is that no SHORT-RANGE variance inflation is detectable.")
    deffs = [r["design_effect"] for r in rows]
    print(f"  design effect over the grid: min {min(deffs):.2f}, median "
          f"{np.median(deffs):.2f}, max {max(deffs):.2f}")

    out = os.path.join(ROOT, "data", "block_inference.json")
    tmp = out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(dict(n_half=N_HALF, block=BLOCK, pub=PUB, n_boot=N_BOOT, delta=DELTA,
                       alpha_per_side=alpha, n_surrogate=n_sur,
                       falsifier_separation_fired=bool(f5), widths_lost=lost,
                       falsifier_design_effect_fired=bool(f9), block_sweep=BLOCK_SWEEP,
                       blocks_confined_within_pub=True,
                       scope="conditional on evaluation pubs 4 and 6; two clusters cannot "
                             "establish drift-robustness in general",
                       rows=rows), fh, indent=1)
    os.replace(tmp, out)
    print(f"\nwrote {out}   ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
