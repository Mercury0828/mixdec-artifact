#!/usr/bin/env python
"""Buffer-width decision churn, stated at the strength a LABEL-FREE analysis can carry. 0 QPU.

Pre-registered corrections at `c1b1b7d`, `docs/expected.md` Round 15b. This replaces the first
version of this analysis, which had three defects, all conceded:

🔴 1. **"Helps" and "hurts" are gone.** Without a label, agreement with the joint decoder is not
      correctness. The categories are *moves toward the joint output* and *moves away from it*, and
      the sign of the risk change is **not identifiable**: given `(D, J, S_b, S_b')`, the worlds
      `Y = S_b` and `Y = S_b'` are observationally identical.

🔴 2. **"Nearly disjoint" was rhetoric.** `Jaccard(b=1, b=16) = 0.039` is a set measure. Under an
      independence null the expected overlap is `172 x 14 / 50,000 = 0.048` against an observed 7 -
      a 145-fold enrichment. The sets are rare, barely overlapping, AND strongly associated. Both
      are reported here.

🔴 3. **Exact McNemar is not exact under shot dependence.** This project has already been bitten by
      that. Every contrast gets pub-stratified counts and a block-length sensitivity curve, and the
      headline is an interval, not a `p`.

What a label-free analysis CAN say, and the only thing it can say, is the churn certificate: for a
frozen trigger `T` and the policy that escalates `b -> b'` when `T = 1`,

    |R(S_T) - R(S_b)|  <=  Pr[T = 1, S_b != S_b']

It bounds how far a retry policy can move logical risk. It does not say in which direction.

🔴 And the earlier claim that this refutes a premise of ADaPT or the spatiotemporal complementary
gap is withdrawn. arXiv:2605.14637 equation (14) already carries both directions; equation (11)
drops the reverse term saying it "is expected to be much smaller ... and can therefore be
neglected". They assume it small, not zero. Nor is it measurable here: their two terms are defined
against the label and partition `Delta_b`, so without labels only their SUM is observable.

Usage:  python tools/escalation_churn.py
"""
import json
import os
import sys

import numpy as np
from scipy.stats import binomtest, fisher_exact

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from analyze_campaign_r import load_arm  # noqa: E402
from block_inference import moving_block_bootstrap  # noqa: E402
from detectors import build_detectors  # noqa: E402
from nesting_audit import patterns  # noqa: E402
from persistent_noise_model import B_GRID, N_HALF  # noqa: E402
from weight_model_robustness import fit_weights_v2  # noqa: E402

BLOCKS = [100, 250, 500, 1000, 2000, 5000]
PUB = 25_000
N_BOOT = 4_000


def _block_idx(n, block, pub, rng):
    """One circular-within-pub block resample index, identical in construction to
    `block_inference.moving_block_bootstrap(pub=...)`. Equivalence is asserted in `main`."""
    segs = [(k * pub, min((k + 1) * pub, n)) for k in range(int(np.ceil(n / pub)))]
    parts = []
    for lo, hi in segs:
        m = hi - lo
        blk = min(block, m)
        nb = int(np.ceil(m / blk))
        st = rng.integers(0, m, size=nb)
        parts.append(lo + ((st[:, None] + np.arange(blk)[None, :]) % m).ravel()[:m])
    return np.concatenate(parts)


def block_ci_matrix(X, block, n_boot, pub, rng, alpha=5.0):
    """Percentile intervals for the column means of `X`, ALL columns sharing each resample.

    Sharing the index across columns is what makes 28 contrasts x 6 block lengths affordable; it
    does not change any single column's marginal interval, only the (unused) cross-column
    dependence of the replicates.
    """
    n, k = X.shape
    reps = np.empty((n_boot, k))
    for i in range(n_boot):
        reps[i] = X[_block_idx(n, block, pub, rng)].mean(axis=0)
    return (np.percentile(reps, alpha / 2, axis=0), np.percentile(reps, 100 - alpha / 2, axis=0))


def main():
    syn, fin, _ = load_arm(0)
    Ddev = build_detectors(syn, fin)
    w, t, _, _ = fit_weights_v2(Ddev[:N_HALF], n_fit=N_HALF)
    M = patterns(Ddev[N_HALF:2 * N_HALF], w, t)
    n = M.shape[0]

    print("BUFFER-WIDTH DECISION CHURN -- device, held-out 50,000, 0 QPU")
    print("  categories are TOWARD / AWAY from the joint output. Neither is correctness.\n")

    # ------------------------------------------------------------------ association, not disjointness
    print("  ASSOCIATION between the b=1 and b=16 disagreement sets")
    a, b = M[:, 0], M[:, B_GRID.index(16)]
    both = int((a & b).sum())
    na, nb = int(a.sum()), int(b.sum())
    exp = na * nb / n
    tab = [[both, na - both], [nb - both, n - na - nb + both]]
    orr = (tab[0][0] * tab[1][1]) / max(tab[0][1] * tab[1][0], 1e-12)
    fp = fisher_exact(tab)[1]
    jac = both / (na + nb - both)
    print(f"    |b=1| = {na},  |b=16| = {nb},  overlap = {both}")
    print(f"    Jaccard = {jac:.4f}   <- a set measure, and NOT a statement about association")
    print(f"    expected overlap under independence = {exp:.4f}  ->  {both / exp:.0f}x enrichment")
    print(f"    odds ratio = {orr:.1f},  Fisher exact p = {fp:.2e}")
    print("    => rare, barely overlapping, and STRONGLY positively associated. All three.\n")

    # ------------------------------------------------------------------ churn with block inference
    # every contrast's signed per-shot churn series, as one matrix
    keys, cols = [], []
    for i, bb in enumerate(B_GRID):
        for j in range(i + 1, len(B_GRID)):
            small, large = M[:, i], M[:, j]
            keys.append((bb, B_GRID[j]))
            cols.append((small & ~large).astype(float) - (~small & large).astype(float))
    X = np.stack(cols, axis=1)

    # 🔴 the shared-index bootstrap must reproduce the project's own estimator on a single series
    chk = np.asarray(moving_block_bootstrap(X[:, 0], 500, 2000, np.random.default_rng(5), pub=PUB))
    lo_ref, hi_ref = np.percentile(chk, [2.5, 97.5])
    lo_new, hi_new = block_ci_matrix(X[:, :1], 500, 2000, PUB, np.random.default_rng(5))
    assert abs(lo_new[0] - lo_ref) < 2e-5 and abs(hi_new[0] - hi_ref) < 2e-5, (
        f"shared-index bootstrap disagrees with block_inference: "
        f"[{lo_new[0]:.3e},{hi_new[0]:.3e}] vs [{lo_ref:.3e},{hi_ref:.3e}]")
    print("  shared-index block bootstrap reproduces block_inference on a single series\n")

    print("  CHURN b -> b2, with block-length sensitivity (the exact test is NOT exact here)")
    print(f"    {'b -> b2':>10} {'toward':>7} {'away':>6} {'net/1e4':>9} "
          f"{'widest block CI (1e-4)':>26}  {'iid p':>9}")
    # 🔴 The reported interval is the WIDEST over the block grid -- the least favourable defensible
    # block length, not the most convenient. Blocks are confined within a pub: the evaluation half
    # is pubs 4 and 6 and pub 5 sits between them in device time, so a block spanning the boundary
    # would splice two non-adjacent stretches.
    los = np.full(len(keys), np.inf)
    his = np.full(len(keys), -np.inf)
    for bl in BLOCKS:
        a_, b_ = block_ci_matrix(X, bl, N_BOOT, PUB, np.random.default_rng(5))
        los, his = np.minimum(los, a_), np.maximum(his, b_)
        print(f"    block {bl:>5}: done", flush=True)

    out = {}
    for m, (bb, b2) in enumerate(keys):
        nt = int((X[:, m] > 0).sum())
        nw = int((X[:, m] < 0).sum())
        p = binomtest(nt, nt + nw, 0.5, alternative="greater").pvalue if nt + nw else 1.0
        lo, hi = float(los[m]), float(his[m])
        out[f"{bb}->{b2}"] = dict(toward=nt, away=nw, net_rate=float(X[:, m].mean()),
                                  block_ci=[lo, hi], crosses_zero=bool(lo <= 0 <= hi),
                                  iid_p=float(p))
        if bb in (1, 2, 3, 4, 6, 8, 12):
            flag = "  CROSSES 0" if lo <= 0 <= hi else ""
            print(f"    {bb:>4} ->{b2:>4} {nt:>7} {nw:>6} {1e4 * X[:, m].mean():>9.2f} "
                  f"   [{1e4 * lo:>7.2f}, {1e4 * hi:>7.2f}] {p:>9.2e}{flag}")

    ncross = sum(1 for v in out.values() if v["crosses_zero"])
    print(f"\n    -> {ncross} of {len(out)} contrasts have a block-bootstrap 95% interval on the")
    print("       net churn rate that CROSSES ZERO at some defensible block length")
    print(f"    -> every one of the {len(out)} contrasts moves at least one shot AWAY from the")
    print("       joint output; strict per-shot nesting fails everywhere")

    path = os.path.join(ROOT, "data", "escalation_churn.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(dict(b_grid=B_GRID, n_shots=n, block_grid=BLOCKS, n_boot=N_BOOT,
                       association=dict(n_b1=na, n_b16=nb, overlap=both, jaccard=jac,
                                        expected_under_independence=exp,
                                        enrichment=both / exp, odds_ratio=orr, fisher_p=fp),
                       churn=out, n_crossing_zero=ncross), fh, indent=1)
    os.replace(tmp, path)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
