#!/usr/bin/env python
"""Is the L~11 temporal correlation real, or an artifact of shot-level heterogeneity?

Round-2 audit, ledger `B11`: "if the temporal correlations behind L ~ 11 were pooled without
subtracting per-layer means, a smooth round-dependent rate profile — or a good-shot/bad-shot mixture
— reproduces them under conditional independence." This bears on the measurement the whole project
rests on, so it is checked here directly, on cached data, at 0 QPU cost.

Two mechanisms, and they need separating:

  (a) PER-LAYER RATE PROFILE. Does NOT apply to this implementation: the statistic is a Pearson
      correlation computed ACROSS SHOTS for each fixed (round, ancilla) pair, so each pair's own
      means are already subtracted. Verified numerically below anyway rather than argued.

  (b) SHOT-LEVEL HETEROGENEITY — the real threat. If some shots are globally noisier than others,
      then D[:,r,j] and D[:,r+dt,j] are positively correlated across shots even when detectors are
      conditionally independent given the shot's quality. Pearson does NOT remove this.

Nulls:
  NULL-0  independent column shuffle          -> destroys everything; must give ~0 (sanity)
  NULL-1  per-(r,j) Bernoulli resample        -> keeps the layer profile, no heterogeneity, no memory
  NULL-2  shot-count-preserving redistribution -> keeps per-shot total AND the layer profile, but
          destroys temporal memory. 🔴 If NULL-2 reproduces the measured curve, L~11 is an artifact.

Plus a stratified analysis: recompute the correlation within quantiles of per-shot detector count.
Survival within strata is direct evidence the signal is not shot heterogeneity.

Usage:  python tools/correlation_nulls.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detectors import build_detectors, load, temporal_corr  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAGS = [1, 2, 3, 5, 8, 10, 15, 20, 25]


def curve(D, lags=LAGS):
    return [temporal_corr(D, dt) for dt in lags]


def null_shuffle(D, rng):
    """NULL-0: shuffle every detector column independently across shots."""
    out = D.copy()
    s, nr, na = D.shape
    for r in range(nr):
        for j in range(na):
            out[:, r, j] = D[rng.permutation(s), r, j]
    return out


def null_bernoulli(D, rng):
    """NULL-1: independent Bernoulli per (r,j) at the empirical per-(r,j) rate."""
    p = D.mean(axis=0)                       # (nr, na) layer profile preserved
    return (rng.random(D.shape) < p).astype(np.uint8)


def null_preserve_shot_counts(D, rng):
    """NULL-2: keep each shot's TOTAL detector count, redistribute positions independently
    according to the global per-(r,j) rate profile. Preserves shot-level heterogeneity and the
    layer profile; destroys temporal memory."""
    s, nr, na = D.shape
    p = D.mean(axis=0).ravel()
    p = p / p.sum()
    counts = D.reshape(s, -1).sum(axis=1)
    out = np.zeros((s, nr * na), dtype=np.uint8)
    idx = np.arange(nr * na)
    for i in range(s):
        k = int(counts[i])
        if k:
            pos = rng.choice(idx, size=k, replace=False, p=p)
            out[i, pos] = 1
    return out.reshape(s, nr, na)


def main():
    syn, fin, meta = load(os.path.join(ROOT, "data", "da7miljsq5js73bk4vtg.npz"))
    D = build_detectors(syn, fin)
    rng = np.random.default_rng(0)
    counts = D.reshape(D.shape[0], -1).sum(axis=1)
    print(f"device job {meta['job_id']}: {D.shape[0]} shots, {D.shape[1]} layers")
    print(f"per-shot detector count: mean {counts.mean():.1f} sd {counts.std():.1f} "
          f"min {counts.min()} max {counts.max()}")
    poisson_sd = np.sqrt(counts.mean())
    print(f"  Poisson sd would be {poisson_sd:.1f}  -> overdispersion factor "
          f"{counts.std() / poisson_sd:.2f}   (>1 means real shot-level heterogeneity exists)")

    obs = curve(D)
    print(f"\n{'dt':>4} " + " ".join(f"{d:>8}" for d in LAGS))
    print(f"{'OBS':>4} " + " ".join(f"{v:>+8.4f}" for v in obs))
    for name, fn in [("NULL-0", null_shuffle), ("NULL-1", null_bernoulli),
                     ("NULL-2", null_preserve_shot_counts)]:
        c = curve(fn(D, rng))
        print(f"{name:>4} " + " ".join(f"{v:>+8.4f}" for v in c))

    print("\nSTRATIFIED by per-shot detector count (signal surviving within strata is not "
          "explained by shot heterogeneity)")
    qs = np.quantile(counts, [0, 0.25, 0.5, 0.75, 1.0])
    print(f"{'stratum':>18} {'n':>6} " + " ".join(f"{d:>8}" for d in LAGS))
    for lo, hi in zip(qs[:-1], qs[1:]):
        sel = (counts >= lo) & (counts <= hi if hi == qs[-1] else counts < hi)
        if sel.sum() < 500:
            continue
        c = curve(D[sel])
        print(f"{f'[{lo:.0f},{hi:.0f}]':>18} {int(sel.sum()):>6} "
              + " ".join(f"{v:>+8.4f}" for v in c))


if __name__ == "__main__":
    main()
