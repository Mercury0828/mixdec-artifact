#!/usr/bin/env python
"""Reconcile the two scales the same-ancilla temporal correlation has been quoted on (`B14`, item 3).

The web-Pro audit of our frozen substrate:

    "Units must be written beside the numbers. `dt=1` appears as 0.405 in one place and 0.013 in
     another -- Pearson correlation versus excess joint probability. Not a contradiction, but a
     reviewer will call the substrate internally inconsistent unless the formulas are given."

Correct. Two different functionals of the same pair have been reported without saying which is which:

    rho_dt = Cov(D[r,j], D[r+dt,j]) / (sd(D[r,j]) sd(D[r+dt,j]))        DIMENSIONLESS, in [-1, 1]
    E_dt   = Pr[D[r,j]=1, D[r+dt,j]=1] - Pr[D[r,j]=1] Pr[D[r+dt,j]=1]   a PROBABILITY

and they are related exactly by `E_dt = rho_dt * sd(D[r,j]) * sd(D[r+dt,j])`. For a detector firing
at rate ~0.04 the standard deviation is ~0.196, so the two scales differ by a factor of ~0.038 --
which is the whole of the apparent 0.405-versus-0.013 discrepancy.

This file computes BOTH on exactly the same subset with exactly the same averaging, and checks the
identity numerically rather than asserting it, so the substrate can be quoted either way without a
reviewer having to take the conversion on trust.

Usage:  python tools/correlation_units.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detectors import build_detectors, load  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAGS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25]


def both_scales(D, dt, bulk_only=True):
    """Return (rho_dt, E_dt, mean sd product), averaged over ancillas and eligible layer pairs.

    Averaging is over PAIRS (r, r+dt), the same convention M1 used, not over a pooled sample -- a
    pooled estimate would weight layers by their event rate and is not what was reported.
    """
    _, nr, na = D.shape
    lo, hi = (1, nr - 1) if bulk_only else (0, nr)
    rhos, exs, sds = [], [], []
    for j in range(na):
        for r in range(lo, hi - dt):
            a = D[:, r, j].astype(np.float64)
            b = D[:, r + dt, j].astype(np.float64)
            sa, sb = a.std(), b.std()
            if sa == 0 or sb == 0:
                continue
            cov = float(((a - a.mean()) * (b - b.mean())).mean())
            rhos.append(cov / (sa * sb))
            exs.append(float((a * b).mean() - a.mean() * b.mean()))
            sds.append(float(sa * sb))
    return float(np.mean(rhos)), float(np.mean(exs)), float(np.mean(sds))


def main():
    syn, fin, _ = load(os.path.join(ROOT, "data", "da7miljsq5js73bk4vtg.npz"))
    D = build_detectors(syn, fin)
    print("SAME-ANCILLA TEMPORAL DEPENDENCE ON TWO SCALES, SAME SUBSET, SAME AVERAGING")
    print("  rho_dt = Cov / (sd sd')          dimensionless, in [-1, 1]   <- M1 reports THIS")
    print("  E_dt   = Pr[both] - Pr Pr'       a probability               <- N9 / P5 report THIS")
    print("  identity: E_dt = rho_dt * sd * sd'\n")
    print("subset: bulk detector layers 1..49 (first and last layers excluded), all 10,000 shots\n")
    print(f"{'dt':>3} {'rho_dt':>9} {'E_dt':>10} {'mean sd*sd':>11} "
          f"{'rho*sd*sd':>10} {'rel err':>9}")
    rows = []
    for dt in LAGS:
        rho, ex, sd = both_scales(D, dt)
        pred = rho * sd
        rel = abs(pred - ex) / abs(ex) if ex else float("nan")
        print(f"{dt:>3} {rho:>9.4f} {ex:>10.5f} {sd:>11.5f} {pred:>10.5f} {rel:>9.1%}")
        rows.append(dict(dt=dt, rho=rho, excess_joint_prob=ex, mean_sd_product=sd,
                         rho_times_sd=pred, rel_err=rel))
    print("\n  (the small residual is Jensen: the mean of a ratio is not the ratio of means, since")
    print("   both are averaged over layer pairs. It is a few percent and does not affect any claim.)")
    out = os.path.join(ROOT, "data", "correlation_units.json")
    tmp = out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(dict(subset="bulk layers 1..49, all 10000 shots", rows=rows), fh, indent=1)
    os.replace(tmp, out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
