#!/usr/bin/env python
"""What the ordinary held-out checks say, beside what the two endpoints say. 0 QPU.

A reader deciding what the two endpoints add needs to know which conventional diagnostics the fitted
model already fails, and this script computes them on the same evaluation records in all four
contexts. Device and model are put side by side on:

  detector rate            the fitting target, so agreement is expected
  detector-count mean      follows from those rates
  count variance/mean      the Fano factor, a one-line dispersion check
  pair-moment residual     the largest standardised residual over the same-stabiliser pair rates
  lag-2 autocorrelation    same-stabiliser, the first lag the detector construction does not force
  Pr[Delta_1]              the E1 observable
  G_same                   the E2 statistic

The model column is that context's own fitted independent model, sampled at the evaluation split's
shot count.

Usage:  python tools/baseline_diagnostics.py [--model-shots N]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detectors import build_detectors  # noqa: E402
from frozen_witness import E2_graphlike_budget  # noqa: E402
from independence_model import sample_independent  # noqa: E402
from parallel_window import two_window  # noqa: E402
from weight_model_robustness import fit_weights_v2  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "baseline_diagnostics.json")
W1 = 25


def load_L0(epoch):
    indir = os.path.join(ROOT, "data", "campaign_v_%s" % epoch.lower())
    out = {}
    for f in sorted(glob.glob(os.path.join(indir, "pub*.npz")),
                    key=lambda p: int(os.path.basename(p)[3:-4])):
        z = np.load(f, allow_pickle=True)
        m = json.loads(str(z["meta"]))
        if m["logical_state"] == 0:
            out.setdefault(m["region"], []).append(build_detectors(z["syn"], z["fin"]))
    return out


def same_stab_lag(D, lag):
    """Pooled binary Pearson correlation between detectors on one stabiliser `lag` layers apart."""
    a = D[:, :-lag, :].reshape(-1).astype(np.float64)
    b = D[:, lag:, :].reshape(-1).astype(np.float64)
    sa, sb = a.std(), b.std()
    if sa == 0 or sb == 0:
        return 0.0
    return float(((a * b).mean() - a.mean() * b.mean()) / (sa * sb))


def pair_residual(Ddev, Dmod):
    """Same-stabiliser pair rates, device against model, as two-sample proportion z-scores.

    A pair that never co-fires in either sample carries no information and scores zero; using the
    model rate alone in the denominator would divide by an empirical zero and report thousands.
    """
    n1, L, A = Ddev.shape
    n2 = len(Dmod)
    worst, over3, total = 0.0, 0, 0
    iu = np.triu_indices(L, 1)
    for a in range(A):
        x = Ddev[:, :, a].astype(np.float64)
        y = Dmod[:, :, a].astype(np.float64)
        k1 = (x.T @ x)[iu]
        k2 = (y.T @ y)[iu]
        p1, p2 = k1 / n1, k2 / n2
        ph = (k1 + k2) / (n1 + n2)
        se = np.sqrt(ph * (1 - ph) * (1.0 / n1 + 1.0 / n2))
        z = np.where(se > 0, (p1 - p2) / np.where(se > 0, se, 1.0), 0.0)
        worst = max(worst, float(np.abs(z).max()))
        over3 += int((np.abs(z) > 3).sum())
        total += z.size
    return worst, over3 / total


def stats(D, label):
    cnt = D.sum(axis=(1, 2)).astype(np.float64)
    return {
        label + "_rate": float(D.mean()),
        label + "_count_mean": float(cnt.mean()),
        label + "_fano": float(cnt.var(ddof=1) / cnt.mean()),
        label + "_lag2": same_stab_lag(D, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-shots", type=int, default=0, help="0 = match the evaluation split")
    args = ap.parse_args()

    rows = []
    for epoch in ("E1", "E2"):
        for region, ds in sorted(load_L0(epoch).items()):
            if len(ds) < 2:
                continue
            fit, ev = ds[0], ds[1]
            w, t, ps, pt = fit_weights_v2(fit, n_fit=len(fit))
            nm = args.model_shots or len(ev)
            mod = sample_independent(ps, pt, nm, np.random.default_rng(3))

            row = dict(epoch=epoch, region=region, shots=len(ev), model_shots=nm)
            row.update(stats(ev, "device"))
            row.update(stats(mod, "model"))
            row["pair_residual_max_z"], row["pair_residual_frac_over_3"] = pair_residual(ev, mod)
            row["G_same_device"] = E2_graphlike_budget(ev)["G_same"]
            row["G_same_model"] = E2_graphlike_budget(mod)["G_same"]
            rd = two_window(ev, W1, 1, logical_j=0, eps=1e-3, seed=0, wtab=w, ttab=t)
            rm = two_window(mod, W1, 1, logical_j=0, eps=1e-3, seed=0, wtab=w, ttab=t)
            row["pr_delta1_device"] = float(rd["diverged_repaired"].mean())
            row["pr_delta1_model"] = float(rm["diverged_repaired"].mean())
            rows.append(row)
            print(f"{epoch} {region}: rate {row['device_rate']:.5f}/{row['model_rate']:.5f}  "
                  f"count {row['device_count_mean']:.2f}/{row['model_count_mean']:.2f}  "
                  f"fano {row['device_fano']:.2f}/{row['model_fano']:.2f}  "
                  f"lag2 {row['device_lag2']:+.4f}/{row['model_lag2']:+.4f}  "
                  f"maxz {row['pair_residual_max_z']:.1f} over3 {row['pair_residual_frac_over_3']*100:.1f}%  "
                  f"Pr[D] {row['pr_delta1_device']*1e4:.2f}/{row['pr_delta1_model']*1e4:.2f}  "
                  f"G {row['G_same_device']:+.2f}/{row['G_same_model']:+.2f}", flush=True)

    with open(OUT, "w") as fh:
        json.dump({"W1": W1, "rows": rows}, fh, indent=2)
    print("\nwrote", os.path.relpath(OUT, ROOT))


if __name__ == "__main__":
    main()
