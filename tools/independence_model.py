#!/usr/bin/env python
"""The independence baseline in its STRONGEST form, and the held-out counterexample it produces.

Guide section 3.1 item 3 calls this the load-bearing exhibit:

  "a held-out counterexample where the independence certificate declares 'safe' while actual
   window-vs-joint divergence exceeds its bound."

A weak version of that test uses a generic i.i.d. simulator matched only on detector event rate — and
the obvious rebuttal is that a defender of independence would never use such a crude model. So this
module builds the independence model **from the device's own data**: take the per-edge probabilities
fitted from the device by the pairwise estimator, assume those edges fire INDEPENDENTLY, sample
synthetic syndromes from exactly that model, and derive the certificate from them.

That is the independence assumption at its most generous: same code, same layout, same per-layer and
per-ancilla error rates, same decoder, same calibration.

🔴 WHAT MAY NOT BE SAID ABOUT IT, and was said here until 2026-08-29. This docstring claimed "the ONLY
thing removed is dependence between edges" and that "any gap that survives is attributable to
dependence and to nothing else". **Both are withdrawn.** They are forbidden by
`docs/R_PREREGISTRATION.md` §5 and by ledger `B4`: the pairwise `pij` estimator assumes an independent
graph-edge model, so under misspecification it returns *pseudo*-probabilities -- correlated pairs read
as one shared edge, hyperedges alias across several pair edges, drift smears into local estimates. The
surrogate therefore differs from the device in an unknown number of ways, not one, and attribution to
dependence is **not identifiable**.

The supported statement is narrow and is the only one to use: **this particular fitted independent
graph-edge surrogate severely under-predicts held-out divergence on this device condition.** One named
alternative -- i.i.d. shot-level rate heterogeneity -- has been excluded by construction in
`tools/heterogeneity_control.py`; that excludes one model, not the space.

Usage:  python tools/independence_model.py
"""
import json
import os
import sys

import numpy as np
from scipy.stats import beta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detectors import build_detectors, load  # noqa: E402
from parallel_window import two_window  # noqa: E402
from weight_model_robustness import fit_weights_v2  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B_GRID = [1, 2, 3, 4, 6, 8, 12, 16]
DELTA = 0.05
W1 = 25


def cp_upper(k, n, alpha):
    return 1.0 if k >= n else float(beta.ppf(1.0 - alpha, k + 1, n - k))


def sample_independent(p_space, p_time, shots, rng):
    """Sample detector patterns from the fitted edge probabilities, firing every edge independently.

    Edge -> detector map (same graph as everywhere else in this project):
      data qubit d_k at layer r : k=0 -> D[r,0]; 1<=k<=na-1 -> D[r,k-1] and D[r,k]; k=na -> D[r,na-1]
      ancilla j between r, r+1  : D[r,j] and D[r+1,j]
    """
    n_layers, n_anc = p_time.shape
    D = np.zeros((shots, n_layers, n_anc), dtype=np.uint8)
    for r in range(n_layers):
        for k in range(n_anc + 1):
            hit = rng.random(shots) < p_space[r, k]
            if k == 0:
                D[hit, r, 0] ^= 1
            elif k == n_anc:
                D[hit, r, n_anc - 1] ^= 1
            else:
                D[hit, r, k - 1] ^= 1
                D[hit, r, k] ^= 1
        if r + 1 < n_layers:
            for j in range(n_anc):
                hit = rng.random(shots) < p_time[r, j]
                D[hit, r, j] ^= 1
                D[hit, r + 1, j] ^= 1
    return D


def main():
    syn, fin, _ = load(os.path.join(ROOT, "data", "da7miljsq5js73bk4vtg.npz"))
    Ddev = build_detectors(syn, fin)
    fit, ev = slice(0, 5000), slice(5000, 10000)
    n = 5000
    alpha = DELTA / len(B_GRID)

    # calibration from the device's FIT half only; the same weights price every decoder below
    wd, td, ps, pt = fit_weights_v2(Ddev[fit])

    # the independence model built from those same fitted probabilities
    rng = np.random.default_rng(0)
    Dind = sample_independent(ps, pt, 20000, rng)

    print("INDEPENDENCE MODEL BUILT FROM THE DEVICE'S OWN FITTED EDGE PROBABILITIES")
    print(f"  device bulk detector rate      {Ddev[:, 1:-1, :].mean():.4f}")
    print(f"  independence-model rate        {Dind[:, 1:-1, :].mean():.4f}")
    print("  (same per-layer, per-ancilla rates and same calibration; ONLY dependence is removed)\n")

    print(f"{'b':>3} {'IND model /20k':>15} {'IND rate':>10} {'IND cert (CP UB)':>17} "
          f"{'DEVICE held-out':>16} {'exceeds cert?':>14}")
    rows = []
    for b in B_GRID:
        ki = int(two_window(Dind, W1, b, logical_j=0, eps=1e-3, seed=0,
                            wtab=wd, ttab=td)["diverged_repaired"].sum())
        kd = int(two_window(Ddev[ev], W1, b, logical_j=0, eps=1e-3, seed=0,
                            wtab=wd, ttab=td)["diverged_repaired"].sum())
        cert = cp_upper(ki, 20000, alpha)
        rate = kd / n
        flag = "** YES **" if rate > cert else "no"
        print(f"{b:>3} {ki:>15} {ki/20000:>10.5f} {cert:>17.2e} {rate:>16.5f} {flag:>14}")
        rows.append(dict(b=b, ind_count=ki, ind_rate=ki / 20000, ind_cert=cert,
                         device_rate=rate, exceeds=rate > cert,
                         ratio=(rate / cert if cert > 0 else None)))

    print("\nRATIO device / independence-certificate, per buffer width:")
    print("  " + "  ".join(f"b={r['b']}: {r['ratio']:.1f}x" for r in rows if r["ratio"]))

    out = os.path.join(ROOT, "data", "independence_model.json")
    tmp = out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(dict(delta=DELTA, alpha_per_test=alpha, n_device_eval=n,
                       n_independence=20000, rows=rows), fh, indent=1)
    os.replace(tmp, out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
