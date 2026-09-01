#!/usr/bin/env python
"""Recompute the E1 bound with a valid 95% error budget, and with circular blocks. 0 QPU.

The pre-registered E1 statistic subtracts a 97.5% Clopper-Pearson upper bound on the fitted model's
rate from a 95% block-bootstrap lower bound on the device's rate. Those two component error
probabilities sum to 7.5%, so the pre-registered difference bound is guaranteed at 92.5%, not at 95%.
The bar it is compared against, eta = 5e-4, is unaffected; what is affected is the level the bound
carries.

This tool recomputes the same difference under three variants, on the same evaluation splits, with
the same decoder and the same frozen widths:

    as-registered   device LCB at 5.0%  minus model UCB at 2.5%   -> level >= 92.5%
    even split      device LCB at 2.5%  minus model UCB at 2.5%   -> level >= 95%
    circular        even split, with CIRCULAR blocks confined within the evaluation pub

The pre-registered numbers are not overwritten. This writes its own artifact, and the paper reports
the as-registered value with its true level beside the corrected one.

It also runs the declared block grid, which includes lengths the campaign's own grid stopped short
of, so the reported minimum is taken over a wider set of block lengths than the campaign used.

Usage:  CAMPAIGN_V_EPOCH=E1 python tools/e1_coverage.py
"""
import glob
import json
import os
import sys
import time

import numpy as np
from scipy.stats import beta as _beta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from block_inference import moving_block_bootstrap  # noqa: E402
from detectors import build_detectors  # noqa: E402
from frozen_witness import E1_disagreement_curve, witness_hash  # noqa: E402
from independence_model import sample_independent  # noqa: E402
from parallel_window import two_window  # noqa: E402
from weight_model_robustness import fit_weights_v2  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPOCH = os.environ.get("CAMPAIGN_V_EPOCH", "E1")
INDIR = os.path.join(ROOT, "data", f"campaign_v_{EPOCH.lower()}")
OUT = os.path.join(ROOT, "data", "e1_coverage.json")
WITNESS_SHA = "d1aed05de62d65b46e2ad18011ef5c7267be99e6d19475c650925f2cd2476ce2"
ETA = 5e-4
GRID = [100, 250, 500, 1000, 2000, 2500, 5000]
N_BOOT = 4000
N_SUR = 200_000
PUB = 25_000


def load_L0():
    out = {}
    for f in sorted(glob.glob(os.path.join(INDIR, "pub*.npz")),
                    key=lambda p: int(os.path.basename(p)[3:-4])):
        z = np.load(f, allow_pickle=True)
        m = json.loads(str(z["meta"]))
        if m["logical_state"] != 0:
            continue
        out.setdefault(m["region"], []).append(build_detectors(z["syn"], z["fin"]))
    return out


def lcb(x, alpha, pub):
    """Minimum over the block grid of the one-sided lower bound at level 1 - alpha."""
    los = []
    for bl in GRID:
        if bl > len(x):
            continue
        arr = np.asarray(moving_block_bootstrap(x, bl, N_BOOT,
                                                np.random.default_rng(5), pub=pub))
        los.append(float(np.percentile(arr, 100 * alpha)))
    return min(los)


def main():
    t0 = time.time()
    if witness_hash() != WITNESS_SHA:
        print("** WITNESS HASH MISMATCH"), sys.exit(1)
    arms = load_L0()
    results = {}
    for region, ds in sorted(arms.items()):
        if len(ds) < 2:
            continue
        fit, ev = ds[0], ds[1]
        w, t, ps, pt = fit_weights_v2(fit, n_fit=len(fit))
        dem = E1_disagreement_curve(
            sample_independent(ps, pt, N_SUR, np.random.default_rng(0)), w, t, two_window)
        for b in (1, 2):
            x = np.asarray(two_window(ev, 25, b, logical_j=0, eps=1e-3, seed=0,
                                      wtab=w, ttab=t)["diverged_repaired"], dtype=float)
            k = int(round(dem[b] * N_SUR))
            ucb975 = float(_beta.ppf(0.975, k + 1, N_SUR - k))
            registered = lcb(x, 0.05, None) - ucb975
            even = lcb(x, 0.025, None) - ucb975
            circ = lcb(x, 0.025, PUB) - ucb975
            key = f"{EPOCH}/{region}/b{b}"
            results[key] = dict(
                device_rate=float(x.mean()), n_events=int(x.sum()), n_shots=int(len(x)),
                model_rate=dem[b], model_ucb_975=ucb975,
                as_registered=registered, as_registered_level=0.925,
                even_split=even, even_split_level=0.95,
                circular_even_split=circ, circular_even_split_level=0.95,
                eta=ETA,
                passes_as_registered=bool(registered > ETA),
                passes_even_split=bool(even > ETA),
                passes_circular=bool(circ > ETA))
            print(f"  {key}: events {int(x.sum())}/{len(x)}  "
                  f"registered {registered * 1e4:+.2f}e-4 (>=92.5%)  "
                  f"even {even * 1e4:+.2f}e-4 (95%)  circular {circ * 1e4:+.2f}e-4 (95%)  "
                  f"bar {ETA * 1e4:.2f}e-4  "
                  f"{'PASS' if circ > ETA else 'FAIL'}")

    prev = {}
    if os.path.exists(OUT):
        with open(OUT) as fh:
            prev = json.load(fh)
    prev.update(results)
    prev["_note"] = ("E1 difference bound under the as-registered 92.5% error budget, under an even "
                     "2.5/2.5 split giving 95%, and under circular within-pub blocks at 95%; "
                     f"block grid {GRID}")
    tmp = OUT + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(prev, fh, indent=1)
    os.replace(tmp, OUT)
    print(f"\nwrote {OUT} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
