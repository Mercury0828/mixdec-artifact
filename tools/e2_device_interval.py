#!/usr/bin/env python
"""Put a sampling interval on the DEVICE side of the E2 margin, and audit the surrogate. 0 QPU.

The pre-registered E2 comparison treats the device's `G_same` as an exact number and attaches an
interval only to the surrogate. `G_same` is a plug-in functional of second moments over 25,000 shots,
so it carries sampling variability of its own, and the pre-registered margin does not account for it.

This tool supplies what was missing, without touching the frozen witness or the pre-registered
artifacts:

  1. a one-sided lower bound on the device's `G_same`, from a CIRCULAR block bootstrap confined
     within the evaluation pub, taken as the minimum over a block-length grid;
  2. the margin recomputed with an even error split, 2.5% on the device side and 2.5% on the
     surrogate side, so the difference bound carries 95% rather than the pre-registered budget;
  3. an audit of the surrogate's realised count dispersion against the device's, since the Gamma
     shape is fixed at the value tuned on the retrospective campaign and is NOT re-tuned per
     context. The surrogate is a fixed-shape mixture driven by each context's own fitted edge
     rates, and what it achieves per context is a measurement, not a design guarantee.

Usage:  CAMPAIGN_V_EPOCH=E1 python tools/e2_device_interval.py
"""
import glob
import json
import os
import sys
import time

import numpy as np
from scipy.stats import t as t_dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detectors import build_detectors  # noqa: E402
from frozen_witness import E2_graphlike_budget, witness_hash  # noqa: E402
from heterogeneity_control import sample_heterogeneous  # noqa: E402
from weight_model_robustness import fit_weights_v2  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPOCH = os.environ.get("CAMPAIGN_V_EPOCH", "E1")
INDIR = os.path.join(ROOT, "data", f"campaign_v_{EPOCH.lower()}")
OUT = os.path.join(ROOT, "data", "e2_device_interval.json")
WITNESS_SHA = "d1aed05de62d65b46e2ad18011ef5c7267be99e6d19475c650925f2cd2476ce2"
GAMMA_SHAPE = 5.701
N_SUR = 32              # surrogate draws. The campaign used 8, which forces a t on seven
                        # degrees of freedom; surrogate generation costs no processor time, so the
                        # recomputation is not obliged to be that ascetic
GRID = [250, 1000, 2500]
N_BOOT = 200
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


def circular_block_index(n, block, rng):
    """Indices of a circular moving-block resample of length n, blocks confined to [0, n)."""
    nb = int(np.ceil(n / block))
    st = rng.integers(0, n, size=nb)
    idx = (st[:, None] + np.arange(block)[None, :]) % n
    return idx.ravel()[:n]


def device_lcb(ev, alpha, rng_seed=11):
    """Minimum over the grid of the alpha-quantile of bootstrapped G_same."""
    n = len(ev)
    los, spread = [], {}
    for bl in GRID:
        rng = np.random.default_rng(rng_seed + bl)
        vals = np.empty(N_BOOT)
        for i in range(N_BOOT):
            idx = circular_block_index(n, bl, rng)
            vals[i] = E2_graphlike_budget(ev[idx])["G_same"]
        los.append(float(np.percentile(vals, 100 * alpha)))
        spread[str(bl)] = dict(mean=float(vals.mean()), sd=float(vals.std(ddof=1)),
                               lcb=los[-1])
    return min(los), spread


def main():
    t0 = time.time()
    if witness_hash() != WITNESS_SHA:
        print("** WITNESS HASH MISMATCH")
        return 1
    arms = load_L0()
    results = {}
    for region, ds in sorted(arms.items()):
        if len(ds) < 2:
            continue
        fit, ev = ds[0], ds[1]
        w, t, ps, pt = fit_weights_v2(fit, n_fit=len(fit))
        g_dev = E2_graphlike_budget(ev)["G_same"]

        sur_D = [sample_heterogeneous(ps, pt, len(ev), np.random.default_rng(200 + s), GAMMA_SHAPE)
                 for s in range(N_SUR)]
        gs = np.array([E2_graphlike_budget(d)["G_same"] for d in sur_D])
        # what the campaign computed: the mean plus 1.895 sample standard deviations. That is
        # neither a t bound on the mean, which would use s/sqrt(n), nor a one-draw prediction
        # bound, which would use s*sqrt(1+1/n). It is reported here for comparison only.
        sur_ucb_pre = float(gs.mean() + 1.895 * gs.std(ddof=1))
        # the estimand for the recomputation is the surrogate POPULATION mean, since the surrogate
        # is a pipeline control rather than the mathematical null boundary; a one-sided 97.5% t
        # upper bound on that mean is the matching quantity
        tcrit = float(t_dist.ppf(0.975, N_SUR - 1))
        sur_ucb_even = float(gs.mean() + tcrit * gs.std(ddof=1) / np.sqrt(N_SUR))
        # and the one-draw prediction bound, for a reader who prefers that estimand
        sur_pred_even = float(gs.mean() + tcrit * gs.std(ddof=1) * np.sqrt(1.0 + 1.0 / N_SUR))

        dc = ev.sum(axis=(1, 2)).astype(float)
        sc = np.concatenate([d.sum(axis=(1, 2)).astype(float) for d in sur_D])
        dev_vm = float(dc.var(ddof=1) / dc.mean())
        sur_vm = float(sc.var(ddof=1) / sc.mean())

        lcb975, spread = device_lcb(ev, 0.025)
        key = f"{EPOCH}/{region}"
        results[key] = dict(
            G_same_device=float(g_dev),
            device_lcb_975=lcb975, device_block_spread=spread,
            surrogate_mean=float(gs.mean()), surrogate_sd=float(gs.std(ddof=1)),
            n_surrogate_draws=N_SUR,
            surrogate_ucb_as_registered=sur_ucb_pre,
            surrogate_ucb_even_split=sur_ucb_even,
            surrogate_prediction_bound=sur_pred_even,
            margin_as_registered=float(g_dev) - sur_ucb_pre,
            margin_even_split_95=lcb975 - sur_ucb_even,
            passes_even_split=bool(lcb975 - sur_ucb_even > 0),
            device_count_var_over_mean=dev_vm,
            surrogate_count_var_over_mean=sur_vm,
            dispersion_ratio=sur_vm / dev_vm)
        print(f"  {key}: G_same {g_dev:+.2f}  device LCB(97.5%) {lcb975:+.2f}  "
              f"surrogate UCB {sur_ucb_even:+.2f}  margin@95% {lcb975 - sur_ucb_even:+.2f}  "
              f"{'PASS' if lcb975 - sur_ucb_even > 0 else 'FAIL'}  "
              f"| dispersion device {dev_vm:.3f} surrogate {sur_vm:.3f} "
              f"ratio {sur_vm / dev_vm:.3f}")

    prev = {}
    if os.path.exists(OUT):
        with open(OUT) as fh:
            prev = json.load(fh)
    prev.update(results)
    prev["_note"] = ("device-side circular block bootstrap on G_same, an even 2.5/2.5 error split "
                     "for the margin, and the surrogate's realised count dispersion against the "
                     f"device's; block grid {GRID}, {N_BOOT} replicates per length")
    tmp = OUT + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(prev, fh, indent=1)
    os.replace(tmp, OUT)
    print(f"\nwrote {OUT} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
