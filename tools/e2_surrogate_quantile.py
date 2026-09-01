#!/usr/bin/env python
"""The mixture reference as a Monte Carlo quantile, and the budget tested against zero. 0 QPU.

Two separate questions were being answered by one number, and this script separates them.

  E2a  CLASS REALIZABILITY.  Theorem~5 says every independent-event model whose faults flip at most
       two detectors has G_same <= 0. The matching test is the device's own G_same against zero,
       with a sampling interval on the device side. A shot-rate mixture is not a member of that
       class, so its own positive G_same is not a nuisance to be subtracted; it is another way of
       failing the same condition.

  E2b  EXCESS OVER A FIXED RATE-MIXTURE CONTROL.  A separate and narrower question: is the device's
       violation larger than one specific global-heterogeneity control reaches? That is what the
       campaign's surrogate answers, and it is reported as a control rather than as the null.

The campaign's reference was the mean of eight draws plus 1.895 sample standard deviations. That is
neither a t bound on the mean, which divides by sqrt(n), nor a one-draw prediction bound, which
multiplies by sqrt(1+1/n). Surrogate generation costs no processor time, so the small-sample
approximation is not worth arguing about: this draws many replicates and reports the empirical upper
quantile together with its own Monte Carlo uncertainty.

Usage:  python tools/e2_surrogate_quantile.py [--draws 200]
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detectors import build_detectors  # noqa: E402
from frozen_witness import E2_graphlike_budget  # noqa: E402
from heterogeneity_control import sample_heterogeneous  # noqa: E402
from weight_model_robustness import fit_weights_v2  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "e2_surrogate_quantile.json")
GAMMA_SHAPE = 5.701
N_BOOT_Q = 2000          # bootstrap replicates for the Monte Carlo error on the quantile


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=200)
    args = ap.parse_args()
    rng_q = np.random.default_rng(11)

    rows = []
    for epoch in ("E1", "E2"):
        for region, ds in sorted(load_L0(epoch).items()):
            if len(ds) < 2:
                continue
            t0 = time.time()
            fit, ev = ds[0], ds[1]
            w, t, ps, pt = fit_weights_v2(fit, n_fit=len(fit))
            g_dev = E2_graphlike_budget(ev)["G_same"]

            gs = np.array([
                E2_graphlike_budget(sample_heterogeneous(
                    ps, pt, len(ev), np.random.default_rng(5000 + s), GAMMA_SHAPE))["G_same"]
                for s in range(args.draws)])

            q95 = float(np.quantile(gs, 0.95))
            boot = np.array([float(np.quantile(rng_q.choice(gs, gs.size, replace=True), 0.95))
                             for _ in range(N_BOOT_Q)])
            rows.append(dict(
                epoch=epoch, region=region, draws=int(args.draws),
                G_same_device=float(g_dev),
                surrogate_mean=float(gs.mean()), surrogate_sd=float(gs.std(ddof=1)),
                surrogate_min=float(gs.min()), surrogate_max=float(gs.max()),
                surrogate_q95=q95,
                surrogate_q95_mc_se=float(boot.std(ddof=1)),
                surrogate_q95_mc_ci=[float(np.quantile(boot, 0.025)),
                                     float(np.quantile(boot, 0.975))],
                margin_over_q95=float(g_dev - q95),
                seconds=round(time.time() - t0, 1)))
            r = rows[-1]
            print(f"{epoch} {region}: device {g_dev:+7.2f} | surrogate mean {gs.mean():+6.2f} "
                  f"sd {gs.std(ddof=1):.3f} q95 {q95:+6.2f} "
                  f"(MC se {r['surrogate_q95_mc_se']:.3f}) | margin {r['margin_over_q95']:+7.2f} "
                  f"[{r['seconds']:.0f}s]", flush=True)

    with open(OUT, "w") as fh:
        json.dump({"gamma_shape": GAMMA_SHAPE, "draws": args.draws, "rows": rows}, fh, indent=2)
    print("\nwrote", os.path.relpath(OUT, ROOT))


if __name__ == "__main__":
    main()
