#!/usr/bin/env python
"""Simulator dry run of the replication campaign's ENTIRE analysis. 0 QPU.

Standing rule: validate on a simulator and produce a runtime estimate before any QPU is spent. This
also discharges item 8 of the admissibility gate -- on synthetic data with known latent structure,
the frozen witness must return the expected null AND the expected alternative before it is allowed
near new hardware data.

Four synthetic contexts, each 100,000 shots, run through the *identical* code path a real context
would take: fit a DEM on the calibration split, freeze, then score both endpoints on the evaluation
split.

    NULL-INDEP     independent fitted edges                -> E1 ~ 0,   E2 <= 0
    NULL-HETERO    + a shot-rate mixture                   -> E1 ~ 0,   E2 > 0   (the confounder)
    ALT-PERSIST    + per-ancilla temporal persistence      -> E1 > 0,   E2 > 0
    DEVICE-REPLAY  the real held-out shots, as a positive control

🔴 `NULL-HETERO` is the load-bearing case. It is the reason the campaign cannot report `E2` against
zero: a shot-rate mixture violates the graphlike budget while producing no decoder disagreement at
all. The campaign's `E2` criterion must therefore be stated against a dispersion-matched surrogate,
and this dry run is what fixes the size of that offset.

Usage:  python tools/campaign_dryrun.py
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_campaign_r import load_arm  # noqa: E402
from detectors import build_detectors  # noqa: E402
from frozen_witness import (B_GRID, E1_disagreement_curve, E2_graphlike_budget,  # noqa: E402
                            witness_hash)
from heterogeneity_control import sample_heterogeneous  # noqa: E402
from independence_model import sample_independent  # noqa: E402
from parallel_window import two_window  # noqa: E402
from persistent_noise_model import N_HALF, counts, sample_persistent  # noqa: E402
from weight_model_robustness import fit_weights_v2  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_CTX = 100_000          # shots per synthetic context, as proposed for each real context
N_CAL = 50_000           # calibration split
GAMMA_SHAPE = 5.701
# the round-12 persistent cell, used ONLY to make a synthetic alternative that E1 can see at all
PERSIST = dict(pi1=0.05, L=16, rho=50.0, gamma=1.796)


def score(D_cal, D_ev, tag):
    """Exactly the path a real context takes: fit on the calibration split, freeze, score on eval."""
    w, t, _, _ = fit_weights_v2(D_cal, n_fit=len(D_cal))
    e1 = E1_disagreement_curve(D_ev, w, t, two_window)
    e2 = E2_graphlike_budget(D_ev)
    c = counts(D_ev)
    print(f"  {tag:<15} E1 b1 = {e1[1] * 1e4:7.2f}e-4   "
          f"E2 G_same = {e2['G_same']:+8.3f}   "
          f"viol {e2['n_detectors_violating']:>3}/{e2['n_detectors']}   "
          f"var/mean {c['var_over_mean']:.3f}", flush=True)
    return dict(tag=tag, E1={str(k): v for k, v in e1.items()}, E2=e2, counts=c)


def main():
    t0 = time.time()
    print("CAMPAIGN DRY RUN -- the whole analysis on the simulator, before any QPU.  0 QPU")
    print(f"  frozen witness sha256 = {witness_hash()}\n")

    syn, fin, _ = load_arm(0)
    Ddev = build_detectors(syn, fin)
    w0, t0_, ps, pt = fit_weights_v2(Ddev[:N_HALF], n_fit=N_HALF)

    def split(D):
        return D[:N_CAL], D[N_CAL:]

    out = {}
    print("  synthetic contexts, each fitted and scored exactly as a real context would be:")
    D = sample_independent(ps, pt, N_CTX, np.random.default_rng(101))
    out["NULL-INDEP"] = score(*split(D), "NULL-INDEP")
    D = sample_heterogeneous(ps, pt, N_CTX, np.random.default_rng(102), GAMMA_SHAPE)
    out["NULL-HETERO"] = score(*split(D), "NULL-HETERO")
    D = sample_persistent(ps, pt, N_CTX, np.random.default_rng(103), **PERSIST)
    out["ALT-PERSIST"] = score(*split(D), "ALT-PERSIST")
    out["DEVICE-REPLAY"] = score(Ddev[:N_CAL], Ddev[N_HALF:2 * N_HALF], "DEVICE-REPLAY")

    ni, nh = out["NULL-INDEP"], out["NULL-HETERO"]
    ap, dv = out["ALT-PERSIST"], out["DEVICE-REPLAY"]
    checks = {
        "NULL-INDEP gives E1 ~ 0": ni["E1"]["1"] * 1e4 < 1.0,
        "NULL-INDEP gives E2 <= 0 (it is a theorem)": ni["E2"]["G_same"] <= 0,
        "NULL-HETERO gives E1 ~ 0": nh["E1"]["1"] * 1e4 < 1.0,
        "NULL-HETERO gives E2 > 0 -- the confounder is real and must be offset":
            nh["E2"]["G_same"] > 0,
        "ALT-PERSIST gives E1 > 0": ap["E1"]["1"] * 1e4 > 5.0,
        "DEVICE-REPLAY reproduces both endpoints":
            dv["E1"]["1"] * 1e4 > 20.0 and dv["E2"]["G_same"] > 40.0,
        "the device exceeds the hetero confounder on E2 by >= 2x":
            dv["E2"]["G_same"] >= 2 * nh["E2"]["G_same"],
    }
    print("\n  ADMISSIBILITY (item 8: synthetic recovery before hardware)")
    for k, v in checks.items():
        print(f"    {'PASS' if v else 'FAIL'}  {k}")
    ok = all(checks.values())

    # ------------------------------------------------------------------ the campaign's E2 offset
    print(f"\n  E2 OFFSET FIXED BY THIS RUN: a dispersion-matched shot-rate mixture reaches "
          f"G_same = {nh['E2']['G_same']:+.3f}")
    print(f"    so the campaign's E2 criterion is stated against that surrogate, never against 0.")

    # ------------------------------------------------------------------ cost, from our own model
    n_pubs, shots, rounds = 32, 800_000, 50
    base = 4.766 * n_pubs + 6.02e-5 * shots + 5.263e-6 * shots * rounds
    print(f"\n  COST, from docs/HARDWARE_PLAN.md's measured model")
    print(f"    8 contexts (2 devices x 2 regions x 2 epochs), {shots:,} shots, "
          f"{n_pubs} pubs, {rounds} rounds")
    print(f"    base                                  {base:7.1f} s")
    print(f"    x1.16 measured multi-pub underprediction {base * 1.16:7.1f} s")
    print(f"    x1.20 headroom                        {base * 1.16 * 1.2:7.1f} s")
    print(f"    => {base * 1.16 * 1.2:.0f} QPU-s, {100 * base * 1.16 * 1.2 / 7200:.1f}% of the "
          f"28-day budget")

    path = os.path.join(ROOT, "data", "campaign_dryrun.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(dict(witness_sha256=witness_hash(), n_context=N_CTX, n_cal=N_CAL,
                       contexts=out, checks={k: bool(v) for k, v in checks.items()},
                       admissible=bool(ok),
                       E2_hetero_offset=nh["E2"]["G_same"],
                       cost=dict(n_pubs=n_pubs, shots=shots, rounds=rounds, base_s=base,
                                 corrected_s=base * 1.16, with_headroom_s=base * 1.16 * 1.2,
                                 pct_of_28day_budget=100 * base * 1.16 * 1.2 / 7200)),
                  fh, indent=1)
    os.replace(tmp, path)
    print(f"\n  {'ADMISSIBLE -- the analysis may be carried to hardware' if ok else 'NOT ADMISSIBLE -- do not propose the campaign'}")
    print(f"wrote {path}   ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
