#!/usr/bin/env python
"""Campaign V epoch E1: score both frozen endpoints in each context. 0 additional QPU.

Pre-registered `docs/expected.md` "Campaign V", amended `b76c17c`. Witness frozen at
`tools/frozen_witness.py`, hash verified before anything is scored.

🔴 DEVIATION FROM THE PRE-REGISTRATION, DECLARED BEFORE ANY RESULT WAS READ.
The pre-registration says 100,000 shots per context, split 50,000 fit / 50,000 evaluation. Only the
**logical-0 arm** is usable, so each context has 50,000 shots and the split is 25,000 / 25,000.

The reason is ledger `B16`, recorded 2026-08-29, not a new finding and not a reaction to these data:
the two logical states of a `d = 9`, 50-round, Z-basis repetition code are **not exchangeable on
this hardware**. On `Y = 1` the bulk detector rate is 2.1x higher and the final readout mean is
0.2175 where it should be ~1.0 -- below the 50% floor any symmetric bit-flip channel can produce, so
the mechanism is amplitude damping (`exp(-T/T1) = 0.2175 => T ~ 1.5 T1`) and the logical-1 memory
experiment is destroyed by relaxation. `B16` already states that the replication "must be read off
the `Y = 0` arm alone". Writing 100,000 into the pre-registration was an oversight in that document,
not a change of plan here. Every number this project has ever reported already uses the `Y = 0` arm.

Consequence: fit and evaluation are **different pubs** rather than halves of one, which is stronger,
not weaker -- no shot is shared and the split falls on a natural collection boundary.

🔴 SECOND CONSEQUENCE, also declared before reading. `G_same`'s reference level cannot be taken from
the 50,000-shot dry run (`+11.891`): `B` and `T` carry different finite-sample bias, so the null
level at 25,000 shots is a different number. Each context's surrogate is therefore generated at
**that context's own shot count**, which is what the pre-registration already required.

Usage:  python tools/analyze_campaign_v.py
"""
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from block_inference import moving_block_bootstrap  # noqa: E402
from detectors import build_detectors  # noqa: E402
from frozen_witness import (B_GRID, E1_disagreement_curve, E2_graphlike_budget,  # noqa: E402
                            witness_hash)
from heterogeneity_control import sample_heterogeneous  # noqa: E402
from independence_model import sample_independent  # noqa: E402
from parallel_window import two_window  # noqa: E402
from persistent_noise_model import counts  # noqa: E402
from weight_model_robustness import fit_weights_v2  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPOCH = os.environ.get("CAMPAIGN_V_EPOCH", "E1")
INDIR = os.path.join(ROOT, "data", f"campaign_v_{EPOCH.lower()}")
WITNESS_SHA = "d1aed05de62d65b46e2ad18011ef5c7267be99e6d19475c650925f2cd2476ce2"
ETA = 5e-4
BLOCKS = [100, 250, 500, 1000, 2500]
N_BOOT = 4000
N_SUR = 200_000
GAMMA_SHAPE = 5.701


def load_L0():
    """Detector arrays per region, logical-0 pubs only, in collection order."""
    out = {}
    for f in sorted(glob.glob(os.path.join(INDIR, "pub*.npz")),
                    key=lambda p: int(os.path.basename(p)[3:-4])):
        z = np.load(f, allow_pickle=True)
        m = json.loads(str(z["meta"]))
        if m["logical_state"] != 0:
            continue
        out.setdefault(m["region"], []).append(build_detectors(z["syn"], z["fin"]))
    return out


def lcb_rate(x, alpha=0.05):
    """One-sided lower bound on the mean of a 0/1 series, widest over the block grid."""
    los = []
    for bl in BLOCKS:
        arr = np.asarray(moving_block_bootstrap(x, bl, N_BOOT, np.random.default_rng(5)))
        los.append(float(np.percentile(arr, 100 * alpha)))
    return min(los)


def main():
    t0 = time.time()
    if witness_hash() != WITNESS_SHA:
        print(f"** WITNESS HASH MISMATCH -- campaign void (kill condition 5)\n"
              f"   expected {WITNESS_SHA}\n   got      {witness_hash()}")
        return 1
    print(f"frozen witness verified: {WITNESS_SHA[:16]}...\n")

    arms = load_L0()
    print(f"CAMPAIGN V EPOCH {EPOCH} -- both frozen endpoints, logical-0 arm (ledger B16)")
    for r, ds in sorted(arms.items()):
        print(f"  {r}: {len(ds)} logical-0 pubs of {len(ds[0]):,} shots")
    print()

    results = {}
    for region, ds in sorted(arms.items()):
        if len(ds) < 2:
            print(f"  {region}: fewer than two logical-0 pubs; cannot split. Skipped.")
            continue
        fit, ev = ds[0], ds[1]
        w, t, ps, pt = fit_weights_v2(fit, n_fit=len(fit))

        # ---------------------------------------------------------------- E1
        dev = E1_disagreement_curve(ev, w, t, two_window)
        dem = E1_disagreement_curve(
            sample_independent(ps, pt, N_SUR, np.random.default_rng(0)), w, t, two_window)
        e1 = {}
        for b in (1, 2):
            x = np.asarray(two_window(ev, 25, b, logical_j=0, eps=1e-3, seed=0,
                                      wtab=w, ttab=t)["diverged_repaired"], dtype=float)
            # conservative: LCB on the device rate minus an upper bound on the DEM's
            k = int(round(dem[b] * N_SUR))
            from scipy.stats import beta as _b
            ucb = float(_b.ppf(0.975, k + 1, N_SUR - k)) if k < N_SUR else 1.0
            lo = lcb_rate(x) - ucb
            e1[b] = dict(device_rate=dev[b], dem_rate=dem[b], dem_ucb=ucb,
                         lcb_difference=lo, passes=bool(lo > ETA))

        # ---------------------------------------------------------------- E2
        g_dev = E2_graphlike_budget(ev)
        sur = [E2_graphlike_budget(
            sample_heterogeneous(ps, pt, len(ev), np.random.default_rng(200 + s), GAMMA_SHAPE))
            for s in range(8)]
        gs = np.array([s["G_same"] for s in sur])
        # one-sided 95% upper bound on the surrogate level, from its own replicate spread
        sur_ucb = float(gs.mean() + 1.895 * gs.std(ddof=1))      # t_7, one-sided 95%
        e2 = dict(G_same_device=g_dev["G_same"],
                  n_violating=g_dev["n_detectors_violating"], n_detectors=g_dev["n_detectors"],
                  max_t_over_omega=g_dev["max_t_over_omega"],
                  surrogate_mean=float(gs.mean()), surrogate_sd=float(gs.std(ddof=1)),
                  surrogate_ucb=sur_ucb,
                  margin=g_dev["G_same"] - sur_ucb,
                  passes=bool(g_dev["G_same"] > sur_ucb))

        c = counts(ev)
        print(f"  === {region}  (fit {len(fit):,} shots, eval {len(ev):,} shots, "
              f"detector rate {ev.mean():.5f}, var/mean {c['var_over_mean']:.3f})")
        print(f"    E1 device:  " + "  ".join(f"b{b}={dev[b] * 1e4:6.2f}" for b in B_GRID))
        print(f"    E1 DEM   :  " + "  ".join(f"b{b}={dem[b] * 1e4:6.2f}" for b in B_GRID))
        for b in (1, 2):
            v = e1[b]
            print(f"    E1 b={b}: device {v['device_rate'] * 1e4:.2f}e-4, DEM "
                  f"{v['dem_rate'] * 1e4:.2f}e-4 (UCB {v['dem_ucb'] * 1e4:.2f}e-4), "
                  f"LCB(diff) = {v['lcb_difference'] * 1e4:+.2f}e-4  vs eta "
                  f"{ETA * 1e4:.2f}e-4  -> {'PASS' if v['passes'] else 'FAIL'}")
        print(f"    E2 G_same device {e2['G_same_device']:+.3f}, "
              f"{e2['n_violating']}/{e2['n_detectors']} detectors, max t/omega "
              f"{e2['max_t_over_omega']:.3f}")
        print(f"    E2 surrogate {e2['surrogate_mean']:+.3f} +- {e2['surrogate_sd']:.3f} "
              f"(UCB {e2['surrogate_ucb']:+.3f}), margin {e2['margin']:+.3f}  -> "
              f"{'PASS' if e2['passes'] else 'FAIL'}", flush=True)
        results[region] = dict(n_fit=len(fit), n_eval=len(ev), counts=c,
                               device_curve={str(k): v for k, v in dev.items()},
                               dem_curve={str(k): v for k, v in dem.items()},
                               E1=e1, E2=e2)

    print("\n" + "=" * 96)
    e1_ok = all(r["E1"][b]["passes"] for r in results.values() for b in (1, 2))
    e2_ok = all(r["E2"]["passes"] for r in results.values())
    print(f"  E1 passes in every context: {e1_ok}")
    print(f"  E2 passes in every context: {e2_ok}")
    if e1_ok and e2_ok:
        print(f"  => EPOCH {EPOCH} REPLICATES BOTH ENDPOINTS in two disjoint regions.")
    else:
        fired = []
        if not e1_ok:
            fired.append("kill condition 1: E1 does not reproduce in every region")
        if not e2_ok:
            fired.append("kill condition 3: E2 does not exceed its dispersion-matched surrogate")
        for f in fired:
            print(f"  => {f}")
        print("     The campaign STOPS here; epoch E2 is not submitted.")

    path = os.path.join(ROOT, "data", f"campaign_v_{EPOCH.lower()}_results.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(dict(witness_sha256=WITNESS_SHA, eta=ETA, n_surrogate=N_SUR,
                       arm="logical-0 only (ledger B16)", contexts=results,
                       E1_all_pass=bool(e1_ok), E2_all_pass=bool(e2_ok),
                       epoch_E2_proceeds=bool(e1_ok and e2_ok)), fh, indent=1)
    os.replace(tmp, path)
    print(f"\nwrote {path}   ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
