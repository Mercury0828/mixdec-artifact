#!/usr/bin/env python
"""Round 14 part A: fit `M2` WITHOUT EVER SEEING `Delta_b`, then predict it. 0 QPU.

Pre-registered at `a8dc617`, `docs/expected.md` Round 14, falsifier 15.

THE OBJECTION THIS ANSWERS, stated by a cross-model consultation on 2026-08-30 and correct:

    `M2` was selected by minimising against the same eight-width `Delta_b` curve it is then credited
    with reproducing. Pre-registration shows the gate was not moved and that failed models were
    kept. It does not convert an in-sample selection score into an out-of-sample prediction.

So here the selection objective is `D_syn` -- autocorrelation curve, dispersion, per-ancilla rate
heterogeneity -- computed by `tools/syndrome_stats.py`, which never touches a decoder. `gamma` is
solved against the **fit half's** detector-count mean, not the evaluation half's, so no statistic of
the evaluation half enters the selection at all. The chosen parameters are frozen to disk. Only then
is the eight-width curve simulated and scored.

🔴 If this file ever computes a disagreement curve before `selected` is written, the experiment is
void. The order is enforced by `_SEALED`.

Usage:  python tools/cross_prediction.py
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_campaign_r import load_arm  # noqa: E402
from detectors import build_detectors  # noqa: E402
from heterogeneity_control import sample_heterogeneous  # noqa: E402
from independence_model import sample_independent  # noqa: E402
from persistent_noise_model import (B_GRID, N_HALF, counts, disagreement_curve,  # noqa: E402
                                    endpoint, sample_persistent, solve_gamma)
from syndrome_stats import all_stats, syndrome_distance  # noqa: E402
from weight_model_robustness import fit_weights_v2  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_SEARCH = 20_000
N_FINAL = 100_000
# the SAME grid as round 12's refinement -- no new model class, no widened search
PI1 = [0.03, 0.05, 0.08]
LS = [8, 12, 16, 24, 32]
RHOS = [50.0, 100.0, 200.0, 400.0]
GAMMA_SHAPE = 5.701

_SEALED = {"open": False}


def _guard():
    if not _SEALED["open"]:
        raise RuntimeError("BLIND PHASE: a decoder observable was requested before the "
                           "syndrome-only selection was frozen. The experiment is void.")


def curve(D, w, t):
    _guard()
    return disagreement_curve(D, w, t)


def main():
    t0 = time.time()
    syn, fin, _ = load_arm(0)
    Ddev = build_detectors(syn, fin)
    fit, ev = Ddev[:N_HALF], Ddev[N_HALF:2 * N_HALF]
    w, t, ps, pt = fit_weights_v2(fit, n_fit=N_HALF)

    print("ROUND 14 PART A -- FIT M2 BLIND TO Delta_b, THEN PREDICT IT.  0 QPU")
    print("  selection uses tools/syndrome_stats.py only; gamma is solved against the FIT half\n")

    dev_fit = all_stats(fit)
    print(f"  device fit half:   count mean {dev_fit['s1_count_mean']:.3f}, "
          f"var/mean {dev_fit['s2_var_over_mean']:.3f}, "
          f"ancilla-rate sd {dev_fit['s4_ancilla_rate_sd']:.5f}")
    print("  s3 same-ancilla autocorr, lags 1..8:  "
          + "  ".join(f"{v:+.4f}" for v in dev_fit["s3_autocorr"][:8]))
    print("  s5 neighbour-pair autocorr, lags 1..8: "
          + "  ".join(f"{v:+.4f}" for v in dev_fit["s5_pair_autocorr"][:8]))

    # ================================================================ BLIND SELECTION
    print(f"\n  BLIND search over {len(PI1) * len(LS) * len(RHOS)} cells, objective D_syn ...",
          flush=True)
    rows, best = [], None
    for pi1 in PI1:
        for L in LS:
            for rho in RHOS:
                gam, sat = solve_gamma(ps, pt, pi1, L, rho, dev_fit["s1_count_mean"],
                                       np.random.default_rng(3))
                D = sample_persistent(ps, pt, N_SEARCH, np.random.default_rng(7),
                                      pi1=pi1, L=L, rho=rho, gamma=gam)
                st = all_stats(D)
                d = syndrome_distance(st, dev_fit)
                r = dict(pi1=pi1, L=L, rho=rho, gamma=gam, gamma_saturated=bool(sat),
                         D_syn=d["total"], d_curve=d["curve"], d_disp=d["dispersion"],
                         d_het=d["heterogeneity"], stats=st)
                rows.append(r)
                if sat:
                    continue
                if best is None or r["D_syn"] < best["D_syn"]:
                    best = r
        print(f"    pi1={pi1}: best D_syn = {best['D_syn']:.4f} at L={best['L']}, "
              f"rho={best['rho']}  (curve {best['d_curve']:.3f}, disp {best['d_disp']:.3f}, "
              f"het {best['d_het']:.3f})", flush=True)

    sel = {k: best[k] for k in ("pi1", "L", "rho", "gamma", "gamma_saturated", "D_syn",
                                "d_curve", "d_disp", "d_het")}
    seal = os.path.join(ROOT, "data", "cross_prediction_selection.json")
    # 🔴 The blind selection is deterministic. If a seal already exists -- this script crashed
    # after sealing on its first run -- a re-run MUST reproduce it exactly, or the blind phase was
    # not blind and the experiment is void.
    if os.path.exists(seal):
        with open(seal) as fh:
            prev = json.load(fh)["selected_blind"]
        for k in ("pi1", "L", "rho"):
            if prev[k] != sel[k]:
                raise RuntimeError(f"BLIND SEAL BROKEN: {k} was {prev[k]}, re-run gives {sel[k]}")
        print(f"    re-run REPRODUCES the existing seal exactly (pi1, L, rho)")
    tmp = seal + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(dict(selected_blind=sel, objective="D_syn (s2, s3, s4); no decoder observable",
                       grid=dict(pi1=PI1, L=LS, rho=RHOS), n_search=N_SEARCH,
                       device_fit_stats=dev_fit), fh, indent=1)
    os.replace(tmp, seal)
    print(f"\n  SEALED: blind selection written to {os.path.basename(seal)}")
    print(f"    pi1={sel['pi1']}, L={sel['L']}, rho={sel['rho']}, gamma={sel['gamma']:.3f}")

    with open(os.path.join(ROOT, "data", "persistent_refine.json")) as fh:
        r12 = json.load(fh)
    d12 = r12["selected"]
    print(f"    round 12 selected against Delta_b: pi1={d12['pi1']}, L={d12['L']}, "
          f"rho={d12['rho']}, gamma={d12['gamma']:.3f}")
    agree = (sel["pi1"], sel["L"], sel["rho"]) == (d12["pi1"], d12["L"], d12["rho"])
    print(f"    the two selections {'AGREE' if agree else 'DIFFER'} "
          "(informative, not a gate condition)")

    # ================================================================ UNSEAL AND SCORE
    _SEALED["open"] = True
    print(f"\n  unsealed. scoring the blind selection at {N_FINAL:,} shots ...", flush=True)
    dev = curve(ev, w, t)
    dev_c = counts(ev)
    e0 = endpoint(dev, curve(sample_independent(ps, pt, N_FINAL, np.random.default_rng(0)), w, t),
                  N_HALF)
    e1 = endpoint(dev, curve(sample_heterogeneous(ps, pt, N_FINAL, np.random.default_rng(1),
                                                  GAMMA_SHAPE), w, t), N_HALF)

    def score(tag, **kw):
        gam, sat = solve_gamma(ps, pt, kw["pi1"], kw["L"], kw["rho"],
                               dev_fit["s1_count_mean"], np.random.default_rng(3),
                               chipwide=kw.get("chipwide", False),
                               memoryless=kw.get("memoryless", False))
        D = sample_persistent(ps, pt, N_FINAL, np.random.default_rng(11), gamma=gam, **kw)
        c = curve(D, w, t)
        return dict(tag=tag, E=endpoint(dev, c, N_HALF), gamma=gam, gamma_saturated=bool(sat),
                    curve={str(k): v for k, v in c.items()}, counts=counts(D), **kw)

    blind = score("blind M2", pi1=sel["pi1"], L=sel["L"], rho=sel["rho"])
    plac = {tag: score(tag, pi1=sel["pi1"], L=sel["L"], rho=sel["rho"], **kw)
            for tag, kw in (("chipwide", dict(chipwide=True)),
                            ("memoryless", dict(memoryless=True)))}

    print(f"\n    E/E0 = {blind['E'] / e0:.4f}   (round 12's Delta_b-fitted value: "
          f"{r12['full']['E'] / r12['E_M0']:.4f})")
    print("    device:  " + "  ".join(f"b{b}={dev[b] * 1e4:6.2f}" for b in B_GRID))
    print("    blind :  " + "  ".join(f"b{b}={blind['curve'][str(b)] * 1e4:6.2f}" for b in B_GRID))
    for tag, p in plac.items():
        print(f"    {tag:>11}: E/E0 = {p['E'] / e0:.4f}  b1 = {p['curve']['1'] * 1e4:.2f}e-4")
    print(f"    counts: mean {blind['counts']['mean']:.3f} (device eval {dev_c['mean']:.3f}), "
          f"var/mean {blind['counts']['var_over_mean']:.3f} ({dev_c['var_over_mean']:.3f})")

    g = {
        "1. E <= 0.50 * E(M0), predicted from statistics it never saw": blind["E"] <= 0.50 * e0,
        "2. beats M0": blind["E"] < e0,
        "3. beats M1": blind["E"] < e1,
        "4. beats the chipwide placebo": blind["E"] < plac["chipwide"]["E"],
        "5. beats the memoryless placebo": blind["E"] < plac["memoryless"]["E"],
        "6. gamma not saturated": not blind["gamma_saturated"],
    }
    print("\n" + "=" * 96)
    print(f"  THE GATE (a8dc617).  blind E/E0 = {blind['E'] / e0:.4f}, E/E1 = {blind['E'] / e1:.4f}")
    for k, v in g.items():
        print(f"    {'PASS' if v else 'FAIL'}  {k}")
    passed = all(g.values())
    print(f"\n  FALSIFIER 15: {'does NOT fire -- the localisation CROSS-PREDICTS' if passed else 'FIRES -- the localisation does not cross-predict; contribution 3 is withdrawn'}")

    out = os.path.join(ROOT, "data", "cross_prediction.json")
    tmp = out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(dict(E_M0=e0, E_M1=e1, device_curve={str(k): v for k, v in dev.items()},
                       device_counts=dev_c, device_fit_stats=dev_fit,
                       selected_blind=sel, selections_agree=bool(agree),
                       round12_selected=d12, blind=blind, placebos=plac, search=rows,
                       n_search=N_SEARCH, n_final=N_FINAL, bar=0.50,
                       gate={k: bool(v) for k, v in g.items()}, gate_passed=bool(passed),
                       falsifier15_fired=bool(not passed)), fh, indent=1)
    os.replace(tmp, out)
    print(f"\nwrote {out}   ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
