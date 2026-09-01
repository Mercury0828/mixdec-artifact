#!/usr/bin/env python
"""Round 14 part B: does `M2` reproduce the SYNDROME process, and is the persistence TIME or SPACE?

Pre-registered at `a8dc617`, amended `c9d69f0`, `docs/expected.md` Round 14, falsifier 16.

Rounds 12-13 compared one fitted dwell (`L = 16`) against one measured correlation length (10-11)
as if they were the same quantity. They are not (`ELIMINATION_LADDER.md` section 2). This runs the
**identical estimator** on the device and on model traces and compares whole curves.

🔴 **`M0` IS THE NULL, NOT ZERO.** The detector construction alone produces autocorrelation: one
time edge flips `d[j, r]` and `d[j, r+1]`, so `s3` at lag 1 is large under *every* model including
the independent one; and two time edges firing on neighbouring ancillas at the same round raise
`s5`. Reading either curve against zero would attribute structural terms to persistence. So every
curve here is reported both raw and as an **excess over `M0`**, and it is the excess that carries
the space-versus-time reading validated in `syndrome_stats.self_test`.

Uncertainty is a block standard error over 10 disjoint shot blocks, not an i.i.d. bootstrap.

Usage:  python tools/syndrome_process.py
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_campaign_r import load_arm  # noqa: E402
from detectors import build_detectors  # noqa: E402
from independence_model import sample_independent  # noqa: E402
from persistent_noise_model import N_HALF, sample_persistent  # noqa: E402
from syndrome_stats import MAX_LAG, all_stats, self_test  # noqa: E402
from weight_model_robustness import fit_weights_v2  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_MODEL = 100_000
N_BLOCKS = 10


def blocked(D, n_blocks=N_BLOCKS):
    """Mean and block standard error of `s3` and `s5`, over disjoint shot blocks."""
    n = D.shape[0] // n_blocks
    s3 = np.array([all_stats(D[i * n:(i + 1) * n])["s3_autocorr"] for i in range(n_blocks)])
    s5 = np.array([all_stats(D[i * n:(i + 1) * n])["s5_pair_autocorr"] for i in range(n_blocks)])
    return dict(s3=s3.mean(axis=0), s3_se=s3.std(axis=0, ddof=1) / np.sqrt(n_blocks),
                s5=s5.mean(axis=0), s5_se=s5.std(axis=0, ddof=1) / np.sqrt(n_blocks))


def show(tag, r, lags=(1, 2, 3, 4, 6, 8, 12, 16, 20, 24)):
    idx = [k - 1 for k in lags]
    print(f"    {tag:<22} s3 " + " ".join(f"{r['s3'][i]:+.4f}" for i in idx))
    print(f"    {'':<22} s5 " + " ".join(f"{r['s5'][i]:+.4f}" for i in idx))


def main():
    t0 = time.time()
    syn, fin, _ = load_arm(0)
    Ddev = build_detectors(syn, fin)
    fit = Ddev[:N_HALF]
    w, t, ps, pt = fit_weights_v2(fit, n_fit=N_HALF)

    print("ROUND 14 PART B -- DOES M2 REPRODUCE THE SYNDROME PROCESS?  0 QPU")
    print("  falsifier 16: M2's autocorrelation must stay within 2x of the device's at every")
    print("  estimable lag. M0 is the null; the excess over M0 is what carries the reading.\n")

    with open(os.path.join(ROOT, "data", "persistent_refine.json")) as fh:
        sel = json.load(fh)["selected"]
    print(f"  M2 at round 12's selection: pi1={sel['pi1']}, L={sel['L']}, rho={sel['rho']}, "
          f"gamma={sel['gamma']:.3f}", flush=True)

    dev = blocked(fit)
    print("  computing M0 ...", flush=True)
    m0 = blocked(sample_independent(ps, pt, N_MODEL, np.random.default_rng(0)))
    print("  computing M2 ...", flush=True)
    m2 = blocked(sample_persistent(ps, pt, N_MODEL, np.random.default_rng(11), pi1=sel["pi1"],
                                   L=sel["L"], rho=sel["rho"], gamma=sel["gamma"]))

    lags = (1, 2, 3, 4, 6, 8, 12, 16, 20, 24)
    print("\n  RAW curves, lags " + " ".join(f"{k:>7}" for k in lags))
    for tag, r in (("device (fit half)", dev), ("M0 independent", m0), ("M2 persistent", m2)):
        show(tag, r)

    print("\n  EXCESS OVER M0 -- this is the persistence signal")
    print("    lag        " + " ".join(f"{k:>8}" for k in lags))
    ex = {}
    for tag, r in (("device", dev), ("M2", m2)):
        e3 = r["s3"] - m0["s3"]
        e5 = r["s5"] - m0["s5"]
        se3 = np.sqrt(r["s3_se"] ** 2 + m0["s3_se"] ** 2)
        se5 = np.sqrt(r["s5_se"] ** 2 + m0["s5_se"] ** 2)
        ex[tag] = dict(s3=e3, s5=e5, s3_se=se3, s5_se=se5)
        idx = [k - 1 for k in lags]
        print(f"    {tag:<6} s3  " + " ".join(f"{e3[i]:+8.4f}" for i in idx))
        print(f"    {'':<6}  +-  " + " ".join(f"{se3[i]:8.4f}" for i in idx))
        print(f"    {tag:<6} s5  " + " ".join(f"{e5[i]:+8.4f}" for i in idx))
        print(f"    {'':<6}  +-  " + " ".join(f"{se5[i]:8.4f}" for i in idx))

    # ---------------------------------------------------------------- space vs time on the device
    d3, d5 = ex["device"]["s3"], ex["device"]["s5"]
    sig3 = d3 > 2 * ex["device"]["s3_se"]
    sig5 = d5 > 2 * ex["device"]["s5_se"]
    last3 = int(np.flatnonzero(sig3).max() + 1) if sig3.any() else 0
    last5 = int(np.flatnonzero(sig5).max() + 1) if sig5.any() else 0
    # 🔴 The ratio must NOT be taken at lag 1. That is exactly where M0's structural term
    # dominates -- one time edge flips d[j,r] and d[j,r+1], so every model has a large s3[1] --
    # and the device's excess there is consistent with zero (it is in fact slightly negative).
    # Forming s5[1]/s3[1] there divides by ~0. Use the lags where BOTH excesses are resolved.
    use = (d3 > 2 * ex["device"]["s3_se"]) & (d5 > 2 * ex["device"]["s5_se"])
    ratio = float(d5[use].sum() / d3[use].sum()) if use.any() else float("nan")
    per_lag = {int(k + 1): float(d5[k] / d3[k]) for k in np.flatnonzero(use)}
    m2_ratio = (float(ex["M2"]["s5"][use].sum() / ex["M2"]["s3"][use].sum())
                if use.any() else float("nan"))
    print("\n  THE SPACE-VERSUS-TIME READING (device excess over M0):")
    print(f"    s3 excess significant (>2 block SE) out to lag {last3}")
    print(f"    s5 excess significant out to lag {last5}")
    print(f"    both resolved at {int(use.sum())} lags: {sorted(per_lag)}")
    print("    per-lag s5/s3: " + " ".join(f"{k}:{v:.3f}" for k, v in
                                           list(per_lag.items())[:12]))
    print(f"    POOLED over those lags:  device s5/s3 = {ratio:.3f}"
          f"     M2 s5/s3 = {m2_ratio:.3f}")
    print("    calibration, COMPUTED by syndrome_stats.self_test on injected ground truth "
          "(not quoted):", flush=True)
    ok, _, cal = self_test(verbose=False)
    tr = cal["time_fault"]["s5_pair_autocorr"][0] / cal["time_fault"]["s3_autocorr"][0]
    sr = cal["space_fault"]["s5_pair_autocorr"][0] / cal["space_fault"]["s3_autocorr"][0]
    print(f"      injected width {cal['width']} rounds; self-test "
          f"{'PASSED' if ok else 'FAILED'}")
    print(f"      a pure TIME  fault gives s5[1]/s3[1] = {tr:.3f}")
    print(f"      a pure SPACE fault gives s5[1]/s3[1] = {sr:.3f}")
    import math
    lt = math.log(max(ratio / tr, 1e-9))
    ls = math.log(max(sr / ratio, 1e-9))
    verdict = ("TIME-like (measurement)" if lt < ls else "SPACE-like (data qubit)")
    print(f"    => on a log scale the device sits {lt:.2f} above pure TIME and {ls:.2f} below")
    print(f"       pure SPACE, i.e. BETWEEN them, marginally closer to {verdict}.")
    print(f"    => and it is {ratio / m2_ratio:.1f}x more space-correlated than M2 is.")

    # ---------------------------------------------------------------- falsifier 16
    est = np.abs(d3) > 2 * ex["device"]["s3_se"]
    bad = []
    for k in range(MAX_LAG):
        if not est[k]:
            continue
        a, b = m2["s3"][k], dev["s3"][k]
        if b == 0 or not (0.5 <= a / b <= 2.0):
            bad.append((k + 1, float(b), float(a)))
    print("\n" + "=" * 96)
    print(f"  FALSIFIER 16: {'FIRES' if bad else 'does NOT fire'} -- "
          f"{len(bad)} of {int(est.sum())} estimable lags outside the 2x band")
    for k, b, a in bad[:10]:
        print(f"    lag {k:>2}: device {b:+.4f}, M2 {a:+.4f}, ratio {a / b:.2f}")

    out = os.path.join(ROOT, "data", "syndrome_process.json")
    tmp = out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(dict(
            max_lag=MAX_LAG, n_blocks=N_BLOCKS, n_model=N_MODEL, m2_params=sel,
            raw={k: {n: [float(x) for x in v[n]] for n in ("s3", "s3_se", "s5", "s5_se")}
                 for k, v in (("device", dev), ("M0", m0), ("M2", m2))},
            excess={k: {n: [float(x) for x in v[n]] for n in ("s3", "s3_se", "s5", "s5_se")}
                    for k, v in ex.items()},
            device_s3_significant_to_lag=last3, device_s5_significant_to_lag=last5,
            device_s5_over_s3_pooled=ratio, m2_s5_over_s3_pooled=m2_ratio,
            per_lag_s5_over_s3=per_lag, lags_used=[int(k) for k in per_lag],
            log_dist_to_time=float(lt), log_dist_to_space=float(ls), calib_time_ratio=float(tr),
            calib_space_ratio=float(sr), calib_self_test_passed=bool(ok),
            space_vs_time_verdict=verdict,
            falsifier16_fired=bool(bad),
            lags_outside_2x=[{"lag": k, "device": b, "M2": a} for k, b, a in bad]), fh, indent=1)
    os.replace(tmp, out)
    print(f"\nwrote {out}   ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
