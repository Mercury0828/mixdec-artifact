#!/usr/bin/env python
"""Round 3 of the TQE #1 self-check: finite-shot honesty and quantified benefit, in units.

Pre-registered in `docs/expected.md` before this file was written (`7ca8256`).

FOUR THINGS, and the first exists because a pre-run check found a hole:

1. WEIGHT-JITTER SENSITIVITY. `two_window(..., eps=..., wtab=w, ttab=t)` never uses `eps` -- the
   jitter is applied inside `make_weights`, which is skipped once fitted weights are supplied. So the
   tie-break this project has been quoting was INERT, and ties are real: the fitted `Y = 0` table has
   859 edges, 757 distinct weights, and 103 of them (12.0%) share the single chain-end boundary price
   that `boundary="symmetry"` assigns to both ends at all 51 layers. Selection among tied optima is
   therefore pymatching's internal ordering. `P7` survives -- it is conditioned on a fixed decoder
   specification and tie handling is part of that -- but the magnitudes have to be measured.

2. RESOURCE ACCOUNTING IN UNITS. Detector-layers decoded per shot, as WORK (total, i.e. throughput
   cost) and CRITICAL PATH (the longer of the two windows, i.e. offline scheduling latency):

       joint          work n,        path n
       fixed split b  work n + 2b,   path n/2 + b
       adaptive       add esc * n to both, since escalated shots pay a full joint decode on top

   🔴 OFFLINE SCHEDULING ONLY, per guide section 7.3. This is not microsecond online throughput and
   must never be packaged as such -- that would need an FPGA control chain this project does not have.

3. BUFFER REDUCTION ON THE PRIOR ART'S OWN FIGURE-OF-MERIT. arXiv:2605.14637 reports ~40% average
   buffer reduction at equal logical error rate. The same quantity is computed here: the smallest
   FIXED buffer reaching a given certificate, against the average buffer the adaptive scheme uses.

4. SEAM-POSITION SWEEP on campaign-R data. The existing sweep is on the cached 10,000-shot job only.

Usage:  python tools/resource_and_stability.py
"""
import json
import os
import sys
import time

import numpy as np
from scipy.stats import beta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_campaign_r import load_arm  # noqa: E402
from detectors import build_detectors  # noqa: E402
from parallel_window import WindowGraph, two_window  # noqa: E402
from weight_model_robustness import fit_weights_v2  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W1 = 25
DELTA = 0.05
N_FIT = 50_000
N_SUB = 20_000
JITTERS = [1e-6, 1e-4, 1e-2]
JITTER_SEEDS = [0, 1, 2]
B_JITTER = [1, 4]
SEAMS = [10, 15, 20, 25, 30, 35, 40]
B_RESOURCE = [1, 2, 3, 4, 6, 8, 12, 16]
LATENCY_SEEDS = 5
LATENCY_SHOTS = 2_000


def cp_upper(k, n, a):
    return 1.0 if k >= n else float(beta.ppf(1.0 - a, k + 1, n - k))


def jitter(wtab, ttab, delta, seed):
    """Multiplicative jitter. delta = 1e-6 breaks ties without moving the objective; 1e-2 moves it."""
    rng = np.random.default_rng(seed)
    return (wtab * (1.0 + delta * rng.standard_normal(wtab.shape)),
            ttab * (1.0 + delta * rng.standard_normal(ttab.shape)))


def main():
    syn, fin, _ = load_arm(0)
    D = build_detectors(syn, fin)
    n_layers = D.shape[1]
    fitsl = slice(0, N_FIT)
    sub = slice(N_FIT, N_FIT + N_SUB)
    w0, t0, _, _ = fit_weights_v2(D[fitsl], n_fit=N_FIT)
    report = {}

    # ------------------------------------------------------------------ 1. tie-break sensitivity
    print("=" * 100)
    print("1. WEIGHT-JITTER SENSITIVITY -- the tie-break this project quoted was never applied")
    g = WindowGraph(0, n_layers, D.shape[2], 0, False, False, wtab=w0, ttab=t0)
    uniq, cnt = np.unique(np.round(g.w, 12), return_counts=True)
    print(f"   fitted table: {len(g.w)} edges, {len(uniq)} distinct weights, "
          f"largest tie group {cnt.max()} edges ({100 * cnt.max() / len(g.w):.1f}%)")
    print(f"{'b':>3} {'delta':>8} {'seed':>5} {'divergences':>12} {'vs baseline':>12} "
          f"{'escalation':>11} {'unflagged':>10}")
    jrows = []
    base = {}
    for b in B_JITTER:
        r = two_window(D[sub], W1, b, logical_j=0, wtab=w0, ttab=t0)
        dv = r["diverged_repaired"].astype(bool)
        base[b] = int(dv.sum())
        fl = r["seam_weight"] > 0
        print(f"{b:>3} {'none':>8} {'-':>5} {base[b]:>12} {'baseline':>12} "
              f"{fl.mean():>11.4f} {int((dv & ~fl).sum()):>10}")
        jrows.append(dict(b=b, delta=0.0, seed=None, divergences=base[b],
                          escalation=float(fl.mean()), unflagged=int((dv & ~fl).sum())))
        for d in JITTERS:
            for s in JITTER_SEEDS:
                wj, tj = jitter(w0, t0, d, s)
                rj = two_window(D[sub], W1, b, logical_j=0, wtab=wj, ttab=tj)
                dvj = rj["diverged_repaired"].astype(bool)
                flj = rj["seam_weight"] > 0
                k = int(dvj.sum())
                print(f"{b:>3} {d:>8.0e} {s:>5} {k:>12} {k / base[b] - 1:>+11.1%} "
                      f"{flj.mean():>11.4f} {int((dvj & ~flj).sum()):>10}")
                jrows.append(dict(b=b, delta=d, seed=s, divergences=k,
                                  rel_change=k / base[b] - 1,
                                  escalation=float(flj.mean()),
                                  unflagged=int((dvj & ~flj).sum())))
    report["jitter"] = jrows
    tiny = [r for r in jrows if r.get("delta") == 1e-6 and r["b"] == 1]
    spread = max(abs(r["rel_change"]) for r in tiny)
    print(f"\n   FALSIFIER 1: does 1e-6 jitter move the b=1 count by more than 25%?  "
          f"max |change| = {spread:.1%}  ->  "
          f"{'FIRES' if spread > 0.25 else 'does NOT fire'}")
    report["falsifier_tiebreak_fired"] = bool(spread > 0.25)
    report["tiebreak_max_rel_change_b1"] = float(spread)

    # ------------------------------------------------------------------ 2. resource accounting
    print("\n" + "=" * 100)
    print("2. RESOURCE ACCOUNTING, IN DETECTOR-LAYERS PER SHOT -- offline scheduling only (guide 7.3)")
    with open(os.path.join(ROOT, "data", "campaign_r_results.json")) as fh:
        cr = json.load(fh)
    dep = {d["b"]: d for d in cr["R6_deployable"]}
    r1 = {d["b"]: d for d in cr["R1_R4"]}
    alpha = DELTA / len(B_RESOURCE)
    print(f"{'scheme':>16} {'b':>3} {'work':>7} {'path':>7} {'esc':>8} {'cert':>9} {'CP UB':>10} "
          f"{'path vs joint':>14}")
    print(f"{'joint':>16} {'-':>3} {n_layers:>7.1f} {n_layers:>7.1f} {'-':>8} {'0':>9} "
          f"{'-':>10} {'1.00x':>14}")
    rrows = []
    for b in B_RESOURCE:
        kd = r1[b]["k_dev"]
        esc = dep[b]["escalation_rate"]
        unf = dep[b]["unflagged"]
        fw, fp = n_layers + 2 * b, n_layers / 2 + b
        aw, ap = fw + esc * n_layers, fp + esc * n_layers
        print(f"{'fixed split':>16} {b:>3} {fw:>7.1f} {fp:>7.1f} {'-':>8} "
              f"{kd / N_FIT:>9.5f} {cp_upper(kd, N_FIT, alpha):>10.2e} "
              f"{n_layers / fp:>13.2f}x")
        print(f"{'adaptive':>16} {b:>3} {aw:>7.1f} {ap:>7.1f} {esc:>8.4f} "
              f"{unf / N_FIT:>9.5f} {cp_upper(unf, N_FIT, alpha):>10.2e} "
              f"{n_layers / ap:>13.2f}x")
        rrows.append(dict(b=b, fixed_work=fw, fixed_path=fp,
                          fixed_cert=kd / N_FIT, fixed_cert_ub=cp_upper(kd, N_FIT, alpha),
                          adaptive_work=aw, adaptive_path=ap, escalation=esc,
                          adaptive_cert=unf / N_FIT,
                          adaptive_cert_ub=cp_upper(unf, N_FIT, alpha),
                          fixed_speedup=n_layers / fp, adaptive_speedup=n_layers / ap))
    report["resource"] = rrows

    # ---- the prior art's own figure-of-merit: average buffer at matched certificate
    print("\n   BUFFER REDUCTION AT MATCHED CERTIFICATE (arXiv:2605.14637 reports ~40%)")
    print(f"{'adaptive b':>11} {'adaptive UB':>12} {'smallest fixed b with UB <= that':>34} "
          f"{'avg buffer':>11} {'reduction':>10}")
    breds = []
    for r in rrows:
        need = next((q["b"] for q in rrows if q["fixed_cert_ub"] <= r["adaptive_cert_ub"]), None)
        if need is None:
            print(f"{r['b']:>11} {r['adaptive_cert_ub']:>12.2e} "
                  f"{'none on the grid -- joint decoding':>34} {'-':>11} {'-':>10}")
            breds.append(dict(b=r["b"], matched_fixed_b=None))
            continue
        avg_b = (1 - r["escalation"]) * r["b"] + r["escalation"] * (n_layers / 2)
        red = 1 - avg_b / need
        print(f"{r['b']:>11} {r['adaptive_cert_ub']:>12.2e} {need:>34} "
              f"{avg_b:>11.2f} {red:>+9.1%}")
        breds.append(dict(b=r["b"], matched_fixed_b=need, avg_buffer=avg_b, reduction=red))
    report["buffer_reduction"] = breds

    # ------------------------------------------------------------------ 3. measured latency
    print("\n" + "=" * 100)
    print(f"3. MEASURED DECODE TIME, {LATENCY_SEEDS} repeats x {LATENCY_SHOTS} shots, mean +/- 95% CI")
    print(f"{'b':>3} {'ms/shot':>20} {'ms/shot, joint-only':>22}")
    lrows = []
    for b in (1, 4, 16):
        ts = []
        for s in range(LATENCY_SEEDS):
            off = N_FIT + s * LATENCY_SHOTS
            t0_ = time.perf_counter()
            two_window(D[off:off + LATENCY_SHOTS], W1, b, logical_j=0, wtab=w0, ttab=t0)
            ts.append((time.perf_counter() - t0_) / LATENCY_SHOTS * 1e3)
        a = np.array(ts)
        ci = 1.96 * a.std(ddof=1) / np.sqrt(len(a))
        print(f"{b:>3} {f'{a.mean():.3f} +/- {ci:.3f}':>20}")
        lrows.append(dict(b=b, ms_per_shot=float(a.mean()), ci95=float(ci),
                          repeats=LATENCY_SEEDS, shots_each=LATENCY_SHOTS))
    report["latency"] = lrows

    # ------------------------------------------------------------------ 4. seam-position sweep
    print("\n" + "=" * 100)
    print(f"4. SEAM-POSITION SWEEP at b = 1, {N_SUB} campaign-R shots")
    print(f"{'W1':>4} {'divergences':>12} {'rate':>9} {'escalation':>11} {'unflagged':>10}")
    srows = []
    for w1 in SEAMS:
        r = two_window(D[sub], w1, 1, logical_j=0, wtab=w0, ttab=t0)
        dv = r["diverged_repaired"].astype(bool)
        fl = r["seam_weight"] > 0
        print(f"{w1:>4} {int(dv.sum()):>12} {dv.mean():>9.5f} {fl.mean():>11.4f} "
              f"{int((dv & ~fl).sum()):>10}")
        srows.append(dict(W1=w1, divergences=int(dv.sum()), rate=float(dv.mean()),
                          escalation=float(fl.mean()), unflagged=int((dv & ~fl).sum())))
    report["seam_sweep"] = srows
    cnts = [r["divergences"] for r in srows]
    ratio = max(cnts) / max(1, min(cnts))
    print(f"\n   FALSIFIER 2: does divergence vary by more than 2x across W1 in [10,40]?  "
          f"max/min = {ratio:.2f}  ->  {'FIRES' if ratio > 2.0 else 'does NOT fire'}")
    report["falsifier_seam_fired"] = bool(ratio > 2.0)
    report["seam_max_over_min"] = float(ratio)

    report["shot_counts"] = dict(fit=N_FIT, subsample=N_SUB, campaign_r_eval=N_FIT)
    out = os.path.join(ROOT, "data", "resource_and_stability.json")
    tmp = out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(report, fh, indent=1)
    os.replace(tmp, out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
