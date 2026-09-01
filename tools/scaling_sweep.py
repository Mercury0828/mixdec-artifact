#!/usr/bin/env python
"""Round 4: does any of this scale? Simulator-first, 0 QPU. Pre-registered at `aaa9078`.

Everything measured so far sits at ONE point -- `d = 9`, 50 rounds, 51 detector layers -- and the
measured wall-clock benefit there is only 1.39x, because a 26-layer window costs 0.324 us/layer
against the 51-layer joint decode's 0.273: the per-call cost does not shrink with the span. Windowing
exists for scale, so the question is whether the benefit grows where windowing is actually needed.

TWO AXES, on stim, noise calibrated so the bulk detector rate matches the device's 0.0399:

    rounds   in {50, 100, 200, 400} at d = 9   -- the axis a TEMPORAL window decoder actually cuts
    distance in {5, 9, 15, 25} at 50 rounds    -- the spatial axis

At each point, `b = 1`, seam at the midpoint: escalation rate (`seam_weight > 0`), divergence rate,
the unflagged-divergence certificate, and bulk-timed `t1`, `t2`, `tj` composed into the same schedule
as `tools/wallclock_benchmark.py` -- latency `max(t1, t2) + p_seam * tj`, which understates the true
`mean(max(...))` and therefore flatters the speedup, exactly as it does there.

Weights are fitted from the simulated data itself by the same `fit_weights_v2` estimator used on the
device, so the decoder is calibrated the same way at every point rather than handed stim's true
probabilities -- otherwise the simulator decoder would be better specified than the device one and
the comparison would be rigged in favour of scale.

Usage:  python tools/scaling_sweep.py
"""
import json
import os
import sys
import time

import numpy as np
from scipy.stats import beta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parallel_window import TBND, WindowGraph, two_window  # noqa: E402
from sim_substrate import make_circuit, sample, to_layers  # noqa: E402
from weight_model_robustness import fit_weights_v2  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P_NOISE = 0.0077                 # calibrated to the device's 0.0399 bulk detector rate
SHOTS_FIT = 10_000
SHOTS_EV = 10_000
SHOTS_TIME = 3_000
REPEATS = 5
DELTA = 0.05
B = 1
ROUNDS_SWEEP = [(9, r) for r in (50, 100, 200, 400)]
DIST_SWEEP = [(d, 50) for d in (5, 9, 15, 25)]


def cp_upper(k, n, a):
    return 1.0 if k >= n else float(beta.ppf(1.0 - a, k + 1, n - k))


def timed_interleaved(ops, reps=REPEATS):
    """Time several operations with their repeats INTERLEAVED, and take the MEDIAN pass.

    🔴 The first run of this sweep did all repeats of t1, then all of t2, then all of tj, and took
    means. Machine load drifted across that span and the result was physically impossible: the
    401-layer joint decode came out FASTER than the 201-layer one. Interleaving puts any load spike
    across all three operations in the same pass, and the median discards the spiked pass instead of
    averaging it in. Absolute times across sweep points are still only as stable as the machine;
    the RATIO within a point is what is quoted.
    """
    acc = {name: [] for name, _, _ in ops}
    for _ in range(reps):
        for name, decode, inputs in ops:
            t0 = time.perf_counter()
            for x in inputs:
                decode(x)
            acc[name].append((time.perf_counter() - t0) / len(inputs) * 1e6)
    # 🔴 The spread across passes used to be discarded, so no timing in this project carried any
    # uncertainty at all while two clean runs of the same sweep differed by ~5% (audit finding 17).
    # Every pass is now returned alongside the median and stored.
    return ({k: float(np.median(v)) for k, v in acc.items()},
            {k: [float(x) for x in v] for k, v in acc.items()})


def one_point(d, rounds, alpha, seed=5):
    n_anc = d - 1
    circ = make_circuit(distance=d, rounds=rounds, p=P_NOISE)
    _, dets_f, _, _ = sample(circ, SHOTS_FIT, seed=seed)
    _, dets_e, _, _ = sample(circ, SHOTS_EV, seed=seed + 1)
    Df = to_layers(dets_f, n_anc=n_anc)
    De = to_layers(dets_e, n_anc=n_anc)
    n_layers = Df.shape[1]
    rate = float(De[:, 1:-1, :].mean())
    w, t, _, _ = fit_weights_v2(Df, n_fit=SHOTS_FIT)

    W1 = n_layers // 2
    r = two_window(De, W1, B, logical_j=n_anc - 1, wtab=w, ttab=t)
    dv = r["diverged_repaired"].astype(bool)
    flag = r["seam_weight"] > 0
    unf = int((dv & ~flag).sum())

    e1, s2 = min(W1 + B, n_layers), max(W1 - B, 0)
    g1 = WindowGraph(0, e1, n_anc, n_anc - 1, False, e1 < n_layers, wtab=w, ttab=t)
    g2 = WindowGraph(s2, n_layers, n_anc, n_anc - 1, s2 > 0, False, wtab=w, ttab=t)
    joint = WindowGraph(0, n_layers, n_anc, n_anc - 1, False, False, wtab=w, ttab=t)
    sub = De[:SHOTS_TIME]
    ts, passes = timed_interleaved([("t1", g1.decode, [x[:e1] for x in sub]),
                                    ("t2", g2.decode, [x[s2:] for x in sub]),
                                    ("tj", joint.decode, list(sub))])
    t1, t2, tj = ts["t1"], ts["t2"], ts["tj"]
    esc = float(flag.mean())
    lat = max(t1, t2) + esc * tj

    # the speedup as computed from EACH pass, so the figure can carry a real interval rather than
    # a point taken from the median pass
    per_pass = [max(a, b) + esc * c for a, b, c in
                zip(passes["t1"], passes["t2"], passes["tj"])]
    sp_pass = [c / L for c, L in zip(passes["tj"], per_pass)]
    ratio_pass = [(max(a / e1, b / (n_layers - s2))) / (c / n_layers)
                  for a, b, c in zip(passes["t1"], passes["t2"], passes["tj"])]

    return dict(d=d, rounds=rounds, n_layers=n_layers, n_anc=n_anc,
                detector_rate=rate, W1=W1,
                divergences=int(dv.sum()), divergence_rate=float(dv.mean()),
                escalation_rate=esc, unflagged=unf, cert=unf / SHOTS_EV,
                cert_ub=cp_upper(unf, SHOTS_EV, alpha),
                t1_us=t1, t2_us=t2, tj_us=tj, latency_adaptive_us=lat,
                speedup_adaptive=tj / lat,
                us_per_layer_w1=t1 / e1,
                us_per_layer_w2=t2 / (n_layers - s2),
                # 🔴 the LATENCY-DETERMINING window is the slower one. An earlier version
                # reported t1/e1 here and got a per-layer ratio of 1.00 at 401 layers, while the
                # window that actually sets the latency was at 1.10. Audit finding 2.
                us_per_layer_window=(t1 / e1 if t1 >= t2 else t2 / (n_layers - s2)),
                us_per_layer_joint=tj / n_layers,
                passes=passes,
                speedup_passes=sp_pass,
                speedup_lo=float(np.min(sp_pass)), speedup_hi=float(np.max(sp_pass)),
                ratio_passes=ratio_pass,
                ratio_lo=float(np.min(ratio_pass)), ratio_hi=float(np.max(ratio_pass)))


def main():
    points = ROUNDS_SWEEP + [p for p in DIST_SWEEP if p not in ROUNDS_SWEEP]
    alpha = DELTA / len(points)
    print("ROUND 4 -- DOES ANY OF THIS SCALE?  simulator, 0 QPU")
    print(f"  stim p = {P_NOISE}, calibrated to the device bulk detector rate 0.0399")
    print(f"  b = {B}, seam at the midpoint, {SHOTS_EV} evaluation shots per point, "
          f"weights fitted from the simulated data by the same estimator used on the device\n")
    print(f"{'d':>3} {'rounds':>7} {'layers':>7} {'det rate':>9} {'div':>6} {'esc':>8} "
          f"{'unflag':>7} {'cert UB':>9} {'t1':>7} {'tj':>8} {'latency':>8} {'speedup':>8} "
          f"{'[pass range]':>14}")
    rows = []
    for d, rounds in points:
        r = one_point(d, rounds, alpha)
        rows.append(r)
        print(f"{r['d']:>3} {r['rounds']:>7} {r['n_layers']:>7} {r['detector_rate']:>9.5f} "
              f"{r['divergences']:>6} {r['escalation_rate']:>8.4f} {r['unflagged']:>7} "
              f"{r['cert_ub']:>9.2e} {r['t1_us']:>7.2f} {r['tj_us']:>8.2f} "
              f"{r['latency_adaptive_us']:>8.2f} {r['speedup_adaptive']:>7.2f}x "
              f"[{r['speedup_lo']:.2f}-{r['speedup_hi']:.2f}]", flush=True)

    r400 = next(r for r in rows if r["d"] == 9 and r["rounds"] == 400)
    f1 = r400["speedup_adaptive"] < 1.5
    f2 = r400["escalation_rate"] > 0.50
    print(f"\n  FALSIFIER 1: speedup at d=9, 400 rounds below 1.5x?  "
          f"{r400['speedup_adaptive']:.2f}x  ->  "
          f"{'FIRES -- the resource claim does not extrapolate' if f1 else 'does NOT fire'}")
    print(f"  FALSIFIER 2: escalation at 400 rounds above 50%?  "
          f"{r400['escalation_rate']:.2%}  ->  "
          f"{'FIRES -- P9 is a small-record result' if f2 else 'does NOT fire'}")

    print("\n  PER-LAYER COST -- why the benefit does or does not grow")
    print(f"{'d':>3} {'rounds':>7} {'w1 us/lay':>10} {'w2 us/lay':>10} {'CRITICAL':>9} "
          f"{'joint':>8} {'ratio':>7}")
    for r in rows:
        print(f"{r['d']:>3} {r['rounds']:>7} {r['us_per_layer_w1']:>10.3f} "
              f"{r['us_per_layer_w2']:>10.3f} {r['us_per_layer_window']:>9.3f} "
              f"{r['us_per_layer_joint']:>8.3f} "
              f"{r['us_per_layer_window'] / r['us_per_layer_joint']:>7.2f} "
              f"[{r['ratio_lo']:.2f}-{r['ratio_hi']:.2f}]")

    out = os.path.join(ROOT, "data", "scaling_sweep.json")
    tmp = out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(dict(p_noise=P_NOISE, b=B, shots_fit=SHOTS_FIT, shots_eval=SHOTS_EV,
                       shots_timed=SHOTS_TIME, delta=DELTA, alpha_per_point=alpha,
                       falsifier_speedup_fired=bool(f1),
                       falsifier_escalation_fired=bool(f2), rows=rows), fh, indent=1)
    os.replace(tmp, out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
