#!/usr/bin/env python
"""Measured decode times composed into a schedule. Forced by the cross-model audit's finding 6.

The audit was right that the reported speedup was a **layer-count proxy** under an assumed linear
cost model, and that nothing in the repo validated it: `two_window` decodes the joint graph on every
shot merely to measure divergence, and runs the two windows serially.

WHAT IS TIMED

    t1     decode window 1 over layers [0, W1+b)
    t2     decode window 2 over layers [W1-b, n)
    tj     decode the joint graph over all n layers
    trep   decode the seam residual on the joint graph -- what seam repair actually costs

With the two windows running CONCURRENTLY on separate hardware:

    joint                 latency  tj
    fixed split           latency  max(t1, t2)                    (no repair: NOT the variant scored)
    fixed split + REPAIR  latency  max(t1, t2) + p_seam * trep    <- matches the accuracy we quote
    adaptive              latency  max(t1, t2) + p_seam * tj

🔴 A MEASUREMENT ARTIFACT WAS FOUND AND IS THE REASON THIS FILE HAS TWO TIMING MODES. The first
version wrapped every individual decode in its own `perf_counter` pair inside one per-shot loop. That
inflated the *window* decodes by ~3 us each while barely touching the joint decode, and it produced
speedups **below 1.0x** -- i.e. it said windowing is slower than joint decoding, which is the
opposite of the truth. A separate-pass control (each operation timed in its own tight loop, inputs
pre-sliced) gives 8.50 / 9.40 / 14.75 us against the in-loop 12.0 / 10.7 / 15.4 us. Both modes are
therefore reported and they BRACKET the answer:

BULK timing is used throughout: each operation in its own tight loop, inputs pre-sliced. The
per-shot mode is NOT reported as a bracket, because its error is instrumentation, not algorithm --
it measures the decoder plus ~3 us/call of `perf_counter` and slicing overhead, which is not a
property of the scheme. Those first numbers are withdrawn outright.

The one caveat that remains and is not measured: latency is composed as `max(mean t1, mean t2)`,
while a real scheduler waits for the slower window ON THAT SHOT, so the true latency is
`mean(max(t1, t2))`, which is larger. The direction of this bias is known -- it **flatters the
speedup** -- and its size is not measured here. State it wherever the number is quoted.

WHAT THIS IS STILL NOT: a schedule composed from single-thread component times on one CPU. Not an
FPGA, no queueing model, no inter-shot pipelining. Guide section 7.3 forbids packaging any of it as
online microsecond throughput.

Usage:  python tools/wallclock_benchmark.py
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_campaign_r import load_arm  # noqa: E402
from detectors import build_detectors  # noqa: E402
from parallel_window import TBND, WindowGraph  # noqa: E402
from weight_model_robustness import fit_weights_v2  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W1 = 25
N_FIT = 50_000
N_TIME = 20_000
REPEATS = 3
B_GRID = [1, 2, 4, 6, 8, 16]


def bulk_us(decode, inputs, reps=REPEATS):
    """Time one operation in its own tight loop. Returns mean us/call and its sd across repeats."""
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for x in inputs:
            decode(x)
        out.append((time.perf_counter() - t0) / len(inputs) * 1e6)
    return float(np.mean(out)), float(np.std(out))


def interleaved_us(ops, reps=REPEATS):
    """Time several operations with their repeats INTERLEAVED; median pass, and every pass kept.

    Round 4 had to throw away a whole sweep because its timing loop did all repeats of one
    operation before starting the next and machine load drifted across that span. This file had the
    same shape -- `bulk_us` per operation, in sequence -- and it produces the project's headline
    device timing, so it gets the same fix: one pass times every operation in turn, the median pass
    is reported, and the full spread is stored instead of discarded (audit finding 17).
    """
    acc = {name: [] for name, _, _ in ops}
    for _ in range(reps):
        for name, decode, inputs in ops:
            if not inputs:
                acc[name].append(0.0)
                continue
            t0 = time.perf_counter()
            for x in inputs:
                decode(x)
            acc[name].append((time.perf_counter() - t0) / len(inputs) * 1e6)
    return ({k: float(np.median(v)) for k, v in acc.items()},
            {k: [float(x) for x in v] for k, v in acc.items()})


def main():
    syn, fin, _ = load_arm(0)
    D = build_detectors(syn, fin)
    n_layers, n_anc = D.shape[1], D.shape[2]
    w, t, _, _ = fit_weights_v2(D[:N_FIT], n_fit=N_FIT)
    Dev = D[N_FIT:N_FIT + N_TIME]
    joint = WindowGraph(0, n_layers, n_anc, 0, False, False, wtab=w, ttab=t)

    print("MEASURED DECODE TIMES, COMPOSED INTO A SCHEDULE")
    print(f"  {N_TIME} held-out shots, {REPEATS} repeats, single thread, one CPU core")
    print("  offline scheduling only (guide 7.3); BULK and PERSHOT bracket the answer\n")

    tj_us, tj_sd = bulk_us(joint.decode, list(Dev))
    print(f"  joint over {n_layers} layers: {tj_us:.2f} us (sd {tj_sd:.2f}), "
          f"{tj_us / n_layers:.3f} us/layer\n")

    print(f"{'b':>3} {'spans':>7} {'t1':>7} {'t2':>7} {'trep':>7} {'seam%':>7} "
          f"{'lat fix+rep':>12} {'lat adapt':>10} {'speedup fix+rep':>16} {'speedup adapt':>14}")
    rows = []
    for b in B_GRID:
        e1, s2 = min(W1 + b, n_layers), max(W1 - b, 0)
        g1 = WindowGraph(0, e1, n_anc, 0, False, e1 < n_layers, wtab=w, ttab=t)
        g2 = WindowGraph(s2, n_layers, n_anc, 0, s2 > 0, False, wtab=w, ttab=t)
        c1 = (g1.kind != TBND) & (g1.glayer < W1)
        c2 = (g2.kind != TBND) & (g2.glayer >= W1)

        in1 = [d[:e1] for d in Dev]
        in2 = [d[s2:] for d in Dev]

        # seam residuals, computed UNTIMED, then their repair decodes timed alongside the rest
        residuals = []
        for s in range(N_TIME):
            st = np.zeros((n_layers, n_anc), dtype=np.uint8)
            st[:e1] ^= g1.boundary_of(g1.decode(in1[s]) & c1)
            st[s2:] ^= g2.boundary_of(g2.decode(in2[s]) & c2)
            r = st ^ Dev[s]
            if r.any():
                residuals.append(r)
        p_seam = len(residuals) / N_TIME

        med, passes = interleaved_us([("t1", g1.decode, in1),
                                      ("t2", g2.decode, in2),
                                      ("tj", joint.decode, list(Dev)),
                                      ("trep", joint.decode, residuals)])
        t1_us, t2_us, trep_us = med["t1"], med["t2"], med["trep"]
        tj_b = med["tj"]          # the joint decode re-timed in the SAME passes as this row
        sp_fix = [c / (max(a, b_) + p_seam * d) for a, b_, c, d in
                  zip(passes["t1"], passes["t2"], passes["tj"], passes["trep"])]
        sp_ad = [c / (max(a, b_) + p_seam * c) for a, b_, c in
                 zip(passes["t1"], passes["t2"], passes["tj"])]

        win = max(t1_us, t2_us)
        lat_fix = win + p_seam * trep_us
        lat_ad = win + p_seam * tj_us
        print(f"{b:>3} {f'{e1}/{n_layers - s2}':>7} {t1_us:>7.2f} {t2_us:>7.2f} {trep_us:>7.2f} "
              f"{p_seam:>7.2%} {lat_fix:>12.2f} {lat_ad:>10.2f} {tj_us / lat_fix:>15.2f}x "
              f"{tj_us / lat_ad:>13.2f}x  [{min(sp_ad):.2f}-{max(sp_ad):.2f}]")
        rows.append(dict(b=b, spans=[e1, n_layers - s2], mode="bulk",
                         t1_us=t1_us, t2_us=t2_us, tj_us=tj_us, trep_us=trep_us,
                         seam_rate=p_seam, latency_fixed_repaired_us=lat_fix,
                         latency_adaptive_us=lat_ad,
                         work_fixed_repaired_us=t1_us + t2_us + p_seam * trep_us,
                         work_adaptive_us=t1_us + t2_us + p_seam * tj_us,
                         speedup_fixed_repaired=tj_us / lat_fix,
                         speedup_adaptive=tj_us / lat_ad,
                         tj_us_same_passes=tj_b, passes=passes,
                         speedup_fixed_repaired_passes=sp_fix,
                         speedup_adaptive_passes=sp_ad,
                         speedup_adaptive_lo=float(np.min(sp_ad)),
                         speedup_adaptive_hi=float(np.max(sp_ad)),
                         speedup_fixed_repaired_lo=float(np.min(sp_fix)),
                         speedup_fixed_repaired_hi=float(np.max(sp_fix))))

    print("\nIS DECODE COST PROPORTIONAL TO WINDOW SPAN? (what the layer proxy assumed)")
    print(f"{'b':>3} {'span1':>6} {'us/layer':>9} {'span2':>6} {'us/layer':>9}")
    for r in rows:
        s1, s2_ = r["spans"]
        print(f"{r['b']:>3} {s1:>6} {r['t1_us'] / s1:>9.3f} {s2_:>6} {r['t2_us'] / s2_:>9.3f}")
    print(f"    joint: {n_layers} layers, {tj_us / n_layers:.3f} us/layer")
    print("  ** A shorter window is NOT proportionally cheaper: there is a per-call cost that does")
    print("     not shrink with the span, so the layer proxy overstates what windowing buys.")

    out = os.path.join(ROOT, "data", "wallclock_benchmark.json")
    tmp = out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(dict(n_shots=N_TIME, repeats=REPEATS, n_layers=n_layers, W1=W1, joint_us=tj_us,
                       note="BULK mode: each operation timed in its own tight loop; latency composed "
                            "as max(mean t1, mean t2) + p_seam * t, which understates latency by a "
                            "Jensen gap and therefore flatters the speedup. Single thread, one CPU, "
                            "no queueing, no pipelining. Not an online real-time claim.",
                       rows=rows), fh, indent=1)
    os.replace(tmp, out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
