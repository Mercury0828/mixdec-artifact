#!/usr/bin/env python
"""Close three `B11` holes in the weight model, on cached data at 0 QPU.

Round-2 audit findings addressed here:

  B11-a  The boundary-edge estimator was "detector rate minus incident edge probabilities", a
         low-rate approximation. The correct independent-edge inversion is
             p_b = 1/2 * [ 1 - (1-2 q_i) / prod_{e != b} (1 - 2 p_e) ]
         Implemented as `invert_boundary`.
  B11-b  Clipping negative residuals to 1e-6 gives weight log((1-p)/p) = 13.82, but a 0/10,000
         binomial only supports p <~ 3e-4, i.e. weight <~ 8.1. Weights are now capped at what the
         sample size supports (`W_CAP`), so the fit cannot fabricate near-forbidden edges.
  B11-c  🔴 THE SHARPEST ONE. The virtual temporal boundary weight (~3.17) sat BELOW the median
         physical time-edge weight (~4.33), so the artificial boundary attracts matches — and the
         joint decoder has no such edges, so "one shared weight table" does not equalise the two
         likelihood models. Here the temporal-boundary weight is SCANNED over a wide range. If the
         device-vs-simulator gap is an artifact of cheap boundaries, it must vanish somewhere in
         that scan.

Also tests per-layer over-fitting (B11-d) by comparing per-layer weights against a single pooled
value and against shrinkage toward the pool.

Usage:  python tools/weight_model_robustness.py
"""
import os
import sys

import numpy as np
import stim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrated_weights import P_CEIL, pij  # noqa: E402
from detectors import build_detectors, load  # noqa: E402
from parallel_window import two_window  # noqa: E402
from sim_substrate import sample, to_layers  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAULTS_PER_LAYER = 17
# 🔴 One-sided 95% upper bound for ZERO events in the FIT-HALF sample size, 1 - alpha^(1/n).
# Round-3 audit caught this: it was justified as 0/10,000 (3.0e-4, weight 8.11) while the fit half
# holds only 5,000 shots. Correct value for n = 5,000 is 5.99e-4, weight 7.42.
N_FIT = 5000
P_MIN_SUPPORTED = float(1 - 0.05 ** (1.0 / N_FIT))   # 5.99e-4
W_CAP = float(np.log((1 - P_MIN_SUPPORTED) / P_MIN_SUPPORTED))   # ~7.42


def invert_boundary(q_i, others):
    """Correct independent-edge inversion for a boundary edge probability."""
    prod = 1.0
    for p in others:
        prod *= (1 - 2 * p)
    if prod <= 0:
        return P_MIN_SUPPORTED
    val = 0.5 * (1 - (1 - 2 * q_i) / prod)
    return float(np.clip(val, P_MIN_SUPPORTED, P_CEIL))


def fit_weights_v2(D, pooled=False, shrink=0.0, tbnd_weight=None, boundary="symmetry",
                   n_fit=None):
    """Weight tables with the corrected boundary inversion and a supported weight cap.

    n_fit: sample size the weight cap is justified against. The cap is
        `log((1-p)/p)` with `p = 1 - 0.05^(1/n_fit)`, the one-sided 95% upper bound for ZERO events
        in `n_fit` shots -- the definition `docs/R_PREREGISTRATION.md` step 4 fixes. Default None
        keeps the module constant (n = 5,000), so every cached result stays reproducible; campaign R
        has a 100,000-shot fit half and must pass its own size, or it would price a genuinely rarer
        edge at a floor 20x too high.
    pooled: use one global space/time probability instead of per-layer values (B11-d).
    shrink: 0 = per-layer, 1 = fully pooled; anything between shrinks toward the pool.
    tbnd_weight: if given, force every virtual temporal boundary edge to this weight (B11-c scan).
    boundary: how to price the chain-end data qubits d_0 and d_8.
        "residual"  -- the old inversion. VALIDATED AGAINST GROUND TRUTH AND FOUND BIASED: on stim
                       data with known DEM probabilities, bulk space edges come out at 1.00x and
                       time edges at 0.99x, but the boundary edges at 1.49x (d_0) and 1.44x (d_8).
                       The residual absorbs every unmodelled mechanism at the chain ends, making
                       those edges too cheap -- and the d_0 boundary is the sole carrier of the
                       logical observable, so that bias converts one-for-one into logical errors.
        "symmetry"  -- DEFAULT. A chain-end data qubit is physically the same kind of object as a
                       bulk data qubit, so price it at the bulk space median. On the simulator this
                       is exactly right: stim's true boundary probability EQUALS the bulk space
                       probability (0.00955 both). On hardware it is an assumption -- that the end
                       data qubits are no worse than bulk ones -- but far better motivated than
                       "absorb all unexplained rate into the end qubit".
    """
    shots, n_layers, n_anc = D.shape
    p_min = (P_MIN_SUPPORTED if n_fit is None else float(1 - 0.05 ** (1.0 / n_fit)))
    w_cap = (W_CAP if n_fit is None else float(np.log((1 - p_min) / p_min)))
    p_space = np.full((n_layers, n_anc + 1), p_min)
    p_time = np.full((n_layers, n_anc), p_min)

    for r in range(n_layers):
        for k in range(1, n_anc):
            p_space[r, k] = pij(D[:, r, k - 1], D[:, r, k])
        if r + 1 < n_layers:
            for j in range(n_anc):
                p_time[r, j] = pij(D[:, r, j], D[:, r + 1, j])

    if pooled or shrink > 0:
        ps = float(np.median(p_space[:, 1:n_anc]))
        pt = float(np.median(p_time[:-1]))
        w = 1.0 if pooled else shrink
        p_space[:, 1:n_anc] = (1 - w) * p_space[:, 1:n_anc] + w * ps
        p_time[:-1] = (1 - w) * p_time[:-1] + w * pt

    # chain-end data qubits d_0 and d_8
    if boundary == "symmetry":
        bulk = float(np.median(p_space[:, 1:n_anc]))
        p_space[:, 0] = bulk
        p_space[:, n_anc] = bulk
    else:
        for r in range(n_layers):
            for j, k in ((0, 0), (n_anc - 1, n_anc)):
                q = float(D[:, r, j].mean())
                others = [p_space[r, 1] if j == 0 else p_space[r, n_anc - 1]]
                if r > 0:
                    others.append(p_time[r - 1, j])
                if r + 1 < n_layers:
                    others.append(p_time[r, j])
                p_space[r, k] = invert_boundary(q, others)

    def w_of(p):
        return float(min(np.log((1 - max(p, p_min)) / max(p, p_min)), w_cap))

    wtab = np.ones((n_layers + 1) * FAULTS_PER_LAYER)
    ttab = np.ones((n_layers + 1) * n_anc)
    for r in range(n_layers):
        for k in range(n_anc + 1):
            wtab[r * FAULTS_PER_LAYER + k] = w_of(p_space[r, k])
        for j in range(n_anc):
            wtab[r * FAULTS_PER_LAYER + 9 + j] = w_of(p_time[r, j])
            ttab[r * n_anc + j] = (tbnd_weight if tbnd_weight is not None
                                   else w_of(max(float(D[:, r, j].mean()), p_min)))
    return wtab, ttab, p_space, p_time


def sim_data(n, seed=17, p=0.005):
    circ = stim.Circuit.generated(
        "repetition_code:memory", distance=9, rounds=50,
        before_measure_flip_probability=p, after_reset_flip_probability=p,
        before_round_data_depolarization=p, after_clifford_depolarization=p)
    _, dets, obs, _ = sample(circ, n, seed=seed)
    return to_layers(dets, n_anc=8), obs


def main():
    syn, fin, _ = load(os.path.join(ROOT, "data", "da7miljsq5js73bk4vtg.npz"))
    D = build_detectors(syn, fin)
    n = D.shape[0]
    Ds, obs = sim_data(n)
    fit, ev = slice(0, 5000), slice(5000, 10000)   # held-out throughout

    wd, td, ps, pt = fit_weights_v2(D[fit])
    ws, ts, _, _ = fit_weights_v2(Ds[fit])
    print(f"weight cap from sample size: {W_CAP:.2f} (was 13.82 with the 1e-6 clip)")
    print(f"device fitted weights: space {wd[[r*17+k for r in range(50) for k in range(9)]].min():.2f}"
          f"..{wd[[r*17+k for r in range(50) for k in range(9)]].max():.2f}  "
          f"time {wd[[r*17+9+j for r in range(50) for j in range(8)]].min():.2f}"
          f"..{wd[[r*17+9+j for r in range(50) for j in range(8)]].max():.2f}  "
          f"tbnd median {np.median(td[:400]):.2f}")

    print("\n=== B11-c: SCAN the virtual temporal boundary weight (b=4, held-out, repaired) ===")
    print("If the device-vs-sim gap is an artifact of cheap artificial boundaries, it must vanish here.")
    print(f"{'tbnd w':>7} {'DEVICE':>8} {'SIM':>6}")
    for tw in [1.0, 2.0, 3.0, 4.33, 6.0, 8.0, 12.0, 20.0]:
        wd2, td2, _, _ = fit_weights_v2(D[fit], tbnd_weight=tw)
        ws2, ts2, _, _ = fit_weights_v2(Ds[fit], tbnd_weight=tw)
        rd = two_window(D[ev], 25, 4, logical_j=0, eps=1e-3, seed=0, wtab=wd2, ttab=td2)
        rs = two_window(Ds[ev], 25, 4, logical_j=7, eps=1e-3, seed=0, wtab=ws2, ttab=ts2)
        print(f"{tw:>7.2f} {int(rd['diverged_repaired'].sum()):>8} "
              f"{int(rs['diverged_repaired'].sum()):>6}")

    print("\n=== B11-a/b/d: weight-model variants (b=4, held-out, repaired) ===")
    print(f"{'variant':<34} {'DEVICE':>8} {'SIM':>6}")
    for label, kw in [("per-layer, corrected boundary", {}),
                      ("shrunk 50% toward pooled", {"shrink": 0.5}),
                      ("fully pooled (no per-layer)", {"pooled": True})]:
        wd2, td2, _, _ = fit_weights_v2(D[fit], **kw)
        ws2, ts2, _, _ = fit_weights_v2(Ds[fit], **kw)
        rd = two_window(D[ev], 25, 4, logical_j=0, eps=1e-3, seed=0, wtab=wd2, ttab=td2)
        rs = two_window(Ds[ev], 25, 4, logical_j=7, eps=1e-3, seed=0, wtab=ws2, ttab=ts2)
        print(f"{label:<34} {int(rd['diverged_repaired'].sum()):>8} "
              f"{int(rs['diverged_repaired'].sum()):>6}")


if __name__ == "__main__":
    main()
