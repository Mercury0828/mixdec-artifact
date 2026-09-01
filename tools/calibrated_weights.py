#!/usr/bin/env python
"""Does the device-vs-simulator divergence gap survive a decoder calibrated to the device?

The obvious confound: the uniform-weight graphlike decoder is roughly right for stim's i.i.d. noise
but wrong for `ibm_cleveland`, whose same-ancilla dt>=2 correlations a graphlike DEM prices at zero.
If the 40-70x gap is just decoder mismatch, fitting edge weights from the device data should close it.
If it survives, the gap is structural — correlation the graphlike model cannot represent at any
weighting.

Edge probabilities use the standard pairwise estimator (Google 2023 / Chen et al.):

    p_ij = 1/2 - 1/2 * sqrt(1 - 4*(<x_i x_j> - <x_i><x_j>) / (1 - 2<x_i> - 2<x_j> + 4<x_i x_j>))

with weight = log((1-p)/p). Boundary edge probabilities come from the per-detector residual:
each detector's own rate minus what its incident bulk edges already explain.

Usage:  python tools/calibrated_weights.py
"""
import os
import sys

import numpy as np
from scipy.stats import fisher_exact

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detectors import build_detectors, load  # noqa: E402
from parallel_window import two_window  # noqa: E402
from sim_substrate import make_circuit, sample, to_layers  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B_GRID = [1, 2, 3, 4, 6, 8, 12, 16]
P_FLOOR, P_CEIL = 1e-6, 0.45
FAULTS_PER_LAYER = 17


def pij(x, y):
    """Pairwise edge-probability estimator for two 0/1 detector columns."""
    xm, ym, xy = x.mean(), y.mean(), (x & y).mean()
    num = xy - xm * ym
    den = 1 - 2 * xm - 2 * ym + 4 * xy
    if den <= 0:
        return P_FLOOR
    v = 1 - 4 * num / den
    if v < 0:
        return P_CEIL
    return float(np.clip(0.5 - 0.5 * np.sqrt(v), P_FLOOR, P_CEIL))


def fit_weights(D):
    """Return (wtab, ttab) global weight tables fitted from detector data.

    Layout matches `parallel_window.WindowGraph`: wtab[layer*17 + k] for k=0..8 the space-like
    faults (data qubits d_0..d_8) and k=9..16 the time-like faults (ancilla j).
    """
    shots, n_layers, n_anc = D.shape
    p_space = np.full((n_layers, n_anc + 1), P_FLOOR)   # data qubit d_0..d_8 per layer
    p_time = np.full((n_layers, n_anc), P_FLOOR)

    for r in range(n_layers):
        for k in range(1, n_anc):                        # bulk data qubits touch two ancillas
            p_space[r, k] = pij(D[:, r, k - 1], D[:, r, k])
        if r + 1 < n_layers:
            for j in range(n_anc):
                p_time[r, j] = pij(D[:, r, j], D[:, r + 1, j])

    # boundary (d_0, d_8): the part of a detector's own rate not explained by its bulk edges
    for r in range(n_layers):
        for j, k in ((0, 0), (n_anc - 1, n_anc)):
            rate = float(D[:, r, j].mean())
            explained = 0.0
            if j == 0:
                explained += p_space[r, 1]
            else:
                explained += p_space[r, n_anc - 1]
            if r > 0:
                explained += p_time[r - 1, j]
            if r + 1 < n_layers:
                explained += p_time[r, j]
            p_space[r, k] = float(np.clip(rate - explained, P_FLOOR, P_CEIL))

    def w(p):
        return float(np.log((1 - p) / p))

    wtab = np.ones((n_layers + 1) * FAULTS_PER_LAYER)
    ttab = np.ones((n_layers + 1) * n_anc)
    for r in range(n_layers):
        for k in range(n_anc + 1):
            wtab[r * FAULTS_PER_LAYER + k] = w(p_space[r, k])
        for j in range(n_anc):
            wtab[r * FAULTS_PER_LAYER + 9 + j] = w(p_time[r, j])
            ttab[r * n_anc + j] = w(np.clip(D[:, r, j].mean(), P_FLOOR, P_CEIL))
    return wtab, ttab, p_space, p_time


def run(D, logical_j, wtab=None, ttab=None):
    n_layers = D.shape[1]
    W1 = n_layers // 2
    out = []
    for b in B_GRID:
        r = two_window(D, W1, b, logical_j=logical_j, eps=1e-3, seed=0, wtab=wtab, ttab=ttab)
        nt = r["seam_nontrivial"].astype(bool)
        dv = r["diverged"].astype(bool)
        out.append((b, int(dv.sum()), int(nt.sum()), int((dv & ~nt).sum())))
    return out


def main():
    syn, fin, meta = load(os.path.join(ROOT, "data", "da7miljsq5js73bk4vtg.npz"))
    Ddev = build_detectors(syn, fin)
    n = Ddev.shape[0]
    dev_rate = float(Ddev[:, 1:-1, :].mean())

    circ = make_circuit(distance=9, rounds=Ddev.shape[1] - 1, p=0.008)
    _, dets, _, _ = sample(circ, n, seed=17)
    Dsim = to_layers(dets, n_anc=8)

    wd, td, ps, pt = fit_weights(Ddev)
    ws, ts, _, _ = fit_weights(Dsim)
    print(f"device bulk detector rate {dev_rate:.4f}; sim {Dsim[:, 1:-1, :].mean():.4f}")
    print(f"fitted device edge p: space bulk median {np.median(ps[:, 1:8]):.5f}, "
          f"time median {np.median(pt[:-1]):.5f}")

    dev_u = run(Ddev, 0)
    dev_c = run(Ddev, 0, wd, td)
    sim_u = run(Dsim, 7)
    sim_c = run(Dsim, 7, ws, ts)

    print(f"\n{'b':>3} | {'DEV uniform':>11} {'DEV calib':>10} | {'SIM uniform':>11} "
          f"{'SIM calib':>10} | {'ratio calib':>11} {'Fisher p':>10}")
    for (b, du, _, _), (_, dc, _, dcf), (_, su, _, _), (_, sc, _, _) in zip(
            dev_u, dev_c, sim_u, sim_c):
        orr, pv = fisher_exact([[dc, n - dc], [sc, n - sc]])
        rs = "inf" if sc == 0 else f"{orr:.1f}"
        print(f"{b:>3} | {du:>11} {dc:>10} | {su:>11} {sc:>10} | {rs:>11} {pv:>10.2e}")

    print("\nseam-free divergence, calibrated decoder:  device",
          [x[3] for x in dev_c], " sim", [x[3] for x in sim_c])


if __name__ == "__main__":
    main()
