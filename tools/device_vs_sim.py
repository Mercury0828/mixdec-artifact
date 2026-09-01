#!/usr/bin/env python
"""Run the parallel two-window analysis on the CACHED DEVICE shots and compare against the
i.i.d. simulator baseline. Costs 0 QPU seconds — both jobs are retrieved by id.

This is the comparison the project is actually about. `stim`'s circuit-level depolarizing noise is
i.i.d. across rounds by construction; `ibm_cleveland` is measurably not (same-ancilla temporal
correlations 0.094 down to 0.036 at dt=2..10, fitted length L ~ 11 rounds, record M1). Any gap
between the two, at matched detector event rate, is what temporal correlation buys or costs.

🔴 What this CANNOT do: the device has no ground truth, so the `B9` weight test is simulator-only.
Here we can only compare event RATES, not physical error weights.

Usage:  python tools/device_vs_sim.py
"""
import json
import os
import sys

import numpy as np
from scipy.stats import beta, fisher_exact

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detectors import build_detectors, load  # noqa: E402
from parallel_window import two_window  # noqa: E402
from sim_substrate import make_circuit, sample, to_layers  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B_GRID = [1, 2, 3, 4, 6, 8, 12, 16]
DELTA = 0.05


def cp_upper(k, n, alpha):
    return 1.0 if k >= n else float(beta.ppf(1.0 - alpha, k + 1, n - k))


def analyse(D, label, logical_j, alpha):
    n_layers = D.shape[1]
    W1 = n_layers // 2
    rows = []
    for b in B_GRID:
        r = two_window(D, W1, b, logical_j=logical_j, eps=1e-3, seed=0)
        nt = r["seam_nontrivial"].astype(bool)
        dv = r["diverged"].astype(bool)
        free = dv & ~nt
        rows.append(dict(label=label, b=b, shots=int(D.shape[0]),
                         k_div=int(dv.sum()), k_seam=int(nt.sum()), k_free=int(free.sum()),
                         U_div=cp_upper(int(dv.sum()), D.shape[0], alpha),
                         U_free=cp_upper(int(free.sum()), D.shape[0], alpha)))
    return rows


def main():
    alpha = DELTA / len(B_GRID)

    syn, fin, meta = load(os.path.join(ROOT, "data", "da7miljsq5js73bk4vtg.npz"))
    Ddev = build_detectors(syn, fin)
    dev_rate = float(Ddev[:, 1:-1, :].mean())
    print(f"DEVICE  ibm_cleveland job {meta['job_id']}: {Ddev.shape[0]} shots, "
          f"{Ddev.shape[1]} layers, bulk detector rate {dev_rate:.4f}")

    # matched-rate simulator: pick p so the simulated bulk detector rate matches the device
    best = None
    for p in np.linspace(0.004, 0.016, 25):
        c = make_circuit(distance=9, rounds=Ddev.shape[1] - 1, p=float(p))
        _, dets, _, _ = sample(c, 4000, seed=5)
        rate = float(to_layers(dets, n_anc=8)[:, 1:-1, :].mean())
        if best is None or abs(rate - dev_rate) < abs(best[1] - dev_rate):
            best = (float(p), rate)
    p_match, rate_match = best
    print(f"SIM     matched p = {p_match:.4f} -> bulk detector rate {rate_match:.4f}\n")

    circ = make_circuit(distance=9, rounds=Ddev.shape[1] - 1, p=p_match)
    _, dets, obs, _ = sample(circ, Ddev.shape[0], seed=17)
    Dsim = to_layers(dets, n_anc=8)

    dev = analyse(Ddev, "device", 0, alpha)
    sim = analyse(Dsim, "sim-iid", 7, alpha)

    print(f"{'b':>3} | {'DEVICE Delta':>12} {'seam':>6} {'free':>5} | "
          f"{'SIM Delta':>10} {'seam':>6} {'free':>5} | {'dev/sim Delta':>13} {'Fisher p':>10}")
    for a, s in zip(dev, sim):
        n = a["shots"]
        orr, pv = fisher_exact([[a["k_div"], n - a["k_div"]], [s["k_div"], n - s["k_div"]]])
        print(f"{a['b']:>3} | {a['k_div']:>12} {a['k_seam']:>6} {a['k_free']:>5} | "
              f"{s['k_div']:>10} {s['k_seam']:>6} {s['k_free']:>5} | "
              f"{orr:>13.2f} {pv:>10.2e}")

    out = os.path.join(ROOT, "data", "device_vs_sim.json")
    tmp = out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(dict(device_bulk_rate=dev_rate, sim_p=p_match, sim_bulk_rate=rate_match,
                       delta=DELTA, alpha_per_test=alpha, device=dev, sim=sim), fh, indent=1)
    os.replace(tmp, out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
