#!/usr/bin/env python
"""Is the device's second-order parity structure representable by ANY independent-event model?

Not pre-registered as a falsifier: this is a DEDUCTION with a measurement attached, and it either
holds or it does not. 0 QPU.

THE ARGUMENT. For any model in which independent events `e` fire with probability `p_e` and flip a
detector set `S_e`, write the parity attenuation

    omega_A  =  -log E[ prod_{i in A} (-1)^{D_i} ]  =  sum_{e : |S_e & A| odd} psi_e,
    psi_e    =  -log(1 - 2 p_e)  >=  0  for  p_e in [0, 1/2].

Then for any pair `{i, j}`,

    omega_i + omega_j - omega_ij  =  2 * sum_{e : S_e contains BOTH i and j} psi_e  >=  0.

🔴 That inequality is **topology-free**. It holds for every hyperedge set, every rate assignment, and
every fitting procedure, provided only that events are independent and each `p_e <= 1/2`. A device
pair whose measured value is significantly negative therefore cannot be reproduced by ANY independent
event model at all -- not by a richer graph, not by hyperedges, not by better fitting.

So this test is not "the fitted DEM is incomplete". It is a test of whether the independent-event
assumption itself is refutable from second-order parity statistics on this hardware.

An earlier estimator in this project clipped the corresponding quantity at zero
(`structure_learned_dem.pij` uses `np.clip(inner, 0, 1)`), which silently discards exactly the
evidence this test needs.

Usage:  python tools/parity_feasibility.py
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
from persistent_noise_model import N_HALF  # noqa: E402
from weight_model_robustness import fit_weights_v2  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_BLOCKS = 10


def omega_excess(D):
    """`omega_i + omega_j - omega_ij` for every detector pair. Returns a (d, d) array.

    `omega_i  = -log(1 - 2<D_i>)`,  `omega_ij = -log(1 - 2<D_i XOR D_j>)`,
    `<D_i XOR D_j> = <D_i> + <D_j> - 2<D_i D_j>`.
    """
    X = D.reshape(D.shape[0], -1).astype(np.float32)
    n = X.shape[0]
    m = X.mean(axis=0, dtype=np.float64)
    G = (X.T @ X).astype(np.float64) / n
    xor = m[:, None] + m[None, :] - 2 * G
    with np.errstate(divide="ignore", invalid="ignore"):
        w1 = -np.log(np.clip(1 - 2 * m, 1e-12, None))
        wij = -np.log(np.clip(1 - 2 * xor, 1e-12, None))
    return w1[:, None] + w1[None, :] - wij


def blocked_excess(D, n_blocks=N_BLOCKS):
    n = D.shape[0] // n_blocks
    E = np.stack([omega_excess(D[i * n:(i + 1) * n]) for i in range(n_blocks)])
    return E.mean(axis=0), E.std(axis=0, ddof=1) / np.sqrt(n_blocks)


def report(tag, mean, se, iu, n_pairs, bonf_z):
    z = np.full(mean.shape, np.nan)
    ok = se[iu] > 0
    zz = np.full(iu[0].shape, np.nan)
    zz[ok] = mean[iu][ok] / se[iu][ok]
    neg = np.nan_to_num(zz, nan=0.0) < -bonf_z
    pos = np.nan_to_num(zz, nan=0.0) > bonf_z
    worst = int(np.nanargmin(zz)) if np.isfinite(zz).any() else -1
    print(f"  {tag}")
    print(f"    pairs with excess significantly NEGATIVE (z < -{bonf_z:.2f}): "
          f"**{int(neg.sum()):,}** of {n_pairs:,}")
    print(f"    pairs with excess significantly positive: {int(pos.sum()):,}")
    if worst >= 0:
        print(f"    most negative pair: excess {mean[iu][worst]:+.5f} "
              f"+- {se[iu][worst]:.5f}   z = {zz[worst]:+.2f}")
    return dict(n_negative=int(neg.sum()), n_positive=int(pos.sum()),
                min_z=float(np.nanmin(zz)), max_z=float(np.nanmax(zz)),
                most_negative_excess=float(mean[iu][worst]) if worst >= 0 else None,
                most_negative_se=float(se[iu][worst]) if worst >= 0 else None)


def main():
    t0 = time.time()
    syn, fin, _ = load_arm(0)
    Ddev = build_detectors(syn, fin)
    fit, ev = Ddev[:N_HALF], Ddev[N_HALF:2 * N_HALF]
    _, _, ps, pt = fit_weights_v2(fit, n_fit=N_HALF)
    d = fit.shape[1] * fit.shape[2]
    iu = np.triu_indices(d, k=1)
    n_pairs = len(iu[0])
    # two-sided Bonferroni over every pair
    from scipy.stats import norm
    bonf_z = float(norm.isf(0.025 / n_pairs))
    print("IS THE DEVICE'S SECOND-ORDER PARITY STRUCTURE REPRESENTABLE AT ALL?  0 QPU")
    print(f"  omega_i + omega_j - omega_ij >= 0 holds for EVERY independent-event model,")
    print(f"  every hyperedge topology, every fitting procedure. {n_pairs:,} pairs, "
          f"Bonferroni z = {bonf_z:.2f}\n")

    out = {}
    m, s = blocked_excess(fit)
    out["device fit half"] = report("device, fit half (50,000)", m, s, iu, n_pairs, bonf_z)
    m2, s2 = blocked_excess(ev)
    out["device eval half"] = report("device, held-out half (50,000)", m2, s2, iu, n_pairs, bonf_z)
    # 🔴 the null control: an actual independent-event model must show ZERO significant negatives,
    # because the inequality is a theorem for it. Any it shows are the estimator's own false rate.
    m0, s0 = blocked_excess(sample_independent(ps, pt, N_HALF, np.random.default_rng(0)))
    out["C0 independent-event control"] = report(
        "C0 independent-event model (50,000) -- MUST be ~0 negatives; it is a theorem for C0",
        m0, s0, iu, n_pairs, bonf_z)

    dev_neg = out["device eval half"]["n_negative"]
    ctl_neg = out["C0 independent-event control"]["n_negative"]
    proved = dev_neg > 0 and ctl_neg == 0
    print("\n" + "=" * 96)
    if proved:
        print("  RESULT: the device violates a necessary condition of EVERY independent-event model.")
        print(f"    {dev_neg:,} pairs significantly negative on held-out data; the control shows "
              f"{ctl_neg}.")
    elif dev_neg > 0:
        print(f"  RESULT: {dev_neg:,} negative pairs on the device, but the control shows "
              f"{ctl_neg} -- the estimator has its own false rate and the argument does NOT go "
              f"through as stated.")
    else:
        print("  RESULT: NO significant violation. The device's second-order parity structure is")
        print("    consistent with some independent-event model; infeasibility must be argued on a")
        print("    declared topology instead, not topology-free.")

    path = os.path.join(ROOT, "data", "parity_feasibility.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(dict(n_pairs=n_pairs, n_blocks=N_BLOCKS, bonferroni_z=bonf_z,
                       results=out, topology_free_violation=bool(proved)), fh, indent=1)
    os.replace(tmp, path)
    print(f"\nwrote {path}   ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
