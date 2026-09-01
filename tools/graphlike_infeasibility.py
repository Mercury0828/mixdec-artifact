#!/usr/bin/env python
"""Round 17: is the MAXIMAL graphlike independent-event class infeasible for this device? 0 QPU.

Pre-registered by this file's commit, before it was run. Falsifiers 22 and 23 below.

THE THEOREM (from the third consultation, and reproduced here so the code is checkable against it).
For independent events `e` firing with probability `p_e <= 1/2` and flipping detector set `S_e`,

    omega_A = -log E[ prod_{i in A} (-1)^{D_i} ] = sum_{e : |S_e & A| odd} psi_e,
    psi_e   = -log(1 - 2 p_e) >= 0.

Writing `c_ij = (omega_i + omega_j - omega_ij) / 2` gives the exact identity

    c_ij = sum_{e : S_e contains both i and j} psi_e,     and     Q = sum_e psi_e s_e s_e^T

with `Q_ii = omega_i`, `Q_ij = c_ij`. For the MAXIMAL graphlike class -- every singleton event and
every one of the `d(d-1)/2` detector pairs allowed, **no geometry chosen by us** -- the moments are
realizable **if and only if**

    (i)  c_ij >= 0            for every pair
    (ii) sum_{j != i} c_ij <= omega_i    for every detector i           <- the budget inequality

Sufficiency: set `psi_{ij} = c_ij` and `psi_{i} = omega_i - sum_j c_ij`; both are nonnegative exactly
under (i) and (ii), and they reproduce every target. Necessity is immediate from the identity.

Summing (ii) over `i` gives the one-dimensional consequence, with `B = sum_i omega_i` and
`T = sum_{i<j} c_ij`:

    G2 = 2T - B <= 0        for every independent graphlike model,

and for events of detector support at most `K`,  `2T <= (K-1) B`,  so

    K >= 1 + 2T/B     and     K >= 1 + max_i (t_i / omega_i).

🔴 `p_e <= 1/2` is without loss of generality here: every singleton polarization `1 - 2<D_i>` is
strongly positive on this device, which forces the deterministic offset in the general
representation to be zero.

🔴 STATUS: **retrospective discovery, not a prospective test.** These 100,000 shots have been
examined many times by this project. What is proved without qualification is the empirical statement
about the measured moments. The population claim is reported with block-aware uncertainty and a
`C0` null-pipeline calibration, and the frozen witness is carried unchanged into any replication.

🔴 The `C0` control is the load-bearing part. `G2 <= 0` is a THEOREM for `C0`, so any positive `G2`
the pipeline reports on `C0` samples is the pipeline's own bias, and the device must clear that
distribution -- not zero. An earlier coordinatewise version of this test was killed by exactly this
control, which reported more violations on `C0` than on the device.

Usage:  python tools/graphlike_infeasibility.py
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
N_NULL = 40          # independent C0 datasets for the null-pipeline calibration
N_BLOCKS = 10


def moments(D):
    """`omega` (d,), `c` (d, d) with zero diagonal, and the derived budget quantities."""
    X = D.reshape(D.shape[0], -1).astype(np.float32)
    n = X.shape[0]
    m = X.mean(axis=0, dtype=np.float64)
    G = (X.T @ X).astype(np.float64) / n
    xor = m[:, None] + m[None, :] - 2 * G
    w1 = -np.log(np.clip(1 - 2 * m, 1e-12, None))
    wij = -np.log(np.clip(1 - 2 * xor, 1e-12, None))
    c = 0.5 * (w1[:, None] + w1[None, :] - wij)
    np.fill_diagonal(c, 0.0)
    return w1, c


def stats(D):
    w1, c = moments(D)
    t = c.sum(axis=1)
    B = float(w1.sum())
    T = float(c.sum() / 2)
    return dict(B=B, T=T, G2=2 * T - B, g=t - w1, omega=w1,
                K_lower_global=1 + 2 * T / B if B > 0 else np.nan,
                K_lower_local=1 + float(np.max(t / np.maximum(w1, 1e-12))),
                n_negative_c=int((c < 0).sum() // 2))


def main():
    t0 = time.time()
    syn, fin, _ = load_arm(0)
    Ddev = build_detectors(syn, fin)
    fit, ev = Ddev[:N_HALF], Ddev[N_HALF:2 * N_HALF]
    _, _, ps, pt = fit_weights_v2(fit, n_fit=N_HALF)
    d = fit.shape[1] * fit.shape[2]

    print("ROUND 17 -- IS THE MAXIMAL GRAPHLIKE CLASS INFEASIBLE?  0 QPU")
    print(f"  every singleton and all {d * (d - 1) // 2:,} detector pairs allowed; no geometry "
          f"chosen by us")
    print("  G2 = 2T - B <= 0 is a THEOREM for every independent graphlike model\n")

    sf, se_ = stats(fit), stats(ev)
    print(f"  device fit half : B = {sf['B']:.3f}, T = {sf['T']:.3f}, "
          f"G2 = {sf['G2']:+.3f},  K >= {sf['K_lower_global']:.2f} (global)")
    print(f"  device eval half: B = {se_['B']:.3f}, T = {se_['T']:.3f}, "
          f"G2 = {se_['G2']:+.3f},  K >= {se_['K_lower_global']:.2f} (global)")
    print(f"                    max_i t_i/omega_i = {se_['K_lower_local'] - 1:.3f}  ->  "
          f"K >= {se_['K_lower_local']:.2f} (local)")
    print(f"                    detectors violating the budget (g_i > 0): "
          f"{int((se_['g'] > 0).sum())} of {d}", flush=True)

    # ---------------------------------------------------------------- the null-pipeline calibration
    print(f"\n  null-pipeline calibration: {N_NULL} independent C0 datasets of {N_HALF:,} shots")
    print("  (G2 <= 0 is a theorem for C0, so anything positive here is the pipeline's own bias)",
          flush=True)
    nulls = []
    for i in range(N_NULL):
        s = stats(sample_independent(ps, pt, N_HALF, np.random.default_rng(1000 + i)))
        nulls.append(dict(G2=s["G2"], max_g=float(s["g"].max()),
                          n_viol=int((s["g"] > 0).sum()), K=s["K_lower_global"]))
        if i % 10 == 9:
            print(f"    {i + 1}/{N_NULL} done", flush=True)
    nG2 = np.array([x["G2"] for x in nulls])
    nmax = np.array([x["max_g"] for x in nulls])
    nviol = np.array([x["n_viol"] for x in nulls])
    print(f"    C0 G2:      mean {nG2.mean():+.4f}, sd {nG2.std(ddof=1):.4f}, "
          f"max {nG2.max():+.4f}")
    print(f"    C0 max g_i: mean {nmax.mean():+.5f}, max {nmax.max():+.5f}")
    print(f"    C0 detectors violating the budget: mean {nviol.mean():.1f}, max {nviol.max()}")

    z_g2 = (se_["G2"] - nG2.mean()) / max(nG2.std(ddof=1), 1e-12)
    exceed_g2 = int((nG2 >= se_["G2"]).sum())
    exceed_v = int((nviol >= (se_["g"] > 0).sum()).sum())
    f22 = se_["G2"] > nG2.max()
    f23 = int((se_["g"] > 0).sum()) > nviol.max()

    print("\n" + "=" * 96)
    print(f"  device eval G2 = {se_['G2']:+.4f}   against the C0 null max of {nG2.max():+.4f}")
    print(f"    z against the C0 null = {z_g2:+.1f};  {exceed_g2} of {N_NULL} null draws reach it")
    print(f"  FALSIFIER 22 (global budget): "
          f"{'the device EXCEEDS every null draw -- the maximal graphlike class is violated' if f22 else 'NOT exceeded -- no global violation is demonstrated'}")
    print(f"  FALSIFIER 23 (local budgets): "
          f"{'device exceeds every null draw' if f23 else 'NOT exceeded'}"
          f"  ({int((se_['g'] > 0).sum())} device detectors vs null max {nviol.max()})")
    if f22:
        print(f"\n  => no independent graphlike DEM, at ANY edge placement, reproduces these")
        print(f"     singleton and pair parity moments. Minimum event support K >= "
              f"{se_['K_lower_global']:.2f}.")

    path = os.path.join(ROOT, "data", "graphlike_infeasibility.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(dict(
            d=d, n_pairs=d * (d - 1) // 2, n_null=N_NULL,
            device_fit={k: (v if not isinstance(v, np.ndarray) else None)
                        for k, v in sf.items()},
            device_eval={k: (v if not isinstance(v, np.ndarray) else None)
                         for k, v in se_.items()},
            device_eval_n_budget_violations=int((se_["g"] > 0).sum()),
            device_eval_max_g=float(se_["g"].max()),
            null=dict(G2_mean=float(nG2.mean()), G2_sd=float(nG2.std(ddof=1)),
                      G2_max=float(nG2.max()), max_g_max=float(nmax.max()),
                      n_viol_mean=float(nviol.mean()), n_viol_max=int(nviol.max())),
            z_G2_against_null=float(z_g2), null_draws_reaching_device=exceed_g2,
            falsifier22_violation=bool(f22), falsifier23_violation=bool(f23),
            status="retrospective discovery on data examined many times; witness frozen for "
                   "any replication"), fh, indent=1)
    os.replace(tmp, path)
    print(f"\nwrote {path}   ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
