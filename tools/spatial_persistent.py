#!/usr/bin/env python
"""Round 13: does SPATIAL EXTENT close what per-ancilla temporal memory left open? 0 QPU.

Pre-registered at `2b4399b`, `docs/expected.md` Round 13, falsifier 14.

WHERE THIS COMES FROM. Round 12 established that per-ancilla temporal persistence generates the
device's window/joint disagreement where the independent model generates exactly none -- but it
closed only ~83% of the decoder-observable error against a pre-registered 90% bar. The two
pre-registered placebos failed from OPPOSITE SIDES:

    memoryless (same pi1, rho, marginals; L = 1)   ->  0.00e-4, i.e. NOTHING
    chip-wide  (one shared state for all ancillas) ->  wildly off, 77.00e-4 and 9.00e-4 across runs

They bracket the answer. So `M4` adds the one parameter that interpolates exactly between them.

THE MODEL. Generate the per-ancilla persistent seed chain as in `M2`, then DILATE it spatially:

    Z[j, r] = OR over the w neighbouring ancillas centred on j

🔴 `w = 1` reproduces `M2` bit-for-bit and `w = n_anc` reproduces the chip-wide placebo, so the two
placebos are the ENDPOINTS of the new parameter's range and the hypothesis under test is that the
truth lies strictly between them -- bursts localised but not confined to one ancilla.

Constraints carried over unchanged: existing edge families with their existing fault ids, no appended
lag edges; `gamma` bisected so the detector-count mean matches, with saturated cells INELIGIBLE.

Usage:  python tools/spatial_persistent.py
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
from persistent_noise_model import (B_GRID, N_HALF, counts, disagreement_curve,  # noqa: E402
                                    endpoint, markov_state)
from weight_model_robustness import fit_weights_v2  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_SEARCH = 20_000
N_FINAL = 100_000
PI_SEED = [0.02, 0.04, 0.08]
LS = [12, 20, 32]
RHOS = [20.0, 50.0, 150.0]
WS = [1, 2, 3, 5, 8]        # 1 == M2 exactly; 8 == n_anc == the chip-wide placebo
GAMMA_SHAPE = 5.701


def dilate(z, w):
    """OR over a window of `w` neighbouring ancillas. `w = 1` is the identity."""
    if w <= 1:
        return z
    n_anc = z.shape[2]
    out = np.zeros_like(z)
    half = w // 2
    for d in range(-half, w - half):
        out |= np.roll(z, d, axis=2) if w >= n_anc else np.roll(z, d, axis=2)
    return out


def sample_spatial(p_space, p_time, shots, rng, pi_seed, L, rho, w, gamma=1.0):
    """`sample_independent` with the TIME edges modulated by a SPATIALLY DILATED persistent state."""
    n_layers, n_anc = p_time.shape
    denom = (1.0 - pi_seed) + pi_seed * rho
    p_lo = gamma * p_time / denom
    p_hi = np.minimum(1.0, rho * gamma * p_time / denom)
    z = dilate(markov_state(shots, n_layers, n_anc, pi_seed, L, rng), w)
    D = np.zeros((shots, n_layers, n_anc), dtype=np.uint8)
    for r in range(n_layers):
        for k in range(n_anc + 1):
            hit = rng.random(shots) < p_space[r, k]
            if k == 0:
                D[hit, r, 0] ^= 1
            elif k == n_anc:
                D[hit, r, n_anc - 1] ^= 1
            else:
                D[hit, r, k - 1] ^= 1
                D[hit, r, k] ^= 1
        if r + 1 < n_layers:
            for j in range(n_anc):
                hit = rng.random(shots) < np.where(z[:, r, j], p_hi[r, j], p_lo[r, j])
                D[hit, r, j] ^= 1
                D[hit, r + 1, j] ^= 1
    return D


def solve_gamma(ps, pt, pi_seed, L, rho, w, target, shots=4_000, lo=0.2, hi=8.0, iters=16):
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        m = float(sample_spatial(ps, pt, shots, np.random.default_rng(31),
                                 pi_seed, L, rho, w, gamma=mid).sum(axis=(1, 2)).mean())
        if m < target:
            lo = mid
        else:
            hi = mid
    g = 0.5 * (lo + hi)
    return g, bool(g >= 0.99 * 8.0 or g <= 1.01 * 0.2)


def evaluate(ps, pt, dev, dev_c, w_, t_, n, seed, pi_seed, L, rho, w):
    gam, sat = solve_gamma(ps, pt, pi_seed, L, rho, w, dev_c["mean"])
    D = sample_spatial(ps, pt, n, np.random.default_rng(seed), pi_seed, L, rho, w, gamma=gam)
    c = disagreement_curve(D, w_, t_)
    return dict(pi_seed=pi_seed, L=L, rho=rho, w=w, gamma=gam, gamma_saturated=sat,
                E=endpoint(dev, c, N_HALF), curve={str(k): v for k, v in c.items()},
                counts=counts(D))


def main():
    t0 = time.time()
    syn, fin, _ = load_arm(0)
    Ddev = build_detectors(syn, fin)
    wt, tt, ps, pt = fit_weights_v2(Ddev[:N_HALF], n_fit=N_HALF)
    with open(os.path.join(ROOT, "data", "persistent_noise_model.json")) as fh:
        st1 = json.load(fh)
    dev = {int(k): v for k, v in st1["device_curve"].items()}
    dev_c, e0 = st1["device_counts"], st1["E_M0"]

    print("ROUND 13 -- DOES SPATIAL EXTENT CLOSE THE REMAINDER?  0 QPU")
    print("  w = 1 is M2 exactly; w = 8 is the chip-wide placebo; the hypothesis is w strictly "
          "between\n")
    e1 = endpoint(dev, disagreement_curve(
        sample_heterogeneous(ps, pt, N_SEARCH, np.random.default_rng(1), GAMMA_SHAPE), wt, tt),
        N_HALF)

    n_cells = len(PI_SEED) * len(LS) * len(RHOS) * len(WS)
    print(f"  searching {n_cells} cells ...", flush=True)
    rows, best, by_w = [], None, {}
    for pi in PI_SEED:
        for L in LS:
            for rho in RHOS:
                for w in WS:
                    r = evaluate(ps, pt, dev, dev_c, wt, tt, N_SEARCH, 7, pi, L, rho, w)
                    rows.append(r)
                    if r["gamma_saturated"]:
                        continue
                    if w not in by_w or r["E"] < by_w[w]["E"]:
                        by_w[w] = r
                    if best is None or r["E"] < best["E"]:
                        best = r
        print(f"    pi_seed={pi}: best E/E0 = {best['E'] / e0:.4f} at L={best['L']}, "
              f"rho={best['rho']}, w={best['w']}, b1={best['curve']['1'] * 1e4:.2f}e-4", flush=True)

    print("\n  BEST AT EACH SPATIAL WIDTH -- the interpolation between the two placebos:")
    print(f"    {'w':>3} {'E/E0':>8} {'b1 (e-4)':>10}   device b1 = {dev[1] * 1e4:.2f}e-4")
    for w in WS:
        if w in by_w:
            print(f"    {w:>3} {by_w[w]['E'] / e0:>8.4f} {by_w[w]['curve']['1'] * 1e4:>10.2f}")
        else:
            print(f"    {w:>3} {'--':>8} {'(all cells saturated)':>10}")

    print(f"\n  selected: pi_seed={best['pi_seed']}, L={best['L']}, rho={best['rho']}, "
          f"w={best['w']};  confirming at {N_FINAL:,} shots ...", flush=True)
    full = evaluate(ps, pt, dev, dev_c, wt, tt, N_FINAL, 11,
                    best["pi_seed"], best["L"], best["rho"], best["w"])
    print(f"    E/E0 = {full['E'] / e0:.4f}")
    print("    device:  " + "  ".join(f"b{b}={dev[b] * 1e4:6.2f}" for b in B_GRID))
    print("    model :  " + "  ".join(f"b{b}={full['curve'][str(b)] * 1e4:6.2f}" for b in B_GRID))
    print(f"    counts: mean {full['counts']['mean']:.3f} (device {dev_c['mean']:.3f}), "
          f"var/mean {full['counts']['var_over_mean']:.3f} (device {dev_c['var_over_mean']:.3f})")

    dev_fall = dev[1] / max(dev[16], 1e-12)
    mod_fall = full["curve"]["1"] / max(full["curve"]["16"], 1e-12)
    ends = [by_w[x]["E"] for x in (1, max(WS)) if x in by_w]
    g = {
        "1. E <= 0.10 * E(M0)": full["E"] <= 0.10 * e0,
        "2. E <= 0.25 * E(M1)": full["E"] <= 0.25 * e1,
        "3. curve falls within 2x of the device's": 0.5 <= mod_fall / dev_fall <= 2.0,
        "4. detector-count mean within 5%":
            abs(full["counts"]["mean"] - dev_c["mean"]) / dev_c["mean"] <= 0.05,
        "5. count variance not degraded":
            full["counts"]["var_over_mean"] >= 0.7 * dev_c["var_over_mean"],
        "6. beats BOTH endpoints w=1 and w=n_anc": all(full["E"] < e for e in ends) and len(ends) == 2,
        "7. no appended zero-observable lag edges": True,
        "8. gamma not saturated": not full["gamma_saturated"],
    }
    print("\n" + "=" * 96)
    print(f"  THE GATE (2b4399b).  E/E0 = {full['E'] / e0:.4f}, device fall {dev_fall:.1f}x "
          f"vs model {mod_fall:.1f}x")
    for k, v in g.items():
        print(f"    {'PASS' if v else 'FAIL'}  {k}")
    passed = all(g.values())
    print(f"\n  FALSIFIER 14: "
          f"{'does NOT fire -- spatial extent closes it' if passed else 'FIRES -- spatial extent is not the missing ingredient either'}")

    out = os.path.join(ROOT, "data", "spatial_persistent.json")
    tmp = out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(dict(E_M0=e0, E_M1=e1, device_curve={str(k): v for k, v in dev.items()},
                       device_counts=dev_c, pi_seed_grid=PI_SEED, L_grid=LS, rho_grid=RHOS,
                       w_grid=WS, n_search=N_SEARCH, n_final=N_FINAL,
                       best_by_width={str(k): v for k, v in by_w.items()},
                       selected=best, full=full, search=rows,
                       gate={k: bool(v) for k, v in g.items()}, gate_passed=bool(passed),
                       falsifier14_fired=bool(not passed)), fh, indent=1)
    os.replace(tmp, out)
    print(f"\nwrote {out}   ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
