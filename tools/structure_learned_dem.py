#!/usr/bin/env python
"""Round 16: does a STRUCTURE-LEARNED DEM also underpredict the decoder observable? 0 QPU.

Pre-registered at `12b1a4d`, `docs/expected.md` Round 16, falsifiers 20 and 21.

The cheapest reviewer objection to this project's headline is that only one weak model class was
tested. So here the baseline is completed the way arXiv:2512.10814 completes it on Google hardware:
estimate every pairwise detector correlation, and add an explicit two-detector edge wherever the
device shows more correlation than the independent model already explains.

    C0  independent fitted graph edges, fixed topology, pairwise pij       -- the existing baseline
    C1  C0 + every significant EXCESS pair, |d ancilla| <= 1, 1 <= |d layer| <= 24
    C2  C1 restricted to same-ancilla pairs

🔴 Every added edge flips exactly TWO detectors, so `C1`'s syndromes stay sums of graph edges. That
is what round 15's voided ladder got wrong -- it sampled detectors directly, produced syndromes no
error process could generate, and inflated the disagreement 27-fold. The property is asserted in
code here, not assumed.

🔴 Falsifier 21 is evaluated BEFORE falsifier 20: a comparator that is quietly broken would
manufacture exactly the conclusion this project wants.

Usage:  python tools/structure_learned_dem.py
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
from persistent_noise_model import (B_GRID, N_HALF, counts, disagreement_curve,  # noqa: E402
                                    endpoint)
from weight_model_robustness import fit_weights_v2  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_MODEL = 200_000
N_PIJ = 100_000          # shots used to estimate the C0 reference correlations
MAX_DLAYER = 24
MAX_DANC = 1
SE_THRESH = 4.0


def pair_list(n_layers, n_anc):
    """The pre-registered neighbourhood, as flat detector index pairs."""
    a, b = [], []
    for j1 in range(n_anc):
        for j2 in range(max(0, j1 - MAX_DANC), min(n_anc, j1 + MAX_DANC + 1)):
            for r1 in range(n_layers):
                for dr in range(1, MAX_DLAYER + 1):
                    r2 = r1 + dr
                    if r2 >= n_layers:
                        break
                    i1, i2 = r1 * n_anc + j1, r2 * n_anc + j2
                    if i1 < i2:
                        a.append(i1)
                        b.append(i2)
    return np.array(a), np.array(b)


def pij(X, ia, ib):
    """Standard pairwise excitation-rate estimator, vectorised over pairs. X is (shots, det)."""
    n = X.shape[0]
    m = X.mean(axis=0, dtype=np.float64)
    xa, xb = X[:, ia], X[:, ib]
    cab = np.einsum("sp,sp->p", xa, xb, optimize=True) / n
    ma, mb = m[ia], m[ib]
    num = cab - ma * mb
    # CORRECTED 2026-08-31: this divided by the product of the two SINGLETON polarizations,
    # where the inversion divides by the PAIR polarization. The two agree only where the
    # numerator vanishes, so the old form was near enough at the low rates this file's own
    # self-test injects and wrong at higher ones. No paper number came from here -- every
    # reported edge rate came from calibrated_weights.pij, which was always correct.
    den = 1 - 2 * ma - 2 * mb + 4 * cab
    inner = 1 - 4 * num / np.where(np.abs(den) < 1e-9, np.nan, den)
    p = 0.5 * (1 - np.sqrt(np.clip(inner, 0.0, 1.0)))
    se = np.sqrt(np.maximum(cab * (1 - cab), 1e-12) / n) / np.maximum(np.abs(den) / 2, 1e-9)
    return np.nan_to_num(p), se


def sample_with_pairs(base, ia, ib, rate, n, rng, shape):
    """`base` shots plus, for each listed pair, an independent flip of BOTH its detectors."""
    D = base.reshape(n, -1).copy()
    for k in np.flatnonzero(rate > 0):
        hit = rng.random(n) < rate[k]
        D[hit, ia[k]] ^= 1
        D[hit, ib[k]] ^= 1
    return D.reshape(shape)


def self_test(verbose=True):
    """Ground truth for the two pieces that could silently break the round.

    1. the pij estimator must recover an injected pair rate;
    2. the pair sampler must flip EXACTLY two detectors per firing -- the property whose absence
       voided round 15, where directly-sampled detectors produced syndromes no error process can
       generate and inflated the disagreement 27-fold.
    """
    n, R, A = 200_000, 51, 8
    ia, ib = pair_list(R, A)
    rng = np.random.default_rng(20260830)
    X = (rng.random((n, R * A)) < 0.04).astype(np.float32)
    true_rate, k = 0.02, 500
    hit = rng.random(n) < true_rate
    X[hit, ia[k]] = 1 - X[hit, ia[k]]
    X[hit, ib[k]] = 1 - X[hit, ib[k]]
    p, se = pij(X, ia[k:k + 1], ib[k:k + 1])

    # a HIGH-rate case as well. At 0.02 the two candidate denominators agree to three decimals, so
    # the low-rate check above passed against the wrong one. At 0.20 they do not.
    hi_rate, k2 = 0.20, 900
    Y = (rng.random((n, R * A)) < 0.04).astype(np.float32)
    hit2 = rng.random(n) < hi_rate
    Y[hit2, ia[k2]] = 1 - Y[hit2, ia[k2]]
    Y[hit2, ib[k2]] = 1 - Y[hit2, ib[k2]]
    p_hi, _ = pij(Y, ia[k2:k2 + 1], ib[k2:k2 + 1])

    base = np.zeros((2_000, R, A), np.uint8)
    rate = np.zeros(len(ia))
    rate[k] = 1.0
    D = sample_with_pairs(base, ia, ib, rate, 2_000, np.random.default_rng(1), base.shape)
    flips = set(int(v) for v in D.reshape(2_000, -1).sum(axis=1))

    checks = {
        "pij recovers a high injected pair rate, which discriminates the denominator":
            abs(float(p_hi[0]) - hi_rate) < 0.01,
        "pij recovers an injected pair rate within 3 SE":
            abs(p[0] - true_rate) < 3 * se[0],
        "the pair sampler flips EXACTLY two detectors": flips == {2},
    }
    if verbose:
        print("structure_learned_dem self-test")
        print(f"  neighbourhood holds {len(ia):,} pairs")
        print(f"  injected pair rate {true_rate:.4f} -> recovered {p[0]:.4f} +- {se[0]:.4f}")
        print(f"  detectors flipped per firing: {sorted(flips)}")
        for kk, v in checks.items():
            print("  %s  %s" % ("PASS" if v else "FAIL", kk))
    return all(checks.values()), checks


def main():
    t0 = time.time()
    syn, fin, _ = load_arm(0)
    Ddev = build_detectors(syn, fin)
    fit, ev = Ddev[:N_HALF], Ddev[N_HALF:2 * N_HALF]
    n_layers, n_anc = fit.shape[1], fit.shape[2]
    w, t, ps, pt = fit_weights_v2(fit, n_fit=N_HALF)

    print("ROUND 16 -- DOES A STRUCTURE-LEARNED DEM ALSO UNDERPREDICT?  0 QPU")
    print(f"  neighbourhood |d anc| <= {MAX_DANC}, 1 <= |d layer| <= {MAX_DLAYER}; "
          f"excess threshold {SE_THRESH} SE\n")
    ok, _ = self_test(verbose=True)
    if not ok:
        raise RuntimeError("self-test FAILED; the comparator is not trustworthy")
    print()

    dev = disagreement_curve(ev, w, t)
    dev_c = counts(ev)
    e0_ref = endpoint(dev, {b: 0.0 for b in B_GRID}, N_HALF)
    print("  device (held-out 50,000):  "
          + "  ".join(f"b{b}={dev[b] * 1e4:6.2f}" for b in B_GRID))

    ia, ib = pair_list(n_layers, n_anc)
    print(f"  {len(ia):,} pairs in the neighbourhood", flush=True)

    print("  estimating device pair rates ...", flush=True)
    p_dev, se_dev = pij(fit.reshape(N_HALF, -1).astype(np.float32), ia, ib)
    print("  estimating C0 reference pair rates ...", flush=True)
    D0 = sample_independent(ps, pt, N_PIJ, np.random.default_rng(0))
    p_c0, se_c0 = pij(D0.reshape(N_PIJ, -1).astype(np.float32), ia, ib)

    exc = p_dev - p_c0
    se = np.sqrt(se_dev ** 2 + se_c0 ** 2)
    sig = exc > SE_THRESH * se
    same_anc = (ia % n_anc) == (ib % n_anc)
    rate1 = np.where(sig, exc, 0.0)
    rate2 = np.where(sig & same_anc, exc, 0.0)
    print(f"  significant excess pairs: {int(sig.sum()):,} of {len(ia):,} "
          f"({int((sig & same_anc).sum()):,} same-ancilla, "
          f"{int((sig & ~same_anc).sum()):,} neighbour)")
    print(f"  total added edge rate per shot: C1 {rate1.sum():.3f}, C2 {rate2.sum():.3f}")

    out = {}
    for tag, rate in (("C0 baseline", None), ("C1 completed", rate1), ("C2 same-ancilla", rate2)):
        base = sample_independent(ps, pt, N_MODEL, np.random.default_rng(3))
        D = base if rate is None else sample_with_pairs(
            base, ia, ib, rate, N_MODEL, np.random.default_rng(4), base.shape)
        c = disagreement_curve(D, w, t)
        cc = counts(D)
        out[tag] = dict(curve={str(k): v for k, v in c.items()},
                        E=endpoint(dev, c, N_HALF), counts=cc,
                        frac_of_device_b1=c[1] / dev[1],
                        n_edges=0 if rate is None else int((rate > 0).sum()))
        print(f"\n  {tag:<16} " + "  ".join(f"b{b}={c[b] * 1e4:6.2f}" for b in B_GRID))
        print(f"  {'':<16} b1 is {100 * c[1] / dev[1]:.1f}% of the device's, "
              f"count mean {cc['mean']:.3f} (device {dev_c['mean']:.3f}), "
              f"var/mean {cc['var_over_mean']:.3f} ({dev_c['var_over_mean']:.3f})", flush=True)
        if rate is not None:
            # falsifier 21: does C1 reproduce the correlations it was built from?
            p_new, _ = pij(D[:N_PIJ].reshape(N_PIJ, -1).astype(np.float32), ia, ib)
            m = sig if tag.startswith("C1") else (sig & same_anc)
            ratio = p_new[m] / np.maximum(p_dev[m], 1e-12)
            out[tag]["recovery_median"] = float(np.median(ratio))
            out[tag]["recovery_frac_within_2x"] = float(np.mean((ratio >= 0.5) & (ratio <= 2.0)))
            print(f"  {'':<16} recovers the fitted pair rates: median ratio "
                  f"{np.median(ratio):.3f}, {100 * np.mean((ratio >= 0.5) & (ratio <= 2.0)):.1f}% "
                  f"within 2x")

    c1 = out["C1 completed"]
    f21 = (abs(c1["counts"]["mean"] - dev_c["mean"]) / dev_c["mean"] > 0.10
           or c1["recovery_frac_within_2x"] < 0.5)
    f20 = c1["frac_of_device_b1"] >= 0.50
    print("\n" + "=" * 96)
    print(f"  FALSIFIER 21 (construction control, read FIRST): "
          f"{'FIRES -- the comparator is not the model it claims to be; round VOID' if f21 else 'does NOT fire -- C1 is a competent structure-learned DEM'}")
    if not f21:
        print(f"  FALSIFIER 20: "
              f"{'FIRES -- the headline narrows to the pairwise-pij DEM specifically' if f20 else 'does NOT fire -- a structure-learned DEM ALSO underpredicts'}")
        print(f"     C1 reproduces {100 * c1['frac_of_device_b1']:.1f}% of the device's b=1 rate, "
              f"against the 50% bar")

    path = os.path.join(ROOT, "data", "structure_learned_dem.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(dict(device_curve={str(k): v for k, v in dev.items()}, device_counts=dev_c,
                       E_zero_reference=e0_ref, n_pairs=int(len(ia)),
                       n_significant=int(sig.sum()),
                       n_significant_same_ancilla=int((sig & same_anc).sum()),
                       max_dlayer=MAX_DLAYER, max_danc=MAX_DANC, se_thresh=SE_THRESH,
                       n_model=N_MODEL, comparators=out, bar_frac=0.50,
                       falsifier20_fired=bool(f20), falsifier21_fired=bool(f21),
                       self_test_passed=bool(ok)), fh, indent=1)
    os.replace(tmp, path)
    print(f"\nwrote {path}   ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
