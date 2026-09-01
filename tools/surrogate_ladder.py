#!/usr/bin/env python
"""Round 15: localise the discrepancy with NO MODEL AT ALL. Constrained randomisation. 0 QPU.

Pre-registered at `c1b1b7d`, `docs/expected.md` Round 15, falsifiers 18 and 19.

Rounds 12-14 tried to explain the device/model gap with a three-parameter persistent process and
failed twice under pre-registered tests. The objection that killed it -- the model was selected
against the very curve it was credited with reproducing -- cannot touch this round, because
**nothing here is fitted**. There is no parameter, no grid and no objective.

Each surrogate carries one kind of dependence over from the device VERBATIM and destroys the rest:

    S0  per-detector Bernoulli at the device's own per-position rates  -- the negative control
    S1  COLUMN shuffle: each ancilla column resampled independently across shots
    S2  LAYER shuffle: each layer resampled independently across shots
    S3  PAIR-column shuffle: adjacent column pairs resampled as units

`S1` preserves each ancilla's entire within-column temporal structure exactly -- non-Markov,
heavy-tailed, any order -- because each column *is* a real device column. It cannot fail through
mis-parameterisation. That is the whole point.

🔴 This is not a fifth model layer and does not break the `2b4399b` commitment: a surrogate has no
family to search, so there is nothing to search until something passes.

Usage:  python tools/surrogate_ladder.py
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_campaign_r import load_arm  # noqa: E402
from detectors import build_detectors  # noqa: E402
from persistent_noise_model import (B_GRID, N_HALF, counts, disagreement_curve,  # noqa: E402
                                    endpoint)
from syndrome_stats import all_stats  # noqa: E402
from weight_model_robustness import fit_weights_v2  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_SUR = 200_000


def s0_bernoulli(src, n, rng):
    """Per-position independent Bernoulli at the source's own empirical rates."""
    p = src.mean(axis=0, dtype=np.float64)
    return (rng.random((n, *p.shape)) < p).astype(np.uint8)


def _shuffle_axis(src, n, rng, axis, groups):
    """Resample whole slices along `axis` independently across shots, in the given groups."""
    out = np.empty((n, src.shape[1], src.shape[2]), dtype=np.uint8)
    for g in groups:
        idx = rng.integers(0, src.shape[0], n)
        if axis == 2:
            out[:, :, g] = src[np.ix_(idx, np.arange(src.shape[1]), np.asarray(g))]
        else:
            out[:, g, :] = src[np.ix_(idx, np.asarray(g), np.arange(src.shape[2]))]
    return out


def s1_column(src, n, rng):
    return _shuffle_axis(src, n, rng, 2, [[j] for j in range(src.shape[2])])


def s2_layer(src, n, rng):
    return _shuffle_axis(src, n, rng, 1, [[r] for r in range(src.shape[1])])


def s3_pair(src, n, rng):
    a = src.shape[2]
    groups = [list(range(j, min(j + 2, a))) for j in range(0, a, 2)]
    return _shuffle_axis(src, n, rng, 2, groups)


def self_test(verbose=True):
    """Ground truth: check each surrogate preserves and destroys exactly what it claims.

    Two injected sources with known structure:
      CROSS  column 3 is a verbatim copy of column 2  -> pure cross-column dependence
      WITHIN every column is constant along time      -> pure within-column temporal dependence

    S0 must destroy both. S1 must keep WITHIN and destroy CROSS. S2 must keep CROSS and destroy
    WITHIN. S3 must keep CROSS for columns inside one pair. Marginal rates must survive all four.
    """
    rng = np.random.default_rng(20260830)
    n, R, A = 5_000, 51, 8

    cross = np.zeros((n, R, A), np.uint8)
    cross[:, :, 2] = rng.random((n, R)) < 0.3
    cross[:, :, 3] = cross[:, :, 2]
    cross[:, :, 0] = rng.random((n, R)) < 0.1

    within = np.zeros((n, R, A), np.uint8)
    on = rng.random((n, A)) < 0.2
    for j in range(A):
        within[on[:, j], :, j] = 1

    def keep_cross(D):
        return float((D[:, :, 2] == D[:, :, 3]).mean())

    def keep_within(D):
        return float(np.all(D == D[:, :1, :], axis=1).mean())

    fns = dict(S0=s0_bernoulli, S1=s1_column, S2=s2_layer, S3=s3_pair)
    c = {k: keep_cross(f(cross, n, np.random.default_rng(1))) for k, f in fns.items()}
    wv = {k: keep_within(f(within, n, np.random.default_rng(2)))
          for k, f in fns.items() if k != "S3"}
    rate = {k: float(f(cross, n, np.random.default_rng(3))[:, :, 0].mean())
            for k, f in fns.items()}
    src_rate = float(cross[:, :, 0].mean())

    checks = {
        "S0 destroys cross-column": c["S0"] < 0.7,
        "S1 destroys cross-column": c["S1"] < 0.7,
        "S2 preserves cross-column exactly": c["S2"] > 0.999,
        "S3 preserves cross-column within a pair": c["S3"] > 0.999,
        "S0 destroys within-column time": wv["S0"] < 0.01,
        "S1 preserves within-column time exactly": wv["S1"] > 0.999,
        "S2 destroys within-column time": wv["S2"] < 0.01,
        "all four preserve marginal rates to 5%":
            all(abs(v / src_rate - 1) < 0.05 for v in rate.values()),
    }
    if verbose:
        print("surrogate_ladder self-test")
        print("  cross-column kept:   " + "  ".join(f"{k}={v:.4f}" for k, v in c.items())
              + "   (source 1.0000)")
        print("  within-column kept:  " + "  ".join(f"{k}={v:.4f}" for k, v in wv.items())
              + "   (source 1.0000)")
        print("  marginal rate:       " + "  ".join(f"{k}={v:.4f}" for k, v in rate.items())
              + f"   (source {src_rate:.4f})")
        for k, v in checks.items():
            print("  %s  %s" % ("PASS" if v else "FAIL", k))
    return all(checks.values()), checks


def main():
    t0 = time.time()
    syn, fin, _ = load_arm(0)
    Ddev = build_detectors(syn, fin)
    fit, ev = Ddev[:N_HALF], Ddev[N_HALF:2 * N_HALF]
    w, t, _, _ = fit_weights_v2(fit, n_fit=N_HALF)

    print("ROUND 15 -- MODEL-FREE LOCALISATION BY CONSTRAINED RANDOMISATION.  0 QPU")
    print("  nothing is fitted; each surrogate carries one dependence over verbatim\n")
    ok, _ = self_test(verbose=True)
    if not ok:
        raise RuntimeError("surrogate self-test FAILED; the ladder is void")
    print()

    dev = disagreement_curve(ev, w, t)
    dev_c, dev_s = counts(ev), all_stats(ev)
    e0_ref = endpoint(dev, {b: 0.0 for b in B_GRID}, N_HALF)
    print("  device (held-out 50,000):  "
          + "  ".join(f"b{b}={dev[b] * 1e4:6.2f}" for b in B_GRID))
    print(f"  device counts: mean {dev_c['mean']:.3f}, var/mean {dev_c['var_over_mean']:.3f}\n")

    # 🔴 The surrogates are built from the FIT half, so the device curve they are compared against
    # is on shots none of them contain.
    rng = np.random.default_rng(20260830)
    out, order = {}, [("S0 bernoulli", s0_bernoulli), ("S1 column", s1_column),
                      ("S2 layer", s2_layer), ("S3 pair-column", s3_pair)]
    for tag, fn in order:
        D = fn(fit, N_SUR, rng)
        c = disagreement_curve(D, w, t)
        st = all_stats(D)
        E = endpoint(dev, c, N_HALF)
        out[tag] = dict(curve={str(k): v for k, v in c.items()}, E=E, E_over_E0=E / e0_ref,
                        counts=counts(D), s3_autocorr=st["s3_autocorr"],
                        s5_pair_autocorr=st["s5_pair_autocorr"],
                        frac_of_device_b1=c[1] / dev[1])
        print(f"  {tag:<16} " + "  ".join(f"b{b}={c[b] * 1e4:6.2f}" for b in B_GRID))
        print(f"  {'':<16} E/E0 = {E / e0_ref:.4f},  b1 is {100 * c[1] / dev[1]:.1f}% of the "
              f"device's,  count mean {st['s1_count_mean']:.2f}, var/mean "
              f"{st['s2_var_over_mean']:.3f}", flush=True)

    f19 = out["S0 bernoulli"]["curve"]["1"] > 3.0 / N_SUR
    f18 = out["S1 column"]["frac_of_device_b1"] < 0.50
    print("\n" + "=" * 96)
    print(f"  FALSIFIER 19 (negative control): "
          f"{'FIRES -- the construction itself generates disagreement; ladder VOID' if f19 else 'does NOT fire'}"
          f"  -- S0 b1 = {out['S0 bernoulli']['curve']['1'] * 1e4:.2f}e-4")
    if not f19:
        print(f"  FALSIFIER 18 (localisation): "
              f"{'FIRES -- within-ancilla temporal dependence does NOT carry the effect' if f18 else 'does NOT fire -- the effect IS carried by within-ancilla temporal dependence'}")
        print(f"     S1 reproduces {100 * out['S1 column']['frac_of_device_b1']:.1f}% of the "
              f"device's b=1 rate, against the 50% bar")
        print(f"     S3 (pairs) reproduces "
              f"{100 * out['S3 pair-column']['frac_of_device_b1']:.1f}%,  "
              f"S2 (layers, temporal destroyed) {100 * out['S2 layer']['frac_of_device_b1']:.1f}%")

    path = os.path.join(ROOT, "data", "surrogate_ladder.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(dict(n_surrogate=N_SUR, device_curve={str(k): v for k, v in dev.items()},
                       device_counts=dev_c, device_s3=dev_s["s3_autocorr"],
                       device_s5=dev_s["s5_pair_autocorr"], E_zero_reference=e0_ref,
                       surrogates=out, bar_frac=0.50,
                       falsifier18_fired=bool(f18), falsifier19_fired=bool(f19),
                       self_test_passed=bool(ok)), fh, indent=1)
    os.replace(tmp, path)
    print(f"\nwrote {path}   ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
