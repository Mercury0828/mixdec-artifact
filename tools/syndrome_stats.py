#!/usr/bin/env python
"""Syndrome-only statistics: the estimators round 14 runs identically on device and on model traces.

Pre-registered at `a8dc617` / amended `c9d69f0`, `docs/expected.md` Round 14.

Nothing here looks at a decoder. That is the whole point: round 14 part A fits `M2` using ONLY these
statistics and then predicts the decoder observable it never saw, which is the answer to the
circularity objection raised against rounds 12-13.

    s1  detector-count mean                          -- solved exactly by `gamma`
    s2  detector-count variance / mean
    s3  same-ancilla detector autocorrelation, lags 1..24            (the TIME signature)
    s4  across-ancilla sd of per-ancilla detector rate
    s5  autocorrelation of the neighbouring-pair coincidence          (the SPACE signature)

`s3` and `s5` are Pearson correlations of binary variables, pooled over the layer and ancilla
positions at which they are estimable. `s5` distinguishes a persistent MEASUREMENT fault on one
ancilla, which does not sustain a neighbouring-pair coincidence, from a persistent DATA-QUBIT fault,
which does -- the hypothesis arXiv:2512.10814 section IV.E.3 advances for its TLS-like events.
"""
import numpy as np

MAX_LAG = 24


def _pearson_lag(X, max_lag=MAX_LAG):
    """Pooled binary Pearson correlation between position `r` and `r + k`, for k = 1..max_lag.

    `X` is (shots, positions, columns) float32. Returns a list of length `max_lag`.
    """
    n_shots, n_pos, _ = X.shape
    out = []
    for k in range(1, max_lag + 1):
        if k >= n_pos:
            out.append(float("nan"))
            continue
        a, b = X[:, : n_pos - k, :], X[:, k:, :]
        ma = a.mean(axis=0, dtype=np.float64)
        mb = b.mean(axis=0, dtype=np.float64)
        mab = np.einsum("slc,slc->lc", a, b, optimize=True) / n_shots
        den = np.sqrt(ma * (1 - ma) * mb * (1 - mb))
        ok = den > 0
        out.append(float(((mab - ma * mb)[ok] / den[ok]).mean()) if ok.any() else float("nan"))
    return out


def autocorr_same_ancilla(D, max_lag=MAX_LAG):
    """`s3` -- the TIME signature. `D` is (shots, layers, ancillas) uint8."""
    return _pearson_lag(np.ascontiguousarray(D, dtype=np.float32), max_lag)


def autocorr_neighbour_pair(D, max_lag=MAX_LAG):
    """`s5` -- the SPACE signature: autocorrelation of `P[j, r] = d[j, r] AND d[j+1, r]`."""
    P = (D[:, :, :-1] & D[:, :, 1:]).astype(np.float32)
    return _pearson_lag(np.ascontiguousarray(P), max_lag)


def all_stats(D, max_lag=MAX_LAG):
    c = D.sum(axis=(1, 2)).astype(np.float64)
    per_anc = D.mean(axis=(0, 1), dtype=np.float64)
    return dict(
        s1_count_mean=float(c.mean()),
        s2_var_over_mean=float(c.var() / c.mean()),
        s3_autocorr=autocorr_same_ancilla(D, max_lag),
        s4_ancilla_rate_sd=float(per_anc.std()),
        s4_ancilla_rates=[float(v) for v in per_anc],
        s5_pair_autocorr=autocorr_neighbour_pair(D, max_lag),
        detector_rate=float(D.mean(dtype=np.float64)),
        n_shots=int(D.shape[0]),
    )


def syndrome_distance(model, device):
    """`D_syn`, the pre-registered selection objective. Uses s2, s3, s4 ONLY -- never a decoder.

    `s1` is not in it: it is solved exactly by `gamma`, so including it would score a constraint
    that is satisfied by construction.
    """
    a = np.asarray(model["s3_autocorr"], dtype=float)
    b = np.asarray(device["s3_autocorr"], dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    curve = float(np.linalg.norm(a[ok] - b[ok]) / max(np.linalg.norm(b[ok]), 1e-12))
    disp = abs(model["s2_var_over_mean"] / max(device["s2_var_over_mean"], 1e-12) - 1.0)
    het = abs(model["s4_ancilla_rate_sd"] / max(device["s4_ancilla_rate_sd"], 1e-12) - 1.0)
    return dict(total=curve + disp + het, curve=curve, dispersion=disp, heterogeneity=het)


def self_test(verbose=True):
    """Ground-truth validation: inject known faults and check the estimators recover them.

    Three cases, all with RANDOM episode start and RANDOM column per shot -- a fault at a FIXED
    position would produce zero across-shot correlation and is not what these estimators are for.

      iid          neither signature fires
      TIME  fault  persistent single-ancilla measurement error  -> s3 fires, s5 does not
      SPACE fault  persistent data-qubit error flipping both     -> both fire, s5 much harder
                   neighbours at the same round

    The SPACE case is the hypothesis arXiv:2512.10814 section IV.E.3 advances for its TLS-like
    events. `s5` is what tells the two apart on the device, and this test is what licenses reading
    it that way.
    """
    import numpy as np
    S, R, A, WIDTH = 20_000, 51, 8, 16
    rng = np.random.default_rng(20260830)

    def _base():
        return (rng.random((S, R, A)) < 0.03).astype(np.uint8)

    iid = all_stats(_base())

    D = _base()
    has = rng.random(S) < 0.4
    st, col = rng.integers(0, R - WIDTH, S), rng.integers(0, A, S)
    for i in np.flatnonzero(has):
        rows = np.arange(st[i], st[i] + WIDTH)[rng.random(WIDTH) < 0.45]
        D[i, rows, col[i]] ^= 1
    time_fault = all_stats(D)

    D = _base()
    has = rng.random(S) < 0.4
    st, col = rng.integers(0, R - WIDTH, S), rng.integers(0, A - 1, S)
    for i in np.flatnonzero(has):
        rows = np.arange(st[i], st[i] + WIDTH)[rng.random(WIDTH) < 0.45]
        D[i, rows, col[i]] ^= 1
        D[i, rows, col[i] + 1] ^= 1
    space_fault = all_stats(D)

    def zero_cross(a):
        a = np.asarray(a)
        neg = np.flatnonzero(a <= 0)
        return int(neg[0] + 1) if neg.size else -1

    checks = {
        "iid: s3 flat": abs(iid["s3_autocorr"][0]) < 0.01,
        "iid: s5 flat": abs(iid["s5_pair_autocorr"][0]) < 0.01,
        "TIME fault raises s3": time_fault["s3_autocorr"][0] > 0.02,
        "TIME fault leaves s5 near zero": abs(time_fault["s5_pair_autocorr"][0]) < 0.01,
        "SPACE fault raises s5 hard": space_fault["s5_pair_autocorr"][0] > 0.1,
        "s5 separates SPACE from TIME by >= 10x":
            space_fault["s5_pair_autocorr"][0] > 10 * abs(time_fault["s5_pair_autocorr"][0]),
        "s3 zero-crossing recovers the injected width (16 +/- 2)":
            abs(zero_cross(time_fault["s3_autocorr"]) - WIDTH) <= 2,
    }
    if verbose:
        print("syndrome_stats self-test -- injected width = %d rounds" % WIDTH)
        print("  iid          s3[1] = %+.4f   s5[1] = %+.4f"
              % (iid["s3_autocorr"][0], iid["s5_pair_autocorr"][0]))
        print("  TIME  fault  s3[1] = %+.4f   s5[1] = %+.4f   s3 zero-crossing at lag %d"
              % (time_fault["s3_autocorr"][0], time_fault["s5_pair_autocorr"][0],
                 zero_cross(time_fault["s3_autocorr"])))
        print("  SPACE fault  s3[1] = %+.4f   s5[1] = %+.4f   s5 zero-crossing at lag %d"
              % (space_fault["s3_autocorr"][0], space_fault["s5_pair_autocorr"][0],
                 zero_cross(space_fault["s5_pair_autocorr"])))
        for k, v in checks.items():
            print("  %s  %s" % ("PASS" if v else "FAIL", k))
    return all(checks.values()), checks, dict(iid=iid, time_fault=time_fault,
                                            space_fault=space_fault, width=WIDTH)


if __name__ == "__main__":
    import sys
    ok, _, _ = self_test()
    print("\nself-test %s" % ("PASSED" if ok else "FAILED"))
    sys.exit(0 if ok else 1)
