#!/usr/bin/env python
"""Is the E1 feature inside the affine span of the E2 features? 0 QPU. Answered exactly.

The non-implication argument in `app_certificate` needs one premise: that the per-shot
disagreement indicator

    f(D) = 1{ Joint(D) != Split_b(D) }

is NOT an affine function of the parity features the E2 statistic reads. Those features are the
singleton detectors `D_i` and the same-stabiliser parities `D_i XOR D_j`. Because

    D_i XOR D_j = D_i + D_j - 2 D_i D_j,

the affine span of {1, D_i, D_i XOR D_j} equals the span of {1, D_i, D_i D_j} over the
same-stabiliser pairs, so the question is whether `f` is a real polynomial of degree <= 2 whose
quadratic terms are confined to same-stabiliser pairs.

WHY THIS IS A DIRECT SOLVE AND NOT AN ITERATIVE ONE. A first version ran LSQR on all 24,896 distinct
records against 10,609 features and reported the residual it reached. Every run stopped at its
iteration limit, and LSQR's residual decreases monotonically, so a non-converged residual is an UPPER
bound on the achievable residual and proves nothing about the minimum. Raising the limit far enough
to converge would take hours per context.

The fix uses a fact about restriction. Let S be any subset of the records. If no affine combination
of the features equals `f` on S, then none equals it on any superset of S, because restriction to S
is a linear map and equality on the superset would force equality on S. So it is enough to answer the
question on a subset -- and a subset only slightly larger than the feature count can be solved
DIRECTLY, by a rank-revealing least squares, where the residual is the true minimum to machine
precision rather than whatever an iteration happened to reach.

A first direct run took only 12,000 distinct records, just above the 10,609 features. That was a
mistake of a different kind: with the record count so close to the feature count the design matrix
has rank near 10,000 and its orthogonal complement is only about 2,000 dimensions wide, so a sparse
target lands in the span for reasons of dimension rather than for any reason about the decoder. Four
of the eight cells returned a residual of exactly zero under that setting.

This version therefore uses EVERY distinct evaluation record, roughly 24,000 against 10,609 features,
which leaves a complement of some 14,000 dimensions. A residual bounded away from zero there is a
statement about the decoder and not about the shape of the matrix.

Reported per context, on the evaluation split, at the two pre-registered widths.

    residual 0 would mean the premise fails and the non-implication argument does not apply.
    residual bounded away from 0 establishes it on that subset, hence on the whole record set.

Usage:  CAMPAIGN_V_EPOCH=E1 python tools/affine_span.py
"""
import glob
import json
import os
import sys
import time

import numpy as np
from scipy.linalg import lstsq as sp_lstsq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detectors import build_detectors  # noqa: E402
from parallel_window import two_window  # noqa: E402
from weight_model_robustness import fit_weights_v2  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPOCH = os.environ.get("CAMPAIGN_V_EPOCH", "E1")
INDIR = os.path.join(ROOT, "data", f"campaign_v_{EPOCH.lower()}")
OUT = os.path.join(ROOT, "data", "affine_span.json")
WIDTHS = (1, 2)
N_SUB = None            # None = every distinct record; the complement must be large
                        # enough that membership is not forced by dimension alone
RCOND = 1e-10           # singular values below this fraction of the largest are treated as zero


def load_L0():
    out = {}
    for f in sorted(glob.glob(os.path.join(INDIR, "pub*.npz")),
                    key=lambda p: int(os.path.basename(p)[3:-4])):
        z = np.load(f, allow_pickle=True)
        m = json.loads(str(z["meta"]))
        if m["logical_state"] != 0:
            continue
        out.setdefault(m["region"], []).append(build_detectors(z["syn"], z["fin"]))
    return out


def design(D):
    """[1, D_i, D_i XOR D_j over same-stabiliser pairs], dense, one row per record."""
    n, n_layers, n_anc = D.shape
    X = D.reshape(n, -1).astype(np.float32)
    ii, jj = np.triu_indices(n_layers, k=1)
    cols = [np.ones((n, 1), dtype=np.float32), X]
    for a in range(n_anc):
        idx = np.arange(n_layers) * n_anc + a
        Xa = X[:, idx]
        cols.append(np.abs(Xa[:, ii] - Xa[:, jj]))
    return np.hstack(cols)


def distinct_index(ev):
    flat = np.ascontiguousarray(ev.reshape(len(ev), -1).astype(np.uint8))
    keys = flat.view(np.dtype((np.void, flat.shape[1])))
    _, first = np.unique(keys, return_index=True)
    return np.sort(first)


def main():
    t0 = time.time()
    arms = load_L0()
    results = {}
    for region, ds in sorted(arms.items()):
        if len(ds) < 2:
            continue
        fit, ev = ds[0], ds[1]
        w, t, _, _ = fit_weights_v2(fit, n_fit=len(fit))
        uniq = distinct_index(ev)
        sub = uniq if N_SUB is None else uniq[:N_SUB]
        A = design(ev[sub]).astype(np.float64)
        n_rows, p = A.shape
        for b in WIDTHS:
            f = np.asarray(two_window(ev, 25, b, logical_j=0, eps=1e-3, seed=0,
                                      wtab=w, ttab=t)["diverged_repaired"], dtype=np.float64)
            fu = f[sub]
            # gelsy is a complete orthogonal factorisation: it returns the exact minimum-norm
            # least-squares solution, and its residual is the true minimum. gelsd would give the
            # same answer through an SVD and takes far longer on a matrix this shape.
            sol, res, rank, sv = sp_lstsq(A, fu, cond=RCOND, lapack_driver="gelsy",
                                          overwrite_a=False, check_finite=False)
            r = fu - A @ sol
            resid = float(np.linalg.norm(r))
            # the residual is orthogonal to the column space at the true minimum; report the
            # departure from that, so the reader can see the solve actually reached the optimum
            orth = float(np.linalg.norm(A.T @ r) / max(np.linalg.norm(A) * resid, 1e-300))
            ss_tot = float(np.linalg.norm(fu - fu.mean()))
            r2 = 1.0 - (resid / ss_tot) ** 2 if ss_tot > 0 else float("nan")
            key = f"{EPOCH}/{region}/b{b}"
            results[key] = dict(
                n_distinct_total=int(len(uniq)), n_rows=int(n_rows), n_features=int(p),
                rank=int(rank), n_disagreements=int(fu.sum()),
                residual_l2=resid, total_l2=ss_tot, r_squared=r2,
                orthogonality=orth, solver="scipy.linalg.lstsq (LAPACK gelsy)", rcond=RCOND)
            print(f"  {key}: rows {n_rows:,}  features {p:,}  rank {rank:,}  "
                  f"disagreements {int(fu.sum())}  residual {resid:.4f}  R2 {r2:.4f}  "
                  f"orthogonality {orth:.2e}")
        del A

    prev = {}
    if os.path.exists(OUT):
        with open(OUT) as fh:
            prev = json.load(fh)
    prev.update(results)
    prev["_note"] = ("exact least-squares residual of the E1 disagreement indicator against the "
                     "affine span of the E2 parity features, on every distinct evaluation record of each "
                     "context, by a direct rank-revealing solve")
    tmp = OUT + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(prev, fh, indent=1)
    os.replace(tmp, OUT)
    print(f"\nwrote {OUT} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
