#!/usr/bin/env python
"""A condition-aware certificate for the affine-span premise, both directions. 0 QPU.

`affine_span.py` reports the least-squares residual of the E1 disagreement indicator against the E2
parity features. A cross-model accuracy check pointed out that a residual plus a tiny relative
`||B^T r||` is not by itself a certificate: if `B` has a very small nonzero singular value, a
sizeable reducible component of `r` along the matching left singular vector produces a small
`B^T r` while the residual is still reducible. This tool closes that gap and supplies the two
witnesses the argument actually needs.

WHAT IT COMPUTES, per context.

  1. A QR factorisation of the column-scaled design `B`, from which
        sigma_max = ||B||_2   by power iteration on R^T R
        sigma_min = smallest singular value of B, by inverse power iteration on R
        kappa     = sigma_max / sigma_min
     R is the same for both buffer widths in a region, so the factorisation is done once.

  2. Per width, with r the residual reported by the direct solve,
        delta          = ||B^T r|| / sigma_min      an upper bound on the reducible part of r
        certified_lcb  = sqrt(||r||^2 - delta^2)    a lower bound on the exact minimum residual
     A positive `certified_lcb` is the statement the paper needs, and it holds regardless of how
     the solver reached `r`.

  3. The signed-measure witness itself. At the least-squares solution `h = r` satisfies
        sum_s h_s = 0                 because the intercept is a column of B
        sum_s h_s phi(s) = 0          because every feature is a column of B
        sum_s h_s f(s) = ||r||^2 > 0
     so `h` is the perturbation direction the non-identification argument perturbs along. The
     residuals of those three identities are reported rather than asserted.

  4. The REVERSE direction, which needs no least squares at all. E1's expectation determines the
     E2 feature expectations only if every feature is constant on each level set of `f`. Exhibiting
     two records with the same `f` and a different feature value refutes that outright, and moving
     mass between them changes an E2 moment while leaving `Pr[Delta_b]` alone. The tool finds such
     a pair and prints it.

Usage:  CAMPAIGN_V_EPOCH=E1 python tools/span_certificate.py
"""
import glob
import json
import os
import sys
import time

import numpy as np
from scipy.linalg import qr, solve_triangular
from scipy.linalg import lstsq as sp_lstsq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detectors import build_detectors  # noqa: E402
from parallel_window import two_window  # noqa: E402
from weight_model_robustness import fit_weights_v2  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPOCH = os.environ.get("CAMPAIGN_V_EPOCH", "E1")
INDIR = os.path.join(ROOT, "data", f"campaign_v_{EPOCH.lower()}")
OUT = os.path.join(ROOT, "data", "span_certificate.json")
WIDTHS = (1, 2)
RCOND = 1e-10
POWER_ITERS = 300
MAX_DEFLATIONS = 8      # a rank deficiency of one was observed; allow a few


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


def sigma_extremes(R, rng):
    """Largest and smallest singular values of B, from its triangular factor R."""
    n = R.shape[1]
    v = rng.standard_normal(n)
    v /= np.linalg.norm(v)
    smax = 0.0
    for _ in range(POWER_ITERS):                       # power iteration on R^T R
        w = R @ v
        v = R.T @ w
        nv = np.linalg.norm(v)
        if nv == 0:
            break
        v /= nv
        smax = np.sqrt(nv)
    # The SMALLEST singular value is the wrong quantity when B is rank deficient: a null direction
    # gives sigma ~ 0 and the bound below collapses, although a null direction cannot hide a smaller
    # residual, since B times it is zero and it moves neither B x nor B^T r. What the bound needs is
    # the smallest RETAINED singular value, so directions below the rank cutoff are deflated away
    # and the iteration is restarted on the orthogonal complement of what it has already found.
    cutoff = smax * RCOND
    deflated = []
    smin = 0.0
    for _ in range(MAX_DEFLATIONS):
        u = rng.standard_normal(n)
        for d in deflated:
            u -= (d @ u) * d
        nu = np.linalg.norm(u)
        if nu == 0:
            break
        u /= nu
        sval = 0.0
        for _ in range(POWER_ITERS):
            y = solve_triangular(R, u, trans="T", lower=False, check_finite=False)
            z = solve_triangular(R, y, trans="N", lower=False, check_finite=False)
            for d in deflated:                         # keep the iterate off the found directions
                z -= (d @ z) * d
            nz = np.linalg.norm(z)
            if not np.isfinite(nz) or nz == 0:
                break
            u = z / nz
            sval = 1.0 / np.sqrt(nz)
        if sval > cutoff:
            smin = sval
            break
        deflated.append(u.copy())
    return float(smax), float(smin), len(deflated), float(cutoff)


def reverse_witness(A, fu):
    """Two records with the same f and a different feature value. Refutes E1 -> E2 outright."""
    for lab in (0.0, 1.0):
        rows = np.flatnonzero(fu == lab)
        if len(rows) < 2:
            continue
        head = A[rows[0]]
        for r in rows[1:]:
            d = np.flatnonzero(A[r] != head)
            if len(d):
                return dict(f_value=float(lab), record_a=int(rows[0]), record_b=int(r),
                            feature_column=int(d[0]),
                            value_a=float(head[d[0]]), value_b=float(A[r][d[0]]),
                            n_features_differing=int(len(d)))
    return None


def main():
    t0 = time.time()
    arms = load_L0()
    results = {}
    for region, ds in sorted(arms.items()):
        if len(ds) < 2:
            continue
        fit, ev = ds[0], ds[1]
        w, t, _, _ = fit_weights_v2(fit, n_fit=len(fit))
        sub = distinct_index(ev)
        A = design(ev[sub]).astype(np.float64)
        scale = np.linalg.norm(A, axis=0)
        scale[scale == 0] = 1.0
        B = A / scale
        print(f"  {EPOCH}/{region}: factorising {B.shape[0]:,} x {B.shape[1]:,} ...", flush=True)
        # mode="r" returns an (m, n) array whose top n rows are the triangular factor
        R = np.ascontiguousarray(qr(B, mode="r", check_finite=False)[0][:B.shape[1], :])
        smax, smin, n_defl, cutoff = sigma_extremes(R, np.random.default_rng(7))
        kappa = smax / smin if smin > 0 else float("inf")
        print(f"    sigma_max {smax:.4e}  sigma_min+ {smin:.4e}  kappa+ {kappa:.3e}  "
              f"deflated {n_defl}  cutoff {cutoff:.2e}", flush=True)
        del R
        for b in WIDTHS:
            f = np.asarray(two_window(ev, 25, b, logical_j=0, eps=1e-3, seed=0,
                                      wtab=w, ttab=t)["diverged_repaired"], dtype=np.float64)
            fu = f[sub]
            sol, _, rank, _ = sp_lstsq(B, fu, cond=RCOND, lapack_driver="gelsy",
                                       check_finite=False)
            r = fu - B @ sol
            rn = float(np.linalg.norm(r))
            btr = float(np.linalg.norm(B.T @ r))
            delta = btr / smin if smin > 0 else float("inf")
            lcb = float(np.sqrt(max(rn * rn - delta * delta, 0.0)))
            key = f"{EPOCH}/{region}/b{b}"
            results[key] = dict(
                n_rows=int(B.shape[0]), n_features=int(B.shape[1]), rank=int(rank),
                n_disagreements=int(fu.sum()), rcond=RCOND,
                residual=rn, sigma_max=smax, sigma_min_retained=smin, kappa_retained=kappa,
                n_deflated=n_defl, rank_cutoff=cutoff,
                Bt_r_norm=btr, delta=delta, certified_residual_lcb=lcb,
                witness_sum=float(abs(r.sum())),
                witness_features=btr,
                witness_dot_f=float(r @ fu),
                witness_dot_f_minus_rsq=float(abs(r @ fu - rn * rn)))
            print(f"    b={b}: residual {rn:.4f}  ||B^T r|| {btr:.3e}  delta {delta:.3e}  "
                  f"certified >= {lcb:.4f}  h.f {float(r @ fu):.4f}", flush=True)
        rw = reverse_witness(A, np.asarray(two_window(
            ev, 25, 1, logical_j=0, eps=1e-3, seed=0, wtab=w, ttab=t)["diverged_repaired"],
            dtype=np.float64)[sub])
        results[f"{EPOCH}/{region}/reverse_witness"] = rw
        print(f"    reverse witness: {rw}", flush=True)
        del A, B

    prev = {}
    if os.path.exists(OUT):
        with open(OUT) as fh:
            prev = json.load(fh)
    prev.update(results)
    prev["_note"] = ("condition-aware certificate for the affine-span premise: the certified "
                     "residual lower bound is sqrt(||r||^2 - (||B^T r||/sigma_min)^2), which is "
                     "positive only if no reducible component of r can account for it; plus the "
                     "signed-measure witness h = r and a two-record witness for the reverse "
                     "direction")
    tmp = OUT + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(prev, fh, indent=1)
    os.replace(tmp, OUT)
    print(f"\nwrote {OUT} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
