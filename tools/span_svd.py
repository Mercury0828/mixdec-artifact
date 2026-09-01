#!/usr/bin/env python
"""Singular values of the design matrix, by SVD, for every context. 0 QPU.

`span_certificate.py` estimates the smallest retained singular value by deflated inverse power
iteration on the triangular factor. A cross-check on one context showed that estimate to be
contaminated by residual weight along the deflated null direction: it reported 2.57e-7 where the
singular value decomposition gives 5.11e-4, pessimistic by about 2000x.

A condition number is a property of the matrix, so the number the paper prints should be the one the
decomposition gives, with the contaminated value kept only as a deliberately pessimistic stress
denominator. This tool supplies the decomposition-based quantities for all four contexts.

The singular values depend only on the design matrix, not on the target, so one decomposition per
context serves both buffer widths.

Usage:  CAMPAIGN_V_EPOCH=E1 python tools/span_svd.py
"""
import glob
import json
import os
import sys
import time

import numpy as np
from scipy.linalg import svdvals

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detectors import build_detectors  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPOCH = os.environ.get("CAMPAIGN_V_EPOCH", "E1")
INDIR = os.path.join(ROOT, "data", f"campaign_v_{EPOCH.lower()}")
OUT = os.path.join(ROOT, "data", "span_svd.json")
RCOND = 1e-10


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


def main():
    t0 = time.time()
    arms = load_L0()
    results = {}
    for region, ds in sorted(arms.items()):
        if len(ds) < 2:
            continue
        ev = ds[1]
        flat = np.ascontiguousarray(ev.reshape(len(ev), -1).astype(np.uint8))
        _, first = np.unique(flat.view(np.dtype((np.void, flat.shape[1]))), return_index=True)
        sub = np.sort(first)
        A = design(ev[sub]).astype(np.float64)
        scale = np.linalg.norm(A, axis=0)
        scale[scale == 0] = 1.0
        B = A / scale
        del A
        print(f"  {EPOCH}/{region}: decomposing {B.shape[0]:,} x {B.shape[1]:,} ...", flush=True)
        sv = svdvals(B, check_finite=False)
        del B
        smax = float(sv[0])
        cut = smax * RCOND
        retained = sv[sv > cut]
        smin = float(retained[-1])
        discarded = sv[sv <= cut]
        key = f"{EPOCH}/{region}"
        results[key] = dict(
            sigma_max=smax, sigma_min_retained=smin, kappa_retained=smax / smin,
            rank=int(len(retained)), n_discarded=int(len(discarded)),
            largest_discarded=float(discarded[0]) if len(discarded) else 0.0,
            cutoff=float(cut), rcond=RCOND)
        print(f"    sigma_max {smax:.4e}  sigma_min_retained {smin:.4e}  "
              f"kappa {smax / smin:.3e}  rank {len(retained):,}  "
              f"discarded {len(discarded)} (largest "
              f"{float(discarded[0]) if len(discarded) else 0.0:.2e})", flush=True)

    prev = {}
    if os.path.exists(OUT):
        with open(OUT) as fh:
            prev = json.load(fh)
    prev.update(results)
    prev["_note"] = ("singular values of the column-scaled design matrix per context, by SVD; the "
                     "smallest retained value is the last above rcond times the largest")
    tmp = OUT + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(prev, fh, indent=1)
    os.replace(tmp, OUT)
    print(f"\nwrote {OUT} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
