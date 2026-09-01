#!/usr/bin/env python
"""Independent cross-checks on the span solve, for one representative context. 0 QPU.

The certificate in `span_certificate.py` rests on one LAPACK driver. This tool re-derives the same
residual three other ways, so the number does not depend on a single code path:

  1. `gelsy` at three rank cutoffs, 1e-12, 1e-10 and 1e-8. A residual that moves with the cutoff
     would mean the answer is a rank decision rather than a fact about the data.
  2. The residual recomputed from the returned coefficients as ||f - B x||, rather than taken from
     the solver's own report.
  3. `gelsd`, the SVD driver, on the same system. This is the expensive one and is run last, so a
     timeout still leaves the first two checks recorded.

Usage:  CAMPAIGN_V_EPOCH=E2 python tools/span_crosscheck.py [REGION]
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
EPOCH = os.environ.get("CAMPAIGN_V_EPOCH", "E2")
REGION = sys.argv[1] if len(sys.argv) > 1 else "R1"
INDIR = os.path.join(ROOT, "data", f"campaign_v_{EPOCH.lower()}")
OUT = os.path.join(ROOT, "data", "span_crosscheck.json")
WIDTH = 1
CUTOFFS = (1e-12, 1e-10, 1e-8)


def load_region():
    out = {}
    for f in sorted(glob.glob(os.path.join(INDIR, "pub*.npz")),
                    key=lambda p: int(os.path.basename(p)[3:-4])):
        z = np.load(f, allow_pickle=True)
        m = json.loads(str(z["meta"]))
        if m["logical_state"] != 0:
            continue
        out.setdefault(m["region"], []).append(build_detectors(z["syn"], z["fin"]))
    return out[REGION]


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
    ds = load_region()
    fit, ev = ds[0], ds[1]
    w, t, _, _ = fit_weights_v2(fit, n_fit=len(fit))
    flat = np.ascontiguousarray(ev.reshape(len(ev), -1).astype(np.uint8))
    _, first = np.unique(flat.view(np.dtype((np.void, flat.shape[1]))), return_index=True)
    sub = np.sort(first)
    A = design(ev[sub]).astype(np.float64)
    scale = np.linalg.norm(A, axis=0)
    scale[scale == 0] = 1.0
    B = A / scale
    del A
    f = np.asarray(two_window(ev, 25, WIDTH, logical_j=0, eps=1e-3, seed=0,
                              wtab=w, ttab=t)["diverged_repaired"], dtype=np.float64)[sub]
    print(f"{EPOCH}/{REGION}/b{WIDTH}: {B.shape[0]:,} x {B.shape[1]:,}, "
          f"{int(f.sum())} disagreements", flush=True)

    res = {"context": f"{EPOCH}/{REGION}/b{WIDTH}", "n_rows": int(B.shape[0]),
           "n_features": int(B.shape[1]), "n_disagreements": int(f.sum()), "gelsy": {}}
    for c in CUTOFFS:
        t1 = time.time()
        sol, _, rank, _ = sp_lstsq(B, f, cond=c, lapack_driver="gelsy", check_finite=False)
        recomputed = float(np.linalg.norm(f - B @ sol))
        res["gelsy"][f"{c:.0e}"] = dict(rank=int(rank), residual_recomputed=recomputed,
                                        seconds=round(time.time() - t1, 1))
        print(f"  gelsy cond={c:.0e}: rank {rank:,}  ||f - Bx|| {recomputed:.6f}  "
              f"{time.time() - t1:.0f}s", flush=True)

    tmp = OUT + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(res, fh, indent=1)
    os.replace(tmp, OUT)
    print(f"  wrote {OUT}", flush=True)

    print("  gelsd (SVD driver), this is the slow one ...", flush=True)
    t1 = time.time()
    sol, _, rank, sv = sp_lstsq(B, f, cond=1e-10, lapack_driver="gelsd", check_finite=False)
    recomputed = float(np.linalg.norm(f - B @ sol))
    res["gelsd"] = dict(rank=int(rank), residual_recomputed=recomputed,
                        singular_min_retained=float(sv[int(rank) - 1]),
                        singular_max=float(sv[0]), seconds=round(time.time() - t1, 1))
    print(f"  gelsd: rank {rank:,}  ||f - Bx|| {recomputed:.6f}  "
          f"sigma_max {sv[0]:.4e}  sigma_min_retained {sv[int(rank) - 1]:.4e}  "
          f"{time.time() - t1:.0f}s", flush=True)
    with open(tmp, "w") as fh:
        json.dump(res, fh, indent=1)
    os.replace(tmp, OUT)
    print(f"wrote {OUT} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
