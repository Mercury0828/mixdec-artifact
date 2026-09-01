#!/usr/bin/env python
"""Is the row order of a pub the order the shots were taken, and is a pub stable across it? 0 QPU.

Every device-side interval in this paper resamples contiguous blocks of rows, which is only a
dependence-robust procedure if adjacent rows are adjacent shots. The primitive returns one bit array
per classical register with row `i` of every register belonging to shot `i` of that pub, so the row
index is the shot index; whether the shot index tracks wall-clock acquisition is not something a
per-shot timestamp is available to confirm. It is testable indirectly, and the test is here.

  ORDER      the autocorrelation of the per-shot detector count along the row index, against a
             permutation null. A permuted pub has none by construction, so any resolved correlation
             is evidence that the row order carries real time structure. Absence is not evidence of
             the reverse: it bounds how much dependence there is to find.

  STABILITY  detector rate, count mean and count dispersion by quarter and by half of each
             evaluation pub, with the spread across segments reported against the binomial scale.

  CONSEQUENCE  if the row order did not track acquisition, the rows would be exchangeable and a
             moving-block bootstrap on an exchangeable sequence returns intervals no narrower than
             the independent-shot ones. The reported envelopes are conservative under that failure,
             not anti-conservative.

Usage:  python tools/shot_order_stability.py
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detectors import build_detectors  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "shot_order_stability.json")
LAGS = [1, 2, 5, 10, 50, 200, 1000]
N_PERM = 200


def load_L0(epoch):
    indir = os.path.join(ROOT, "data", "campaign_v_%s" % epoch.lower())
    out = {}
    for f in sorted(glob.glob(os.path.join(indir, "pub*.npz")),
                    key=lambda p: int(os.path.basename(p)[3:-4])):
        z = np.load(f, allow_pickle=True)
        m = json.loads(str(z["meta"]))
        if m["logical_state"] == 0:
            out.setdefault(m["region"], []).append(build_detectors(z["syn"], z["fin"]))
    return out


def acf(x, lag):
    a, b = x[:-lag], x[lag:]
    sa, sb = a.std(), b.std()
    if sa == 0 or sb == 0:
        return 0.0
    return float(((a * b).mean() - a.mean() * b.mean()) / (sa * sb))


def segments(D, k):
    n = len(D)
    out = []
    for s in range(k):
        seg = D[s * n // k:(s + 1) * n // k]
        cnt = seg.sum(axis=(1, 2)).astype(np.float64)
        out.append(dict(rate=float(seg.mean()), count_mean=float(cnt.mean()),
                        fano=float(cnt.var(ddof=1) / cnt.mean()), shots=int(len(seg))))
    return out


def main():
    rng = np.random.default_rng(3)
    rows = []
    for epoch in ("E1", "E2"):
        for region, ds in sorted(load_L0(epoch).items()):
            if len(ds) < 2:
                continue
            ev = ds[1]                                   # the evaluation split
            cnt = ev.sum(axis=(1, 2)).astype(np.float64)

            obs = {str(L): acf(cnt, L) for L in LAGS}
            null = np.empty((N_PERM, len(LAGS)))
            for i in range(N_PERM):
                p = cnt[rng.permutation(len(cnt))]
                null[i] = [acf(p, L) for L in LAGS]
            band = {str(L): [float(np.percentile(null[:, j], 2.5)),
                             float(np.percentile(null[:, j], 97.5))]
                    for j, L in enumerate(LAGS)}
            outside = [L for j, L in enumerate(LAGS)
                       if not (band[str(L)][0] <= obs[str(L)] <= band[str(L)][1])]

            row = dict(epoch=epoch, region=region, shots=int(len(ev)),
                       count_acf=obs, permutation_band=band,
                       lags_outside_null=outside,
                       quarters=segments(ev, 4), halves=segments(ev, 2))
            rows.append(row)
            q = row["quarters"]
            print(f"{epoch} {region}: n={len(ev)}")
            print("   count ACF   " + "  ".join("L%d %+0.4f" % (L, obs[str(L)]) for L in LAGS))
            print("   outside the permutation null at lags: %s"
                  % (outside if outside else "none"))
            print("   quarters    rate " + " ".join("%.5f" % s["rate"] for s in q)
                  + "   fano " + " ".join("%.2f" % s["fano"] for s in q))

    with open(OUT, "w") as fh:
        json.dump({"lags": LAGS, "n_permutations": N_PERM, "rows": rows}, fh, indent=2)
    print("\nwrote", os.path.relpath(OUT, ROOT))


if __name__ == "__main__":
    main()
