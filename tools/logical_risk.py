#!/usr/bin/env python
"""How much of the certified envelope the substitution actually spends. 0 QPU.

Theorem~2 bounds $|R(S_b) - R(J)|$ by $\\Pr[\\Delta_b]$, and the bound is attained only when every
disagreement favours the same decoder. On a prepared logical state the two risks are measurable, so
the bound's tightness on hardware is a measurement rather than a supposition. This script makes it,
in all four campaign V contexts, at both pre-registered buffer widths.

Scoring is at codeword level, the form validated in commit `a19cce3`: track the per-data-qubit
correction each decoder applies, apply it to the final data readout, take the majority. The corrected
readout is a valid codeword on every shot, and the rule reproduces the single-qubit convention
exactly, so the observable is a gauge choice and not a scoring assumption.

The majority vote over the nine raw data qubits is reported beside the two decoders as a
terminal-readout baseline, not as an operational competitor: it reads every data qubit transversally
at the end and supplies no syndrome-based correction, so it does not answer the streaming question
the two decoders here are compared on.

INFERENCE. The risk difference is the mean of a per-shot signed series

    Z_t = 1{S_b(D_t) != Y_t} - 1{J(D_t) != Y_t},

so it carries the same serial dependence every other device quantity in this project carries. The
interval is a circular block bootstrap confined within the evaluation pub, over the same block-length
grid the campaign used, with the envelope taken across the grid. The spent ratio is bootstrapped
jointly, resampling the same blocks for the numerator and the denominator so the ratio is not built
from two independently resampled quantities.

🔴 This analysis is POST-REGISTRATION. The twelve judgments frozen before campaign V are eight E1
and four E2; codeword-level logical scoring is not one of them, and this table is a label-based
descriptive analysis added afterwards.

Usage:  python tools/logical_risk.py [--shots N]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detectors import build_detectors  # noqa: E402
from parallel_window import two_window  # noqa: E402
from weight_model_robustness import fit_weights_v2  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "logical_risk.json")
W1 = 25
B_GRID = (1, 2)
BLOCKS = [100, 250, 500, 1000, 2500]      # the campaign's own grid, within one 25,000-shot pub
N_BOOT = 4000


def circular_block_index(n, block, rng):
    """Indices of a circular moving-block resample of length n, blocks confined to [0, n)."""
    nb = int(np.ceil(n / block))
    st = rng.integers(0, n, size=nb)
    idx = (st[:, None] + np.arange(block)[None, :]) % n
    return idx.ravel()[:n]


def block_envelope(z, dis, seed=17):
    """Envelope over the block grid of percentile intervals for E[Z] and for |E[Z]|/Pr[Delta]."""
    n = len(z)
    d_lo, d_hi, s_lo, s_hi = [], [], [], []
    for bl in BLOCKS:
        rng = np.random.default_rng(seed + bl)
        dv = np.empty(N_BOOT)
        sv = np.empty(N_BOOT)
        for i in range(N_BOOT):
            idx = circular_block_index(n, bl, rng)
            zz, dd = z[idx], dis[idx]
            m = zz.mean()
            pr = dd.mean()
            dv[i] = m
            sv[i] = abs(m) / pr if pr > 0 else np.nan
        d_lo.append(float(np.percentile(dv, 2.5)))
        d_hi.append(float(np.percentile(dv, 97.5)))
        s_lo.append(float(np.nanpercentile(sv, 2.5)))
        s_hi.append(float(np.nanpercentile(sv, 97.5)))
    return (min(d_lo), max(d_hi)), (min(s_lo), max(s_hi))


def logical_estimate(fin, corr):
    """Codeword-level logical readout: majority of the corrected data-qubit values."""
    fixed = fin ^ corr
    return (fixed.sum(axis=1) * 2 > fixed.shape[1]).astype(np.uint8)


def load(epoch):
    """Logical-zero pubs per region, in collection order, detectors and final readout together."""
    indir = os.path.join(ROOT, "data", "campaign_v_%s" % epoch.lower())
    out = {}
    for f in sorted(glob.glob(os.path.join(indir, "pub*.npz")),
                    key=lambda p: int(os.path.basename(p)[3:-4])):
        z = np.load(f, allow_pickle=True)
        m = json.loads(str(z["meta"]))
        if m["logical_state"] != 0:
            continue
        out.setdefault(m["region"], []).append(
            (build_detectors(z["syn"], z["fin"]), z["fin"].astype(np.uint8)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=0, help="cap the evaluation split, 0 = all")
    args = ap.parse_args()

    rows = []
    for epoch in ("E1", "E2"):
        arms = load(epoch)
        for region, ds in sorted(arms.items()):
            if len(ds) < 2:
                continue
            (Dfit, _), (Dev, finev) = ds[0], ds[1]
            if args.shots:
                Dev, finev = Dev[:args.shots], finev[:args.shots]
            w, t, _, _ = fit_weights_v2(Dfit, n_fit=len(Dfit))

            truth = np.zeros(len(finev), np.uint8)          # the logical-zero arm
            floor = float((logical_estimate(finev, np.zeros_like(finev)) != truth).mean())

            for b in B_GRID:
                r = two_window(Dev, W1, b, logical_j=0, eps=1e-3, seed=0, wtab=w, ttab=t)
                yj = logical_estimate(finev, r["corr_joint"])
                ys = logical_estimate(finev, r["corr_repaired"])
                dis = r["diverged_repaired"].astype(bool)
                wrong_j, wrong_s = (yj != truth), (ys != truth)
                harmful = int((dis & wrong_s & ~wrong_j).sum())
                beneficial = int((dis & wrong_j & ~wrong_s).sum())
                n = len(finev)
                rj, rs = float(wrong_j.mean()), float(wrong_s.mean())
                pdel = float(dis.mean())
                z = wrong_s.astype(np.int8) - wrong_j.astype(np.int8)
                (dlo, dhi), (slo, shi) = block_envelope(z, dis.astype(np.uint8))
                rows.append(dict(
                    epoch=epoch, region=region, b=b, shots=n,
                    disagreements=int(dis.sum()), pr_delta=pdel,
                    harmful=harmful, beneficial=beneficial,
                    risk_joint=rj, risk_split=rs, risk_difference=rs - rj,
                    risk_difference_ci=[dlo, dhi],
                    envelope_spent=(abs(rs - rj) / pdel) if pdel > 0 else None,
                    envelope_spent_ci=[slo, shi],
                    blocks=BLOCKS, n_boot=N_BOOT, majority_floor=floor))
                print(f"{epoch} {region} b={b}: Pr[D]={pdel*1e4:6.2f}e-4  "
                      f"R(J)={rj*1e4:6.2f}e-4  R(S)={rs*1e4:6.2f}e-4  "
                      f"diff={(rs-rj)*1e4:+6.2f}e-4 [{dlo*1e4:+6.2f},{dhi*1e4:+6.2f}]  "
                      f"spent={abs(rs-rj)/pdel:5.3f} [{slo:.3f},{shi:.3f}]  "
                      f"h/b={harmful}/{beneficial}", flush=True)

    with open(OUT, "w") as fh:
        json.dump({"W1": W1, "rows": rows}, fh, indent=2)
    print("\nwrote", os.path.relpath(OUT, ROOT))


if __name__ == "__main__":
    main()
