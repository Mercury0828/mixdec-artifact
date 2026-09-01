#!/usr/bin/env python
"""Pilot: joint vs windowed MWPM divergence on real ibm_cleveland syndrome data.

Executes exactly the design frozen in docs/PILOT_PREREGISTRATION.md. Answers one question:
do seam-free divergences (Repair_b) occur?

Resumable per (slice, b) with atomic checkpoints (guide section 7.5).

Usage:
    python tools/pilot_divergence.py                 # both slices, full grid
    python tools/pilot_divergence.py --slice 50      # one slice
    python tools/pilot_divergence.py --fresh         # ignore checkpoint
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pymatching
from scipy.stats import beta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detectors import build_detectors, load  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rel(p):
    """Resolve a repo-relative path so the script works from any cwd."""
    return p if os.path.isabs(p) else os.path.join(ROOT, p)

# ---- frozen grid (PILOT_PREREGISTRATION section 5) ----------------------------------------
B_GRID = [1, 2, 3, 4, 5, 6, 8, 10, 12]
DELTA = 0.05
K_TESTS = len(B_GRID)

SLICES = {
    "50": {"job": "da7miljsq5js73bk4vtg", "W": 10},
    "12": {"job": "da7mi6bsq5js73bk4veg", "W": 4},
}

N_ANC = 8


FAULTS_PER_LAYER = 17  # 9 space-like (data d_0..d_8) + 8 time-like (ancilla j)


def build_graph(n_layers, n_anc=N_ANC, offset=0, wtab=None, logical_j=0):
    """Matching graph for n_layers detector layers, per PILOT_PREREGISTRATION section 3.

    Node id = r * n_anc + j. Boundary edges exist only at j=0 (logical flag 1, data qubit d_0)
    and j=n_anc-1 (logical flag 0, data qubit d_8). Edge insertion order is frozen: per round,
    space-like k ascending, then time-like j ascending.

    wtab is None (the FROZEN pre-registered path) -> all weights uniform 1.0.
    Otherwise wtab is a global weight table indexed by GLOBAL fault id
    (global_layer * FAULTS_PER_LAYER + k for space-like, + 9 + j for time-like) and `offset` is
    this window's first global layer. Indexing globally is essential: the joint decoder and every
    window must see the SAME weight for the same physical fault, or divergences are manufactured.
    """
    def w(glayer, fid):
        return 1.0 if wtab is None else float(wtab[glayer * FAULTS_PER_LAYER + fid])

    m = pymatching.Matching()
    for r in range(n_layers):
        base = r * n_anc
        g = offset + r
        # space-like faults: data qubit d_k flips in round r
        lo = {0} if logical_j == 0 else set()
        hi = {0} if logical_j == n_anc - 1 else set()
        m.add_boundary_edge(base + 0, fault_ids=lo, weight=w(g, 0))                 # data qubit d_0
        for k in range(1, n_anc):
            m.add_edge(base + k - 1, base + k, fault_ids=set(), weight=w(g, k))
        m.add_boundary_edge(base + n_anc - 1, fault_ids=hi, weight=w(g, n_anc))     # data qubit d_8
        # time-like faults: ancilla j mismeasured between layer r and r+1
        if r + 1 < n_layers:
            for j in range(n_anc):
                m.add_edge(base + j, base + n_anc + j, fault_ids=set(), weight=w(g, 9 + j))
    return m


def _edge_info(u, v, n_anc=N_ANC, logical_j=0):
    """(anchor_layer, flipped_nodes, logical_flag) for a solution edge; -1 means boundary.

    `logical_j` selects which boundary carries the observable. j=0 (data qubit d_0) is this
    project's frozen convention; stim's generated repetition code puts it on j=n_anc-1 (d_8).
    The two were verified on 2026-08-29 to give bit-identical divergence on every device shot,
    so the choice is a gauge — but joint and split MUST use the same one.
    """
    if v == -1:
        return u // n_anc, (u,), 1 if (u % n_anc) == logical_j else 0
    if u == -1:
        return v // n_anc, (v,), 1 if (v % n_anc) == logical_j else 0
    return min(u, v) // n_anc, (u, v), 0


def run_slice(D, W, b, graphs):
    """Return per-shot arrays: diverged, seam_nontrivial, logical parity of each decoder."""
    shots, n_layers, n_anc = D.shape
    joint_graph = graphs[n_layers]

    par_joint = np.empty(shots, dtype=np.uint8)
    par_split = np.empty(shots, dtype=np.uint8)
    seam_flag = np.zeros(shots, dtype=np.uint8)

    for s in range(shots):
        par_joint[s] = joint_graph.decode(D[s].ravel())[0]

        residual = D[s].copy()
        parity = 0
        seam = False
        t = 0
        while t < n_layers:
            commit_end = min(t + W, n_layers)
            decode_end = min(commit_end + b, n_layers)
            span = decode_end - t
            edges = graphs[span].decode_to_edges_array(residual[t:decode_end].ravel())

            flipped = np.zeros((span, n_anc), dtype=np.uint8)
            reached_forward = False
            for u, v in edges:
                anchor, nodes, flag = _edge_info(int(u), int(v), n_anc)
                if t + anchor >= commit_end:
                    continue  # belongs to a later window; not committed here
                parity ^= flag
                for nd in nodes:
                    rl, j = divmod(nd, n_anc)
                    flipped[rl, j] ^= 1
                    if t + rl >= commit_end:
                        reached_forward = True

            residual[t:decode_end] ^= flipped
            assert not residual[t:commit_end].any(), (
                f"commit region not fully explained at shot {s}, t={t}"
            )
            if reached_forward:
                seam = True
            t = commit_end

        par_split[s] = parity
        seam_flag[s] = seam

    diverged = (par_joint != par_split).astype(np.uint8)
    return diverged, seam_flag, par_joint, par_split


def cp_upper(k, n, alpha):
    """One-sided Clopper-Pearson upper bound at level alpha."""
    if k >= n:
        return 1.0
    return float(beta.ppf(1.0 - alpha, k + 1, n - k))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", choices=list(SLICES), action="append")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--ckpt", default="data/pilot_checkpoint.json")
    ap.add_argument("--out", default="data/pilot_results.json")
    args = ap.parse_args()
    wanted = args.slice or list(SLICES)

    results = {}
    args.ckpt = rel(args.ckpt)
    args.out = rel(args.out)
    if os.path.exists(args.ckpt) and not args.fresh:
        results = json.load(open(args.ckpt))
        print(f"resuming from {args.ckpt}: {len(results)} cell(s) done")

    alpha = DELTA / K_TESTS
    for sl in wanted:
        cfg = SLICES[sl]
        syn, fin, meta = load(rel(f"data/{cfg['job']}.npz"))
        D = build_detectors(syn, fin)
        shots, n_layers, _ = D.shape
        W = cfg["W"]
        print(f"\n=== slice {sl}: {shots} shots, {n_layers} detector layers, W={W} ===")

        spans = {n_layers} | {
            min(min(t + W, n_layers) + b, n_layers) - t
            for b in B_GRID
            for t in range(0, n_layers, W)
        }
        graphs = {sp: build_graph(sp) for sp in sorted(spans)}

        for b in B_GRID:
            key = f"{sl}/b{b}"
            if key in results:
                print(f"  b={b:2d}  (cached)")
                continue
            t0 = time.time()
            div, seam, pj, ps = run_slice(D, W, b, graphs)

            k_div = int(div.sum())
            k_seam = int((div & seam).sum())
            k_rep = int((div & ~seam).sum())
            cell = {
                "slice": sl, "b": b, "W": W, "shots": shots, "n_layers": n_layers,
                "k_divergence": k_div, "k_seam": k_seam, "k_repair": k_rep,
                "rate_divergence": k_div / shots,
                "U_divergence": cp_upper(k_div, shots, alpha),
                "U_seam": cp_upper(k_seam, shots, alpha),
                "U_repair": cp_upper(k_rep, shots, alpha),
                "n_seam_nontrivial_any": int(seam.sum()),
                "logical_err_joint": int((pj ^ fin[:, 0]).sum()),
                "logical_err_split": int((ps ^ fin[:, 0]).sum()),
                "seconds": round(time.time() - t0, 1),
            }
            results[key] = cell
            print(
                f"  b={b:2d}  Delta={k_div:5d} ({k_div/shots:.4f})"
                f"  Seam={k_seam:5d}  Repair={k_rep:5d}"
                f"  U_repair={cell['U_repair']:.2e}  [{cell['seconds']}s]"
            )
            tmp = args.ckpt + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(results, fh, indent=1)
            os.replace(tmp, args.ckpt)

    tmp = args.out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump({"delta": DELTA, "K": K_TESTS, "alpha_per_test": alpha,
                   "b_grid": B_GRID, "cells": results}, fh, indent=1)
    os.replace(tmp, args.out)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
