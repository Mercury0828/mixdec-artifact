#!/usr/bin/env python
"""POST-HOC robustness check — NOT part of the frozen PILOT_PREREGISTRATION design.

Question: is `Repair_b` an artifact of tie degeneracy under uniform weights?

Zhang et al. App. E Assumption 1 requires a unique minimum-weight correction, and states it
"should hold for MWPM decoders if edge weights are perturbed to break ties". This script does
exactly that: it perturbs every edge weight by an independent tiny amount, so the minimum-weight
solution is generically unique, and re-runs the pilot.

🔴 The perturbation is indexed by GLOBAL fault id, so the joint decoder and every window see the
SAME weight for the same physical fault. Perturbing per-window would manufacture divergences.

If Repair events vanish under perturbation, they were tie artifacts. If they survive, they are
genuine seam-free re-pairings and Zhang's Assumption 1 does not remove them.

Usage:  python tools/pilot_robustness.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detectors import build_detectors, load  # noqa: E402
from pilot_divergence import (  # noqa: E402
    DELTA, FAULTS_PER_LAYER, K_TESTS, SLICES, _edge_info, build_graph, cp_upper, rel,
)

EPS = [0.0, 1e-6, 1e-3, 1e-2]
SEEDS = [1, 2, 3]
B_SHOW = [1, 2, 4, 8, 12]


def graphs_for(n_layers, W, b, wtab):
    """Window graphs keyed by (span, offset) so each window gets its own global weight offset."""
    g = {("joint", 0): build_graph(n_layers, offset=0, wtab=wtab)}
    for t in range(0, n_layers, W):
        span = min(min(t + W, n_layers) + b, n_layers) - t
        g[(span, t)] = build_graph(span, offset=t, wtab=wtab)
    return g


def run(D, W, b, wtab):
    """run_slice with per-window offsets; mirrors pilot_divergence.run_slice exactly otherwise."""
    shots, n_layers, n_anc = D.shape
    g = graphs_for(n_layers, W, b, wtab)

    par_joint = np.empty(shots, dtype=np.uint8)
    par_split = np.empty(shots, dtype=np.uint8)
    seam_flag = np.zeros(shots, dtype=np.uint8)
    joint_graph = g[("joint", 0)]

    for s in range(shots):
        par_joint[s] = joint_graph.decode(D[s].ravel())[0]
        residual = D[s].copy()
        parity, seam, t = 0, False, 0
        while t < n_layers:
            commit_end = min(t + W, n_layers)
            decode_end = min(commit_end + b, n_layers)
            span = decode_end - t
            edges = g[(span, t)].decode_to_edges_array(residual[t:decode_end].ravel())
            flipped = np.zeros((span, n_anc), dtype=np.uint8)
            fwd = False
            for u, v in edges:
                anchor, nodes, flag = _edge_info(int(u), int(v), n_anc)
                if t + anchor >= commit_end:
                    continue
                parity ^= flag
                for nd in nodes:
                    rl, j = divmod(nd, n_anc)
                    flipped[rl, j] ^= 1
                    if t + rl >= commit_end:
                        fwd = True
            residual[t:decode_end] ^= flipped
            assert not residual[t:commit_end].any()
            seam = seam or fwd
            t = commit_end
        par_split[s] = parity
        seam_flag[s] = seam

    diverged = (par_joint != par_split).astype(np.uint8)
    return diverged, seam_flag


def main():
    cfg = SLICES["50"]
    syn, fin, _ = load(rel(f"data/{cfg['job']}.npz"))
    D = build_detectors(syn, fin)
    W = cfg["W"]
    n_layers = D.shape[1]
    n_faults = (n_layers + 1) * FAULTS_PER_LAYER
    alpha = DELTA / K_TESTS

    out = {}
    print("PRIMARY 50-round slice, W=10. Repair counts under weight perturbation.")
    print("(eps=0.0 with any seed == the frozen uniform-weight pilot)\n")
    header = "  ".join(f"b={b}" for b in B_SHOW)
    print(f"{'eps':>8} {'seed':>5}   {header}      (Delta / Seam / Repair)")

    for eps in EPS:
        for seed in (SEEDS if eps > 0 else [0]):
            if eps == 0:
                wtab = None
            else:
                rng = np.random.default_rng(seed)
                wtab = 1.0 + eps * rng.random(n_faults)
            cells = []
            for b in B_SHOW:
                div, seam = run(D, W, b, wtab)
                kd, kr = int(div.sum()), int((div & ~seam).sum())
                cells.append((b, kd, int((div & seam).sum()), kr))
                out[f"eps{eps}_seed{seed}_b{b}"] = {
                    "eps": eps, "seed": seed, "b": b,
                    "k_divergence": kd, "k_seam": int((div & seam).sum()), "k_repair": kr,
                    "U_repair": cp_upper(kr, D.shape[0], alpha),
                }
            row = "  ".join(f"{d:3d}/{s:3d}/{r:2d}" for _, d, s, r in cells)
            print(f"{eps:>8} {seed:>5}   {row}")

    path = rel("data/pilot_robustness.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=1)
    os.replace(tmp, path)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
