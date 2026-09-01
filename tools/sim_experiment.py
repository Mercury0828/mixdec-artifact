#!/usr/bin/env python
"""Confirmatory simulator sweep: parallel two-window decoding, with ground-truth fault weights.

Executes docs/SIM_PREREGISTRATION.md. Resumable per (config, b) with atomic checkpoints.

Usage:
    python tools/sim_experiment.py                # full sweep
    python tools/sim_experiment.py --shots 20000  # smaller
    python tools/sim_experiment.py --fresh
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.stats import beta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parallel_window import TBND, WindowGraph, make_weights, two_window  # noqa: E402
from sim_substrate import edge_table, make_circuit, reduced_weights, sample, to_layers  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

B_GRID = [1, 2, 3, 4, 6, 8, 12, 16]
DELTA = 0.05
CONFIGS = [
    # label, distance, rounds, p  -- p=0.008 reproduces the device's ~0.04 detector event rate
    ("device-matched", 9, 40, 0.008),
    ("mid-noise", 9, 40, 0.02),
    ("high-noise", 9, 40, 0.05),
]


def rel(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def cp_upper(k, n, alpha):
    if k >= n:
        return 1.0
    return float(beta.ppf(1.0 - alpha, k + 1, n - k))


def weighted_buffer(n_layers, n_anc, W1, b, wtab, ttab, logical_j):
    """w_b = shortest weighted distance from a seam vertex to the virtual time boundary, on
    window 1's decoding graph — exactly as Zhang defines it.

    🔴 Boundary vertices are SINKS, not transit vertices. An earlier version let paths route
    seam -> spatial boundary -> some far detector -> temporal boundary, three edges, which pinned
    w_b at 3.00 for every b >= 3. A matching cannot travel through the boundary and reappear
    elsewhere. With that fixed the result agrees with Zhang's own gloss — "the weighted buffer size
    should be the buffer size times the weight of a vertical edge" — i.e. w_b ~ b under unit
    vertical weights.
    """
    import heapq

    e1 = min(W1 + b, n_layers)
    if e1 >= n_layers:
        return float("inf")  # window 1 reaches the real end: no virtual time boundary exists
    g = WindowGraph(0, e1, n_anc, logical_j, False, True, wtab=wtab, ttab=ttab)
    n_det = g.span * n_anc
    tb = n_det + 1

    adj = [[] for _ in range(n_det)]
    to_tb = [float("inf")] * n_det          # terminal hop onto the virtual time boundary
    for i in range(g.n_edges):
        ds = sorted(g.dets[i])
        w = float(g.m.get_edge_data(*ds)["weight"]) if len(ds) == 2 else None
        if len(ds) == 2:
            adj[ds[0]].append((ds[1], w))
            adj[ds[1]].append((ds[0], w))
        elif g.kind[i] == TBND:
            # terminal hop only; its weight lives in ttab
            to_tb[ds[0]] = min(to_tb[ds[0]], float(ttab[g.glayer[i] * n_anc + (ds[0] % n_anc)]))
        # spatial boundary edges are sinks and are NOT traversable — deliberately dropped

    dist = [float("inf")] * n_det
    pq = []
    for j in range(n_anc):                  # every seam vertex starts at distance 0
        sn = W1 * n_anc + j
        dist[sn] = 0.0
        heapq.heappush(pq, (0.0, sn))
    best = float("inf")
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u] or d >= best:
            continue
        if to_tb[u] < float("inf"):
            best = min(best, d + to_tb[u])
        for v, w in adj[u]:
            if d + w < dist[v]:
                dist[v] = d + w
                heapq.heappush(pq, (dist[v], v))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=200_000)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--ckpt", default="data/sim_checkpoint.json")
    ap.add_argument("--out", default="data/sim_results.json")
    args = ap.parse_args()
    ckpt, out = rel(args.ckpt), rel(args.out)

    results = {}
    if os.path.exists(ckpt) and not args.fresh:
        results = json.load(open(ckpt))
        print(f"resuming: {len(results)} cell(s) done")

    alpha = DELTA / len(B_GRID)
    t_start = time.time()

    for label, d, rounds, p in CONFIGS:
        na = d - 1
        circ = make_circuit(distance=d, rounds=rounds, p=p)
        t0 = time.time()
        dem, dets, obs, errs = sample(circ, args.shots, seed=hash(label) % 2**31)
        D = to_layers(dets, n_anc=na)
        w_phys, _ = reduced_weights(errs, edge_table(dem, n_anc=na))
        t_prep = time.time() - t0
        n_layers = D.shape[1]
        W1 = n_layers // 2
        wtab, ttab = make_weights(n_layers, na, eps=1e-3, seed=0)
        print(f"\n=== {label}: d={d} rounds={rounds} p={p} shots={args.shots} "
              f"det_rate={D.mean():.4f} mean_weight={w_phys.mean():.1f} prep={t_prep:.1f}s ===")
        print(f"{'b':>3} {'w_b':>6} {'Delta':>7} {'seamNT':>7} {'D&~seam':>8} {'U(D&~seam)':>11} "
              f"{'minW|D&~seam':>13} {'THM1':>5} {'sec':>6}")

        for b in B_GRID:
            key = f"{label}/b{b}"
            if key in results:
                print(f"{b:>3}  (cached)")
                continue
            t0 = time.time()
            r = two_window(D, W1, b, logical_j=na - 1, eps=1e-3, seed=0)
            wb = weighted_buffer(n_layers, na, W1, b, wtab, ttab, na - 1)
            nt = r["seam_nontrivial"].astype(bool)
            dv = r["diverged"].astype(bool)
            free = dv & ~nt
            thm1_min = int(w_phys[nt].min()) if nt.any() else -1
            thm1 = "OK" if (not nt.any() or thm1_min >= wb / 2) else "VIOL"
            cell = dict(
                label=label, d=d, rounds=rounds, p=p, b=b, shots=int(args.shots),
                W1=int(W1), n_layers=int(n_layers), w_b=(None if wb == float("inf") else wb),
                k_divergence=int(dv.sum()), k_seam_nontrivial=int(nt.sum()),
                k_seam_free_divergence=int(free.sum()),
                U_divergence=cp_upper(int(dv.sum()), args.shots, alpha),
                U_seam_free=cp_upper(int(free.sum()), args.shots, alpha),
                min_weight_given_seam=thm1_min,
                theorem1=thm1,
                # THE B9 TEST: physical weight of seam-free divergent shots vs w_b/2
                seam_free_weights=sorted(int(x) for x in w_phys[free][:200]),
                min_weight_seam_free=int(w_phys[free].min()) if free.any() else -1,
                logical_err_joint=int((r["par_joint"] ^ obs[:, 0]).sum()),
                logical_err_split=int((r["par_split"] ^ obs[:, 0]).sum()),
                seconds=round(time.time() - t0, 1),
            )
            results[key] = cell
            wbs = "inf" if wb == float("inf") else f"{wb:.2f}"
            print(f"{b:>3} {wbs:>6} {cell['k_divergence']:>7} {cell['k_seam_nontrivial']:>7} "
                  f"{cell['k_seam_free_divergence']:>8} {cell['U_seam_free']:>11.2e} "
                  f"{cell['min_weight_seam_free']:>13} {thm1:>5} {cell['seconds']:>6}")
            tmp = ckpt + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(results, fh, indent=1)
            os.replace(tmp, ckpt)

    tmp = out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(dict(delta=DELTA, K=len(B_GRID), alpha_per_test=alpha,
                       b_grid=B_GRID, configs=[c[0] for c in CONFIGS],
                       total_seconds=round(time.time() - t_start, 1), cells=results), fh, indent=1)
    os.replace(tmp, out)
    print(f"\ntotal wall time {time.time() - t_start:.1f}s -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
