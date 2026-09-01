#!/usr/bin/env python
"""Round 2 of the TQE #1 self-check: OUR TRIGGER AGAINST THE ACTUAL PRIOR ART, head-to-head.

Scan S-5 established that runtime buffer escalation is prior art:

  STCG   Mishima, Toshio, Kishi, Fujisaki, Oshima, Sato & Fujii, arXiv:2605.14637 v1 (14 May 2026).
         Decode with a small buffer, compute a *complementary gap* as soft information, enlarge the
         buffer and re-decode when the gap is small.
  ADaPT  arXiv:2605.01149. Same idea with a cluster-based confidence `Q` and sequential windows.

`tools/baselines.py` labels its EMPLER row "(The ADaPT / Complementary-Gap style)" but implements
only fixed-buffer selection by point estimate, which is not what either paper does. That is a
strawman, and TQE lens 2 requires the named prior method IMPLEMENTED AND RUN. This file does that.

WHAT IS COMPARED. Three runtime triggers, all deciding per shot whether to escalate to joint
decoding, all evaluated at MATCHED escalation rates with thresholds set on the fit half:

  SEAMW   ours. Seam syndrome weight: how many detectors the stitched correction fails to explain.
          Needs no weights, no LLRs, no second decode, no noise model.
  STCG    the complementary gap, computed EXACTLY via `WindowGraph.complementary_gap`
          (observable promoted to a real node; two constrained decodes give both class minima).
          Taken per window and minimised over the two windows, which is the windowed soft
          information their scheme uses. The joint-graph gap is reported as a variant.
  ADAPT   cluster confidence over the connected components of the committed solution edges,
          `(sum_i (sum_{e in C_i} w_e)^alpha)^(1/alpha)` normalised either by the selected weight
          (scale-free) or by the graph's total weight (scale-dependent). 🔴 The normalisation is
          ambiguous in the source we could read, so BOTH are computed and `alpha` is swept, and
          ADaPT is credited with its BEST cell -- steel-man the baseline, never beat a mistuned one.

Plus the `route_a.py` controls (`RATE`, `RANDLAG`, `PERMUTE`, `FLAT`) so the battery stays honest.

🔴 THE AXIS THAT MATTERS IS MISSPECIFICATION. Both prior triggers are computed FROM THE FITTED EDGE
WEIGHTS, so they can be no better than the noise model. A seam-consistency check reads only whether
the stitched correction explains the syndrome, which is a fact about the record. Ledger `B15` claims
this is what survives; that claim is tested here by re-running everything under deliberately wrong
weights (pooled, and fitted on the other logical arm).

Pre-registered falsifier (`docs/expected.md`): if STCG matches or beats SEAMW at matched escalation
UNDER MISSPECIFICATION, `B15` loses its item 2 and `P9` reduces to the bound alone.

Usage:  python tools/prior_art_triggers.py
"""
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
from scipy.stats import beta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_campaign_r import load_arm  # noqa: E402
from detectors import build_detectors  # noqa: E402
from parallel_window import TBND, WindowGraph, two_window  # noqa: E402
from route_a import build_scores, flag_at  # noqa: E402
from weight_model_robustness import fit_weights_v2  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B_NOMINAL = [1, 2, 4, 8]
B_STRESS = [1, 4]
PHIS = [0.02, 0.05, 0.10]
ALPHAS = [1.5, 2.0, 3.0, 5.0]
# Selection half-widths for the PATH-SELECTED gap. Their paper selects a spatiotemporal region;
# the region's extent is a free parameter of the transfer, so it is swept and the baseline is
# credited with its BEST held-out cell -- the same steel-man protocol ADaPT's alpha gets. The
# whole-window endpoint is STCG_win, already computed.
PATH_H = [1, 2, 4, 8]
SELECT_SHOTS = 10_000            # fit-half subsample the selection width is chosen on
W1 = 25
DELTA = 0.05
N_HALF = 50_000
N_STRESS = 20_000


def cp_upper(k, n, a):
    return 1.0 if k >= n else float(beta.ppf(1.0 - a, k + 1, n - k))


def cluster_weights(sel, dets_list, w):
    """Total weight of each connected component of the selected edge set.

    Components are taken over shared detector nodes -- boundary nodes are excluded from `dets`, so
    two edges that both end on the boundary are NOT merged, which is the right notion here: they are
    separate error chains that happen to both terminate outside.
    """
    idx = np.flatnonzero(sel)
    if idx.size == 0:
        return []
    parent = {int(i): int(i) for i in idx}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    owner = {}
    for i in idx:
        i = int(i)
        for d in dets_list[i]:
            if d in owner:
                a, b = find(i), find(owner[d])
                if a != b:
                    parent[a] = b
            else:
                owner[d] = i
    tot = defaultdict(float)
    for i in idx:
        tot[find(int(i))] += w[int(i)]
    return list(tot.values())


def adapt_scores(cw, w_selected_total, w_graph_total, alphas):
    """ADaPT-style cluster confidence, both normalisations, every alpha. Higher = escalate."""
    out = {}
    if not cw:
        for a in alphas:
            out[(a, "sel")] = 0.0
            out[(a, "all")] = 0.0
        return out
    arr = np.asarray(cw, dtype=np.float64)
    for a in alphas:
        norm = float((arr ** a).sum() ** (1.0 / a))
        out[(a, "sel")] = norm / w_selected_total if w_selected_total > 0 else 0.0
        out[(a, "all")] = norm / w_graph_total
    return out


def trigger_scores(D, W1_, b, wtab, ttab, n_layers, n_anc, logical_j=0, path_h=None):
    """Every runtime trigger, per shot, plus the divergence label. One pass over the shots."""
    shots = D.shape[0]
    e1 = min(W1_ + b, n_layers)
    s2 = max(W1_ - b, 0)
    g1 = WindowGraph(0, e1, n_anc, logical_j, False, e1 < n_layers, wtab=wtab, ttab=ttab,
                     obs_as_detector=True)
    g2 = WindowGraph(s2, n_layers, n_anc, logical_j, s2 > 0, False, wtab=wtab, ttab=ttab,
                     obs_as_detector=True)
    # plain (non-observable-node) copies, to get the committed solutions ADaPT clusters over
    p1 = WindowGraph(0, e1, n_anc, logical_j, False, e1 < n_layers, wtab=wtab, ttab=ttab)
    p2 = WindowGraph(s2, n_layers, n_anc, logical_j, s2 > 0, False, wtab=wtab, ttab=ttab)
    joint = WindowGraph(0, n_layers, n_anc, logical_j, False, False, wtab=wtab, ttab=ttab,
                        obs_as_detector=True)
    # PATH-SELECTED spatiotemporal gap (arXiv:2605.14637). The whole-window gap above asks how
    # confident the decoder is about the observable over the ENTIRE window, which with a small
    # buffer is dominated by faults nowhere near the seam. The path-selected gap asks the question
    # the buffer actually exists to answer: how confident is the decoder about the observable
    # RESTRICTED to the 2b-layer seam neighbourhood. Same two class-constrained decodes, same
    # exactness, different selected region.
    qs = {}
    for H in (PATH_H if path_h is None else path_h):
        sel = (max(W1_ - H, 0), min(W1_ + H, n_layers))
        qs[H] = (WindowGraph(0, e1, n_anc, logical_j, False, e1 < n_layers, wtab=wtab, ttab=ttab,
                             obs_as_detector=True, obs_layers=sel),
                 WindowGraph(s2, n_layers, n_anc, logical_j, s2 > 0, False, wtab=wtab, ttab=ttab,
                             obs_as_detector=True, obs_layers=sel))
    c1 = (p1.kind != TBND) & (p1.glayer < W1_)
    c2 = (p2.kind != TBND) & (p2.glayer >= W1_)
    d1 = [tuple(s) for s in p1.dets]
    d2 = [tuple(s) for s in p2.dets]
    wt1, wt2 = p1.w, p2.w
    gtot = float(p1.w.sum() + p2.w.sum())

    stcg_win = np.empty(shots)
    stcg_path = {H: np.empty(shots) for H in qs}
    stcg_joint = np.empty(shots)
    adapt = {k: np.empty(shots) for k in [(a, n) for a in ALPHAS for n in ("sel", "all")]}
    for s in range(shots):
        stcg_win[s] = min(g1.complementary_gap(D[s, :e1])[1],
                          g2.complementary_gap(D[s, s2:])[1])
        for H, (qa, qb) in qs.items():
            stcg_path[H][s] = min(qa.complementary_gap(D[s, :e1])[1],
                                  qb.complementary_gap(D[s, s2:])[1])
        stcg_joint[s] = joint.complementary_gap(D[s])[1]
        k1 = p1.decode(D[s, :e1]) & c1
        k2 = p2.decode(D[s, s2:]) & c2
        cw = cluster_weights(k1, d1, wt1) + cluster_weights(k2, d2, wt2)
        wsel = float(wt1[k1.astype(bool)].sum() + wt2[k2.astype(bool)].sum())
        sc = adapt_scores(cw, wsel, gtot, ALPHAS)
        for k, v in sc.items():
            adapt[k][s] = v
    return stcg_win, stcg_path, stcg_joint, adapt


def select_path_h(D, fitsl, b, wtab, ttab, n_layers, n_anc, n_sel=SELECT_SHOTS):
    """Pick the path-selected gap's selection width ON THE FIT HALF, as a fitting step.

    Sweeping all of `PATH_H` over both full halves costs four extra class-constrained decode pairs
    per shot per width and does not finish in reasonable time. It is also not what a user of the
    baseline would do: they would tune the selection on data they are allowed to see. So the sweep
    runs on the first `n_sel` FIT-half shots and is scored by fit-half recall at the smallest
    budget; the winner alone is then computed on both full halves. Nothing about the evaluation half
    enters the choice.
    """
    sub = slice(fitsl.start, fitsl.start + n_sel)
    r = two_window(D[sub], W1, b, logical_j=0, eps=1e-3, seed=0, wtab=wtab, ttab=ttab)
    dv = r["diverged_repaired"].astype(bool)
    _, sp, _, _ = trigger_scores(D[sub], W1, b, wtab, ttab, n_layers, n_anc, path_h=PATH_H)
    if not dv.any():
        return PATH_H[-1], {}
    scored = {}
    for H, v in sp.items():
        fl = flag_at(-v, -v, PHIS[0], rng=np.random.default_rng(0))
        scored[H] = int((dv & fl).sum())
    best = max(scored, key=lambda H: scored[H])
    return best, scored


def evaluate(tag, D, fitsl, evsl, wtab, ttab, b_grid, n, rows):
    """Matched-escalation head-to-head at one weight table."""
    n_layers, n_anc = D.shape[1], D.shape[2]
    for b in b_grid:
        t0 = time.time()
        h_star, h_scores = select_path_h(D, fitsl, b, wtab, ttab, n_layers, n_anc)
        print(f"  [{tag}] b={b}: path-selection width chosen on {SELECT_SHOTS} fit shots "
              f"-> h = {h_star}   (fit-half catches by width: {h_scores})", flush=True)
        rf = two_window(D[fitsl], W1, b, logical_j=0, eps=1e-3, seed=0, wtab=wtab, ttab=ttab)
        re_ = two_window(D[evsl], W1, b, logical_j=0, eps=1e-3, seed=0, wtab=wtab, ttab=ttab)
        dv = re_["diverged_repaired"].astype(bool)
        nd = int(dv.sum())
        sw_f, sp_f, sj_f, ad_f = trigger_scores(D[fitsl], W1, b, wtab, ttab, n_layers, n_anc,
                                                path_h=[h_star])
        sw_e, sp_e, sj_e, ad_e = trigger_scores(D[evsl], W1, b, wtab, ttab, n_layers, n_anc,
                                                path_h=[h_star])
        ctl_f, _ = build_scores(D[fitsl], rf["seam_weight"], np.random.default_rng(1000 + b))
        ctl_e, _ = build_scores(D[evsl], re_["seam_weight"], np.random.default_rng(1000 + b))

        # every trigger as a RISK score: larger = escalate first
        fitS = {"SEAMW": rf["seam_weight"].astype(float),
                "STCG_win": -sw_f, "STCG_joint": -sj_f,
                "RATE": ctl_f["RATE"], "RANDLAG": ctl_f["RANDLAG"],
                "PERMUTE": ctl_f["PERMUTE"], "FLAT": ctl_f["FLAT"]}
        evS = {"SEAMW": re_["seam_weight"].astype(float),
               "STCG_win": -sw_e, "STCG_joint": -sj_e,
               "RATE": ctl_e["RATE"], "RANDLAG": ctl_e["RANDLAG"],
               "PERMUTE": ctl_e["PERMUTE"], "FLAT": ctl_e["FLAT"]}
        for k in ad_f:
            fitS[f"ADAPT_a{k[0]}_{k[1]}"] = ad_f[k]
            evS[f"ADAPT_a{k[0]}_{k[1]}"] = ad_e[k]
        for H in sp_f:
            fitS[f"STCG_path_h{H}"] = -sp_f[H]
            evS[f"STCG_path_h{H}"] = -sp_e[H]

        alpha_cp = DELTA / (len(b_grid) * len(PHIS))
        print(f"\n=== [{tag}] b = {b}   held-out divergences {nd}/{n}   "
              f"({time.time() - t0:.0f}s)")
        print(f"{'trigger':>18} {'phi':>6} {'esc':>7} {'caught':>7} {'UNFLAGGED':>10} "
              f"{'cert':>9} {'CP UB':>10}")
        best_adapt, best_path = {}, {}
        for phi in PHIS:
            for name in sorted(fitS):
                flag = flag_at(fitS[name], evS[name], phi, n)
                unf = int((dv & ~flag).sum())
                rec = dict(tag=tag, b=b, trigger=name, phi=phi,
                           escalation=float(flag.mean()), n_div=nd,
                           caught=int((dv & flag).sum()), unflagged=unf,
                           cert=unf / n, cert_ub=cp_upper(unf, n, alpha_cp))
                rows.append(rec)
                if name.startswith("ADAPT"):
                    k = (b, phi)
                    if k not in best_adapt or unf < best_adapt[k]["unflagged"]:
                        best_adapt[k] = rec
                if name.startswith("STCG_path_h") or name == "STCG_win":
                    # the credited gap baseline is the BEST member of the family, and the family
                    # INCLUDES the whole-window endpoint. A selection width can only help it.
                    k = (b, phi)
                    if k not in best_path or unf < best_path[k]["unflagged"]:
                        best_path[k] = rec
                if name in ("SEAMW", "STCG_win", "STCG_joint", "RATE", "FLAT"):
                    print(f"{name:>18} {phi:>6.2f} {flag.mean():>7.4f} {rec['caught']:>7} "
                          f"{unf:>10} {rec['cert']:>9.5f} {rec['cert_ub']:>10.2e}")
            bp = best_path[(b, phi)]
            print(f"{'STCG (best of fam)':>18} {phi:>6.2f} {bp['escalation']:>7.4f} "
                  f"{bp['caught']:>7} {bp['unflagged']:>10} {bp['cert']:>9.5f} "
                  f"{bp['cert_ub']:>10.2e}   <- {bp['trigger']}")
            ba = best_adapt[(b, phi)]
            print(f"{'ADAPT (best cell)':>18} {phi:>6.2f} {ba['escalation']:>7.4f} "
                  f"{ba['caught']:>7} {ba['unflagged']:>10} {ba['cert']:>9.5f} "
                  f"{ba['cert_ub']:>10.2e}   <- {ba['trigger']}")

        # the threshold-free rule, which needs no fit half at all
        flag0 = re_["seam_weight"] > 0
        unf0 = int((dv & ~flag0).sum())
        print(f"{'SEAMW>0 (no fit)':>18} {'-':>6} {flag0.mean():>7.4f} "
              f"{int((dv & flag0).sum()):>7} {unf0:>10} {unf0 / n:>9.5f} "
              f"{cp_upper(unf0, n, alpha_cp):>10.2e}")
        rows.append(dict(tag=tag, b=b, trigger="SEAMW_gt0", phi=None,
                         escalation=float(flag0.mean()), n_div=nd,
                         caught=int((dv & flag0).sum()), unflagged=unf0,
                         cert=unf0 / n, cert_ub=cp_upper(unf0, n, alpha_cp)))


def main():
    syn, fin, _ = load_arm(0)
    D0 = build_detectors(syn, fin)
    syn1, fin1, _ = load_arm(1)
    D1 = build_detectors(syn1, fin1)

    rows = []
    print("ROUND 2 -- OUR TRIGGER AGAINST THE ACTUAL PRIOR ART")
    print("  STCG  = arXiv:2605.14637 complementary gap, computed exactly")
    print("  ADAPT = arXiv:2605.01149 cluster confidence, both normalisations, alpha swept, "
          "best cell credited")
    print("  SEAMW = ours: does the stitched correction explain the syndrome\n")

    # ---------------- nominal: the correct weights for this arm
    w_ok, t_ok, _, _ = fit_weights_v2(D0[:N_HALF], n_fit=N_HALF)
    evaluate("nominal", D0, slice(0, N_HALF), slice(N_HALF, 2 * N_HALF),
             w_ok, t_ok, B_NOMINAL, N_HALF, rows)

    # ---------------- stress: the weights are deliberately wrong
    print("\n" + "=" * 100)
    print("MISSPECIFICATION STRESS -- the axis on which B15 claims our trigger survives and theirs")
    print("does not. Both prior triggers are functions of the fitted weights; SEAMW is not.")
    w_pool, t_pool, _, _ = fit_weights_v2(D0[:N_STRESS], n_fit=N_STRESS, pooled=True)
    w_wrong, t_wrong, _, _ = fit_weights_v2(D1[:N_STRESS], n_fit=N_STRESS)
    for tag, wt, tt in (("pooled-weights", w_pool, t_pool),
                        ("wrong-arm-weights", w_wrong, t_wrong)):
        evaluate(tag, D0, slice(0, N_STRESS), slice(N_HALF, N_HALF + N_STRESS),
                 wt, tt, B_STRESS, N_STRESS, rows)

    out = os.path.join(ROOT, "data", "prior_art_triggers.json")
    tmp = out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(dict(b_nominal=B_NOMINAL, b_stress=B_STRESS, phis=PHIS, alphas=ALPHAS,
                       n_half=N_HALF, n_stress=N_STRESS, delta=DELTA, rows=rows), fh, indent=1)
    os.replace(tmp, out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
