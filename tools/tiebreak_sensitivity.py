#!/usr/bin/env python
"""Round 7: every matched-escalation number in this project, re-run under an ONLINE tie-break.

🔴 WHY. A figure audit found that `route_a.flag_at`
admitted tied shots as `tie[:need]` -- **the earliest evaluation shots in collection order**. Integer
scores tie enormously (`SEAMW` at phi = 2% would otherwise flag ~30% of the sample), so that
admission is load-bearing, and taking the earliest shots is neither an online decision rule nor an
exchangeable tie-break. This project has already measured block drift, so it can bias catch rates.

THE REPLACEMENT, in `route_a.tie_rule`: calibrate on the FIT half alone,

    thr = quantile(fit, 1 - phi)      q = (phi - Pr_fit[s > thr]) / Pr_fit[s == thr]

then decide each evaluation shot independently -- flag if `s > thr`, and flag a tied shot with
probability `q`. Nothing consults the shot's position, its outcome, or the evaluation half's own tie
count, so this is deployable as written. The realized flag rate is now RANDOM around `phi` instead of
being forced to it, and that randomness is what this file measures: `SEEDS` independent tie streams
per cell, mean and sd for every recall number.

Pre-registered at `a54bc69`, `docs/expected.md` Round 7, with three falsifiers:

  1. Route A's predictor -- SEAMW's mean recall at phi = 2%, b = 1 no longer above the free control
     RATE by more than 2 seed-sd  =>  Route A's prediction claim was a tie-ordering artifact.
  2. N10 -- LAGEXC beats RATE at >= 5 of 15 cells under Bonferroni-corrected McNemar  =>  N10 wrong.
  3. Round 2 -- the SEAMW / STCG_win / ADaPT ranking reverses in any nominal cell once seed
     variability is included  =>  round 2 restated with intervals.

Device data only, campaign R, `Y = 0`, the pre-registered 50,000 / 50,000 split in collection order.
Weights are the pooled Y-blind table from `audit_reanalysis.py`, so the decoder here is the corrected
one. **0 QPU.**

Usage:  python tools/tiebreak_sensitivity.py
"""
import json
import os
import sys
import time

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_campaign_r import load_arm  # noqa: E402
from detectors import build_detectors  # noqa: E402
from parallel_window import two_window  # noqa: E402
from prior_art_triggers import trigger_scores  # noqa: E402
from route_a import build_scores, flag_at, tie_rule  # noqa: E402
from weight_model_robustness import fit_weights_v2  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B_MAIN = [1, 2, 4, 6, 8]
PHIS = [0.02, 0.05, 0.10, 0.20]
W1 = 25
N_HALF = 50_000
DELTA = 0.05
SEEDS = 20
CONTROLS = ("LAGEXC", "LAGSUM", "RATE", "SEAMW", "RANDLAG", "PERMUTE", "FLAT")


def mcnemar_two(b, c):
    return 1.0 if b + c == 0 else float(binomtest(b, b + c, 0.5).pvalue)


def main():
    t_start = time.time()
    syn0, fin0, _ = load_arm(0)
    syn1, fin1, _ = load_arm(1)
    D0 = build_detectors(syn0, fin0)
    D1 = build_detectors(syn1, fin1)
    n_layers, n_anc = D0.shape[1], D0.shape[2]
    fitsl, evsl = slice(0, N_HALF), slice(N_HALF, 2 * N_HALF)

    # the same pooled Y-blind table the corrected re-analysis uses
    w, t, _, _ = fit_weights_v2(np.concatenate([D0[fitsl], D1[fitsl]]), n_fit=2 * N_HALF)

    print("ROUND 7 -- ONLINE TIE-BREAK, SEED VARIABILITY.  device, 0 QPU")
    print(f"  campaign R, Y=0, {N_HALF} fit / {N_HALF} evaluation in collection order")
    print(f"  pooled Y-blind weight table; {SEEDS} independent tie streams per cell\n")

    rows, n10, prior = [], [], []
    for b in B_MAIN:
        t0 = time.time()
        rf = two_window(D0[fitsl], W1, b, logical_j=0, eps=1e-3, seed=0, wtab=w, ttab=t)
        re_ = two_window(D0[evsl], W1, b, logical_j=0, eps=1e-3, seed=0, wtab=w, ttab=t)
        dv = re_["diverged_repaired"].astype(bool)
        nd = int(dv.sum())

        cf, _ = build_scores(D0[fitsl], rf["seam_weight"], np.random.default_rng(1000 + b))
        ce, _ = build_scores(D0[evsl], re_["seam_weight"], np.random.default_rng(1000 + b))
        # the path-selected gap at the b-MATCHED half-width; round 2 sweeps the width and
        # credits the baseline with its best cell, which is a different question from this one
        sw_f, sp_f, sj_f, ad_f = trigger_scores(D0[fitsl], W1, b, w, t, n_layers, n_anc,
                                               path_h=[b])
        sw_e, sp_e, sj_e, ad_e = trigger_scores(D0[evsl], W1, b, w, t, n_layers, n_anc,
                                               path_h=[b])
        fitS = dict(cf, STCG_win=-sw_f, STCG_path=-sp_f[b], STCG_joint=-sj_f)
        evS = dict(ce, STCG_win=-sw_e, STCG_path=-sp_e[b], STCG_joint=-sj_e)
        # ADaPT: round 7's F3 pre-registered `STCG_win` OR ADaPT `Q` and only ever ran the
        # gap. Re-gate finding 6. Every swept cell enters, and the best is credited.
        for kk in ad_f:
            nm = f"ADAPT_a{kk[0]}_{kk[1]}"
            fitS[nm] = ad_f[kk]
            evS[nm] = ad_e[kk]
        adapt_names = [f"ADAPT_a{kk[0]}_{kk[1]}" for kk in ad_f]
        names = list(CONTROLS) + ["STCG_win", "STCG_path", "STCG_joint"] + adapt_names

        print(f"=== b = {b}   held-out disagreements {nd}/{N_HALF}   ({time.time() - t0:.0f}s)")
        print(f"{'score':>11} {'phi':>6} {'q_tie':>7} {'esc mean':>9} {'esc sd':>7} "
              f"{'recall %':>9} {'sd':>6} {'unflagged':>10}")
        for phi in PHIS:
            flags = {}
            for nm in names:
                _, q = tie_rule(fitS[nm], phi)
                fl = [flag_at(fitS[nm], evS[nm], phi, rng=np.random.default_rng(7000 + s))
                      for s in range(SEEDS)]
                flags[nm] = fl
                esc = np.array([f.mean() for f in fl])
                cau = np.array([int((dv & f).sum()) for f in fl], dtype=float)
                rec = 100 * cau / nd if nd else np.zeros(SEEDS)
                r = dict(b=b, score=nm, phi=phi, n_div=nd, q_tie=q,
                         esc_mean=float(esc.mean()), esc_sd=float(esc.std(ddof=1)),
                         caught_mean=float(cau.mean()), caught_sd=float(cau.std(ddof=1)),
                         recall_mean=float(rec.mean()), recall_sd=float(rec.std(ddof=1)),
                         unflagged_mean=float(nd - cau.mean()))
                rows.append(r)
                if nm in ("SEAMW", "RATE", "LAGEXC", "STCG_win", "STCG_path", "STCG_joint",
                          "FLAT"):
                    print(f"{nm:>11} {phi:>6.2f} {q:>7.3f} {r['esc_mean']:>9.4f} "
                          f"{r['esc_sd']:>7.4f} {r['recall_mean']:>9.2f} {r['recall_sd']:>6.2f} "
                          f"{r['unflagged_mean']:>10.1f}")

            # ---- N10, paired, per seed
            ps, ol, orr = [], [], []
            for s in range(SEEDS):
                fl_l, fl_r = flags["LAGEXC"][s], flags["RATE"][s]
                a_ = int((dv & fl_l & ~fl_r).sum())
                b_ = int((dv & fl_r & ~fl_l).sum())
                ol.append(a_)
                orr.append(b_)
                ps.append(mcnemar_two(a_, b_))
            thr_bonf = DELTA / (len(B_MAIN) * len(PHIS))
            wins = sum(1 for s in range(SEEDS)
                       if ol[s] > orr[s] and ps[s] < thr_bonf)
            n10.append(dict(b=b, phi=phi, n_div=nd,
                            lagexc_only_mean=float(np.mean(ol)),
                            rate_only_mean=float(np.mean(orr)),
                            p_median=float(np.median(ps)),
                            seeds_lagexc_better=wins,
                            lagexc_better=bool(wins > SEEDS // 2)))

            # ---- round 2 ranking, per seed
            opps = ["STCG_win", "STCG_path", "STCG_joint"]
            # ADaPT is credited with its BEST swept cell in each seed, the same steel-man
            # protocol it gets in round 2.
            # 🔴 ADaPT's hyperparameter is chosen ONCE on the FIT half and frozen. Choosing it
            # per seed on evaluation labels made it an evaluation-set oracle rather than a
            # deployable baseline. Gate-3 finding 9.
            fit_dv = rf['diverged_repaired'].astype(bool)
            ad_star = min(adapt_names,
                          key=lambda nm: int((fit_dv & ~flag_at(
                              fitS[nm], fitS[nm], phi,
                              rng=np.random.default_rng(11))).sum()))
            best_ad = [ad_star] * SEEDS
            d_ad = np.array([int((dv & ~flags['SEAMW'][s2]).sum())
                             - int((dv & ~flags[best_ad[s2]][s2]).sum())
                             for s2 in range(SEEDS)], dtype=float)
            prior.append(dict(b=b, phi=phi, opponent='ADAPT(fit-selected)',
                              seamw_minus_opp_unflagged_mean=float(d_ad.mean()),
                              sd=float(d_ad.std(ddof=1)),
                              adapt_cell_frozen_on_fit_half=ad_star,
                              seamw_better_seeds=int((d_ad < 0).sum()),
                              sign_stable=bool((d_ad < 0).all() or (d_ad > 0).all()
                                               or (d_ad == 0).all())))
            for opp in opps:
                d = np.array([int((dv & ~flags["SEAMW"][s]).sum())
                              - int((dv & ~flags[opp][s]).sum()) for s in range(SEEDS)],
                             dtype=float)
                prior.append(dict(b=b, phi=phi, opponent=opp,
                                  seamw_minus_opp_unflagged_mean=float(d.mean()),
                                  sd=float(d.std(ddof=1)),
                                  seamw_better_seeds=int((d < 0).sum()),
                                  sign_stable=bool((d < 0).all() or (d > 0).all() or (d == 0).all())))
        print()

    # ---------------------------------------------------------------- falsifiers
    print("=" * 96)
    f1_cell = next(r for r in rows if r["b"] == 1 and r["phi"] == 0.02 and r["score"] == "SEAMW")
    f1_ctl = next(r for r in rows if r["b"] == 1 and r["phi"] == 0.02 and r["score"] == "RATE")
    margin = f1_cell["recall_mean"] - f1_ctl["recall_mean"]
    sd_pool = float(np.hypot(f1_cell["recall_sd"], f1_ctl["recall_sd"]))
    f1 = margin <= 2 * sd_pool
    print(f"  F1  Route A predictor at b=1, phi=2%: SEAMW {f1_cell['recall_mean']:.2f}% "
          f"+/- {f1_cell['recall_sd']:.2f} vs RATE {f1_ctl['recall_mean']:.2f}% "
          f"+/- {f1_ctl['recall_sd']:.2f};  margin {margin:+.2f} pp, 2sd = {2 * sd_pool:.2f} pp")
    print(f"      -> {'FIRES -- Route A predictor withdrawn' if f1 else 'does NOT fire'}")

    n_better = sum(1 for r in n10 if r["lagexc_better"])
    f2 = n_better >= 5
    print(f"  F2  N10: LAGEXC beats RATE (majority of seeds, Bonferroni McNemar) at "
          f"{n_better} of {len(n10)} cells")
    print(f"      -> {'FIRES -- N10 withdrawn' if f2 else 'does NOT fire; N10 stands'}")

    unstable = [p for p in prior if not p["sign_stable"]]
    f3 = len(unstable) > 0
    print(f"  F3  round-2 ranking: {len(unstable)} of {len(prior)} SEAMW-vs-opponent cells change "
          f"sign across seeds")
    print(f"      -> {'FIRES -- round 2 must be restated with intervals' if f3 else 'does NOT fire'}")
    for p in unstable:
        print(f"        b={p['b']} phi={p['phi']:.2f} vs {p['opponent']}: "
              f"{p['seamw_minus_opp_unflagged_mean']:+.2f} +/- {p['sd']:.2f} unflagged, "
              f"SEAMW better in {p['seamw_better_seeds']}/{SEEDS} seeds")

    out = os.path.join(ROOT, "data", "tiebreak_sensitivity.json")
    tmp = out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(dict(seeds=SEEDS, b_grid=B_MAIN, phis=PHIS, n_half=N_HALF, arm=0,
                       weight_table="pooled(Y-blind)", delta=DELTA,
                       bonferroni_cells=len(B_MAIN) * len(PHIS),
                       falsifier_routeA_fired=bool(f1), falsifier_N10_fired=bool(f2),
                       falsifier_round2_fired=bool(f3),
                       rows=rows, n10=n10, ranking=prior), fh, indent=1)
    os.replace(tmp, out)
    print(f"\nwrote {out}   ({time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
