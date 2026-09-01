#!/usr/bin/env python
"""Re-analysis forced by an independent audit of the first campaign.

Three of my claims did not survive. This file redoes them properly.

A. R-8 WITH A SINGLE Y-BLIND DECODER.
   `analyze_campaign_r.py` fitted a SEPARATE weight table per arm and applied `theta_0` to Y=0 and
   `theta_1` to Y=1. That uses the arm label, contradicts the pre-registration's "the decoder is
   never told the prepared state", and compares (D0,theta0) against (D1,theta1) -- not two noise arms
   under one fixed theta, which is the condition `P7` is stated under. Here a single POOLED table is
   fitted on the union of the two fit halves (100,000 shots, Y-blind) and applied to both arms, and
   the full cross-weight matrix theta_0 / theta_1 / theta_pooled is reported on both arms so the
   confound is visible rather than hidden.
   Inference is PAIRED: `h` and `g` are two outcomes of the same shots, so McNemar's exact test, not
   a one-sample binomial.
   Codeword validity is checked SHOTWISE on campaign R rather than inherited from a cached-data check
   -- if `fin ^ corr` is a valid codeword then majority scoring equals reading any single corrected
   qubit, and if it is not, the scoring rule matters and must be justified.

B. RESOURCE ACCOUNTING, CORRECTED THREE WAYS.
   (a) The windows are `W1 + b` and `n - W1 + b` layers, i.e. 26 and 27 at b=1 -- so the path is 27,
       not `n/2 + b = 26.5`.
   (b) 🔴 SEAM REPAIR RUNS A FULL JOINT-GRAPH DECODE, and the fixed-split accuracy numbers are the
       REPAIRED ones. The old accounting charged the adaptive scheme for a joint decode and the fixed
       scheme for nothing, while quoting repaired accuracy for both. Repair runs exactly when
       `seam_weight > 0` -- which is exactly the escalation set -- so the two schemes cost the SAME to
       first order and the honest comparison is against the repaired baseline.
   (c) `base + esc*n` is an expected branch length, not a critical path. Per-shot paths are reported.

C. N10 AS A PAIRED TEST.
   `binomtest(caught_LAGEXC, n_div, caught_RATE/n_div)` treats RATE's observed catch fraction as a
   fixed null and ignores that both predictors act on the SAME divergence shots. McNemar on the
   discordant cases is the correct test. And "never beats RATE" was literally false -- at
   b=16, phi=0.02 LAGEXC caught 9 against RATE's 8.

Usage:  python tools/audit_reanalysis.py
"""
import json
import os
import sys

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_campaign_r import load_arm, logical_estimate  # noqa: E402
from detectors import build_detectors  # noqa: E402
from parallel_window import two_window  # noqa: E402
from route_a import build_scores, flag_at  # noqa: E402
from weight_model_robustness import fit_weights_v2  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W1 = 25
N_HALF = 50_000
B_MAIN = [1, 2, 4, 6, 8]
B_CROSS = [1]
PHIS = [0.02, 0.05, 0.10]


def mcnemar(b, c):
    """Exact McNemar: given discordant counts, is b > c? Two-sided p and the one-sided direction."""
    n = b + c
    if n == 0:
        return 1.0, 1.0
    two = float(binomtest(b, n, 0.5).pvalue)
    one = float(binomtest(b, n, 0.5, alternative="greater").pvalue)
    return two, one


def main():
    syn0, fin0, _ = load_arm(0)
    syn1, fin1, _ = load_arm(1)
    D0, D1 = build_detectors(syn0, fin0), build_detectors(syn1, fin1)
    n_layers = D0.shape[1]
    fitsl, evsl = slice(0, N_HALF), slice(N_HALF, 2 * N_HALF)

    # the Y-BLIND table: fitted on the union of the two fit halves, never told the arm
    Dpool_fit = np.concatenate([D0[fitsl], D1[fitsl]])
    w_pool, t_pool, _, _ = fit_weights_v2(Dpool_fit, n_fit=2 * N_HALF)
    w0, t0, _, _ = fit_weights_v2(D0[fitsl], n_fit=N_HALF)
    w1, t1, _, _ = fit_weights_v2(D1[fitsl], n_fit=N_HALF)
    tables = {"pooled(Y-blind)": (w_pool, t_pool), "theta_0": (w0, t0), "theta_1": (w1, t1)}
    arms = {0: (D0, fin0), 1: (D1, fin1)}
    report = {}

    # ------------------------------------------------------------------ A. R-8, blinded and paired
    print("=" * 104)
    print("A. R-8 WITH ONE Y-BLIND DECODER, AND PAIRED INFERENCE")
    print("   pooled table fitted on 100,000 shots (both fit halves), the arm label never used\n")
    print(f"{'table':>16} {'Y':>2} {'b':>3} {'Delta':>7} {'harmful h':>10} {'benef g':>8} "
          f"{'h-g':>10} {'McNemar p':>11} {'LER joint':>10} {'LER split':>10} {'codeword ok':>12}")
    rows = []
    for tag in ("pooled(Y-blind)",):
        wt, tt = tables[tag]
        for y in (0, 1):
            Dy, finy = arms[y]
            for b in B_MAIN:
                r = two_window(Dy[evsl], W1, b, logical_j=0, wtab=wt, ttab=tt)
                yj = logical_estimate(finy[evsl], r["corr_joint"])
                yr = logical_estimate(finy[evsl], r["corr_repaired"])
                h = int(((yj == y) & (yr != y)).sum())
                g = int(((yj != y) & (yr == y)).sum())
                two, one = mcnemar(h, g)
                fixed = finy[evsl] ^ r["corr_repaired"]
                ok = float(((fixed.sum(axis=1) == 0) | (fixed.sum(axis=1) == fixed.shape[1])).mean())
                print(f"{tag:>16} {y:>2} {b:>3} {int((yj != yr).sum()):>7} {h:>10} {g:>8} "
                      f"{(h - g) / N_HALF:>+10.5f} {two:>11.2e} {(yj != y).mean():>10.5f} "
                      f"{(yr != y).mean():>10.5f} {ok:>11.1%}")
                rows.append(dict(table=tag, Y=y, b=b, delta=int((yj != yr).sum()),
                                 harmful=h, beneficial=g, net_regret=(h - g) / N_HALF,
                                 mcnemar_p_two_sided=two, mcnemar_p_h_gt_g=one,
                                 ler_joint=float((yj != y).mean()),
                                 ler_split=float((yr != y).mean()),
                                 codeword_valid_frac=ok))
    report["R8_blinded"] = rows

    print("\n   CROSS-WEIGHT MATRIX at b = 1 -- makes the old confound visible")
    print(f"{'table':>16} {'Y':>2} {'Delta':>7} {'h':>6} {'g':>6} {'h-g':>10} {'LER joint':>10} "
          f"{'LER split':>10}")
    cross = []
    for tag, (wt, tt) in tables.items():
        for y in (0, 1):
            Dy, finy = arms[y]
            for b in B_CROSS:
                r = two_window(Dy[evsl], W1, b, logical_j=0, wtab=wt, ttab=tt)
                yj = logical_estimate(finy[evsl], r["corr_joint"])
                yr = logical_estimate(finy[evsl], r["corr_repaired"])
                h = int(((yj == y) & (yr != y)).sum())
                g = int(((yj != y) & (yr == y)).sum())
                print(f"{tag:>16} {y:>2} {int((yj != yr).sum()):>7} {h:>6} {g:>6} "
                      f"{(h - g) / N_HALF:>+10.5f} {(yj != y).mean():>10.5f} "
                      f"{(yr != y).mean():>10.5f}")
                cross.append(dict(table=tag, Y=y, b=b, delta=int((yj != yr).sum()),
                                  harmful=h, beneficial=g, net_regret=(h - g) / N_HALF,
                                  ler_joint=float((yj != y).mean()),
                                  ler_split=float((yr != y).mean())))
    report["R8_cross_weight"] = cross

    # ------------------------------------------------------------------ B. resource accounting
    print("\n" + "=" * 104)
    print("B. RESOURCE ACCOUNTING, CORRECTED. Seam repair is a FULL JOINT DECODE and runs exactly")
    print("   when seam_weight > 0 -- which is exactly the escalation set. So the repaired fixed")
    print("   split and the adaptive scheme cost the SAME to first order.")
    print(f"{'b':>3} {'win lens':>10} {'path(win)':>10} {'seam rate':>10} "
          f"{'path fixed+repair':>18} {'path adaptive':>14} {'speedup fixed':>14} "
          f"{'speedup adapt':>14}")
    res = []
    wt, tt = tables["pooled(Y-blind)"]
    for b in B_MAIN:
        r = two_window(D0[evsl], W1, b, logical_j=0, wtab=wt, ttab=tt)
        seam_rate = float((r["seam_weight"] > 0).mean())
        e1, s2 = min(W1 + b, n_layers), max(W1 - b, 0)
        len1, len2 = e1, n_layers - s2
        path_win = max(len1, len2)
        path_fixed = path_win + seam_rate * n_layers      # repair decodes on the joint graph
        path_adapt = path_win + seam_rate * n_layers      # escalation replaces the repair decode
        print(f"{b:>3} {f'{len1}/{len2}':>10} {path_win:>10} {seam_rate:>10.4f} "
              f"{path_fixed:>18.2f} {path_adapt:>14.2f} {n_layers / path_fixed:>13.2f}x "
              f"{n_layers / path_adapt:>13.2f}x")
        res.append(dict(b=b, window_lengths=[len1, len2], path_windows=path_win,
                        seam_rate=seam_rate, path_fixed_repaired=path_fixed,
                        path_adaptive=path_adapt,
                        speedup_fixed_repaired=n_layers / path_fixed,
                        speedup_adaptive=n_layers / path_adapt,
                        per_shot_path_accepted=path_win,
                        per_shot_path_escalated=path_win + n_layers))
    report["resource_corrected"] = res
    print("\n   PER-SHOT paths (an expected branch length is not a critical path):")
    for r_ in res:
        print(f"     b={r_['b']:>2}  accepted {r_['per_shot_path_accepted']:>3} layers, "
              f"escalated {r_['per_shot_path_escalated']:>3} layers, "
              f"escalated on {r_['seam_rate']:.2%} of shots")

    # ------------------------------------------------------------------ C. N10, paired
    print("\n" + "=" * 104)
    print("C. N10 AS A PAIRED TEST -- McNemar on the discordant divergence shots")
    print(f"{'b':>3} {'phi':>6} {'div':>5} {'LAGEXC':>7} {'RATE':>6} {'LAG-only':>9} "
          f"{'RATE-only':>10} {'McNemar p':>11} {'LAGEXC better?':>15}")
    n10 = []
    for b in B_MAIN:
        rf = two_window(D0[fitsl], W1, b, logical_j=0, wtab=wt, ttab=tt)
        re_ = two_window(D0[evsl], W1, b, logical_j=0, wtab=wt, ttab=tt)
        dv = re_["diverged_repaired"].astype(bool)
        nd = int(dv.sum())
        sf, _ = build_scores(D0[fitsl], rf["seam_weight"], np.random.default_rng(1000 + b))
        se, _ = build_scores(D0[evsl], re_["seam_weight"], np.random.default_rng(1000 + b))
        for phi in PHIS:
            fl_l = flag_at(sf["LAGEXC"], se["LAGEXC"], phi, N_HALF)
            fl_r = flag_at(sf["RATE"], se["RATE"], phi, N_HALF)
            only_l = int((dv & fl_l & ~fl_r).sum())
            only_r = int((dv & fl_r & ~fl_l).sum())
            two, one = mcnemar(only_l, only_r)
            better = "YES" if (only_l > only_r and two < 0.05) else "no"
            print(f"{b:>3} {phi:>6.2f} {nd:>5} {int((dv & fl_l).sum()):>7} "
                  f"{int((dv & fl_r).sum()):>6} {only_l:>9} {only_r:>10} {two:>11.4f} "
                  f"{better:>15}")
            n10.append(dict(b=b, phi=phi, n_div=nd,
                            caught_lagexc=int((dv & fl_l).sum()),
                            caught_rate=int((dv & fl_r).sum()),
                            lagexc_only=only_l, rate_only=only_r,
                            mcnemar_p=two, lagexc_better=bool(better == "YES"),
                            flag_rate_lagexc=float(fl_l.mean()),
                            flag_rate_rate=float(fl_r.mean())))
    report["N10_paired"] = n10
    wins = sum(1 for r_ in n10 if r_["lagexc_better"])
    numeric = sum(1 for r_ in n10 if r_["caught_lagexc"] > r_["caught_rate"])
    print(f"\n   LAGEXC significantly better at {wins} of {len(n10)} cells; "
          f"numerically ahead at {numeric}.")
    print("   'never beats RATE' was literally false and is withdrawn; the supported statement is")
    print("   'no superiority demonstrated on this grid'.")

    out = os.path.join(ROOT, "data", "audit_reanalysis.json")
    tmp = out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(report, fh, indent=1)
    os.replace(tmp, out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
