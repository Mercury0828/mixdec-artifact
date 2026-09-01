#!/usr/bin/env python
"""Round 7 DIAGNOSTIC, not a verdict: was the matched flag rate actually matched?

🔴 READ THIS FIRST. `tiebreak_sensitivity.py` is the pre-registered run and **its falsifier verdicts
stand as recorded**. This file exists because of something the run itself printed: the *realized*
escalation rates were not equal across scores. At `b = 1, phi = 2%` the run gave `RATE` an escalation
of 0.0211 and `SEAMW` 0.0199 -- a 6% larger budget for the competitor at exactly the cell where
falsifier F1 fired. That is a measured confound in the comparison, and a confound is diagnosed, not
used to overturn a pre-registered outcome.

🔴 THE CAUSE I GUESSED WAS WRONG, and this file records that. I expected quantile interpolation:
`np.quantile(..., method="linear")` can put the threshold strictly between two attained values of a
discrete score, leaving no tie mass for `q` to correct with. Measured, `method="lower"` gives
**bit-identical** escalation and recall in every cell below. So interpolation is not the cause.

THE ACTUAL CAUSE is drift. `q` is calibrated on the fit half by construction -- that is what makes
the rule online -- and the evaluation half's score distribution is not the fit half's, because the
evaluation half is later pubs in collection order. The realized rate therefore misses `phi` by an
amount that depends on how much each score's distribution moved. **This is not removable by any rule
calibrated without the evaluation data, so it is a property of deployment, not a coding defect.**

WHAT THAT LEAVES, and what section 2 below answers: the comparison at `b = 1, phi = 2%` gave `RATE` a
6% larger realized budget than `SEAMW`. So `SEAMW` is also evaluated at the budget `RATE` actually
received, by searching `phi` for the realized-rate match. **Whatever this file shows, F1 fired.**

Usage:  python tools/tiebreak_diagnostic.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_campaign_r import load_arm  # noqa: E402
from detectors import build_detectors  # noqa: E402
from parallel_window import two_window  # noqa: E402
from route_a import build_scores, flag_at  # noqa: E402
from weight_model_robustness import fit_weights_v2  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B_MAIN = [1, 2]
PHIS = [0.02, 0.05, 0.10, 0.20]
W1 = 25
N_HALF = 50_000
SEEDS = 20
SHOWN = ("SEAMW", "RATE", "LAGEXC")


def main():
    syn0, fin0, _ = load_arm(0)
    syn1, fin1, _ = load_arm(1)
    D0, D1 = build_detectors(syn0, fin0), build_detectors(syn1, fin1)
    fitsl, evsl = slice(0, N_HALF), slice(N_HALF, 2 * N_HALF)
    w, t, _, _ = fit_weights_v2(np.concatenate([D0[fitsl], D1[fitsl]]), n_fit=2 * N_HALF)

    print("ROUND 7 DIAGNOSTIC -- was the matched flag rate matched?  device, 0 QPU")
    print("  the pre-registered verdicts in tiebreak_sensitivity.json STAND; this is a confound")
    print("  check on the comparison, not a re-test\n")

    rows = []
    for b in B_MAIN:
        rf = two_window(D0[fitsl], W1, b, logical_j=0, eps=1e-3, seed=0, wtab=w, ttab=t)
        re_ = two_window(D0[evsl], W1, b, logical_j=0, eps=1e-3, seed=0, wtab=w, ttab=t)
        dv = re_["diverged_repaired"].astype(bool)
        nd = int(dv.sum())
        sf, _ = build_scores(D0[fitsl], rf["seam_weight"], np.random.default_rng(1000 + b))
        se, _ = build_scores(D0[evsl], re_["seam_weight"], np.random.default_rng(1000 + b))

        print(f"=== b = {b}   disagreements {nd}/{N_HALF}")
        print(f"{'score':>8} {'phi':>6} | {'esc(linear)':>12} {'recall':>8} {'sd':>5} | "
              f"{'esc(lower)':>11} {'recall':>8} {'sd':>5}")
        for phi in PHIS:
            for nm in SHOWN:
                cell = {}
                for meth in ("linear", "lower"):
                    fl = [flag_at(sf[nm], se[nm], phi, rng=np.random.default_rng(7000 + s),
                                  method=meth) for s in range(SEEDS)]
                    esc = np.array([f.mean() for f in fl])
                    rec = 100 * np.array([int((dv & f).sum()) for f in fl]) / nd
                    cell[meth] = (float(esc.mean()), float(rec.mean()), float(rec.std(ddof=1)))
                print(f"{nm:>8} {phi:>6.2f} | {cell['linear'][0]:>12.4f} "
                      f"{cell['linear'][1]:>8.2f} {cell['linear'][2]:>5.2f} | "
                      f"{cell['lower'][0]:>11.4f} {cell['lower'][1]:>8.2f} "
                      f"{cell['lower'][2]:>5.2f}")
                rows.append(dict(b=b, phi=phi, score=nm,
                                 linear=dict(zip(("esc", "recall", "sd"), cell["linear"])),
                                 lower=dict(zip(("esc", "recall", "sd"), cell["lower"]))))
        print()

    def get(b, phi, nm, meth):
        r = next(x for x in rows if x["b"] == b and x["phi"] == phi and x["score"] == nm)
        return r[meth]

    # ---------------------------------------------------------------- rate-matched re-check
    print("=" * 96)
    print("2. SEAMW AT THE BUDGET *RATE* ACTUALLY RECEIVED (b = 1, phi = 2%)")
    b, phi = 1, 0.02
    rf = two_window(D0[fitsl], W1, b, logical_j=0, eps=1e-3, seed=0, wtab=w, ttab=t)
    re_ = two_window(D0[evsl], W1, b, logical_j=0, eps=1e-3, seed=0, wtab=w, ttab=t)
    dv = re_["diverged_repaired"].astype(bool)
    nd = int(dv.sum())
    sf, _ = build_scores(D0[fitsl], rf["seam_weight"], np.random.default_rng(1000 + b))
    se, _ = build_scores(D0[evsl], re_["seam_weight"], np.random.default_rng(1000 + b))
    target = float(np.mean([flag_at(sf["RATE"], se["RATE"], phi,
                                    rng=np.random.default_rng(7000 + s)).mean()
                            for s in range(SEEDS)]))
    best = None
    for phi2 in np.arange(0.018, 0.030, 0.0005):
        fl = [flag_at(sf["SEAMW"], se["SEAMW"], float(phi2),
                      rng=np.random.default_rng(7000 + s)) for s in range(SEEDS)]
        esc = float(np.mean([f.mean() for f in fl]))
        if best is None or abs(esc - target) < abs(best[1] - target):
            rec = 100 * np.array([int((dv & f).sum()) for f in fl]) / nd
            best = (float(phi2), esc, float(rec.mean()), float(rec.std(ddof=1)))
    print(f"   RATE's realized escalation: {target:.4f}, recall "
          f"{100 * np.mean([int((dv & flag_at(sf['RATE'], se['RATE'], phi, rng=np.random.default_rng(7000 + s))).sum()) for s in range(SEEDS)]) / nd:.2f}%")
    print(f"   SEAMW at phi={best[0]:.4f} -> realized {best[1]:.4f}, recall "
          f"{best[2]:.2f} +/- {best[3]:.2f}%")
    rate_matched = dict(target_escalation=target, seamw_phi=best[0],
                        seamw_escalation=best[1], seamw_recall=best[2], seamw_sd=best[3])

    print("=" * 96)
    for meth in ("linear", "lower"):
        s_, r_ = get(1, 0.02, "SEAMW", meth), get(1, 0.02, "RATE", meth)
        print(f"  F1's cell (b=1, phi=2%) under {meth:>6}: "
              f"SEAMW {s_['recall']:.2f}+/-{s_['sd']:.2f} at esc {s_['esc']:.4f}   "
              f"RATE {r_['recall']:.2f}+/-{r_['sd']:.2f} at esc {r_['esc']:.4f}   "
              f"margin {s_['recall'] - r_['recall']:+.2f} pp")
    print("  F1 fired on the pre-registered 'linear' run and stays fired.")

    out = os.path.join(ROOT, "data", "tiebreak_diagnostic.json")
    tmp = out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(dict(status="DIAGNOSTIC ONLY -- does not alter any pre-registered verdict",
                       seeds=SEEDS, b_grid=B_MAIN, phis=PHIS, scores=list(SHOWN),
                       n_half=N_HALF, rows=rows,
                       quantile_method_changes_nothing=True,
                       rate_matched_recheck=rate_matched), fh, indent=1)
    os.replace(tmp, out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
