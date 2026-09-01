#!/usr/bin/env python
"""Route A -- use the measured long-range temporal structure to PREDICT divergence, not correct it.

Web-Pro round 1, section 4. `N9` is what makes this the interesting route: adding long-range matching
edges to the decoder supplied new *detector* explanations with no independently validated logical
action, and every placebo reproduced the effect exactly. Route A never touches the correction. It
uses the same measured structure only to decide, per shot, whether a small buffer is safe:

    accept the split decode when the acceptance rule `A_b = 1`, otherwise escalate to joint.

The certificate is the same pointwise argument as `P7`, restricted to the accepted set. On escalated
shots the adaptive output IS the joint output, so those shots contribute nothing:

    R(adaptive) - R(Joint)  <=  Pr[A_b = 1, Delta_b]        (the UNFLAGGED-divergence probability)

Directly measurable offline, needs no ground truth, and valid under complete decoder
misspecification -- exactly like `P7`, of which it is a restriction.

THE POINT OF THE FILE IS THE PLACEBO BATTERY. `N9` was retracted because random lags, permuted
weights and deliberately expensive weights all scored identically. So every score below is run
against the same controls at the SAME fallback rate:

    LAGSUM   raw same-ancilla coincidence counts at the measured excess lags {2,3,4,6,8,12}
    LAGEXC   the same counts MINUS their within-shot random-placement expectation -- the only
             feature that isolates long-memory structure from the shot's own event rate
    RATE     total detector count. NOT a placebo: a legitimate, free competitor. If RATE matches
             LAGEXC then the long-range structure contributes nothing and Route A is rate-selection
             wearing a correlation costume.
    SEAMW    seam syndrome weight from the windowed decode itself. Also not a placebo: the strongest
             honest competitor, observable at decode time with no model claim at all.
    RANDLAG  the LAGEXC construction at lags drawn from {17..40}, where the measured excess is at
             the noise floor. The direct analogue of the `N9` random-lag placebo.
    PERMUTE  the LAGEXC construction on a per-ancilla time-permuted copy of the record: same counts,
             temporal order destroyed.
    FLAT     constant score, i.e. flag a uniformly random fraction. The null.

Thresholds are set on the FIT half (shots 0..4999) as a quantile of the score -- a
distribution-only quantity, no labels -- and applied to the HELD-OUT half. Weights are the frozen
fit-half calibration used everywhere else in this project.

Usage:  python tools/route_a.py
"""
import json
import os
import sys

import numpy as np
from scipy.stats import beta, binomtest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detectors import build_detectors, load  # noqa: E402
from parallel_window import two_window  # noqa: E402
from weight_model_robustness import fit_weights_v2  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B_GRID = [1, 2, 3, 4, 6, 8]
MEASURED_LAGS = [2, 3, 4, 6, 8, 12]      # lags with measured device excess (ledger `P4`)
NULL_LAGS_POOL = list(range(17, 41))     # lags where the measured excess is at the noise floor
SCORES = ("LAGEXC", "LAGSUM", "RATE", "SEAMW", "RANDLAG", "PERMUTE", "FLAT")
PHIS = [0.02, 0.05, 0.10, 0.20]
W1 = 25
DELTA = 0.05


def cp_upper(k, n, alpha):
    return 1.0 if k >= n else float(beta.ppf(1.0 - alpha, k + 1, n - k))


def lag_counts(D, lags):
    """Per-shot same-ancilla coincidence counts at each lag, and their within-shot expectation.

    obs[s, i] = sum over ancillas j and layers r of D[s,r,j] * D[s,r+lag_i,j].
    exp[s, i] = the same quantity if that ancilla's firings were placed uniformly at random over its
                own layers holding the per-ancilla count fixed: (nr - lag) * c/nr * (c-1)/(nr-1).

    The difference isolates temporal STRUCTURE from the shot's own event rate, which is exactly the
    separation `N9` failed to make.
    """
    shots, nr, na = D.shape
    Df = D.astype(np.float64)
    c = Df.sum(axis=1)                                    # (shots, na) per-ancilla counts
    pair = c * (c - 1.0) / (nr * (nr - 1.0))
    obs = np.empty((shots, len(lags)))
    exp = np.empty((shots, len(lags)))
    for i, k in enumerate(lags):
        obs[:, i] = (Df[:, :-k, :] * Df[:, k:, :]).sum(axis=(1, 2))
        exp[:, i] = ((nr - k) * pair).sum(axis=1)
    return obs, exp


def permute_time(D, rng):
    """Per-shot, per-ancilla random permutation of the layer axis. Counts kept, order destroyed."""
    shots, nr, na = D.shape
    out = np.empty_like(D)
    for j in range(na):
        idx = np.argsort(rng.random((shots, nr)), axis=1)
        out[:, :, j] = np.take_along_axis(D[:, :, j], idx, axis=1)
    return out


def build_scores(D, seam_w, rng):
    """Every score, on one half of the data. Higher score = more suspicious = flagged first."""
    obs, exp = lag_counts(D, MEASURED_LAGS)
    null_lags = sorted(rng.choice(NULL_LAGS_POOL, size=len(MEASURED_LAGS), replace=False).tolist())
    obs_n, exp_n = lag_counts(D, null_lags)
    obs_p, exp_p = lag_counts(permute_time(D, rng), MEASURED_LAGS)
    return {
        "LAGSUM": obs.sum(axis=1),
        "LAGEXC": (obs - exp).sum(axis=1),
        "RATE": D.sum(axis=(1, 2)).astype(np.float64),
        "SEAMW": seam_w.astype(np.float64),
        "RANDLAG": (obs_n - exp_n).sum(axis=1),
        "PERMUTE": (obs_p - exp_p).sum(axis=1),
        "FLAT": rng.random(D.shape[0]),
    }, null_lags


def tie_rule(score_fit, phi, method="linear"):
    """Calibrate the randomized threshold rule on the FIT half alone.

    Returns `(thr, q)`: flag when `s > thr`, and flag a tied shot with independent probability `q`.
    `q` is the fit half's own deficit, `(phi - Pr[s > thr]) / Pr[s == thr]`, so the rule is fixed
    before a single evaluation shot is seen.

    `method` is the quantile interpolation. `"linear"` (numpy's default, and what round 7 was
    pre-registered and run with) can put `thr` strictly between two attained values of a discrete
    score, in which case `Pr[s == thr] = 0`, `q = 0`, and the realized rate is whatever
    `Pr[s > thr]` happens to be -- it missed `phi` by up to 7% relative in round 7, differently for
    different scores. `"lower"` snaps `thr` to an attained value so `q` can correct the rate
    exactly. Round 7 reports `"lower"` as a labelled post-hoc diagnostic, never as the verdict.
    """
    thr = float(np.quantile(score_fit, 1 - phi, method=method))
    p_above = float((score_fit > thr).mean())
    p_tie = float((score_fit == thr).mean())
    q = 0.0 if p_tie <= 0 else min(1.0, max(0.0, (phi - p_above) / p_tie))
    return thr, q


def flag_at(score_fit, score_ev, phi, n=None, rng=None, method="linear"):
    """Flag the top-phi fraction under an ONLINE, outcome-independent rule.

    🔴 The previous version admitted ties as `tie[:need]` -- **the earliest evaluation shots in
    collection order**. That is not a decision rule (it needs the whole evaluation half in hand and
    it needs `need`, which depends on the evaluation half's own tie count), and under the block drift
    measured in this project it can bias catch rates. Figure-audit finding 5.

    The randomized threshold rule below is calibrated on the fit half only and decides each shot
    independently, so the realized flag rate is random around `phi` rather than forced to it. `rng`
    is that tie stream; sweeping it is how the seed sd in `tiebreak_sensitivity.py` is produced.
    `n` is accepted and ignored, so callers frozen against the old signature keep working.
    """
    thr, q = tie_rule(score_fit, phi, method=method)
    flag = score_ev > thr
    if q > 0.0:
        tie = score_ev == thr
        if tie.any():
            rng = np.random.default_rng(0) if rng is None else rng
            flag = flag | (tie & (rng.random(score_ev.shape) < q))
    return flag


def main():
    syn, fin, _ = load(os.path.join(ROOT, "data", "da7miljsq5js73bk4vtg.npz"))
    Ddev = build_detectors(syn, fin)
    fit, ev = slice(0, 5000), slice(5000, 10000)
    n = 5000
    wd, td, _, _ = fit_weights_v2(Ddev[fit])
    alpha = DELTA / (len(B_GRID) * len(PHIS))

    print("ROUTE A -- PREDICT DIVERGENCE, DO NOT CORRECT IT")
    print("  certificate: R(adaptive) - R(Joint) <= Pr[A_b = 1, Delta_b]   (unflagged divergence)")
    print("  thresholds from the FIT half as score quantiles; evaluated on the HELD-OUT half")
    print(f"  measured lags {MEASURED_LAGS}; null-lag pool "
          f"{NULL_LAGS_POOL[0]}..{NULL_LAGS_POOL[-1]}\n")

    rows = []
    for b in B_GRID:
        rf = two_window(Ddev[fit], W1, b, logical_j=0, eps=1e-3, seed=0, wtab=wd, ttab=td)
        re_ = two_window(Ddev[ev], W1, b, logical_j=0, eps=1e-3, seed=0, wtab=wd, ttab=td)
        dv = re_["diverged_repaired"].astype(bool)
        n_div = int(dv.sum())
        sf, null_lags = build_scores(Ddev[fit], rf["seam_weight"], np.random.default_rng(1000 + b))
        se, _ = build_scores(Ddev[ev], re_["seam_weight"], np.random.default_rng(1000 + b))

        print(f"=== b = {b}   held-out divergences: {n_div}/{n}   "
              f"random-lag draw: {null_lags}")
        print(f"{'score':>9} {'phi':>6} {'flagged':>8} {'caught':>7} {'UNFLAGGED':>10} "
              f"{'cert U/n':>10} {'CP UB':>10} {'p vs FLAT':>10}")
        for phi in PHIS:
            for name in SCORES:
                flag = flag_at(sf[name], se[name], phi, n)
                caught = int((dv & flag).sum())
                unflag = n_div - caught
                fr = float(flag.mean())
                ub = cp_upper(unflag, n, alpha)
                pv = (float(binomtest(caught, n_div, min(fr, 1.0), alternative="greater").pvalue)
                      if n_div > 0 else float("nan"))
                print(f"{name:>9} {fr:>6.3f} {int(flag.sum()):>8} {caught:>7} {unflag:>10} "
                      f"{unflag / n:>10.5f} {ub:>10.2e} {pv:>10.4f}")
                rows.append(dict(b=b, score=name, phi=phi, flag_rate=fr,
                                 n_flagged=int(flag.sum()), n_div=n_div, caught=caught,
                                 unflagged=unflag, cert=unflag / n, cert_ub=ub, p_vs_flat=pv,
                                 null_lags=null_lags))
            print()

    # ------------------------------------------------------------------ the deployable rule
    # A quantile threshold is a fitted object and needs a fit half. The seam syndrome weight admits
    # a THRESHOLD-FREE rule -- "escalate whenever the stitched correction fails to explain the
    # syndrome by more than t detectors" -- with t a small integer fixed in advance. That is what
    # would actually be shipped, so it is what the certificate should be quoted for.
    n_layers = Ddev.shape[1]
    print("=" * 100)
    print("DEPLOYABLE RULE: escalate when seam weight > t. No fitting, no threshold search.")
    print("Critical path in DETECTOR-LAYERS per shot, offline scheduling only (guide 7.3):")
    print("    joint         n")
    print("    fixed split   n/2 + b        (the longer of the two windows, decoded concurrently)")
    print("    adaptive      n/2 + b + esc*n")
    print("** The escalated shots pay the windowed decode AND THEN the joint one -- the seam is not")
    print("   visible until the windows have been decoded, so escalation ADDS to the path, it does")
    print("   not replace it. An earlier version of this file divided instead of adding and reported")
    print("   1.79x at b=1 where the correct figure is 1.64x.")
    print(f"{'b':>3} {'t':>3} {'escalated':>10} {'esc rate':>9} {'div':>4} {'UNFLAGGED':>10} "
          f"{'cert U/n':>10} {'CP UB':>10} {'path(b)':>8} {'path(adapt)':>12} {'speedup':>8}")
    dep = []
    for b in B_GRID:
        re_ = two_window(Ddev[ev], W1, b, logical_j=0, eps=1e-3, seed=0, wtab=wd, ttab=td)
        dv = re_["diverged_repaired"].astype(bool)
        sw = re_["seam_weight"]
        n_div = int(dv.sum())
        path_fixed = n_layers / 2.0 + b
        for t in (0, 1, 2, 4):
            flag = sw > t
            unflag = int((dv & ~flag).sum())
            esc = float(flag.mean())
            path_adapt = path_fixed + esc * n_layers
            ub = cp_upper(unflag, n, alpha)
            print(f"{b:>3} {t:>3} {int(flag.sum()):>10} {esc:>9.3f} {n_div:>4} {unflag:>10} "
                  f"{unflag / n:>10.5f} {ub:>10.2e} {path_fixed:>8.1f} {path_adapt:>12.1f} "
                  f"{n_layers / path_adapt:>7.2f}x")
            dep.append(dict(b=b, t=t, n_escalated=int(flag.sum()), escalation_rate=esc,
                            n_div=n_div, unflagged=unflag, cert=unflag / n, cert_ub=ub,
                            path_layers_fixed=path_fixed, path_layers_adaptive=path_adapt,
                            speedup_fixed=n_layers / path_fixed,
                            speedup_adaptive=n_layers / path_adapt))
        print()

    out = os.path.join(ROOT, "data", "route_a.json")
    tmp = out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(dict(b_grid=B_GRID, phis=PHIS, measured_lags=MEASURED_LAGS,
                       null_lag_pool=[NULL_LAGS_POOL[0], NULL_LAGS_POOL[-1]],
                       delta=DELTA, alpha_per_cell=alpha, n_eval=n, rows=rows,
                       deployable=dep), fh, indent=1)
    os.replace(tmp, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
