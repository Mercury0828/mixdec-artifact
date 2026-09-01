#!/usr/bin/env python
"""The device-vs-surrogate comparison as ONE 5% familywise procedure (ledger `B14`, item 4).

Why this file exists. The original comparison did two things in sequence: it put a Clopper-Pearson
upper limit on the SURROGATE's divergence probability at level `delta/K`, then tested the DEVICE
count against that limit at level `delta/K` again. The web-Pro audit rejected the composition:

    "Using a random CP upper limit as if it were a fixed null and then spending a second full error
     budget on the device count does not compose. Use a direct two-sample exact comparison per `b`
     with simultaneous multiplicity control, or a simultaneous interval for the difference, or split
     the budget explicitly. `b=1` survives easily; the weakest `p = 2.8e-3` conclusion must be
     recomputed under the joint procedure."

The CP limit is a random quantity computed from the same 20,000 surrogate shots that the second
stage then re-uses, so the two stages are neither independent nor nested, and the nominal 5% is not
the procedure's actual familywise level.

What is done instead — one error budget, spent once:

🔴 CORRECTED 2026-08-29 UNDER CROSS-MODEL AUDIT. The first version of this file called the two
constructions below "one 5% familywise procedure" with "delta spent once". They were not: the Fisher
family received a full delta through delta/K, and the simultaneous interval independently received
another full delta through delta/(2K) per side -- two separately 5%-controlled families, so joint
validity of every decision AND every interval was never at 5%. The budget is now SPLIT explicitly,
which is what makes the "one budget" statement true.

  PRIMARY   half the budget, DELTA/2. Per buffer width, Fisher's exact one-sided test of the 2x2
            table
                          diverged   not
                device      k_dev     n_dev - k_dev
                surrogate   k_ind     n_ind - k_ind
            against `H0: q_dev <= q_ind`, Bonferroni-corrected over the K widths at (delta/2)/K.

  EFFECT    the other half, DELTA/2. A simultaneous one-sided lower bound on `q_dev - q_ind`: an
            exact CP lower bound on `q_dev` and an exact CP upper bound on `q_ind`, each at
            (delta/2)/(2K). Valid jointly over all K widths and both arms by the union bound, and a
            bound on a DIFFERENCE rather than a comparison of a point estimate against a random
            limit.

Every decision and every interval below therefore holds jointly at 5%. Two further limits the audit
recorded, which no re-splitting fixes and which must be stated wherever this is quoted:
  - Fisher is exact CONDITIONAL on the disjoint fit half that produced the surrogate's parameters. It
    is NOT a goodness-of-fit test of the composite independent-graph-edge model -- estimation error
    in those parameters is not propagated.
  - Exactness also needs i.i.d. evaluation shots. The evaluation half is later pubs in collection
    order and drift is acknowledged elsewhere in this project, so the quoted p-values and CP coverage
    are not justified over device time without a block or drift analysis.


Usage:  python tools/joint_surrogate_test.py
"""
import json
import os
import sys

import numpy as np
from scipy.stats import beta, fisher_exact

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detectors import build_detectors, load  # noqa: E402
from independence_model import sample_independent  # noqa: E402
from parallel_window import two_window  # noqa: E402
from weight_model_robustness import fit_weights_v2  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B_GRID = [1, 2, 3, 4, 6, 8, 12, 16]
# Self-describing multiplicity, so an artifact can never disagree with the figure that
# reads it or with the pre-registration. Round 7.
MULTIPLICITY = dict(
    budget="delta = 0.05 split in half across the two families reported here",
    primary=dict(family="Fisher exact, device vs surrogate, one per width",
                 members=len(B_GRID), sided="two", level_per_member="(delta/2)/K"),
    effect=dict(family="Clopper-Pearson endpoints paired into a difference lower bound",
                members=2 * len(B_GRID), sided="one per endpoint",
                level_per_member="(delta/2)/(2K)"))
DELTA = 0.05
W1 = 25
N_IND = 20000


def cp_lower(k, n, alpha):
    """Exact one-sided lower confidence bound on a binomial probability."""
    return 0.0 if k == 0 else float(beta.ppf(alpha, k, n - k + 1))


def cp_upper(k, n, alpha):
    """Exact one-sided upper confidence bound on a binomial probability."""
    return 1.0 if k >= n else float(beta.ppf(1.0 - alpha, k + 1, n - k))


def main():
    syn, fin, _ = load(os.path.join(ROOT, "data", "da7miljsq5js73bk4vtg.npz"))
    Ddev = build_detectors(syn, fin)
    fit, ev = slice(0, 5000), slice(5000, 10000)
    n_dev = 5000
    K = len(B_GRID)
    # the budget is SPLIT between the two families, so the pair holds jointly at DELTA
    alpha_test = (DELTA / 2) / K            # PRIMARY: half the budget, Bonferroni over K
    alpha_side = (DELTA / 2) / (2 * K)      # EFFECT:  half the budget, K widths x 2 arms

    wd, td, ps, pt = fit_weights_v2(Ddev[fit])
    Dind = sample_independent(ps, pt, N_IND, np.random.default_rng(0))

    print("DEVICE vs FITTED-SURROGATE, AS ONE 5% FAMILYWISE PROCEDURE")
    print(f"  device rate 0-1 detectors {Ddev[:, 1:-1, :].mean():.4f}   "
          f"surrogate {Dind[:, 1:-1, :].mean():.4f}")
    print(f"  K = {K} widths, total delta = {DELTA} SPLIT between the two families: "
          f"primary alpha/test = {alpha_test:.6f}, "
          f"effect-bound alpha/side = {alpha_side:.6f}\n")
    print(f"{'b':>3} {'k_dev/n':>10} {'k_ind/n':>11} {'Fisher p':>11} {'p x K':>10} "
          f"{'sig@5% FWER':>12} {'LB(q_dev)':>11} {'UB(q_ind)':>11} {'LB(diff)':>11} {'diff>0':>7}")

    rows = []
    for b in B_GRID:
        k_dev = int(two_window(Ddev[ev], W1, b, logical_j=0, eps=1e-3, seed=0,
                               wtab=wd, ttab=td)["diverged_repaired"].sum())
        k_ind = int(two_window(Dind, W1, b, logical_j=0, eps=1e-3, seed=0,
                               wtab=wd, ttab=td)["diverged_repaired"].sum())
        table = [[k_dev, n_dev - k_dev], [k_ind, N_IND - k_ind]]
        p = float(fisher_exact(table, alternative="greater")[1])
        p_adj = min(1.0, p * K)
        lo_dev = cp_lower(k_dev, n_dev, alpha_side)
        up_ind = cp_upper(k_ind, N_IND, alpha_side)
        lb_diff = lo_dev - up_ind
        sig = p_adj < DELTA / 2
        print(f"{b:>3} {f'{k_dev}/{n_dev}':>10} {f'{k_ind}/{N_IND}':>11} {p:>11.2e} "
              f"{p_adj:>10.2e} {('** YES **' if sig else 'no'):>12} {lo_dev:>11.2e} "
              f"{up_ind:>11.2e} {lb_diff:>11.2e} {('YES' if lb_diff > 0 else 'no'):>7}")
        rows.append(dict(b=b, k_dev=k_dev, n_dev=n_dev, k_ind=k_ind, n_ind=N_IND,
                         fisher_p=p, p_bonferroni=p_adj, significant=bool(sig),
                         cp_lower_device=lo_dev, cp_upper_surrogate=up_ind,
                         lower_bound_difference=lb_diff))

    sig_b = [r["b"] for r in rows if r["significant"]]
    eff_b = [r["b"] for r in rows if r["lower_bound_difference"] > 0]
    worst = max((r["p_bonferroni"] for r in rows if r["significant"]), default=None)
    print(f"\n  significant at 5% FWER (primary): b = {sig_b}")
    print(f"  simultaneous lower bound on the difference > 0 at 5%: b = {eff_b}")
    print(f"  weakest surviving adjusted p: {worst:.2e}" if worst else "  none survive")

    out = os.path.join(ROOT, "data", "joint_surrogate_test.json")
    tmp = out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(dict(delta=DELTA, K=K, alpha_primary_per_test=alpha_test,
                       alpha_effect_per_side=alpha_side, W1=W1,
                       multiplicity=MULTIPLICITY,
                       significant_widths=sig_b, effect_positive_widths=eff_b,
                       weakest_adjusted_p=worst, rows=rows), fh, indent=1)
    os.replace(tmp, out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
