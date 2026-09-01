#!/usr/bin/env python
"""Campaign R — the pre-registered analysis. Nothing here may be changed now the shots exist.

Pipeline frozen in `docs/R_PREREGISTRATION.md` (commit `6b759df`, amended `775ecb8`, both before
submission). Falsifiers R-1 .. R-8 are evaluated verbatim and reported whether or not they fire.

🔴 ONE THING THE PRE-REGISTRATION GOT WRONG, DISCOVERED AT RETRIEVAL AND DECLARED HERE.
It assumed the prepared logical state is invisible in the noise, so that the two arms would be
exchangeable and a blinded decoder would see one distribution. On this device they are NOT
exchangeable. Measured over 100,000 shots per arm:

                              Y = 0        Y = 1
    bulk detector rate        0.03993      0.08527      (2.1x)
    detectors per shot        16.22        35.26
    final readout mean        0.0066       0.2175       <- should be ~1.0 for Y = 1
    raw majority wrong        0/100,000    97,372/100,000

The logical-1 state decays toward |0...0> under amplitude damping over the 50 rounds: only 21.8% of
data qubits still read 1 at the end, which is *below* the 50% floor any symmetric bit-flip channel
could produce, so this is T1, not depolarising noise. `exp(-T/T1) = 0.2175` gives `T ~ 1.5 T1`.

Consequences, both declared rather than absorbed:
  - The Y = 0 arm is a faithful replication of the cached condition (detector rate 0.03993 against
    the frozen job's 0.0402), so **R-1 .. R-4 and R-6/R-7 are evaluated on the Y = 0 arm**, with its
    own 50,000-shot fit half, exactly the protocol being replicated.
  - The Y = 1 arm is not a second draw from the same population. It is analysed separately, and it
    turns out to be a far **stronger** test of ledger `B13` than the one designed: the raw majority
    fails on 97.4% of its shots, so any decoder that has learned to defer to the terminal readout is
    now visibly, catastrophically wrong. R-5's falsifier ("raw majority correct on all 200,000") does
    not fire.
  - 🔴 "Randomize the logical state" is therefore **necessary but not sufficient AND not free**: on
    superconducting hardware the two logical states of a bit-flip code are not equally protected.
    A symmetric randomization needs a phase-flip (X-basis) code, or few enough rounds that
    `T << T1`. Recorded as a barrier.

Usage:  python tools/analyze_campaign_r.py
"""
import json
import os
import sys

import numpy as np
from scipy.stats import beta, binomtest, fisher_exact

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detectors import build_detectors, load  # noqa: E402
from independence_model import sample_independent  # noqa: E402
from parallel_window import two_window  # noqa: E402
from route_a import build_scores, flag_at  # noqa: E402
from weight_model_robustness import fit_weights_v2  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "campaign_r")
B_GRID = [1, 2, 3, 4, 6, 8, 12, 16]
PHIS = [0.02, 0.05, 0.10, 0.20]
W1 = 25
DELTA = 0.05
N_HALF = 50_000
N_SURR = 200_000
SCORES = ("LAGEXC", "LAGSUM", "RATE", "SEAMW", "RANDLAG", "PERMUTE", "FLAT")


def cp_upper(k, n, a):
    return 1.0 if k >= n else float(beta.ppf(1.0 - a, k + 1, n - k))


def cp_lower(k, n, a):
    return 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))


def load_arm(y):
    """Concatenate the four pubs of one logical state IN COLLECTION ORDER.

    Pub index is collection order, so the pre-registered "first 50,000 = fit, last 50,000 =
    evaluation" split is a split in time, which is what makes the drift claim in R-2 meaningful.
    """
    with open(os.path.join(ROOT, "data", "campaign_r_manifest.json")) as fh:
        man = json.load(fh)
    idx = [i for i, yy in enumerate(man["pub_logical_states"]) if yy == y]
    syn, fin = [], []
    for i in idx:
        s, f, _ = load(os.path.join(CACHE, f"pub{i}.npz"))
        syn.append(s)
        fin.append(f)
    return np.concatenate(syn), np.concatenate(fin), idx


def logical_estimate(fin, corr):
    """Codeword-level logical readout: majority of the corrected data-qubit values."""
    fixed = fin ^ corr
    return (fixed.sum(axis=1) * 2 > fixed.shape[1]).astype(np.uint8)


def decode_grid(D, wtab, ttab, tag):
    """Run the frozen two-window decode at every buffer width. Returns {b: result dict}."""
    out = {}
    for b in B_GRID:
        out[b] = two_window(D, W1, b, logical_j=0, eps=1e-3, seed=0, wtab=wtab, ttab=ttab)
        print(f"    [{tag}] b={b:>2}  diverged {int(out[b]['diverged_repaired'].sum()):>6}",
              flush=True)
    return out


def main():
    report = {}
    K = len(B_GRID)
    alpha_test = DELTA / K
    alpha_side = DELTA / (2 * K)

    # ---------------------------------------------------------------- section 0: the two arms
    print("=" * 100)
    print("SECTION 0 -- THE TWO ARMS, AND R-5")
    arms = {}
    for y in (0, 1):
        syn, fin, idx = load_arm(y)
        D = build_detectors(syn, fin)
        maj = (fin.sum(axis=1) * 2 > fin.shape[1]).astype(np.uint8)
        arms[y] = dict(D=D, fin=fin, maj=maj, pubs=idx)
        print(f"  Y={y}  pubs {idx}  n={len(D)}  bulk detector rate {D[:, 1:-1, :].mean():.5f}  "
              f"detectors/shot {D.sum(axis=(1, 2)).mean():.2f}  fin mean {fin.mean():.4f}  "
              f"raw majority wrong {int((maj != y).sum())}")
        report[f"arm_Y{y}"] = dict(n=len(D), pubs=idx,
                                   bulk_detector_rate=float(D[:, 1:-1, :].mean()),
                                   detectors_per_shot=float(D.sum(axis=(1, 2)).mean()),
                                   fin_mean=float(fin.mean()),
                                   raw_majority_wrong=int((maj != y).sum()))
    n_maj_fail = sum(report[f"arm_Y{y}"]["raw_majority_wrong"] for y in (0, 1))
    print(f"\n  R-5: raw majority fails on {n_maj_fail}/200,000 shots. "
          f"Falsifier ('correct on all 200,000') {'FIRES' if n_maj_fail == 0 else 'does NOT fire'}.")
    report["R5"] = dict(majority_failures=n_maj_fail, falsifier_fired=bool(n_maj_fail == 0))

    # ------------------------------------------------- section 1: R-1..R-4 on the Y=0 arm
    print("\n" + "=" * 100)
    print("SECTION 1 -- R-1 .. R-4, THE REPLICATION, ON THE Y=0 ARM")
    D0 = arms[0]["D"]
    fitsl, evsl = slice(0, N_HALF), slice(N_HALF, 2 * N_HALF)
    print("  fitting weights on the 50,000-shot fit half (cap justified at n_fit=50,000)...",
          flush=True)
    wd, td, ps, pt = fit_weights_v2(D0[fitsl], n_fit=N_HALF)
    print("  sampling the independence surrogate (200,000 shots)...", flush=True)
    Dind = sample_independent(ps, pt, N_SURR, np.random.default_rng(0))
    print(f"  device bulk rate {D0[:, 1:-1, :].mean():.5f}   "
          f"surrogate {Dind[:, 1:-1, :].mean():.5f}", flush=True)

    print("  decoding the device evaluation half...", flush=True)
    dev_ev = decode_grid(D0[evsl], wd, td, "dev-eval")
    print("  decoding the device fit half (needed for Route A thresholds)...", flush=True)
    dev_fit = decode_grid(D0[fitsl], wd, td, "dev-fit")
    print("  decoding the surrogate...", flush=True)
    surr = decode_grid(Dind, wd, td, "surrogate")

    print(f"\n{'b':>3} {'k_dev/50k':>12} {'k_ind/200k':>12} {'dev rate':>10} {'Fisher p':>11} "
          f"{'p x 8':>10} {'sig':>5} {'LB(diff)':>11}")
    rows = []
    for b in B_GRID:
        kd = int(dev_ev[b]["diverged_repaired"].sum())
        ki = int(surr[b]["diverged_repaired"].sum())
        p = float(fisher_exact([[kd, N_HALF - kd], [ki, N_SURR - ki]], alternative="greater")[1])
        padj = min(1.0, p * K)
        lb = cp_lower(kd, N_HALF, alpha_side) - cp_upper(ki, N_SURR, alpha_side)
        print(f"{b:>3} {f'{kd}/{N_HALF}':>12} {f'{ki}/{N_SURR}':>12} {kd / N_HALF:>10.5f} "
              f"{p:>11.2e} {padj:>10.2e} {('YES' if padj < DELTA else 'no'):>5} {lb:>11.2e}")
        rows.append(dict(b=b, k_dev=kd, k_ind=ki, dev_rate=kd / N_HALF, ind_rate=ki / N_SURR,
                         fisher_p=p, p_bonferroni=padj, significant=bool(padj < DELTA),
                         lower_bound_difference=lb))
    report["R1_R4"] = rows

    sig = [r["b"] for r in rows if r["significant"]]
    core = [b for b in (1, 2, 3, 4, 6, 8) if b in sig]
    r1_fires = len(core) < 5
    print(f"\n  R-1: {len(core)}/6 of b in (1,2,3,4,6,8) significant at 5% FWER. "
          f"Falsifier (<5) {'FIRES -- P5 does not replicate' if r1_fires else 'does NOT fire'}.")
    cached = {1: 0.00360, 2: 0.00320, 3: 0.00340, 4: 0.00320, 6: 0.00220, 8: 0.00120,
              12: 0.00060, 16: 0.00040}
    print(f"  R-2: point estimates vs cached held-out (factor-of-2 band):")
    r2_fires = False
    for r in rows:
        ratio = r["dev_rate"] / cached[r["b"]] if cached[r["b"]] else float("nan")
        bad = not (0.5 <= ratio <= 2.0)
        r2_fires = r2_fires or (r["b"] in (1, 8) and bad)
        print(f"      b={r['b']:>2}  {r['dev_rate']:.5f} vs {cached[r['b']]:.5f}  "
              f"ratio {ratio:.2f}{'   <- outside band' if bad else ''}")
    k12, k16 = rows[6]["k_dev"], rows[7]["k_dev"]
    print(f"  R-3: b=12 -> {k12} events, b=16 -> {k16}. "
          f"Falsifier (<40 at both) {'FIRES' if (k12 < 40 and k16 < 40) else 'does NOT fire'}.")
    print(f"  R-4: surrogate at b>=12 -> {rows[6]['k_ind']}, {rows[7]['k_ind']}. "
          f"Departure {'noted' if (rows[6]['k_ind'] or rows[7]['k_ind']) else 'none'}.")
    report["verdicts"] = dict(R1_fired=bool(r1_fires), R1_significant_core=core,
                              R2_fired=bool(r2_fires),
                              R3_fired=bool(k12 < 40 and k16 < 40),
                              R4_departure=bool(rows[6]["k_ind"] or rows[7]["k_ind"]))

    # ------------------------------------------------- section 2: R-6, R-7 -- Route A
    print("\n" + "=" * 100)
    print("SECTION 2 -- R-6 AND R-7, ROUTE A, ON THE Y=0 ARM")
    print(f"{'b':>3} {'escalated':>10} {'esc rate':>9} {'div':>5} {'UNFLAGGED':>10} "
          f"{'cert':>9} {'CP UB':>10}")
    dep = []
    a_dep = DELTA / len(B_GRID)
    for b in B_GRID:
        dv = dev_ev[b]["diverged_repaired"].astype(bool)
        flag = dev_ev[b]["seam_weight"] > 0
        unf = int((dv & ~flag).sum())
        esc = float(flag.mean())
        ub = cp_upper(unf, N_HALF, a_dep)
        print(f"{b:>3} {int(flag.sum()):>10} {esc:>9.4f} {int(dv.sum()):>5} {unf:>10} "
              f"{unf / N_HALF:>9.5f} {ub:>10.2e}")
        dep.append(dict(b=b, n_escalated=int(flag.sum()), escalation_rate=esc,
                        n_div=int(dv.sum()), unflagged=unf, cert=unf / N_HALF, cert_ub=ub))
    report["R6_deployable"] = dep
    e1 = dep[0]
    r6_fires = (e1["escalation_rate"] > 0.15) or (e1["cert_ub"] > 2 * cp_upper(
        int(dev_ev[1]["diverged_repaired"].sum()), N_HALF, a_dep))
    print(f"\n  R-6: b=1 escalation {e1['escalation_rate']:.2%}, unflagged bound "
          f"{e1['cert_ub']:.2e} (target < 5e-4). Falsifier "
          f"{'FIRES -- P9 withdrawn' if r6_fires else 'does NOT fire'}.")

    print("\n  R-7 -- the control N10 must keep failing. Divergences caught, LAGEXC vs RATE:")
    print(f"{'b':>3} {'phi':>6} {'LAGEXC':>8} {'LAGSUM':>8} {'RATE':>7} {'SEAMW':>7} "
          f"{'RANDLAG':>8} {'PERMUTE':>8} {'FLAT':>6} {'p(LAGEXC>RATE)':>15}")
    beats, cells = 0, []
    for b in B_GRID:
        dv = dev_ev[b]["diverged_repaired"].astype(bool)
        nd = int(dv.sum())
        sf, _ = build_scores(D0[fitsl], dev_fit[b]["seam_weight"], np.random.default_rng(1000 + b))
        se, _ = build_scores(D0[evsl], dev_ev[b]["seam_weight"], np.random.default_rng(1000 + b))
        for phi in PHIS:
            got = {}
            for nm in SCORES:
                fl = flag_at(sf[nm], se[nm], phi, N_HALF)
                got[nm] = (int((dv & fl).sum()), float(fl.mean()))
            cl, _ = got["LAGEXC"]
            cr, fr = got["RATE"]
            # does LAGEXC catch more than RATE, at RATE's own flag rate, as a binomial excess?
            pv = float(binomtest(cl, nd, cr / nd if nd else 0.5,
                                 alternative="greater").pvalue) if nd and cr else float("nan")
            hit = bool(cl > cr and pv == pv and pv * len(B_GRID) * len(PHIS) < DELTA)
            beats += hit
            print(f"{b:>3} {phi:>6.2f} {got['LAGEXC'][0]:>8} {got['LAGSUM'][0]:>8} "
                  f"{cr:>7} {got['SEAMW'][0]:>7} {got['RANDLAG'][0]:>8} "
                  f"{got['PERMUTE'][0]:>8} {got['FLAT'][0]:>6} "
                  f"{pv:>15.4f}{'  <- BEATS' if hit else ''}")
            cells.append(dict(b=b, phi=phi, n_div=nd,
                              caught={k: v[0] for k, v in got.items()},
                              flag_rate={k: v[1] for k, v in got.items()},
                              p_lagexc_beats_rate=pv, beats=hit))
    r7_fires = beats >= 5
    print(f"\n  R-7: LAGEXC beats RATE with corrected significance at {beats} of "
          f"{len(B_GRID) * len(PHIS)} cells. Falsifier (>=5) "
          f"{'FIRES -- N10 is WRONG and is withdrawn' if r7_fires else 'does NOT fire; N10 stands'}.")
    report["R7_cells"] = cells
    report["verdicts"]["R6_fired"] = bool(r6_fires)
    report["verdicts"]["R7_fired"] = bool(r7_fires)

    # ------------------------------------------------- section 3: R-8 -- harm, with Y known
    print("\n" + "=" * 100)
    print("SECTION 3 -- R-8, THE HARMFUL/BENEFICIAL TABLE, WITH THE LOGICAL TRUTH KNOWN")
    harm = {}
    for y in (0, 1):
        Dy = arms[y]["D"]
        finy = arms[y]["fin"]
        wy, ty, _, _ = fit_weights_v2(Dy[fitsl], n_fit=N_HALF)
        print(f"\n  Y = {y}   (weights fitted on this arm's own fit half)")
        print(f"{'b':>3} {'Delta':>7} {'harmful':>8} {'benef':>7} {'h-g':>10} "
              f"{'LER joint':>10} {'LER split':>10}")
        rowsy = []
        for b in B_GRID:
            r = two_window(Dy[evsl], W1, b, logical_j=0, eps=1e-3, seed=0, wtab=wy, ttab=ty)
            yj = logical_estimate(finy[evsl], r["corr_joint"])
            yr = logical_estimate(finy[evsl], r["corr_repaired"])
            h = int(((yj == y) & (yr != y)).sum())
            g = int(((yj != y) & (yr == y)).sum())
            d = int((yj != yr).sum())
            print(f"{b:>3} {d:>7} {h:>8} {g:>7} {(h - g) / N_HALF:>10.5f} "
                  f"{(yj != y).mean():>10.5f} {(yr != y).mean():>10.5f}")
            rowsy.append(dict(b=b, delta_codeword=d, harmful=h, beneficial=g,
                              net_regret=(h - g) / N_HALF,
                              ler_joint=float((yj != y).mean()),
                              ler_split=float((yr != y).mean())))
        harm[y] = rowsy
    report["R8"] = {str(k): v for k, v in harm.items()}
    frac1 = harm[0][0]["harmful"] / max(1, harm[0][0]["harmful"] + harm[0][0]["beneficial"])
    r8_fires = (frac1 < 0.60) or any(r["net_regret"] <= 0 for r in harm[0] if r["b"] <= 4)
    print(f"\n  R-8 (Y=0 arm): {frac1:.1%} of divergences harmful at b=1. Falsifier "
          f"{'FIRES' if r8_fires else 'does NOT fire'}.")
    report["verdicts"]["R8_fired"] = bool(r8_fires)

    # ------------------------------------------------- section 4: the deference test
    print("\n" + "=" * 100)
    print("SECTION 4 -- LEDGER B13: DOES THE DECODER ACTUALLY USE THE SYNDROME?")
    print("  Override value R(T) - R(F) = Pr[C=1,E=1] - Pr[C=1,E=0], C = decoder overrides the")
    print("  terminal-only majority, E = that majority was wrong. Positive => the syndrome helps.")
    ov = {}
    for y in (0, 1):
        Dy, finy = arms[y]["D"], arms[y]["fin"]
        wy, ty, _, _ = fit_weights_v2(Dy[fitsl], n_fit=N_HALF)
        r = two_window(Dy[evsl], W1, 16, logical_j=0, eps=1e-3, seed=0, wtab=wy, ttab=ty)
        term = arms[y]["maj"][evsl]
        yj = logical_estimate(finy[evsl], r["corr_joint"])
        C = yj != term
        E = term != y
        helpful = int((C & E).sum())
        harmful_o = int((C & ~E).sum())
        print(f"  Y={y}: terminal-only wrong {int(E.sum())}/{N_HALF}   decoder overrides "
              f"{int(C.sum())}   helpful {helpful}   harmful {harmful_o}   "
              f"R(T)-R(J) = {(helpful - harmful_o) / N_HALF:+.5f}")
        ov[y] = dict(terminal_wrong=int(E.sum()), overrides=int(C.sum()), helpful=helpful,
                     harmful=harmful_o, override_value=(helpful - harmful_o) / N_HALF,
                     ler_decoder=float((yj != y).mean()), ler_terminal=float(E.mean()))
    report["override_value"] = {str(k): v for k, v in ov.items()}

    out = os.path.join(ROOT, "data", "campaign_r_results.json")
    tmp = out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(report, fh, indent=1)
    os.replace(tmp, out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
