#!/usr/bin/env python
"""Round 8: `P9` with MANY seams, and escalation that stays local. Simulator, 0 QPU.

🔴 THE BARRIER THIS CLOSES, raised in review against ourselves: everything
measured so far has ONE seam. A real streaming decoder has many. With `S` seams and a per-seam flag
probability `p`, the probability that some seam is nontrivial is `1 - (1-p)^S`, so at `S = 20` and
the device's `p ~ 9%` almost every record contains a flagged seam. The two-window scheme escalates
the WHOLE RECORD, so at that point it degenerates into joint decoding and `P9` buys nothing.

The fix is obvious and was, until this file, unimplemented and unmeasured: escalate LOCALLY.

    Split `[0, n)` at cuts `c_1 < ... < c_S`. Window `i` decodes `[c_i - b, c_{i+1} + b)` and commits
    `[c_i, c_{i+1})`. Stitch. The residual `stitched XOR D` is the seam syndrome. For each seam `k`
    whose neighbourhood `[c_k - L, c_k + L)` carries residual, re-decode THAT NEIGHBOURHOOD ONLY,
    on a graph with virtual time boundaries at both ends, and add the correction.

Cost is then `sum over FLAGGED seams of 2L` instead of `n`, and it is the flagged count that grows
with `S`, not the record length. `P7`'s pointwise argument restricts to the neighbourhood unchanged,
so the certificate should carry over -- "should" is what this file replaces with a measurement.

WHAT IS MEASURED, at each `(S, noise)`:

  per-seam flag rate            `p` above, directly
  any-seam flag rate            `1 - (1-p)^S` if seams were independent; measured instead
  escalated layers, whole-record   the two-window policy: `n` whenever any seam fires
  escalated layers, local          `2L` per flagged seam -- the quantity that must not reach `n`
  divergence vs the joint decode, after local repair
  unflagged divergence            the `P9` certificate, `Pr[A = 1, Delta]`
  residual outside every neighbourhood  local repair CANNOT fix it; measured, not assumed

TWO NOISE LEVELS, because the simulator at device-calibrated noise is too easy to be a test: at
`p = 0.0077` round 4 found **zero** divergences in 10,000 shots at 401 layers, so a certificate
comparison there is vacuous. The second level is tuned so the PER-SEAM flag rate matches the
device's, which is the regime where whole-record escalation is predicted to collapse.

Usage:  python tools/multi_seam.py [--calibrate]
"""
import json
import os
import sys
import time

import numpy as np
from scipy.stats import beta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parallel_window import TBND, WindowGraph  # noqa: E402
from sim_substrate import make_circuit, sample, to_layers  # noqa: E402
from weight_model_robustness import fit_weights_v2  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST = 9
ROUNDS = 400
B = 1
L = 4                            # local re-decode half-width, in detector layers
S_GRID = [1, 2, 4, 8, 20]
SHOTS_FIT = 5_000
SHOTS_EV = 10_000
DELTA = 0.05
P_DEVICE_RATE = 0.0077           # calibrated to the device's 0.0399 bulk detector rate
DEVICE_SEAM_RATE = 0.09          # the device's measured per-seam flag rate at b = 1


def cp_upper(k, n, a):
    return 1.0 if k >= n else float(beta.ppf(1.0 - a, k + 1, n - k))


class MultiWindow:
    """S seams, S+1 parallel windows, and both escalation policies on the same decode."""

    def __init__(self, n_layers, n_anc, S, b, L_, logical_j, wtab, ttab):
        self.n_layers, self.n_anc, self.S, self.b, self.L = n_layers, n_anc, S, b, L_
        # evenly spaced interior cuts; commit regions are [cuts[i], cuts[i+1])
        self.cuts = [0] + [int(round((i + 1) * n_layers / (S + 1))) for i in range(S)] + [n_layers]
        self.win = []
        for i in range(S + 1):
            lo_c, hi_c = self.cuts[i], self.cuts[i + 1]
            lo = max(lo_c - b, 0)
            hi = min(hi_c + b, n_layers)
            g = WindowGraph(lo, hi, n_anc, logical_j, lo > 0, hi < n_layers, wtab=wtab, ttab=ttab)
            own = (g.kind != TBND) & (g.glayer >= lo_c) & (g.glayer < hi_c)
            self.win.append((lo, hi, g, own))
        self.joint = WindowGraph(0, n_layers, n_anc, logical_j, False, False, wtab=wtab, ttab=ttab)
        # 🔴 A LADDER of repair graphs per seam, not one. A repair chain that terminates on the
        # neighbourhood's VIRTUAL time boundary is a placeholder for a fault continuing outside it;
        # `boundary_of` drops that endpoint, so the old syndrome check passed on a projection that
        # never wrote the detector the real fault would also have flipped -- circular. Re-gate
        # finding 1. The fix is to grow the neighbourhood until the repair uses no virtual-time
        # edge, charging the layers actually used, and fall back to a whole-record joint repair when
        # growth reaches the ends.
        self.local = []
        for k in range(1, S + 1):
            c = self.cuts[k]
            ladder = []
            half = L_
            while True:
                lo, hi = max(c - half, 0), min(c + half, n_layers)
                # 🔴 a rung that spans the WHOLE record has no TBND and is therefore always
                # accepted, which made the explicit whole-record fallback unreachable. The previous
                # guard stopped the ladder by half-width, but for seams near the record ends the
                # clamp reached [0, n) earlier than that -- 11 of 35 ladders in the tested grid.
                # Gate-4 finding 8.
                if lo == 0 and hi == n_layers:
                    break
                g = WindowGraph(lo, hi, n_anc, logical_j, lo > 0, hi < n_layers,
                                wtab=wtab, ttab=ttab)
                ladder.append((lo, hi, g, (g.kind == TBND)))
                # 🔴 the ladder STOPS before the whole record. It used to include the
                # full-record graph, which has no TBND and is therefore always accepted, so
                # the explicit whole-record fallback below was unreachable and its measured
                # rate of 0 meant nothing. Gate-3 finding 7.
                if 2 * half >= n_layers:
                    break
                half *= 2
            self.local.append(ladder)

    def shot(self, d):
        """One shot. Returns the joint parity, the locally repaired parity, and the accounting."""
        n_layers, n_anc = self.n_layers, self.n_anc
        sol_j = self.joint.decode(d)
        par_j = int(self.joint.logical[sol_j.astype(bool)].sum() & 1)

        stitched = np.zeros((n_layers, n_anc), dtype=np.uint8)
        par_s = 0
        for lo, hi, g, own in self.win:
            k = g.decode(d[lo:hi]) & own
            par_s ^= int(g.logical[k.astype(bool)].sum() & 1)
            stitched[lo:hi] ^= g.boundary_of(k)

        residual = stitched ^ d
        res_layers = np.flatnonzero(residual.any(axis=1))

        # which seams are flagged, and what the local repair can reach
        flagged, covered = [], np.zeros(n_layers, dtype=bool)
        for idx, ladder in enumerate(self.local):
            lo, hi, _, _ = ladder[0]
            if residual[lo:hi].any():
                flagged.append(idx)
                covered[lo:hi] = True
        uncovered = int(sum(1 for r in res_layers if not covered[r]))

        # LOCAL escalation: re-decode each flagged neighbourhood on the residual alone.
        # The repair is APPLIED to the stitched correction, not merely counted -- the first version
        # of this file XORed only the fix's logical parity and never touched `stitched`, so its
        # claim that the repaired correction explained the syndrome was never tested.
        # Audit finding: the flagged fraction has to be counted per record, not per seam.
        par_r = par_s
        repair_layers = 0
        grew = 0
        fell_back = 0
        for idx in flagged:
            done = False
            for rung, (lo, hi, gl, is_tbnd) in enumerate(self.local[idx]):
                fix = gl.decode(residual[lo:hi])
                repair_layers += hi - lo
                if not (fix.astype(bool) & is_tbnd).any():
                    # every edge of this repair is a real fault: admissible
                    par_r ^= int(gl.logical[fix.astype(bool)].sum() & 1)
                    stitched[lo:hi] ^= gl.boundary_of(fix)
                    grew += rung
                    done = True
                    break
            if not done:
                # growth reached the record ends and the chain still wants to leave it: the only
                # admissible completion is a whole-record repair
                fix = self.joint.decode(residual)
                repair_layers += n_layers
                par_r ^= int(self.joint.logical[fix.astype(bool)].sum() & 1)
                stitched ^= self.joint.boundary_of(fix)
                fell_back += 1

        # Does the stitched-plus-repaired correction explain the syndrome EXACTLY? A local graph can
        # "explain" a detector by running a chain into its own virtual time boundary, which explains
        # nothing globally, so this can fail and has to be measured rather than assumed.
        final = stitched ^ d
        return dict(par_joint=par_j, par_local=par_r,
                    n_flagged=len(flagged), any_flagged=bool(flagged),
                    residual_weight=int(residual.sum()),
                    uncovered_layers=uncovered,
                    explains=bool(not final.any()),
                    final_residual_weight=int(final.sum()),
                    # 🔴 the MEASURED layers the repair actually decoded, including every rung of
                    # the growth ladder that was tried and every whole-record fallback -- not
                    # `2L x flagged`, which assumed the first rung always worked.
                    cost_local=repair_layers,
                    grew=grew, fell_back=fell_back,
                    cost_whole=(n_layers if flagged else 0))


def run(S, D_ev, w, t, n_anc, alpha):
    n_layers = D_ev.shape[1]
    mw = MultiWindow(n_layers, n_anc, S, B, L, n_anc - 1, w, t)
    out = [mw.shot(D_ev[s]) for s in range(D_ev.shape[0])]
    n = len(out)
    anyf = np.array([o["any_flagged"] for o in out])
    nf = np.array([o["n_flagged"] for o in out])
    dv = np.array([o["par_joint"] != o["par_local"] for o in out])
    unf = int((dv & ~anyf).sum())
    unc = np.array([o["uncovered_layers"] for o in out])
    exp_ok = np.array([o["explains"] for o in out])
    # 🔴 THE CERTIFICATE IS Pr[local != joint], OVER ALL SHOTS. `P9` may drop escalated shots only
    # because escalation there makes the output EXACTLY the joint output. Local escalation does not
    # -- an escalated shot outputs the stitched correction plus a local repair -- so `P7` bounds this
    # policy by the whole disagreement probability. The first version of this file reported the
    # unflagged count as the certificate and omitted 1,133 of 1,135 disagreements at S = 20.
    # Audit finding 1. `unflagged` is kept below as a diagnostic and is NOT a certificate.
    return dict(S=S, n_windows=S + 1, commit_span=n_layers / (S + 1), shots=n,
                per_seam_flag_rate=float(nf.sum() / (S * n)),
                any_seam_flag_rate=float(anyf.mean()),
                mean_flagged_seams=float(nf.mean()),
                divergences=int(dv.sum()), divergence_rate=float(dv.mean()),
                cert=float(dv.mean()), cert_ub=cp_upper(int(dv.sum()), n, alpha),
                explains_syndrome_rate=float(exp_ok.mean()),
                shots_not_explained=int((~exp_ok).sum()),
                unflagged_diagnostic_not_a_certificate=unf,
                cost_local_layers=float(np.mean([o["cost_local"] for o in out])),
                cost_whole_layers=float(np.mean([o["cost_whole"] for o in out])),
                joint_layers=float(n_layers),
                shots_with_uncovered_residual=int((unc > 0).sum()),
                uncovered_rate=float((unc > 0).mean()),
                # 🔴 TOTAL decoded layers: the S+1 initial windows PLUS the repair. The
                # previously reported figure was repair overhead only, so a scheme costing
                # ~460 layers was being compared against Joint's 401 as though it cost 19.
                # Gate-3 finding 7.
                cost_windows_layers=float(sum(hi - lo for lo, hi, _, _ in mw.win)),
                cost_total_layers=float(sum(hi - lo for lo, hi, _, _ in mw.win)
                                        + np.mean([o["cost_local"] for o in out])),
                mean_growth_steps=float(np.mean([o["grew"] for o in out])),
                fallback_rate=float(np.mean([o["fell_back"] > 0 for o in out])),
                mean_fallbacks=float(np.mean([o["fell_back"] for o in out])))


def make_data(p, seed=5):
    circ = make_circuit(distance=DIST, rounds=ROUNDS, p=p)
    _, df, _, _ = sample(circ, SHOTS_FIT, seed=seed)
    _, de, _, _ = sample(circ, SHOTS_EV, seed=seed + 1)
    Df = to_layers(df, n_anc=DIST - 1)
    De = to_layers(de, n_anc=DIST - 1)
    w, t, _, _ = fit_weights_v2(Df, n_fit=SHOTS_FIT)
    return Df, De, w, t


def seam_rate_at(p, shots=1000, S=4):
    # 🔴 seed 900: the calibration used to draw from the SAME seed as the evaluation sample,
    # so its first 1,000 shots were reused in the test despite the file saying they never
    # were. Gate-3 finding 7.
    _, De, w, t = make_data(p, seed=900)
    De = De[:shots]
    mw = MultiWindow(De.shape[1], DIST - 1, S, B, L, DIST - 2, w, t)
    o = [mw.shot(De[s]) for s in range(De.shape[0])]
    return (float(np.mean([x["n_flagged"] for x in o]) / S),
            float(De[:, 1:-1, :].mean()),
            float(np.mean([x["par_joint"] != x["par_local"] for x in o])))


def calibrate():
    """Find the noise level whose PER-SEAM flag rate matches the device's ~9% at b = 1.

    A tuning step, reported in full: the grid it was tuned on, the bisection, and the achieved
    rate. Tuned at S = 4 on its own 1000 shots, never on the shots the test evaluates.
    """
    print("CALIBRATION -- per-seam flag rate vs stim noise, S = 4, 1000 shots")
    print(f"{'p':>8} {'det rate':>9} {'per-seam flag':>14} {'divergence':>11}")
    grid = {}
    for p in (0.0077, 0.015, 0.025, 0.04, 0.06):
        nf, dr, dv = seam_rate_at(p)
        grid[p] = nf
        print(f"{p:>8.4f} {dr:>9.5f} {nf:>14.4f} {dv:>11.4f}", flush=True)

    lo = max(p for p, r in grid.items() if r <= DEVICE_SEAM_RATE)
    hi = min(p for p, r in grid.items() if r > DEVICE_SEAM_RATE)
    print(f"\n  bisecting for a per-seam rate of {DEVICE_SEAM_RATE} in [{lo}, {hi}]")
    best = None
    for _ in range(5):
        mid = (lo + hi) / 2
        nf, dr, dv = seam_rate_at(mid)
        print(f"{mid:>8.4f} {dr:>9.5f} {nf:>14.4f} {dv:>11.4f}", flush=True)
        best = (mid, nf, dr)
        if nf < DEVICE_SEAM_RATE:
            lo = mid
        else:
            hi = mid
    out = os.path.join(ROOT, "data", "multi_seam_noise.json")
    tmp = out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(dict(target_per_seam_flag_rate=DEVICE_SEAM_RATE, p_matched=best[0],
                       achieved_per_seam_flag_rate=best[1], detector_rate=best[2],
                       tuned_at_S=4, tuning_shots=1000,
                       grid={str(k): v for k, v in grid.items()}), fh, indent=1)
    os.replace(tmp, out)
    print(f"\n  p_matched = {best[0]:.5f}  ->  per-seam {best[1]:.4f} "
          f"(target {DEVICE_SEAM_RATE})\n  wrote {out}")


def main():
    if "--calibrate" in sys.argv:
        calibrate()
        return
    with open(os.path.join(ROOT, "data", "multi_seam_noise.json")) as fh:
        p_high = json.load(fh)["p_matched"]

    alpha = DELTA / (2 * len(S_GRID))
    print("ROUND 8 -- P9 WITH MANY SEAMS AND LOCAL ESCALATION.  simulator, 0 QPU")
    print(f"  d = {DIST}, {ROUNDS} rounds ({ROUNDS + 1} detector layers), b = {B}, "
          f"local half-width L = {L}")
    print(f"  {SHOTS_EV} evaluation shots per cell; alpha = {alpha:.5f} over "
          f"{2 * len(S_GRID)} cells\n")

    rows = []
    for tag, p in (("device detector rate", P_DEVICE_RATE), ("device SEAM rate", p_high)):
        _, De, w, t = make_data(p)
        print(f"--- {tag}:  stim p = {p:.4f},  detector rate {De[:, 1:-1, :].mean():.5f}")
        print(f"{'S':>3} {'commit':>7} {'per-seam':>9} {'any-seam':>9} {'div':>5} "
              f"{'CERT':>8} {'cert UB':>9} {'explains':>9} {'local':>8} {'whole':>8} "
              f"{'joint':>7} {'unflag*':>8}")
        for S in S_GRID:
            t0 = time.time()
            r = run(S, De, w, t, DIST - 1, alpha)
            r["noise"] = p
            r["regime"] = tag
            rows.append(r)
            print(f"{S:>3} {r['commit_span']:>7.1f} {r['per_seam_flag_rate']:>9.4f} "
                  f"{r['any_seam_flag_rate']:>9.4f} {r['divergences']:>5} "
                  f"{r['cert']:>8.5f} {r['cert_ub']:>9.2e} "
                  f"{r['explains_syndrome_rate']:>9.4f} {r['cost_local_layers']:>8.2f} "
                  f"{r['cost_whole_layers']:>8.1f} {r['joint_layers']:>7.0f} "
                  f"{r['unflagged_diagnostic_not_a_certificate']:>8}   "
                  f"({time.time() - t0:.0f}s)", flush=True)
        print()

    hi = [r for r in rows if r["regime"] == "device SEAM rate"]
    r20 = next(r for r in hi if r["S"] == 20)
    f1 = r20["cost_local_layers"] > 0.5 * r20["joint_layers"]
    f2 = r20["uncovered_rate"] > 0.01
    # round 10 falsifier 6: after growth, does the correction actually explain the syndrome, and is
    # it still cheap? Either failure withdraws P10 entirely.
    f6 = (r20["explains_syndrome_rate"] < 0.99
          or r20["cost_local_layers"] > 0.5 * r20["joint_layers"])
    print("=" * 100)
    print(f"  F1  local escalation cost at S=20, device seam rate: "
          f"{r20['cost_local_layers']:.2f} layers against the joint decode's "
          f"{r20['joint_layers']:.0f}  (whole-record policy: {r20['cost_whole_layers']:.1f})")
    print(f"      -> {'FIRES -- local escalation does not solve the collapse' if f1 else 'does NOT fire'}")
    print(f"  F2  residual outside every escalated neighbourhood at S=20 (as pre-registered): "
          f"{r20['uncovered_rate']:.4f} of shots")
    print(f"      -> {'FIRES -- the local repair is incomplete' if f2 else 'does NOT fire'}")
    print(f"  F6  after growth at S=20: explains {r20['explains_syndrome_rate']:.4f}, "
          f"cost {r20['cost_local_layers']:.2f} of {r20['joint_layers']:.0f} layers, "
          f"growth {r20['mean_growth_steps']:.3f} steps/shot, "
          f"whole-record fallback {r20['fallback_rate']:.4f}")
    print(f"      -> {'FIRES -- P10 is withdrawn entirely' if f6 else 'does NOT fire'}")
    print(f"  F2' the CORRECTED form of the same question -- does the stitched-plus-repaired "
          f"correction explain the syndrome exactly?")
    for r in rows:
        print(f"        {r['regime']:>22} S={r['S']:>2}  explains {r['explains_syndrome_rate']:.4f}"
              f"  ({r['shots_not_explained']} shots not explained)")
    print("" + chr(10) + "  CERTIFICATE, corrected: Pr[local != joint] over ALL shots, not the unflagged subset")
    for r in rows:
        print(f"        {r['regime']:>22} S={r['S']:>2}  cert {r['cert']:.5f}  "
              f"UB {r['cert_ub']:.2e}   (unflagged diagnostic: "
              f"{r['unflagged_diagnostic_not_a_certificate']})")

    out = os.path.join(ROOT, "data", "multi_seam.json")
    tmp = out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(dict(distance=DIST, rounds=ROUNDS, b=B, local_halfwidth=L,
                       s_grid=S_GRID, shots_eval=SHOTS_EV, shots_fit=SHOTS_FIT,
                       delta=DELTA, alpha_per_cell=alpha,
                       p_device_rate=P_DEVICE_RATE, p_seam_matched=p_high,
                       device_seam_rate_target=DEVICE_SEAM_RATE,
                       falsifier_cost_fired=bool(f1), falsifier_coverage_fired=bool(f2),
                       falsifier_admissible_repair_fired=bool(f6),
                       rows=rows), fh, indent=1)
    os.replace(tmp, out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
