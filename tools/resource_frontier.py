#!/usr/bin/env python
"""Round 11: the resource comparison, with every decode charged and JOINT in the comparison.

🔴 THIS FILE EXISTS BECAUSE THE THIRD GATE FOUND FIVE FATALS THAT ALL POINT THE SAME WAY.
an independent audit of the resource claims, findings 1-4 and 6:

  1. `work_accounting` was never wired in, and could not have caught an omission anyway.
  2. `enlarge_policy`'s "local" seam repair reproduced the virtual-time-boundary defect that round
     10A had just corrected in `multi_seam` -- decode once, never check for a `TBND` edge, never
     grow, never apply the repair to the stitched boundary, never re-check the global syndrome.
  3. "the gap trigger costs 3x" charged TWO constrained decodes per window. The ordinary window
     decode is already performed and already yields the winning class and its weight, so only the
     OPPOSITE class needs an extra constrained decode. The honest surcharge is half what was charged.
  4. dominance was computed only among window policies. **Joint is (n layers, certificate 0)** and
     has to be in the comparison; excluding the thing you are trying to beat is not a frontier.
  6. empirical rates were reported as certificates. Every number here carries its one-sided
     Clopper-Pearson upper bound at a declared simultaneous level.

WHAT IS COMPARED, on identical held-out device shots, every decode charged through an instrumented
graph that cannot be bypassed:

    JOINT              decode the whole record.                              <- the comparator
    FIXED b            two windows, stitch, admissible local seam repair.
    ENLARGE b->b'      FIXED, and on trigger re-decode both windows at b'.   <- the prior art
    ESCALATE b         FIXED, and on trigger decode jointly.                 <- ours

Triggers: the seam residual (ours; free, a by-product of stitching) and the whole-window
complementary gap (a proxy for the prior art's; charged reuse-aware).

Usage:  python tools/resource_frontier.py
"""
import json
import os
import sys
import time

import numpy as np
from scipy.stats import beta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_campaign_r import load_arm  # noqa: E402
from detectors import build_detectors  # noqa: E402
from parallel_window import TBND, WindowGraph  # noqa: E402
from weight_model_robustness import fit_weights_v2  # noqa: E402
from work_accounting import RAW, WorkLedger, audit, instrument  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W1 = 25
B_SMALL = 1
B_LARGE = [4, 8, 16]
N_HALF = 50_000
N_EVAL = 20_000
N_TAU = 5_000
L0 = 8                      # first rung of the seam-repair ladder, half-width in layers
DELTA = 0.05


def cp_upper(k, n, a):
    return 1.0 if k >= n else float(beta.ppf(1.0 - a, k + 1, n - k))


def ladder(centre, n_layers, n_anc, wtab, ttab, box, half0=L0):
    """Repair graphs of growing half-width, instrumented. Last rung is the whole record."""
    out, half = [], half0
    while True:
        lo, hi = max(centre - half, 0), min(centre + half, n_layers)
        g = WindowGraph(lo, hi, n_anc, 0, lo > 0, hi < n_layers, wtab=wtab, ttab=ttab)
        instrument(g, box, "seam-repair", hi - lo)
        out.append((lo, hi, g, g.kind == TBND))
        if lo == 0 and hi == n_layers:
            return out
        half *= 2


class Windows:
    """One buffer width, fully instrumented, with an ADMISSIBLE local seam repair."""

    def __init__(self, n_layers, n_anc, b, wtab, ttab, box):
        self.b, self.box = b, box
        self.e1, self.s2 = min(W1 + b, n_layers), max(W1 - b, 0)
        self.n_layers, self.n_anc = n_layers, n_anc
        self.g1 = WindowGraph(0, self.e1, n_anc, 0, False, self.e1 < n_layers, wtab=wtab, ttab=ttab)
        self.g2 = WindowGraph(self.s2, n_layers, n_anc, 0, self.s2 > 0, False, wtab=wtab, ttab=ttab)
        instrument(self.g1, box, "window", self.e1)
        instrument(self.g2, box, "window", n_layers - self.s2)
        self.c1 = (self.g1.kind != TBND) & (self.g1.glayer < W1)
        self.c2 = (self.g2.kind != TBND) & (self.g2.glayer >= W1)
        # class-constrained copies for the gap. Charged ONCE per call: the ordinary decode already
        # gives the winning class, so only the opposite class costs extra. Gate-3 finding 3.
        self.q1 = WindowGraph(0, self.e1, n_anc, 0, False, self.e1 < n_layers, wtab=wtab, ttab=ttab,
                              obs_as_detector=True)
        self.q2 = WindowGraph(self.s2, n_layers, n_anc, 0, self.s2 > 0, False, wtab=wtab, ttab=ttab,
                              obs_as_detector=True)
        instrument(self.q1, box, "gap-trigger", self.e1)
        instrument(self.q2, box, "gap-trigger", n_layers - self.s2)
        self.rep = ladder(W1, n_layers, n_anc, wtab, ttab, box)

    def run(self, d):
        """Full windowed decode: stitch, then grow the seam repair until it is admissible.

        Returns (parity, seam weight, explains, growth steps). The repair is APPLIED to the stitched
        boundary and the GLOBAL syndrome is re-checked, which is what `enlarge_policy`'s version
        never did.
        """
        # the INSTRUMENTED weighted decode. Calling `self.g1.m.decode` directly and charging by
        # hand is exactly the bypass this module exists to prevent -- and `audit()` caught it doing
        # so on the first run of this file, off by 160,000 decodes to the unit.
        s1, wt1 = self.g1.decode_weighted(d[: self.e1].ravel())
        s2_, wt2 = self.g2.decode_weighted(d[self.s2:].ravel())
        s1 = s1.astype(np.uint8)
        s2_ = s2_.astype(np.uint8)
        # the ordinary solution already identifies the WINNING logical class and its weight, so the
        # gap only needs the opposite class. Gate-3 finding 3.
        self.last = (int(self.g1.logical[s1.astype(bool)].sum() & 1), float(wt1),
                     int(self.g2.logical[s2_.astype(bool)].sum() & 1), float(wt2))
        k1 = s1 & self.c1
        k2 = s2_ & self.c2
        par = (self.g1.logical[k1.astype(bool)].sum()
               + self.g2.logical[k2.astype(bool)].sum()) & 1
        st = np.zeros((self.n_layers, self.n_anc), dtype=np.uint8)
        st[: self.e1] ^= self.g1.boundary_of(k1)
        st[self.s2:] ^= self.g2.boundary_of(k2)
        seam = st ^ d
        sw = int(seam.sum())
        grew = 0
        if sw:
            for rung, (lo, hi, gl, tb) in enumerate(self.rep):
                fix = gl.decode(seam[lo:hi])
                if not (fix.astype(bool) & tb).any():
                    par ^= int(gl.logical[fix.astype(bool)].sum() & 1)
                    st[lo:hi] ^= gl.boundary_of(fix)
                    grew = rung
                    break
        return int(par), sw, bool(not (st ^ d).any()), grew

    def gap(self, d):
        """Complementary gap at its MARGINAL cost: ONE constrained decode per window.

        🔴 The previous version ran both class-constrained decodes and charged both, making the gap
        trigger look 2x more expensive than it is. The ordinary window decode has already been
        performed by `run()` and already yields the winning class and its weight (this repository
        verified zero mismatches between the unconstrained winner and the two-class minimum), so the
        marginal cost of the gap is the OPPOSITE class only. Gate-3 finding 3.

        `run(d)` must have been called on the same shot first; that is how a deployed policy would
        order it, and it is what makes the reuse legitimate rather than assumed.
        """
        c1, w1, c2, w2 = self.last
        _, w1o = self.q1.decode_class(d[: self.e1], 1 - c1)
        _, w2o = self.q2.decode_class(d[self.s2:], 1 - c2)
        return min(abs(w1o - w1), abs(w2o - w2))


def main():
    t0 = time.time()
    syn, fin, _ = load_arm(0)
    D = build_detectors(syn, fin)
    n_layers, n_anc = D.shape[1], D.shape[2]
    w, t, _, _ = fit_weights_v2(D[:N_HALF], n_fit=N_HALF)
    Dev = D[N_HALF:N_HALF + N_EVAL]
    Dfit = D[:N_TAU]

    box = [WorkLedger()]
    joint = WindowGraph(0, n_layers, n_anc, 0, False, False, wtab=w, ttab=t)
    instrument(joint, box, "joint", n_layers)
    small = Windows(n_layers, n_anc, B_SMALL, w, t, box)
    large = {b: Windows(n_layers, n_anc, b, w, t, box) for b in B_LARGE}

    print("ROUND 11 -- THE RESOURCE COMPARISON, WITH JOINT IN IT.  device, 0 QPU")
    print(f"  campaign R Y=0, {N_EVAL} held-out shots; every decode charged through an")
    print("  instrumented graph; the gap trigger charged reuse-aware; JOINT is a policy.\n")

    par_j = np.empty(N_EVAL, np.uint8)
    c_j = np.empty(N_EVAL)
    par_s = np.empty(N_EVAL, np.uint8)
    c_s = np.empty(N_EVAL)
    seam = np.empty(N_EVAL, np.int32)
    expl = np.empty(N_EVAL, bool)
    grew = np.empty(N_EVAL, np.int32)
    gap = np.empty(N_EVAL)
    c_gap = np.empty(N_EVAL)
    par_l = {b: np.empty(N_EVAL, np.uint8) for b in B_LARGE}
    c_l = {b: np.empty(N_EVAL) for b in B_LARGE}

    print("  decoding ...", flush=True)
    charged_decodes = 0
    for s in range(N_EVAL):
        box[0] = WorkLedger()
        par_j[s] = joint.logical[joint.decode(Dev[s]).astype(bool)].sum() & 1
        c_j[s] = box[0].total_layers
        charged_decodes += box[0].n_decodes

        box[0] = WorkLedger()
        par_s[s], seam[s], expl[s], grew[s] = small.run(Dev[s])
        c_s[s] = box[0].total_layers
        charged_decodes += box[0].n_decodes

        box[0] = WorkLedger()          # `small.run` above left the winning class on `small.last`
        gap[s] = small.gap(Dev[s])
        c_gap[s] = box[0].total_layers
        charged_decodes += box[0].n_decodes

        for b in B_LARGE:
            box[0] = WorkLedger()
            par_l[b][s] = large[b].run(Dev[s])[0]
            c_l[b][s] = box[0].total_layers
            charged_decodes += box[0].n_decodes
        if s and s % 5000 == 0:
            print(f"    {s}/{N_EVAL}  ({time.time() - t0:.0f}s)", flush=True)

    print(f"\n  windowed decode explains the syndrome in {expl.mean():.4%} of shots; "
          f"growth {grew.mean():.3f} rungs/shot")


    # 🔴 the audit the first version could not fail: every decode the decoders performed must have
    # been charged to some ledger. RAW.calls is incremented inside the instrumented method itself,
    # so a mismatch means a decoder was never instrumented. Gate-3 finding 1.
    box[0] = WorkLedger()
    gap_fit = []
    for s in range(N_TAU):
        small.run(Dfit[s])
        gap_fit.append(small.gap(Dfit[s]))
    gap_fit = np.array(gap_fit)
    charged_decodes += box[0].n_decodes

    # 🔴 the audit runs AFTER every decode in the file, including the fit-side threshold pass. The
    # previous version audited mid-run and then made 20,419 further calls, so the stored call count
    # did not match the equality it advertised. Gate-4 finding 1.
    audit(charged_decodes, "all decodes in this file")
    print(f"  AUDIT PASSED: {charged_decodes} charged == {RAW.calls} decoder calls "
          f"(every bypass the auditor named is caught: see the self-test in work_accounting)")
    p_seam = float((seam > 0).mean())
    tau = float(np.quantile(gap_fit, p_seam))
    trig = {"seam residual (ours)": ((seam > 0), np.zeros(N_EVAL)),
            "whole-window gap (proxy)": ((gap <= tau), c_gap)}

    K = 2 + 2 * (len(B_LARGE) + 2)       # joint, fixed, and each policy under each trigger
    alpha = DELTA / K
    rows = [dict(trigger="-", policy="JOINT (comparator)", fires=0.0, disagree=0,
                 cert=0.0, cert_ub=0.0, cost=float(c_j.mean()))]
    for tname, (fires, tcost) in trig.items():
        base = c_s + tcost
        defs = [(f"FIXED b={B_SMALL}", par_s, base)]
        for b in B_LARGE:
            defs.append((f"ENLARGE {B_SMALL}->{b}", np.where(fires, par_l[b], par_s),
                         base + fires * c_l[b]))
        defs.append(("ESCALATE -> joint", np.where(fires, par_j, par_s), base + fires * c_j))
        for pol, out, c in defs:
            dis = int((out != par_j).sum())
            rows.append(dict(trigger=tname, policy=pol, fires=float(fires.mean()),
                             disagree=dis, cert=dis / N_EVAL,
                             cert_ub=cp_upper(dis, N_EVAL, alpha), cost=float(np.mean(c))))

    print(f"\n{'trigger':>24} {'policy':>20} {'fires':>7} {'!=J':>5} "
          f"{'cert (pt)':>10} {'CERT UB':>10} {'cost':>8}")
    for r in rows:
        print(f"{r['trigger']:>24} {r['policy']:>20} {r['fires']:>7.2%} {r['disagree']:>5} "
              f"{r['cert']:>10.5f} {r['cert_ub']:>10.2e} {r['cost']:>8.1f}")

    # 🔴 the PAIRED test, computed and stored. The 6.8e-13 quoted for the trigger contrast was a
    # hardcoded manual number, and the marginals alone cannot distinguish 46-vs-1 discordant from
    # 45-vs-0. Gate-4 finding 5.
    from scipy.stats import binomtest
    fs = trig["seam residual (ours)"][0]
    fg = trig["whole-window gap (proxy)"][0]
    esc_seam = np.where(fs, par_j, par_s) != par_j
    esc_gap = np.where(fg, par_j, par_s) != par_j
    b_only = int((esc_gap & ~esc_seam).sum())      # gap wrong where the seam residual is right
    c_only = int((esc_seam & ~esc_gap).sum())      # seam residual wrong where the gap is right
    both = int((esc_seam & esc_gap).sum())
    mcp = float(binomtest(b_only, b_only + c_only, 0.5).pvalue) if b_only + c_only else 1.0
    paired = dict(gap_only_wrong=b_only, seam_only_wrong=c_only, both_wrong=both,
                  mcnemar_p_two_sided=mcp)
    print(f"\n  PAIRED, escalate-to-joint under each trigger: gap wrong where seam right "
          f"{b_only}, seam wrong where gap right {c_only}, both wrong {both}; "
          f"McNemar p = {mcp:.3e}")

    nd = [r for r in rows if not any(
        (o["cost"] <= r["cost"] and o["cert_ub"] <= r["cert_ub"] and o is not r
         and (o["cost"] < r["cost"] or o["cert_ub"] < r["cert_ub"])) for o in rows)]
    print("\n" + "=" * 96)
    print("  NON-DOMINATED on (cost, CERTIFIED bound), with JOINT included:")
    for r in sorted(nd, key=lambda x: x["cost"]):
        print(f"    {r['trigger']:>24} | {r['policy']:>20} | cost {r['cost']:>7.1f} | "
              f"UB {r['cert_ub']:.2e}")
    only_joint = len(nd) == 1 and nd[0]["policy"].startswith("JOINT")
    print(f"\n  F10  does JOINT dominate every windowed policy?  "
          f"{'YES -- there is no resource frontier at this workload' if only_joint else 'no'}")

    gapf = next(r for r in rows if r["trigger"].startswith("whole-window")
                and r["policy"].startswith("FIXED"))
    seamf = next(r for r in rows if r["trigger"].startswith("seam")
                 and r["policy"].startswith("FIXED"))
    print(f"  F11  gap-trigger cost multiple, charged reuse-aware: "
          f"{gapf['cost'] / seamf['cost']:.2f}x  (round 10 claimed 2.95x)")

    out = os.path.join(ROOT, "data", "resource_frontier.json")
    tmp = out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(dict(n_eval=N_EVAL, b_small=B_SMALL, b_large=B_LARGE, W1=W1, n_layers=n_layers,
                       gap_tau=tau, n_tau=N_TAU, seam_escalation_rate=p_seam,
                       delta=DELTA, alpha_per_cell=alpha, K=K,
                       explains_syndrome_rate=float(expl.mean()),
                       growth_rungs_per_shot=float(grew.mean()),
                       raw_decoder_calls=int(RAW.calls),
                       gap_cost_multiple=gapf["cost"] / seamf["cost"],
                       joint_dominates_all=bool(only_joint),
                       joint_dominates_note="dominates every EVALUATED policy row; ten "
                                           "window policies were compared",
                       paired_trigger_test=paired,
                       non_dominated=[dict(trigger=r["trigger"], policy=r["policy"],
                                           cost=r["cost"], cert_ub=r["cert_ub"]) for r in nd],
                       rows=rows), fh, indent=1)
    os.replace(tmp, out)
    print(f"\nwrote {out}   ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
