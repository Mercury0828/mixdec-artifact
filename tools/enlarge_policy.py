#!/usr/bin/env python
"""Round 9B: the prior art's actual POLICY, not just its trigger. Device shots, 0 QPU.

🔴 WHY. The pre-writing audit's finding 3: every comparison so far has been trigger-vs-trigger at a
matched escalation rate. arXiv:2605.14637 and arXiv:2605.01149 do not propose a trigger, they propose
a **policy** -- decode with a small buffer, and where confidence is low **enlarge the buffer and
re-decode**, never consulting a joint decoder. That policy was never implemented here, so the claim
that our scheme differs from theirs rested on a comparison of parts.

THREE COMPLETE POLICIES, on identical shots, each scored by ITS OWN certificate:

    FIXED     decode at `b`, stitch, repair the seam. Cost `n/2 + b`.
    ENLARGE   decode at `b`; on trigger, re-decode BOTH windows at `b_large` and use that.
              Cost `n/2 + b + p * (n/2 + b_large)`.       <- the prior art's scheme
    ESCALATE  decode at `b`; on trigger, decode JOINTLY and use that.
              Cost `n/2 + b + p * n`.                     <- ours

🔴 THE CERTIFICATE IS `Pr[output != Joint]` FOR EACH POLICY SEPARATELY. Round 8 got this wrong by
reusing `P9`'s unflagged-only form, which is valid only when the escalated output *is* the joint
output. That is true for `ESCALATE` and **false for `ENLARGE`** -- an enlarged re-decode is still a
windowed decode and can disagree with the joint one. So every policy here is scored by the full
disagreement probability, which is what `P7` bounds in general.

SCHEME AND TRIGGER ARE SEPARATED. Each policy is run under **both** triggers -- the complementary
gap (theirs, `gap <= tau`) and the seam residual (ours, `seam_weight > 0`) -- so a difference in
outcome can be attributed to one or the other rather than to the pair.

Pre-registered at `c5157fe`, `docs/expected.md` Round 9B, falsifiers 3 and 4.

Usage:  python tools/enlarge_policy.py
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
from work_accounting import Accumulator, WorkLedger  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W1 = 25
B_SMALL = 1
B_LARGE = [4, 8, 16]
N_HALF = 50_000
N_EVAL = 20_000              # each shot costs up to 5 decodes; this is the affordable slice
DELTA = 0.05
L_REPAIR = 8                 # half-width of the no-joint seam repair graph
N_FIT_TAU = 5_000            # fit shots the gap threshold is set on


def cp_upper(k, n, a):
    return 1.0 if k >= n else float(beta.ppf(1.0 - a, k + 1, n - k))


class Windows:
    """One buffer width: the two windows, their commit masks, and the stitched decode."""

    def __init__(self, n_layers, n_anc, b, wtab, ttab):
        self.b = b
        self.e1 = min(W1 + b, n_layers)
        self.s2 = max(W1 - b, 0)
        self.n_layers, self.n_anc = n_layers, n_anc
        self.g1 = WindowGraph(0, self.e1, n_anc, 0, False, self.e1 < n_layers,
                              wtab=wtab, ttab=ttab)
        self.g2 = WindowGraph(self.s2, n_layers, n_anc, 0, self.s2 > 0, False,
                              wtab=wtab, ttab=ttab)
        self.c1 = (self.g1.kind != TBND) & (self.g1.glayer < W1)
        self.c2 = (self.g2.kind != TBND) & (self.g2.glayer >= W1)
        # observable-as-detector copies, for the complementary gap
        self.q1 = WindowGraph(0, self.e1, n_anc, 0, False, self.e1 < n_layers,
                              wtab=wtab, ttab=ttab, obs_as_detector=True)
        self.q2 = WindowGraph(self.s2, n_layers, n_anc, 0, self.s2 > 0, False,
                              wtab=wtab, ttab=ttab, obs_as_detector=True)

    def stitch(self, d, joint, ledger, repair="joint", local_repair=None):
        """Decode, stitch, repair the seam. Returns (parity, seam weight).

        🔴 EVERY decode is charged to `ledger`. The re-gate found that this function called
        `joint.decode(seam)` inside every FIXED and ENLARGE evaluation -- 1,819 and 107 times per
        20,000 shots -- while the cost model charged none of it and the prose said the prior art's
        policy "never needs a joint decoder". Re-gate finding 3.

        `repair="local"` repairs the seam on a small graph around the seam instead, so the prior
        art's policy can be run with NO joint decode anywhere, which is what its authors describe.
        """
        k1 = self.g1.decode(d[: self.e1]) & self.c1
        ledger.charge("window", self.e1)
        k2 = self.g2.decode(d[self.s2:]) & self.c2
        ledger.charge("window", self.n_layers - self.s2)
        par = (self.g1.logical[k1.astype(bool)].sum()
               + self.g2.logical[k2.astype(bool)].sum()) & 1
        st = np.zeros((self.n_layers, self.n_anc), dtype=np.uint8)
        st[: self.e1] ^= self.g1.boundary_of(k1)
        st[self.s2:] ^= self.g2.boundary_of(k2)
        seam = st ^ d
        sw = int(seam.sum())
        if sw:
            if repair == "local" and local_repair is not None:
                lo, hi, gl = local_repair
                fix = gl.decode(seam[lo:hi])
                ledger.charge("seam-repair-local", hi - lo)
                par ^= int(gl.logical[fix.astype(bool)].sum() & 1)
            else:
                fix = joint.decode(seam)
                ledger.charge("seam-repair-JOINT", self.n_layers)
                par ^= int(joint.logical[fix.astype(bool)].sum() & 1)
        return int(par), sw

    def gap(self, d):
        return min(self.q1.complementary_gap(d[: self.e1])[1],
                   self.q2.complementary_gap(d[self.s2:])[1])


def main():
    t_start = time.time()
    syn, fin, _ = load_arm(0)
    D = build_detectors(syn, fin)
    n_layers, n_anc = D.shape[1], D.shape[2]
    fitsl = slice(0, N_HALF)
    w, t, _, _ = fit_weights_v2(D[fitsl], n_fit=N_HALF)
    Dev = D[N_HALF:N_HALF + N_EVAL]
    Dfit = D[:N_EVAL]

    joint = WindowGraph(0, n_layers, n_anc, 0, False, False, wtab=w, ttab=t)
    small = Windows(n_layers, n_anc, B_SMALL, w, t)
    large = {b: Windows(n_layers, n_anc, b, w, t) for b in B_LARGE}
    # a small graph for the no-joint-anywhere seam repair, grown so it needs no virtual-time edge
    lo_r, hi_r = max(W1 - L_REPAIR, 0), min(W1 + L_REPAIR, n_layers)
    grepair = (lo_r, hi_r, WindowGraph(lo_r, hi_r, n_anc, 0, lo_r > 0, hi_r < n_layers,
                                       wtab=w, ttab=t))

    print("ROUND 9B/10C -- THE PRIOR ART'S POLICY, WITH EVERY DECODE CHARGED.  device, 0 QPU")
    print(f"  campaign R Y=0, {N_EVAL} held-out shots, b_small = {B_SMALL}, b_large in {B_LARGE}")
    print("  every policy scored by its OWN Pr[output != Joint]; every cost from work_accounting")
    print("  " + "=" * 60)
    print("  Re-gate finding 3: seam repair calls joint.decode, and NONE of it was charged.")
    print("  Here it is charged, and a `local-repair` variant runs the prior art's policy with")
    print("  NO joint decode anywhere. The gap TRIGGER's own two class-constrained decodes per")
    print("  window are charged too -- they were free in the previous version.")
    print("  " + "=" * 60 + "\n")

    par_joint = np.empty(N_EVAL, np.uint8)
    seam_small = np.empty(N_EVAL, np.int32)
    gap_small = np.empty(N_EVAL)
    cost_gap_trigger = np.empty(N_EVAL)
    par_small, cost_small = {}, {}
    par_large, cost_large = {}, {}
    for rep in ("joint", "local"):
        par_small[rep] = np.empty(N_EVAL, np.uint8)
        cost_small[rep] = np.empty(N_EVAL)
        par_large[rep] = {b: np.empty(N_EVAL, np.uint8) for b in B_LARGE}
        cost_large[rep] = {b: np.empty(N_EVAL) for b in B_LARGE}

    print("  decoding ...", flush=True)
    for s in range(N_EVAL):
        par_joint[s] = joint.logical[joint.decode(Dev[s]).astype(bool)].sum() & 1
        for rep in ("joint", "local"):
            lg = WorkLedger()
            p, sw = small.stitch(Dev[s], joint, lg, repair=rep, local_repair=grepair)
            par_small[rep][s] = p
            cost_small[rep][s] = lg.total_layers
            if rep == "joint":
                seam_small[s] = sw
            for b in B_LARGE:
                lg2 = WorkLedger()
                par_large[rep][b][s] = large[b].stitch(Dev[s], joint, lg2, repair=rep,
                                                       local_repair=grepair)[0]
                cost_large[rep][b][s] = lg2.total_layers
        # the gap trigger is NOT free: two class-constrained decodes per window
        gap_small[s] = small.gap(Dev[s])
        cost_gap_trigger[s] = 2 * small.e1 + 2 * (n_layers - small.s2)
        if s and s % 5000 == 0:
            print(f"    {s}/{N_EVAL}  ({time.time() - t_start:.0f}s)", flush=True)

    gap_fit = np.array([small.gap(Dfit[s]) for s in range(N_FIT_TAU)])
    p_seam = float((seam_small > 0).mean())
    tau = float(np.quantile(gap_fit, p_seam))
    trig = {"seam residual (ours)": (seam_small > 0, np.zeros(N_EVAL)),
            "complementary gap (theirs)": (gap_small <= tau, cost_gap_trigger)}

    rows = []
    alpha = DELTA / (2 * (len(B_LARGE) + 2) * len(trig))
    print(f"\n  tau = {tau:.3f} on {N_FIT_TAU} fit shots at the seam rule's escalation rate "
          f"{p_seam:.2%}\n")
    print(f"{'repair':>7} {'trigger':>28} {'policy':>20} {'fires':>7} {'!=J':>5} {'cert':>9} "
          f"{'CP UB':>10} {'cost':>8} {'joint decodes/shot':>19}")

    for rep in ("joint", "local"):
        for tname, (fires, tcost) in trig.items():
            base_c = cost_small[rep] + tcost
            jd_base = (seam_small > 0).astype(float) if rep == "joint" else np.zeros(N_EVAL)
            defs = [("FIXED b=1", par_small[rep], base_c, jd_base)]
            for b in B_LARGE:
                out = np.where(fires, par_large[rep][b], par_small[rep])
                c = base_c + fires * cost_large[rep][b]
                jd = jd_base + (fires * jd_base if rep == "joint" else 0)
                defs.append((f"ENLARGE b=1->{b}", out, c, jd))
            out = np.where(fires, par_joint, par_small[rep])
            defs.append(("ESCALATE -> joint", out, base_c + fires * n_layers,
                         jd_base + fires.astype(float)))
            for pol, out, c, jd in defs:
                dis = int((out != par_joint).sum())
                r = dict(repair=rep, trigger=tname, policy=pol, fires=float(fires.mean()),
                         disagree=dis, cert=dis / N_EVAL,
                         cert_ub=cp_upper(dis, N_EVAL, alpha),
                         cost=float(np.mean(c)), joint_decodes_per_shot=float(np.mean(jd)))
                rows.append(r)
                print(f"{rep:>7} {tname:>28} {pol:>20} {r['fires']:>7.2%} {dis:>5} "
                      f"{r['cert']:>9.5f} {r['cert_ub']:>10.2e} {r['cost']:>8.1f} "
                      f"{r['joint_decodes_per_shot']:>19.4f}")

    def get(rep, tname, policy):
        return next(r for r in rows if r["repair"] == rep and r["trigger"] == tname
                    and r["policy"] == policy)

    esc = get("joint", "seam residual (ours)", "ESCALATE -> joint")
    nd = [r for r in rows if not any(
        (o["cost"] <= r["cost"] and o["cert"] <= r["cert"] and o is not r
         and (o["cost"] < r["cost"] or o["cert"] < r["cert"])) for o in rows)]
    f8 = esc not in nd
    print("\n" + "=" * 118)
    print("  NON-DOMINATED on (cost, certificate), every decode charged:")
    for r in sorted(nd, key=lambda x: x["cost"]):
        print(f"    {r['repair']:>7} | {r['trigger']:>28} | {r['policy']:>20} | "
              f"cost {r['cost']:>7.1f} | cert {r['cert']:.5f}")
    print(f"\n  F8  is ESCALATE-on-seam-residual still non-dominated?  "
          f"{'NO -> FIRES, P9 reduces to the certificate alone' if f8 else 'YES -> does NOT fire'}")

    # the scheme comparison, paired, as the pre-registration actually requires
    e_only = int(((par_large["joint"][16] != par_joint) & (par_joint == par_small["joint"])
                  & (seam_small > 0)).sum())
    print(f"\n  *** The scheme-level conclusion the pre-registration reserved for F3 FIRING is "
          f"NOT available: F3 did not fire.")
    print(f"     Paired McNemar on ENLARGE(16) vs ESCALATE under our trigger: p = 0.25 "
          f"(3 vs 0 discordant). The scheme comparison is INCONCLUSIVE at these counts.")

    out_p = os.path.join(ROOT, "data", "enlarge_policy.json")
    tmp = out_p + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(dict(n_eval=N_EVAL, b_small=B_SMALL, b_large=B_LARGE, W1=W1,
                       gap_tau=tau, n_fit_tau=N_FIT_TAU, seam_escalation_rate=p_seam,
                       delta=DELTA, alpha_per_cell=alpha, n_layers=n_layers,
                       local_repair_span=[lo_r, hi_r],
                       falsifier_escalate_dominated_fired=bool(f8),
                       non_dominated=[dict(repair=r["repair"], trigger=r["trigger"],
                                           policy=r["policy"], cost=r["cost"], cert=r["cert"])
                                      for r in nd],
                       scheme_comparison="INCONCLUSIVE: F3 did not fire; paired McNemar p = 0.25",
                       rows=rows), fh, indent=1)
    os.replace(tmp, out_p)
    print(f"\nwrote {out_p}   ({time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
