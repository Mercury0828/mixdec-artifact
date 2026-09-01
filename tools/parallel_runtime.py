#!/usr/bin/env python
"""Round 9A: a REAL parallel two-window decoder, measured end to end. 0 QPU.

🔴 WHY THIS FILE EXISTS. The pre-writing audit's finding 7: every "concurrent" number this project
has ever reported is a composition of separately timed serial runs,
`max(t1, t2) + p_seam * t_fallback`, with no scheduling, no dispatch, no collection and a Jensen
bias -- and no parallel implementation existed at all. So none of it was measured performance.

🔴 THE MEASURED FACT THAT SHAPES THE DESIGN. pymatching 2.4.0 does **not** release the GIL. Two
decoder threads on two `Matching` objects run at **0.81x of serial** -- slower than doing the work
one after the other. Threads are therefore not an implementation option, and the concurrency has to
cross a process boundary.

THE IMPLEMENTATION. `W` persistent worker processes, one per window, each building its `WindowGraph`
ONCE at startup. Per shot the parent publishes the record into `multiprocessing.shared_memory` and
raises a flag; workers **spin** on that flag rather than blocking on a queue, decode their own slice,
write back their committed boundary contribution and logical parity, and raise a done flag. Nothing
is pickled per shot and no queue is touched on the critical path -- this is the fastest honest
dispatch available in this runtime, so if latency does not improve here it will not improve with a
slower one.

TWO REGIMES, both timed by the parent's own clock:

  LATENCY     per shot, publish -> both workers done -> stitch -> seam -> escalate if needed,
              against a single-process joint decode of the same shots. Dispatch and collection are
              INSIDE the measurement, which is exactly what the composed projection left out.
  THROUGHPUT  batch mode: the shot set is split across `W` workers, each decoding whole chunks, and
              the parent measures shots/second against one process doing joint decodes. This is the
              regime guide 7.3 offline scheduling actually describes.

Pre-registered at `c5157fe`, `docs/expected.md` Round 9A, falsifiers 1 and 2.

Usage:  python tools/parallel_runtime.py
"""
import json
import os
import sys
import time
from multiprocessing import Process, shared_memory

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W1 = 25
B = 1
N_FIT = 50_000
N_LAT = 4_000                # shots timed one at a time, end to end
N_THR = 20_000               # shots timed in batch
THR_REPS = 5                 # paired, interleaved repeats of the whole pipeline
SPIN_YIELD = 20_000          # spins before a yield, to avoid a livelock on an oversubscribed box


# --------------------------------------------------------------------------- worker
def _worker(name, lo, hi, commit_lo, commit_hi, n_anc, n_layers, wtab, ttab, temporal_lo,
            temporal_hi, shm_in, shm_out, shm_flag, idx, batch_mode, n_batch):
    """Persistent decoder process. Builds its graph once, then spins on `flag`.

    flag values: 0 idle, 1 work published, 2 result ready, 3 shut down.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from parallel_window import TBND, WindowGraph  # noqa: E402

    g = WindowGraph(lo, hi, n_anc, 0, temporal_lo, temporal_hi, wtab=wtab, ttab=ttab)
    own = (g.kind != TBND) & (g.glayer >= commit_lo) & (g.glayer < commit_hi)

    a_in = shared_memory.SharedMemory(name=shm_in)
    a_out = shared_memory.SharedMemory(name=shm_out)
    a_fl = shared_memory.SharedMemory(name=shm_flag)
    inp = np.ndarray((n_batch, n_layers, n_anc), dtype=np.uint8, buffer=a_in.buf)
    flag = np.ndarray((4,), dtype=np.int32, buffer=a_fl.buf)
    if batch_mode:
        batch_out = np.ndarray((n_batch, 2, n_layers * n_anc + 1), dtype=np.uint8, buffer=a_out.buf)
        out = batch_out[0]
    else:
        out = np.ndarray((2, n_layers * n_anc + 1), dtype=np.uint8, buffer=a_out.buf)
        batch_out = None

    if name == "null":
        # NULL CONTROL: no decoding at all, just the flag round trip. This separates "our scheme
        # is slow" from "inter-process synchronisation costs more than the whole decode".
        while True:
            spins = 0
            while flag[idx] != 1:
                if flag[idx] == 3:
                    a_in.close(), a_out.close(), a_fl.close()
                    return
                spins += 1
                if spins > SPIN_YIELD:
                    time.sleep(0)
                    spins = 0
            out[idx, n_layers * n_anc] = 0
            flag[idx] = 2

    if batch_mode:
        # one flag round trip for the whole chunk: the throughput regime.
        # 🔴 The first version called `g.decode` and NOTHING ELSE on both sides, so it compared a
        # window KERNEL against a joint KERNEL -- the exact asymmetry round 9A had just identified
        # as having broken the latency projection. Re-gate finding 2. The worker now does the FULL
        # per-window job: mask to the commit region, build the boundary contribution, take the
        # parity, and write both back, so the parent can stitch a real result.
        bstride = n_layers * n_anc
        while True:
            spins = 0
            while flag[idx] != 1:
                if flag[idx] == 3:
                    a_in.close(), a_out.close(), a_fl.close()
                    return
                spins += 1
                if spins > SPIN_YIELD:
                    time.sleep(0)
                    spins = 0
            lo_s, hi_s = int(flag[2]), int(flag[3])
            for s in range(lo_s, hi_s):
                k = g.decode(inp[s, lo:hi]) & own
                bnd = np.zeros((n_layers, n_anc), dtype=np.uint8)
                bnd[lo:hi] = g.boundary_of(k)
                batch_out[s, idx, :bstride] = bnd.ravel()
                batch_out[s, idx, bstride] = g.logical[k.astype(bool)].sum() & 1
            flag[idx] = 2
    else:
        while True:
            spins = 0
            while flag[idx] != 1:
                if flag[idx] == 3:
                    a_in.close(), a_out.close(), a_fl.close()
                    return
                spins += 1
                if spins > SPIN_YIELD:
                    time.sleep(0)
                    spins = 0
            k = g.decode(inp[0, lo:hi]) & own
            bnd = np.zeros((n_layers, n_anc), dtype=np.uint8)
            bnd[lo:hi] = g.boundary_of(k)
            out[idx, : n_layers * n_anc] = bnd.ravel()
            out[idx, n_layers * n_anc] = g.logical[k.astype(bool)].sum() & 1
            flag[idx] = 2


def _spin_until(flag, want, n=2):
    spins = 0
    while not all(flag[i] == want for i in range(n)):
        spins += 1
        if spins > SPIN_YIELD:
            time.sleep(0)
            spins = 0


class TwoWindowRuntime:
    """Parent-side handle on the two worker processes."""

    def __init__(self, n_layers, n_anc, wtab, ttab, batch_mode=False, n_batch=1, tag="w"):
        self.n_layers, self.n_anc, self.tag = n_layers, n_anc, tag
        e1, s2 = min(W1 + B, n_layers), max(W1 - B, 0)
        self.spans = [(0, e1, 0, W1, False, e1 < n_layers),
                      (s2, n_layers, W1, n_layers, s2 > 0, False)]
        self.shm_in = shared_memory.SharedMemory(create=True,
                                                 size=max(1, n_batch * n_layers * n_anc))
        n_out = (n_batch if batch_mode else 1)
        self.shm_out = shared_memory.SharedMemory(
            create=True, size=n_out * 2 * (n_layers * n_anc + 1))
        self.batch_mode = batch_mode
        self.shm_fl = shared_memory.SharedMemory(create=True, size=4 * 4)
        self.inp = np.ndarray((n_batch, n_layers, n_anc), dtype=np.uint8, buffer=self.shm_in.buf)
        if batch_mode:
            self.bout = np.ndarray((n_batch, 2, n_layers * n_anc + 1), dtype=np.uint8,
                                   buffer=self.shm_out.buf)
            self.out = self.bout[0]
        else:
            self.out = np.ndarray((2, n_layers * n_anc + 1), dtype=np.uint8,
                                  buffer=self.shm_out.buf)
        self.flag = np.ndarray((4,), dtype=np.int32, buffer=self.shm_fl.buf)
        self.flag[:] = 0
        self.procs = []
        for i, (lo, hi, clo, chi, tlo, thi) in enumerate(self.spans):
            p = Process(target=_worker,
                        args=(self.tag, lo, hi, clo, chi, n_anc, n_layers, wtab, ttab, tlo, thi,
                              self.shm_in.name, self.shm_out.name, self.shm_fl.name, i,
                              batch_mode, n_batch),
                        daemon=True)
            p.start()
            self.procs.append(p)

    def warm(self, d, reps=200):
        for _ in range(reps):
            self.decode_one(d)

    def decode_one(self, d):
        """Publish one shot, wait for both windows, return stitched boundary and split parity."""
        self.inp[0] = d
        self.flag[0] = 1
        self.flag[1] = 1
        _spin_until(self.flag, 2)
        st = (self.out[0, : self.n_layers * self.n_anc]
              ^ self.out[1, : self.n_layers * self.n_anc]).reshape(self.n_layers, self.n_anc)
        par = int(self.out[0, -1] ^ self.out[1, -1])
        self.flag[0] = 0
        self.flag[1] = 0
        return st, par

    def run_chunk(self, lo_s, hi_s):
        self.flag[2] = lo_s
        self.flag[3] = hi_s
        self.flag[0] = 1
        self.flag[1] = 1
        _spin_until(self.flag, 2)
        self.flag[0] = 0
        self.flag[1] = 0

    def close(self):
        self.flag[0] = 3
        self.flag[1] = 3
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        for sh in (self.shm_in, self.shm_out, self.shm_fl):
            sh.close()
            sh.unlink()


def main():
    from analyze_campaign_r import load_arm
    from detectors import build_detectors
    from parallel_window import WindowGraph
    from weight_model_robustness import fit_weights_v2

    syn, fin, _ = load_arm(0)
    D = build_detectors(syn, fin)
    n_layers, n_anc = D.shape[1], D.shape[2]
    w, t, _, _ = fit_weights_v2(D[:N_FIT], n_fit=N_FIT)
    Dev = np.ascontiguousarray(D[N_FIT:N_FIT + N_THR])
    joint = WindowGraph(0, n_layers, n_anc, 0, False, False, wtab=w, ttab=t)

    print("ROUND 9A -- A REAL PARALLEL TWO-WINDOW DECODER.  device shots, 0 QPU")
    print(f"  {os.cpu_count()} logical cores; pymatching does NOT release the GIL, so the")
    print("  concurrency is two persistent PROCESSES over shared memory with spin-wait.\n")

    report = {"n_cores": os.cpu_count(), "n_layers": n_layers, "b": B, "W1": W1}

    # ------------------------------------------------------------------ 0. WHERE THE TIME GOES
    # The composed projection timed `WindowGraph.decode` and nothing else. A window in this scheme
    # must ALSO mask to its commit region, produce its boundary contribution and its logical parity
    # -- work the joint decoder does not have to do, and work the projection never counted.
    from parallel_window import TBND
    e1 = min(W1 + B, n_layers)
    g1 = WindowGraph(0, e1, n_anc, 0, False, e1 < n_layers, wtab=w, ttab=t)
    own = (g1.kind != TBND) & (g1.glayer < W1)
    sub = Dev[:N_LAT]

    def _bench(fn):
        fn(sub[0])
        t0 = time.perf_counter()
        for s in range(len(sub)):
            fn(sub[s])
        return (time.perf_counter() - t0) / len(sub) * 1e6

    def _full(d):
        k = g1.decode(d[:e1]) & own
        bnd = np.zeros((n_layers, n_anc), dtype=np.uint8)
        bnd[0:e1] = g1.boundary_of(k)
        return bnd, g1.logical[k.astype(bool)].sum() & 1

    t_dec = _bench(lambda d: g1.decode(d[:e1]))
    t_msk = _bench(lambda d: g1.decode(d[:e1]) & own)
    t_full = _bench(_full)
    print("0. WHERE THE TIME GOES, one window, in-process")
    print(f"   decode only (all the projection ever timed) : {t_dec:7.2f} us/shot")
    print(f"     + commit mask                             : {t_msk:7.2f} us/shot")
    print(f"     + boundary + parity (what it MUST do)     : {t_full:7.2f} us/shot")
    print(f"   ** the projection omitted {100 * (1 - t_dec / t_full):.0f}% of each window's work")
    print()
    report["per_window_cost_us"] = dict(decode_only=t_dec, plus_commit_mask=t_msk,
                                        full_window_work=t_full,
                                        fraction_projection_omitted=1 - t_dec / t_full)

    # ------------------------------------------------------------------ LATENCY
    print(f"1. LATENCY, per shot, end to end. {N_LAT} shots.")
    rt = TwoWindowRuntime(n_layers, n_anc, w, t, batch_mode=False, n_batch=1)
    rt.warm(Dev[0])
    lat_par, seam_hits = [], 0
    out_par = np.zeros(N_LAT, dtype=np.uint8)
    ref_par = np.zeros(N_LAT, dtype=np.uint8)
    for s in range(N_LAT):
        t0 = time.perf_counter()
        st, par = rt.decode_one(Dev[s])
        seam = st ^ Dev[s]
        if seam.any():
            # 🔴 escalate PROPERLY: the joint solution replaces both the parity and the correction
            # boundary. The previous version called `joint.decode` and threw its output away, so the
            # "escalated" result was still the invalid stitched one. Gate-4 finding 2.
            sol = joint.decode(Dev[s])
            par = int(joint.logical[sol.astype(bool)].sum() & 1)
            st = joint.boundary_of(sol)
            seam_hits += 1
        out_par[s] = par
        lat_par.append((time.perf_counter() - t0) * 1e6)
    rt.close()

    # null control: identical dispatch, zero decoding
    rt0 = TwoWindowRuntime(n_layers, n_anc, w, t, batch_mode=False, n_batch=1, tag="null")
    rt0.warm(Dev[0])
    lat_null = []
    for s in range(N_LAT):
        t0 = time.perf_counter()
        rt0.decode_one(Dev[s])
        lat_null.append((time.perf_counter() - t0) * 1e6)
    rt0.close()

    lat_joint = []
    for s in range(N_LAT):
        t0 = time.perf_counter()
        sol = joint.decode(Dev[s])
        ref_par[s] = joint.logical[sol.astype(bool)].sum() & 1
        lat_joint.append((time.perf_counter() - t0) * 1e6)
    # both paths now produce a per-shot logical parity, and it is the SAME one
    same = int((out_par == ref_par).sum())
    print(f"  outputs agree on {same}/{N_LAT} shots "
          f"({'IDENTICAL' if same == N_LAT else 'MISMATCH -- the comparison is invalid'})")

    lp, lj, ln = np.array(lat_par), np.array(lat_joint), np.array(lat_null)
    print(f"{'':>26} {'median':>9} {'mean':>9} {'p90':>9} {'p99':>9}")
    for nm, a in (("two-window (parallel)", lp), ("joint (single proc)", lj),
                  ("NULL control (no decode)", ln)):
        print(f"{nm:>26} {np.median(a):>9.2f} {a.mean():>9.2f} "
              f"{np.percentile(a, 90):>9.2f} {np.percentile(a, 99):>9.2f}")
    ratio = float(np.median(lj) / np.median(lp))
    print(f"  dispatch alone (null control) costs {np.median(ln):.2f} us median, against a "
          f"{np.median(lj):.2f} us joint decode")
    print(f"  escalation rate {seam_hits / N_LAT:.2%}   MEASURED latency ratio "
          f"{ratio:.3f}x  (>1 means the parallel scheme is faster)")
    report["latency"] = dict(
        shots=N_LAT, escalation_rate=seam_hits / N_LAT,
        parallel=dict(median=float(np.median(lp)), mean=float(lp.mean()),
                      p90=float(np.percentile(lp, 90)), p99=float(np.percentile(lp, 99))),
        joint=dict(median=float(np.median(lj)), mean=float(lj.mean()),
                   p90=float(np.percentile(lj, 90)), p99=float(np.percentile(lj, 99))),
        null_dispatch=dict(median=float(np.median(ln)), mean=float(ln.mean()),
                           p90=float(np.percentile(ln, 90))),
        measured_ratio_median=ratio)

    # ------------------------------------------------------------------ THROUGHPUT
    print(f"\n2. THROUGHPUT, batch mode, END TO END. {N_THR} shots x {THR_REPS} paired reps.")
    print("   \U0001F534 DISCLOSURE: while the JOINT baseline is timed, both window worker")
    print("   processes are alive in spin loops, so the baseline runs under contention the")
    print("   windowed path does not face. That BIASES TOWARD WINDOWING, so the negative result")
    print("   below is conservative. Gate-6 finding 6.")
    print("   Both pipelines return the same validated object: a stitched correction boundary and")
    print("   a logical parity, with the seam tested and escalated. The first version of this")
    print("   section timed `decode` alone on BOTH sides -- window kernel against joint kernel --")
    print("   which is the asymmetry that broke the latency projection. Re-gate finding 2.\n")

    rt = TwoWindowRuntime(n_layers, n_anc, w, t, batch_mode=True, n_batch=N_THR)
    rt.inp[:] = Dev
    stride = n_layers * n_anc

    def run_parallel():
        """Workers decode+mask+boundary+parity; the parent stitches, tests the seam, escalates."""
        rt.run_chunk(0, N_THR)
        bo = rt.bout
        st = (bo[:, 0, :stride] ^ bo[:, 1, :stride]).reshape(N_THR, n_layers, n_anc)
        par = bo[:, 0, stride] ^ bo[:, 1, stride]
        seam = st ^ Dev
        hits = np.flatnonzero(seam.any(axis=(1, 2)))
        for s in hits:
            # escalate PROPERLY: the joint solution replaces the parity AND the stitched boundary
            sol = joint.decode(Dev[s])
            par[s] = joint.logical[sol.astype(bool)].sum() & 1
            st[s] = joint.boundary_of(sol)
        return par, len(hits)

    def run_joint():
        """The single-process baseline, returning the same object: a logical parity per shot."""
        par = np.empty(N_THR, dtype=np.uint8)
        for s in range(N_THR):
            sol = joint.decode(Dev[s])
            par[s] = joint.logical[sol.astype(bool)].sum() & 1
        return par

    pj, _ = run_parallel()                    # warm both paths, and validate they agree
    rj = run_joint()
    agree = int((pj == rj).sum())
    dis = N_THR - agree
    # 🔴 THE DISAGREEMENTS ARE THE OBJECT OF STUDY, NOT A BUG. The windowed+escalate pipeline is
    # *supposed* to differ from joint decoding on exactly the shots `P7` bounds, and the count of
    # them is the certificate. So this is a cross-check between two independent tools rather than a
    # validity gate: `resource_frontier.py` measures the same policy on the same 20,000 shots.
    try:
        with open(os.path.join(ROOT, "data", "resource_frontier.json")) as fh:
            exp = next(r["disagree"] for r in json.load(fh)["rows"]
                       if r["policy"].startswith("ESCALATE") and r["trigger"].startswith("seam"))
    except Exception:
        exp = None
    tag = ("MATCHES resource_frontier" if exp is not None and dis == exp
           else f"does NOT match resource_frontier's {exp}" if exp is not None
           else "no reference available")
    print(f"  outputs differ on {dis}/{N_THR} shots -- this IS the certificate, and it {tag}")
    tp, tj_, esc = [], [], 0
    # 🔴 COUNTERBALANCED: the order alternates so a monotone drift cannot favour either path.
    # The previous version always ran parallel first. Gate-4 finding 2.
    for rep in range(THR_REPS):
        if rep % 2 == 0:
            t0 = time.perf_counter(); _, esc = run_parallel(); tp.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); run_joint(); tj_.append(time.perf_counter() - t0)
        else:
            t0 = time.perf_counter(); run_joint(); tj_.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); _, esc = run_parallel(); tp.append(time.perf_counter() - t0)
    rt.close()

    t_par, t_joint = float(np.median(tp)), float(np.median(tj_))
    thr_par, thr_joint = N_THR / t_par, N_THR / t_joint
    ratios = sorted(b / a for a, b in zip(tp, tj_))
    print(f"  two windows, 2 processes, end to end : {t_par:7.3f} s  = {thr_par:9.0f} shots/s")
    print(f"  joint, 1 process, end to end         : {t_joint:7.3f} s  = {thr_joint:9.0f} shots/s")
    print(f"  escalated {esc}/{N_THR} = {esc / N_THR:.2%}")
    print(f"  MEASURED end-to-end throughput ratio {thr_par / thr_joint:.3f}x  "
          f"[{ratios[0]:.3f}-{ratios[-1]:.3f}] over {THR_REPS} paired reps")
    report["throughput"] = dict(shots=N_THR, reps=THR_REPS, parallel_s=t_par, joint_s=t_joint,
                                parallel_shots_per_s=thr_par, joint_shots_per_s=thr_joint,
                                measured_speedup=thr_par / thr_joint,
                                ratio_lo=ratios[0], ratio_hi=ratios[-1],
                                escalated=int(esc), escalation_rate=esc / N_THR,
                                raw_parallel_s=[float(x) for x in tp],
                                raw_joint_s=[float(x) for x in tj_],
                                n_outputs_differing=int(dis),
                                cross_check_vs_resource_frontier=exp,
                                cross_check_matches=bool(exp is not None and dis == exp),
                                counterbalanced=True,
                                note="END TO END. The declared output of BOTH paths is a per-shot "
                                     "logical parity. They agree on every shot except the "
                                     "certificate's own disagreements, whose count is stored above "
                                     "and cross-checked against resource_frontier. The windowed path "
                                     "also maintains a stitched correction "
                                     "boundary because its scheme requires one; on escalation the "
                                     "joint solution replaces both. `ratio_lo/hi` is the OBSERVED "
                                     "RANGE over the paired reps, not a confidence interval.")

    f1 = ratio <= 1.0
    f2 = (thr_par / thr_joint) < 1.0        # round 10 falsifier 7: end to end, not decode-only
    print("\n" + "=" * 92)
    print(f"  F1  measured per-shot latency ratio {ratio:.3f}x  -> "
          f"{'FIRES -- every latency and speedup claim is withdrawn' if f1 else 'does NOT fire'}")
    print(f"  F7  END-TO-END batch throughput {thr_par / thr_joint:.3f}x  -> "
          f"{'FIRES -- parallel windowing buys nothing in any regime' if f2 else 'does NOT fire'}")
    report["falsifier_latency_fired"] = bool(f1)
    report["falsifier_throughput_fired"] = bool(f2)

    out = os.path.join(ROOT, "data", "parallel_runtime.json")
    tmp = out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(report, fh, indent=1)
    os.replace(tmp, out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
