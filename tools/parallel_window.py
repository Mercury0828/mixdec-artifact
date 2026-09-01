#!/usr/bin/env python
"""Parallel two-window decoding with Zhang's ACTUAL seam syndrome, on simulated data.

This replaces the sequential pilot, which ledger `N5` records as FATAL: a one-sided sequential
decoder propagates its residual forward, so the stitched correction always explains the syndrome and
Zhang's seam syndrome does not exist in it. Here the two windows decode INDEPENDENTLY over the
ORIGINAL syndrome (parallel, "without merging"), each commits its own region, and the stitched
correction genuinely can fail to explain the syndrome. That leftover IS the seam syndrome.

Implementation notes that matter:
- Every edge gets a UNIQUE fault id, so `Matching.decode` returns the exact solution edge set with
  full type information. `decode_to_edges_array` cannot be used: it collapses every boundary edge to
  `-1`, and at a corner node (truncation layer AND j in {0, n_anc-1}) a spatial and a virtual
  temporal boundary edge are then indistinguishable.
- A truncated window gets **virtual time boundary** edges on its cut layer (Zhang's "virtual time
  boundary vertex"). They are placeholders for out-of-window errors: they are NOT committed, and
  dropping them is exactly what leaves a nontrivial seam syndrome.
- Zhang's Assumption 1 (unique minimum-weight correction) is enforced by perturbing weights; with
  uniform weights the optimum is massively degenerate and the assumption simply fails.

Usage:  python tools/parallel_window.py        # Theorem-1 validation self-check
"""
import os
import sys

import numpy as np
import pymatching

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilot_divergence import N_ANC  # noqa: E402

# edge kinds
SPACE, TIME, SBND, TBND, LTIME = "space", "time", "sbnd", "tbnd", "ltime"


class WindowGraph:
    """Matching graph over detector layers [lo, hi), with optional virtual temporal boundaries."""

    def __init__(self, lo, hi, n_anc=N_ANC, logical_j=0, temporal_lo=False, temporal_hi=False,
                 wtab=None, ttab=None, extra_lags=None, obs_as_detector=False, obs_layers=None):
        self.lo, self.hi, self.n_anc = lo, hi, n_anc
        self.span = hi - lo
        self.kind, self.glayer, self.dets, self.logical = [], [], [], []
        self.dq = []            # data-qubit index for space-like faults; -1 for time-like
        self.w = []             # per-edge weight, needed by cluster-confidence metrics (ADaPT)

        def node(r, j):
            return (r - lo) * n_anc + j

        # Two SEPARATE boundary nodes. A corner node (a truncation layer with j in {0, n_anc-1})
        # carries both a spatial and a virtual temporal boundary edge; sharing one boundary node
        # would make those parallel edges, which pymatching rejects outright.
        bnd_s = self.span * n_anc          # spatial boundary (data-qubit errors off the chain ends)
        bnd_t = self.span * n_anc + 1      # virtual TIME boundary (Zhang's virtual time vertex)
        # OBSERVABLE-AS-DETECTOR (`obs_as_detector`). Promoting the logical observable to an extra
        # REAL node makes the minimum weight in each logical class exactly computable: every
        # observable-carrying edge is routed to `obs_node` instead of to the spatial boundary, so a
        # correction touches `obs_node` an odd number of times exactly when it is in logical class 1.
        # Decoding the syndrome with that bit pinned to 0 and to 1 then gives both class minima, and
        # their difference is the complementary gap -- the soft information STCG (arXiv:2605.14637)
        # is built from. Exact for this code, two decodes per shot, nothing to approximate.
        # PATH-SELECTED variant (`obs_layers = (a, b)`). arXiv:2605.14637's stated problem is that
        # the ordinary gap is not usable for window decoding "especially with a small buffer",
        # because the whole-window observable is dominated by faults far from the seam that the
        # buffer was never meant to protect. Their fix is to select the spatiotemporal region the
        # confidence is about. Here that is exact and needs no approximation: route ONLY the
        # observable-carrying edges whose global layer lies in `[a, b)` to `obs_node`, leaving the
        # rest at the spatial boundary. Pinning `obs_node` then constrains the parity of the
        # correction RESTRICTED TO THAT REGION, and the gap is the weight the decoder must give up
        # to flip its decision there. `obs_layers=None` keeps the whole-window observable.
        obs_node = self.span * n_anc + 2
        self.obs_as_detector = obs_as_detector
        self.obs_layers = obs_layers
        self.obs_node = obs_node
        self.n_obs_edges = 0
        self.syn_len = self.span * n_anc + 3
        self.bnd = (bnd_s, bnd_t)
        m = pymatching.Matching()
        eid = 0

        def add(kind, glayer, a, b, flag, w, dq=-1):
            nonlocal eid
            if obs_as_detector and flag == 1 and (
                    obs_layers is None or obs_layers[0] <= glayer < obs_layers[1]):
                # only chain-end boundary edges ever carry the observable, so this rewires exactly
                # those and leaves every bulk edge untouched
                if a == bnd_s or b == bnd_s:
                    self.n_obs_edges += 1
                a = obs_node if a == bnd_s else a
                b = obs_node if b == bnd_s else b
            m.add_edge(a, b, fault_ids={eid}, weight=w)
            self.kind.append(kind)
            self.glayer.append(glayer)
            self.dets.append(frozenset(x for x in (a, b) if x not in (bnd_s, bnd_t, obs_node)))
            self.logical.append(flag)
            self.dq.append(dq)
            self.w.append(float(w))
            eid += 1

        def w_of(g, fid):
            # GLOBAL indexing: joint and every window must price the SAME physical fault
            # identically, or the decoders optimise different objectives and divergence is
            # manufactured out of nothing (and does not vanish as the buffer grows).
            return 1.0 if wtab is None else float(wtab[g * 17 + fid])

        def t_of(g, j):
            return 1.0 if ttab is None else float(ttab[g * n_anc + j])

        for r in range(lo, hi):
            for k in range(n_anc + 1):  # data qubits d_0 .. d_8
                if k == 0:
                    add(SBND, r, node(r, 0), bnd_s, 1 if logical_j == 0 else 0, w_of(r, 0), dq=0)
                elif k == n_anc:
                    add(SBND, r, node(r, n_anc - 1), bnd_s,
                        1 if logical_j == n_anc - 1 else 0, w_of(r, n_anc), dq=n_anc)
                else:
                    add(SPACE, r, node(r, k - 1), node(r, k), 0, w_of(r, k), dq=k)
            if r + 1 < hi:
                for j in range(n_anc):
                    add(TIME, r, node(r, j), node(r + 1, j), 0, w_of(r, 9 + j))

        # CORRELATION-AWARE extension: explicit same-ancilla long-range time edges. The FITTED LOCAL
        # graph contains no direct single-edge explanation for a lag-dt pair (it can still explain
        # one through several finite-weight edges -- see ledger `B14`; "priced at infinity" was
        # wrong), yet on `ibm_cleveland` the measured excess JOINT PROBABILITY at dt=2 is 0.0035,
        # 2.5x the bulk space-edge probability. Passing `extra_lags = {dt: weight}` adds them.
        # 🔴 RETRACTED AS A DECODING IMPROVEMENT (ledger `N9`): every placebo -- random lags,
        # permuted weights, flat weights, deliberately expensive weights -- reproduced the gain
        # exactly. Kept only so that refutation stays reproducible.
        if extra_lags:
            for dt, wt in sorted(extra_lags.items()):
                for r in range(lo, hi - dt):
                    for j in range(n_anc):
                        add(LTIME, r, node(r, j), node(r + dt, j), 0, float(wt))

        if temporal_lo:
            for j in range(n_anc):
                add(TBND, lo, node(lo, j), bnd_t, 0, t_of(lo, j))
        if temporal_hi:
            for j in range(n_anc):
                add(TBND, hi - 1, node(hi - 1, j), bnd_t, 0, t_of(hi - 1, j))

        m.set_boundary_nodes({bnd_s, bnd_t})
        self.m = m
        self.kind = np.array(self.kind)
        self.glayer = np.array(self.glayer)
        self.logical = np.array(self.logical, dtype=np.uint8)
        self.dq = np.array(self.dq)
        self.w = np.array(self.w, dtype=np.float64)
        self.n_edges = eid
        # edge -> data-qubit incidence, for codeword-level scoring (see `correction_on_data`)
        self.dq_inc = np.zeros((eid, n_anc + 1), dtype=np.uint8)
        for i, k in enumerate(self.dq):
            if k >= 0:
                self.dq_inc[i, k] ^= 1
        # dense edge->detector incidence, so boundary_of is a matmul instead of a Python loop
        self.inc = np.zeros((eid, self.span * n_anc), dtype=np.uint8)
        for i, ds in enumerate(self.dets):
            for d in ds:
                self.inc[i, d] ^= 1
        # SPARSE form of the same incidence. The dense matmul is O(n_edges x n_detectors) per call
        # -- 32M integer ops per shot on a 401-layer graph, which is what made the many-seam sweep
        # unrunnable. A matching solution touches only a handful of edges, so scattering their
        # endpoints is O(selected). `_flat` lists each (edge, detector) incidence once; equality
        # with the matmul is asserted in the self-check below.
        self._inc_e = np.array([i for i, ds in enumerate(self.dets) for _ in ds], dtype=np.int64)
        self._inc_d = np.array([d for ds in self.dets for d in ds], dtype=np.int64)
        self._inc_ptr = np.zeros(eid + 1, dtype=np.int64)
        np.cumsum([len(ds) for ds in self.dets], out=self._inc_ptr[1:])

    def decode(self, D_window):
        """D_window is (span, n_anc). Returns the solution edge indicator vector."""
        return self.m.decode(D_window.ravel()).astype(np.uint8)

    def decode_class(self, D_window, c):
        """Minimum-weight correction CONSTRAINED to logical class `c`, and its weight.

        Requires `obs_as_detector=True`. Pinning the observable node's syndrome bit to `c` forces
        the matching to use an odd (c=1) or even (c=0) number of observable-carrying edges, so the
        two calls return the exact class minima. `gap = |w_1 - w_0|` is the complementary gap.
        """
        if not self.obs_as_detector:
            raise ValueError("decode_class needs a graph built with obs_as_detector=True")
        if self.n_obs_edges == 0:
            raise ValueError(f"obs_layers={self.obs_layers} selects no observable-carrying edge in "
                             f"[{self.lo}, {self.hi}); class 1 is unreachable")
        z = np.zeros(self.syn_len, dtype=np.uint8)
        z[: self.span * self.n_anc] = D_window.ravel()
        z[self.obs_node] = c
        sol, w = self.m.decode(z, return_weight=True)
        return sol.astype(np.uint8), float(w)

    def complementary_gap(self, D_window):
        """(argmin class, gap). Gap is the weight the decoder would have to give up to flip class."""
        _, w0 = self.decode_class(D_window, 0)
        _, w1 = self.decode_class(D_window, 1)
        return (0 if w0 <= w1 else 1), abs(w1 - w0)

    def correction_on_data(self, sel):
        """Parity of the correction applied to each data qubit d_0..d_8 by an edge subset.

        A bit-flip on d_k at ANY round persists to the final readout, so the total correction on
        d_k is the parity of that qubit's space-like faults over all rounds. Used for CODEWORD-LEVEL
        scoring, which replaces the old 'value of d_0' rule: that rule put the entire logical
        observable on one boundary edge, so a mis-estimated boundary probability translated
        one-for-one into logical errors (device diagnosis 2026-08-29: the decoder fixed 16/16 real
        d_0 flips but spuriously applied 13 d_0 corrections on noisy shots, because the fitted d_0
        boundary weight 5.92 came out CHEAPER than a single bulk space edge 6.58).
        """
        return ((sel.astype(np.int64) @ self.dq_inc) & 1).astype(np.uint8)

    def boundary_of(self, sel):
        """Detector-flip pattern (span, n_anc) of the selected edge subset.

        Scatters the selected edges' endpoints instead of multiplying by the dense incidence.
        Identical output; O(selected edges) instead of O(all edges x all detectors).
        """
        out = np.zeros(self.span * self.n_anc, dtype=np.uint8)
        sel_e = np.flatnonzero(sel)
        if sel_e.size:
            take = np.concatenate([np.arange(self._inc_ptr[i], self._inc_ptr[i + 1])
                                   for i in sel_e]) if sel_e.size else np.empty(0, np.int64)
            np.bitwise_xor.at(out, self._inc_d[take], 1)
        return out.reshape(self.span, self.n_anc)


def make_weights(n_layers, n_anc, eps=1e-3, seed=0):
    """One global weight table shared by joint and every window.

    `eps > 0` breaks minimum-weight ties, which is what Zhang's Assumption 1 requires ("should hold
    for MWPM decoders if edge weights are perturbed to break ties"). Under uniform weights the
    optimum is massively degenerate and Assumption 1 simply fails.
    """
    rng = np.random.default_rng(seed)
    wtab = 1.0 + eps * rng.random((n_layers + 1) * 17)
    ttab = 1.0 + eps * rng.random((n_layers + 1) * n_anc)
    return wtab, ttab


def two_window(D, W1, b, logical_j=0, eps=1e-3, seed=0, wtab=None, ttab=None,
               extra_lags=None):
    """Parallel two-window decode of every shot.

    Returns dict of per-shot arrays: par_joint, par_split, seam_nontrivial, seam_weight.
    """
    shots, n_layers, n_anc = D.shape
    if wtab is None or ttab is None:
        wtab, ttab = make_weights(n_layers, n_anc, eps=eps, seed=seed)
    # SYMMETRIC buffers: each window looks b layers PAST its commit region, on the side where it
    # abuts the other window. Window 2 needs its buffer too — without it the seam syndrome is not
    # suppressed by b at all, which is the bug that made the first run's seam counts flat in b.
    e1 = min(W1 + b, n_layers)      # window 1 decodes [0, e1),  commits [0, W1)
    s2 = max(W1 - b, 0)             # window 2 decodes [s2, n),  commits [W1, n)

    joint = WindowGraph(0, n_layers, n_anc, logical_j, False, False, wtab=wtab, ttab=ttab,
                        extra_lags=extra_lags)
    g1 = WindowGraph(0, e1, n_anc, logical_j, False, e1 < n_layers, wtab=wtab, ttab=ttab,
                     extra_lags=extra_lags)
    g2 = WindowGraph(s2, n_layers, n_anc, logical_j, s2 > 0, False, wtab=wtab, ttab=ttab,
                     extra_lags=extra_lags)

    # which committed edges each window owns
    c1 = (g1.kind != TBND) & (g1.glayer < W1)
    c2 = (g2.kind != TBND) & (g2.glayer >= W1)

    par_j = np.empty(shots, np.uint8)
    par_s = np.empty(shots, np.uint8)
    par_r = np.empty(shots, np.uint8)
    seam_nt = np.zeros(shots, np.uint8)
    seam_w = np.zeros(shots, np.int32)
    # per-data-qubit correction parity, for CODEWORD-LEVEL scoring against a known logical truth.
    # The d_0 parity above puts the whole observable on one boundary edge; `correction_on_data`
    # spreads it over all n_anc+1 data qubits, which is the scoring rebuilt in commit a19cce3.
    corr_j = np.zeros((shots, n_anc + 1), np.uint8)
    corr_r = np.zeros((shots, n_anc + 1), np.uint8)

    for s in range(shots):
        sol_j = joint.decode(D[s])
        par_j[s] = joint.logical[sol_j.astype(bool)].sum() & 1
        corr_j[s] = joint.correction_on_data(sol_j)

        k1 = g1.decode(D[s, :e1]) & c1
        k2 = g2.decode(D[s, s2:]) & c2
        par_s[s] = (g1.logical[k1.astype(bool)].sum() + g2.logical[k2.astype(bool)].sum()) & 1
        corr_r[s] = g1.correction_on_data(k1) ^ g2.correction_on_data(k2)

        # stitched correction's detector boundary, in global layer coordinates
        stitched = np.zeros((n_layers, n_anc), dtype=np.uint8)
        stitched[:e1] ^= g1.boundary_of(k1)
        stitched[s2:] ^= g2.boundary_of(k2)

        seam = stitched ^ D[s]          # detectors the stitched correction fails to explain
        seam_w[s] = int(seam.sum())
        seam_nt[s] = 1 if seam_w[s] else 0

        # SEAM REPAIR. Without it the stitched object is not a valid correction when the seam is
        # nontrivial, and its "logical class" is not canonical -- the round-2 auditor's central
        # objection. A complete parallel-window decoder repairs the residual; decode it on the full
        # graph and add it, so C_repaired explains the syndrome exactly and its parity IS a class.
        if seam_w[s]:
            fix = joint.decode(seam)
            par_r[s] = par_s[s] ^ (joint.logical[fix.astype(bool)].sum() & 1)
            corr_r[s] ^= joint.correction_on_data(fix)
        else:
            par_r[s] = par_s[s]

    return dict(par_joint=par_j, par_split=par_s, par_repaired=par_r,
                corr_joint=corr_j, corr_repaired=corr_r,
                seam_nontrivial=seam_nt, seam_weight=seam_w,
                diverged=(par_j != par_s).astype(np.uint8),
                diverged_repaired=(par_j != par_r).astype(np.uint8), e1=e1, s2=s2)


if __name__ == "__main__":
    from sim_substrate import edge_table, make_circuit, reduced_weights, sample, to_layers

    print("Zhang Theorem 1 validation on simulated data with GROUND TRUTH")
    print("Theorem: nontrivial seam syndrome => physical error weight >= w_b/2.")
    print("Uniform vertical weight => w_b = b, so the threshold is b/2.\n")
    print(f"{'d':>2} {'p':>6} {'b':>3} {'shots':>6} {'seamNT':>7} {'minW|seamNT':>12} "
          f"{'b/2':>5} {'THM1':>6} {'Delta':>6} {'D&~seam':>8}")

    for d, p, rounds in [(9, 0.005, 40), (9, 0.02, 40), (9, 0.05, 40)]:
        circ = make_circuit(distance=d, rounds=rounds, p=p)
        dem, dets, obs, errs = sample(circ, 3000, seed=7)
        na = d - 1
        D = to_layers(dets, n_anc=na)
        w, _ = reduced_weights(errs, edge_table(dem, n_anc=na))
        W1 = D.shape[1] // 2
        for b in (1, 2, 4, 8, 12, 16):
            r = two_window(D, W1, b, logical_j=na - 1, eps=1e-3, seed=11)
            nt = r["seam_nontrivial"].astype(bool)
            dv = r["diverged"].astype(bool)
            minw = int(w[nt].min()) if nt.any() else -1
            ok = "OK" if (not nt.any() or minw >= b / 2) else "VIOLATED"
            print(f"{d:>2} {p:>6} {b:>3} {D.shape[0]:>6} {int(nt.sum()):>7} {minw:>12} "
                  f"{b/2:>5.1f} {ok:>6} {int(dv.sum()):>6} {int((dv & ~nt).sum()):>8}")
