#!/usr/bin/env python
"""🔴 THE FROZEN WITNESS. Do not edit after `WITNESS_HASH` is recorded in a pre-registration.

Two endpoints, computed on the SAME shots, with no free parameter between them. This file is
carried unchanged into any replication campaign; that is the whole point of freezing it.

    E1  the operational endpoint -- Pr[Delta_b], window/joint decoder disagreement at the frozen
        eight buffer widths, device against a DEM fitted to that context's own calibration split.

    E2  the structural endpoint -- the same-ancilla attenuation budget
            G_same = 2 * sum_{i<j, same ancilla} c_ij  -  sum_i omega_i,
        which is <= 0 for EVERY independent graphlike model, at any edge placement and any rates.

🔴 The two are NOT interchangeable and neither implies the other. Round 17 proved that in this
project's own data: `M1`, a one-parameter shot-rate mixture, violates `E2` at `G_same ~ +11.5` and
produces **exactly zero** `E1` disagreement in 200,000 shots. A paper that lets one stand in for the
other is wrong. They are reported as two separate findings that happen to be measurable on one
dataset.

🔴 `E2` needs `p_e <= 1/2`, which costs nothing while every singleton polarization `1 - 2<D_i>` is
positive. `check_polarization_positive` asserts it rather than assuming it.

The definitions here are parameter-free. Everything tunable -- shot counts, block lengths, the
comparison surrogate -- lives in the caller, not here.
"""
import hashlib
import os

import numpy as np

B_GRID = [1, 2, 3, 4, 6, 8, 12, 16]
W1 = 25


# ---------------------------------------------------------------------------- E2, structural
def attenuations(D):
    """`omega` (d,) and the shared-attenuation matrix `c` (d, d), zero diagonal.

    `omega_i  = -log(1 - 2<D_i>)`;  `c_ij = (omega_i + omega_j - omega_ij)/2`
    where `omega_ij = -log(1 - 2<D_i XOR D_j>)` and `<D_i XOR D_j> = <D_i> + <D_j> - 2<D_i D_j>`.
    """
    X = D.reshape(D.shape[0], -1).astype(np.float32)
    n = X.shape[0]
    m = X.mean(axis=0, dtype=np.float64)
    G = (X.T @ X).astype(np.float64) / n
    xor = m[:, None] + m[None, :] - 2 * G
    w = -np.log(np.clip(1 - 2 * m, 1e-12, None))
    wij = -np.log(np.clip(1 - 2 * xor, 1e-12, None))
    c = 0.5 * (w[:, None] + w[None, :] - wij)
    np.fill_diagonal(c, 0.0)
    return w, c


def check_polarization_positive(D):
    """`p_e <= 1/2` is w.l.o.g. only while every singleton polarization is positive. Assert it."""
    m = D.reshape(D.shape[0], -1).mean(axis=0, dtype=np.float64)
    worst = float(np.max(m))
    if worst >= 0.5:
        raise RuntimeError(f"a detector fires at {worst:.4f} >= 1/2; the nonnegative-attenuation "
                           "parameterisation is no longer without loss of generality")
    return worst


def same_ancilla_mask(n_layers, n_anc):
    a = np.arange(n_layers * n_anc) % n_anc
    M = a[:, None] == a[None, :]
    np.fill_diagonal(M, False)
    return M


def E2_graphlike_budget(D):
    """`G_same`, plus the per-detector budget slacks. `G_same <= 0` for every graphlike model."""
    check_polarization_positive(D)
    n_layers, n_anc = D.shape[1], D.shape[2]
    w, c = attenuations(D)
    S = same_ancilla_mask(n_layers, n_anc)
    t = (c * S).sum(axis=1)
    return dict(G_same=float(t.sum() - w.sum()),
                n_detectors_violating=int((t > w).sum()),
                n_detectors=int(len(w)),
                max_t_over_omega=float(np.max(t / np.maximum(w, 1e-12))),
                B=float(w.sum()), T_same=float(t.sum() / 2))


# ---------------------------------------------------------------------------- E1, operational
def E1_disagreement_curve(D, wtab, ttab, two_window):
    """`Pr[Delta_b]` at the frozen widths. `two_window` is injected so this file imports nothing."""
    return {b: float(np.asarray(
        two_window(D, W1, b, logical_j=0, eps=1e-3, seed=0,
                   wtab=wtab, ttab=ttab)["diverged_repaired"]).mean()) for b in B_GRID}


# ---------------------------------------------------------------------------- provenance
def witness_hash():
    """SHA-256 of this file's source. Recorded in the pre-registration; verified before every run."""
    with open(os.path.abspath(__file__), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


if __name__ == "__main__":
    print("frozen witness")
    print("  B_GRID =", B_GRID, " W1 =", W1)
    print("  sha256 =", witness_hash())
