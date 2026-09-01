#!/usr/bin/env python
"""Detector construction for the d=9 bit-flip repetition code, plus the M1 reproduction check.

Convention (bit-flip repetition code, logical |0>, ancilla reset each round):
  syn[s, r, j]      ancilla j measured in round r, shot s
  D[s, 0, j]      = syn[s, 0, j]                        (first round; deterministic 0 if noiseless)
  D[s, r, j]      = syn[s, r, j] XOR syn[s, r-1, j]     (bulk, 1 <= r <= R-1)
  D[s, R, j]      = sfin[s, j]   XOR syn[s, R-1, j]     (final, from data readout)
  sfin[s, j]      = fin[s, j] XOR fin[s, j+1]           (stabilizer from final data readout)

So D has R+1 detector rounds for R measurement rounds.

Run directly to reproduce the frozen M1 numbers:  python tools/detectors.py
"""
import json

import numpy as np


def load(path):
    z = np.load(path, allow_pickle=False)
    return z["syn"], z["fin"], json.loads(str(z["meta"]))


def build_detectors(syn, fin):
    """Return D of shape (shots, R+1, n_anc), uint8."""
    shots, R, n_anc = syn.shape
    sfin = (fin[:, :-1] ^ fin[:, 1:]).astype(np.uint8)
    assert sfin.shape[1] == n_anc, f"{sfin.shape[1]} stabilizers vs {n_anc} ancillas"
    D = np.empty((shots, R + 1, n_anc), dtype=np.uint8)
    D[:, 0, :] = syn[:, 0, :]
    D[:, 1:R, :] = syn[:, 1:, :] ^ syn[:, :-1, :]
    D[:, R, :] = sfin ^ syn[:, -1, :]
    return D


def logical_flip(fin):
    """Logical observable for the bit-flip repetition code: the first data qubit's readout.

    Any single data qubit works as the observable representative once the correction has been
    applied; index 0 is the convention used throughout this project.
    """
    return fin[:, 0].astype(np.uint8)


def _pearson(a, b):
    """Pearson correlation of two 0/1 vectors; nan when either is constant."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    sa, sb = a.std(), b.std()
    if sa == 0 or sb == 0:
        return np.nan
    return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))


def temporal_corr(D, dt, bulk_only=True):
    """Mean Pearson correlation between same-ancilla detectors at time lag dt."""
    _, nr, n_anc = D.shape
    lo, hi = (1, nr - 1) if bulk_only else (0, nr)
    vals = []
    for j in range(n_anc):
        for r in range(lo, hi - dt):
            c = _pearson(D[:, r, j], D[:, r + dt, j])
            if not np.isnan(c):
                vals.append(c)
    return float(np.mean(vals)) if vals else np.nan


def spatial_corr(D, ds, bulk_only=True):
    """Mean Pearson correlation between same-round detectors at ancilla distance ds."""
    _, nr, n_anc = D.shape
    lo, hi = (1, nr - 1) if bulk_only else (0, nr)
    vals = []
    for r in range(lo, hi):
        for j in range(n_anc - ds):
            c = _pearson(D[:, r, j], D[:, r + ds, j] if False else D[:, r, j + ds])
            if not np.isnan(c):
                vals.append(c)
    return float(np.mean(vals)) if vals else np.nan


if __name__ == "__main__":
    syn, fin, meta = load("data/da7miljsq5js73bk4vtg.npz")
    D = build_detectors(syn, fin)
    print(f"job {meta['job_id']} ({meta['label']}): D shape {D.shape}")

    bulk = D[:, 1:-1, :]
    print(f"\ndetector event rate (bulk): {bulk.mean():.4f}   [M1 frozen: ~0.039]")
    print(f"  per detector-round: first={D[:,0,:].mean():.4f} final={D[:,-1,:].mean():.4f}")

    print("\ntemporal, same ancilla   (M1 frozen values in brackets)")
    frozen_t = {1: .405, 2: .094, 3: .078, 5: .059, 8: .042, 10: .036,
                15: .022, 20: .015, 25: .011}
    for dt, want in frozen_t.items():
        got = temporal_corr(D, dt)
        flag = "OK " if abs(got - want) < 0.012 else "DIFF"
        print(f"  dt={dt:3d}  {got:+.4f}   [{want:+.3f}]  {flag}")

    print("\nspatial, same round")
    frozen_s = {1: .0653, 2: .0011, 3: -.0004, 4: -.0015}
    for ds, want in frozen_s.items():
        got = spatial_corr(D, ds)
        flag = "OK " if abs(got - want) < 0.012 else "DIFF"
        print(f"  ds={ds:3d}  {got:+.4f}   [{want:+.4f}]  {flag}")
