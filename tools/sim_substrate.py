#!/usr/bin/env python
"""Simulator substrate: stim repetition-code circuits with GROUND-TRUTH fault configurations.

Why this exists (owner rule, 2026-08-29): validate on a simulator before spending QPU seconds.
And it unlocks a test hardware cannot do — ledger `B9` says real syndrome shots do not identify the
underlying physical error configuration, so no shot-count experiment on hardware can falsify a
minimum-weight statement. A simulator hands you the injected faults directly.

The hand-built matching graph in `pilot_divergence.build_graph` was cross-validated against stim's
DEM on 2026-08-29: 408 detectors = 51 layers x 8 ancillas; boundary edges only at j=0 and j=7; every
two-detector error is space-like (same layer, adjacent ancilla) or time-like (same ancilla, adjacent
layer). The logical observable sits on the j=7 boundary in stim and on j=0 here; the two were shown
to give bit-identical divergence on every device shot, so the choice is a gauge.

Usage:  python tools/sim_substrate.py        # self-check
"""
import os
import sys

import numpy as np
import stim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilot_divergence import N_ANC  # noqa: E402

DEFAULT_NOISE = dict(
    before_measure_flip_probability=0.02,
    after_reset_flip_probability=0.02,
    before_round_data_depolarization=0.02,
)


def make_circuit(distance=9, rounds=50, p=None, **noise):
    """Repetition-code memory circuit. `p` sets all three noise channels at once."""
    if p is not None:
        noise = dict(
            before_measure_flip_probability=p,
            after_reset_flip_probability=p,
            before_round_data_depolarization=p,
        )
    elif not noise:
        noise = DEFAULT_NOISE
    return stim.Circuit.generated(
        "repetition_code:memory", distance=distance, rounds=rounds, **noise
    )


def edge_table(dem, n_anc=N_ANC):
    """Map each DEM error mechanism to a canonical graph edge.

    Returns a list, one entry per mechanism: (frozenset_of_detectors, flips_observable).
    A boundary edge has a single detector. Used to reduce a fired-mechanism vector to the
    edge set it actually represents, so that two faults on the same edge cancel.
    """
    table = []
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        dets = frozenset(t.val for t in inst.targets_copy() if t.is_relative_detector_id())
        obs = any(t.is_logical_observable_id() for t in inst.targets_copy())
        table.append((dets, obs))
    return table


def sample(circuit, shots, seed=0):
    """Sample detectors, observables and the GROUND-TRUTH fired-mechanism matrix."""
    dem = circuit.detector_error_model(decompose_errors=True)
    sampler = dem.compile_sampler(seed=seed)
    dets, obs, errs = sampler.sample(shots=shots, return_errors=True)
    return dem, dets.astype(np.uint8), obs.astype(np.uint8), errs.astype(np.uint8)


def reduced_weights(errs, table):
    """Ground-truth physical error weight per shot, as the size of the XOR-reduced edge set.

    Two faults on the same graph edge cancel, so the honest 'total weight' of the configuration is
    the number of distinct edges surviving the XOR — not the raw count of fired mechanisms.
    """
    keys = [d for d, _ in table]
    out = np.empty(errs.shape[0], dtype=np.int32)
    raw = np.empty(errs.shape[0], dtype=np.int32)
    for s in range(errs.shape[0]):
        fired = np.flatnonzero(errs[s])
        raw[s] = len(fired)
        acc = {}
        for i in fired:
            k = keys[i]
            acc[k] = acc.get(k, 0) ^ 1
        out[s] = sum(acc.values())
    return out, raw


def to_layers(dets, n_anc=N_ANC):
    """(shots, n_detectors) -> (shots, n_layers, n_anc)."""
    shots, nd = dets.shape
    assert nd % n_anc == 0, f"{nd} detectors is not a multiple of {n_anc}"
    return dets.reshape(shots, nd // n_anc, n_anc)


if __name__ == "__main__":
    import pymatching

    from pilot_divergence import build_graph

    print("stim", stim.__version__)
    for p in (0.005, 0.01, 0.02):
        c = make_circuit(distance=9, rounds=50, p=p)
        dem, dets, obs, errs = sample(c, 2000, seed=1)
        D = to_layers(dets)
        table = edge_table(dem)
        w, raw = reduced_weights(errs, table)

        g = build_graph(D.shape[1], logical_j=N_ANC - 1)  # stim puts the observable on d_8
        pred = np.array([g.decode(D[s].ravel())[0] for s in range(D.shape[0])], dtype=np.uint8)
        ler = float((pred ^ obs[:, 0]).mean())

        # cross-check against pymatching built straight from stim's DEM
        m_dem = pymatching.Matching.from_detector_error_model(dem)
        pred_dem = m_dem.decode_batch(dets)[:, 0].astype(np.uint8)
        ler_dem = float((pred_dem ^ obs[:, 0]).mean())

        print(
            f"p={p:<6} shots=2000  det_rate={D.mean():.4f}  "
            f"weight: mean={w.mean():.1f} max={w.max()} (raw mean {raw.mean():.1f})  "
            f"LER uniform-graph={ler:.4f}  LER stim-DEM-weights={ler_dem:.4f}"
        )
