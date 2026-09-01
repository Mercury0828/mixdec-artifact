#!/usr/bin/env python
"""Translate the frozen ISA circuit gate for gate into a stim circuit. 0 QPU.

The premise behind Theorem 5 is a property of the circuit that ran, not of a textbook circuit with
the same name, so the enumeration has to run on the submitted instruction-set circuit. That circuit
is retrieved by `tools/fetch_frozen_circuit.py` and is entirely Clifford: every `rz` in it is exactly
$\\pi/2$, so each instruction has an exact stim counterpart.

  rz(pi/2) -> S          sx -> SQRT_X          cz -> CZ
  reset    -> R          measure -> M          barrier -> dropped

Detectors are built to match `detectors.build_detectors` exactly, so the detector indices here are
the same objects the paper's statistics are computed on:

  D[0][a]  = m[0][a]
  D[r][a]  = m[r][a] xor m[r-1][a]                      for 1 <= r < R
  D[R][a]  = (fin[a] xor fin[a+1]) xor m[R-1][a]

and the observable is the first data qubit's final readout, which is `detectors.logical_flip`.

Circuit-level noise is attached to the native operations: depolarising noise after every one-qubit
Clifford and after every `cz`, a flip after every reset and before every measurement, and idle
depolarising noise on the data qubits once per round.

Usage:  python tools/isa_to_stim.py            # self-check against the frozen circuit
"""
import os
import sys

import numpy as np
import stim

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QPY = os.path.join(ROOT, "data", "frozen_circuit", "r1_l0.qpy")
R1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 19]
DATA_POS = [0, 2, 4, 6, 8, 10, 12, 14, 16]
ANC_POS = [1, 3, 5, 7, 9, 11, 13, 15]
ONE_Q = {"rz": "S", "sx": "SQRT_X"}


def load_isa(path=QPY):
    from qiskit import qpy
    with open(path, "rb") as fh:
        return qpy.load(fh)[0]


def translate(circ, p1=0.0, p2=0.0, pr=0.0, pm=0.0, pidle=0.0):
    """Gate for gate, into a stim circuit with the detectors the paper uses.

    `p1` one-qubit gate depolarisation, `p2` two-qubit, `pr` reset flip, `pm` measurement flip,
    `pidle` per-round idle depolarisation on the data qubits. All zero gives the noiseless circuit,
    which is what the determinism self-check needs.
    """
    pos = {phys: i for i, phys in enumerate(R1)}          # physical qubit -> position on the line
    data_q = [pos_i for pos_i in DATA_POS]
    anc_q = [pos_i for pos_i in ANC_POS]
    anc_index = {a: k for k, a in enumerate(anc_q)}        # position -> ancilla ordinal 0..7

    lines = []
    nmeas = 0
    mrec = {}                                              # (round, ancilla ordinal) -> record index
    frec = {}                                              # data ordinal -> record index
    rnd = 0
    seen_this_round = 0
    final = False

    for inst in circ.data:
        name = inst.operation.name
        qs = [pos[circ.find_bit(q).index] for q in inst.qubits]

        if name == "barrier":
            if pidle and not final:
                lines.append("DEPOLARIZE1(%g) %s" % (pidle, " ".join(str(q) for q in data_q)))
            continue

        if name in ONE_Q:
            lines.append("%s %d" % (ONE_Q[name], qs[0]))
            if p1:
                lines.append("DEPOLARIZE1(%g) %d" % (p1, qs[0]))
        elif name == "cz":
            lines.append("CZ %d %d" % (qs[0], qs[1]))
            if p2:
                lines.append("DEPOLARIZE2(%g) %d %d" % (p2, qs[0], qs[1]))
        elif name == "reset":
            lines.append("R %d" % qs[0])
            if pr:
                lines.append("X_ERROR(%g) %d" % (pr, qs[0]))
        elif name == "measure":
            if pm:
                lines.append("X_ERROR(%g) %d" % (pm, qs[0]))
            lines.append("M %d" % qs[0])
            if qs[0] in anc_index:
                mrec[(rnd, anc_index[qs[0]])] = nmeas
                seen_this_round += 1
                if seen_this_round == len(anc_q):
                    rnd += 1
                    seen_this_round = 0
            else:
                final = True
                frec[data_q.index(qs[0])] = nmeas
            nmeas += 1
        else:
            raise ValueError("unhandled instruction %r" % name)

    R = rnd
    assert len(mrec) == R * len(anc_q), (len(mrec), R)
    assert len(frec) == len(data_q), len(frec)

    def rec(i):
        return "rec[%d]" % (i - nmeas)

    det = []
    for a in range(len(anc_q)):
        det.append("DETECTOR %s" % rec(mrec[(0, a)]))
    for r in range(1, R):
        for a in range(len(anc_q)):
            det.append("DETECTOR %s %s" % (rec(mrec[(r, a)]), rec(mrec[(r - 1, a)])))
    for a in range(len(anc_q)):
        det.append("DETECTOR %s %s %s"
                   % (rec(frec[a]), rec(frec[a + 1]), rec(mrec[(R - 1, a)])))
    det.append("OBSERVABLE_INCLUDE(0) %s" % rec(frec[0]))

    return stim.Circuit("\n".join(lines + det)), R, len(anc_q)


def main():
    circ = load_isa()
    clean, R, n_anc = translate(circ)
    print("translated the frozen ISA circuit")
    print("  rounds %d, ancillas %d, detectors %d, observables %d"
          % (R, n_anc, clean.num_detectors, clean.num_observables))
    print("  stim instructions %d" % len(clean))

    # determinism: with no noise every detector must be zero on every shot
    D = clean.compile_detector_sampler(seed=1).sample(64)
    print("  noiseless detector firings over 64 shots: %d  (must be 0)" % int(D.sum()))
    assert D.sum() == 0

    noisy, _, _ = translate(circ, p1=1e-3, p2=1e-3, pr=1e-3, pm=1e-3, pidle=1e-3)
    dem = noisy.detector_error_model(decompose_errors=False)
    print("  noisy DEM mechanisms: %d" % sum(1 for i in dem.flattened() if i.type == "error"))
    print("self-check passed")


if __name__ == "__main__":
    sys.exit(main())
