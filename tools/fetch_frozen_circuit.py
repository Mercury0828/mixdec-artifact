#!/usr/bin/env python
"""Retrieve the frozen ISA circuit the campaign submitted, and archive it. 0 QPU.

`submit_campaign_v.py` fetches the circuit from job `da7miljsq5js73bk4vtg` at submission time and
never writes it down, so the repository has every shot the circuit produced and not the circuit. That
is a hole in the reproduction manifest and it is also what a detector-support enumeration needs: the
premise behind Theorem~5 is a property of the compiled circuit, so the enumeration has to run on the
circuit that ran.

Retrieving a job's inputs costs no processor time. It does need network access and IBM Quantum
credentials, which is why this is a separate script the owner runs rather than a step inside an
analysis.

What it writes, into `data/frozen_circuit/`:

  r1_l0.qpy          the submitted ISA circuit, byte for byte, in QPY
  r2_l0.qpy          the same circuit relabelled onto the second line, as the campaign built it
  summary.json       gate histogram, depth, size, qubit lists, the two-qubit connectivity used,
                     and the sha256 of each QPY payload

Usage:  python tools/fetch_frozen_circuit.py
"""
import collections
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(ROOT, "data", "frozen_circuit")
SOURCE_JOB = "da7miljsq5js73bk4vtg"


def describe(circ, name, qubits):
    hist = collections.Counter(i.operation.name for i in circ.data)
    edges = set()
    for inst in circ.data:
        if inst.operation.name == "barrier" or len(inst.qubits) != 2:
            continue
        a, b = (circ.find_bit(q).index for q in inst.qubits)
        edges.add(tuple(sorted((a, b))))
    return {
        "name": name,
        "num_qubits": circ.num_qubits,
        "depth": circ.depth(),
        "size": circ.size(),
        "gate_histogram": dict(sorted(hist.items())),
        "two_qubit_edges": sorted(edges),
        "line": qubits,
    }


def main():
    from qiskit import qpy
    from qiskit_ibm_runtime import QiskitRuntimeService

    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from submit_campaign_v import R1, R2, relabel

    os.makedirs(OUTDIR, exist_ok=True)
    svc = QiskitRuntimeService()
    print(f"retrieving job {SOURCE_JOB} (inputs only, no processor time)...")
    c0 = svc.job(SOURCE_JOB).inputs["pubs"][0][0]
    c2 = relabel(c0, R1, R2)
    assert c2.size() == c0.size() and c2.depth() == c0.depth()

    summary = {"source_job": SOURCE_JOB, "circuits": []}
    for circ, name, line in ((c0, "r1_l0", R1), (c2, "r2_l0", R2)):
        path = os.path.join(OUTDIR, name + ".qpy")
        with open(path, "wb") as fh:
            qpy.dump(circ, fh)
        with open(path, "rb") as fh:
            sha = hashlib.sha256(fh.read()).hexdigest()
        row = describe(circ, name, line)
        row["qpy_sha256"] = sha
        summary["circuits"].append(row)
        print(f"  {name}: depth {row['depth']} size {row['size']} "
              f"gates {row['gate_histogram']}")
        print(f"    sha256 {sha}")

    with open(os.path.join(OUTDIR, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwrote {os.path.relpath(OUTDIR, ROOT)}")
    print("next:  python tools/circuit_support_audit.py --isa")


if __name__ == "__main__":
    sys.exit(main())
