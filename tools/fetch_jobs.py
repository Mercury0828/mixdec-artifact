#!/usr/bin/env python
"""Retrieve the frozen 2026-08-26 ibm_cleveland repetition-code jobs and cache them locally.

Retrieval by job id is FREE (no QPU charge). This script is idempotent: it skips any job whose
cache file already exists, so a re-run never re-fetches. Writes atomically (temp then rename) per
guide section 7.5.

Cache layout, one .npz per job:
    syn      uint8  (shots, rounds, n_anc)   ancilla measurement per round
    fin      uint8  (shots, n_data)          final data-qubit readout
    meta     json   layout, qubit chain, job id, backend, creation date

Usage:  python tools/fetch_jobs.py [--out data/]
"""
import argparse
import json
import os
import sys

import numpy as np

JOBS = {
    "da7miljsq5js73bk4vtg": "50-round",
    "da7mi6bsq5js73bk4veg": "12-round",
}


def extract_layout(circuit):
    """Recover the physical data/ancilla chain from the circuit's measure instructions.

    Data qubits are those measured into the 'fin' register, ancillas those measured into 'r0'.
    Register bit order is the chain order the campaign used, so we key on it rather than on the
    physical index, which is DFS-found and not monotonic.
    """
    creg = {r.name: r for r in circuit.cregs}
    data_q = [None] * creg["fin"].size
    anc_q = [None] * creg["r0"].size
    for inst in circuit.data:
        if inst.operation.name != "measure":
            continue
        qubit = circuit.find_bit(inst.qubits[0]).index
        bit = circuit.find_bit(inst.clbits[0])
        reg, idx = bit.registers[0]
        if reg.name == "fin":
            data_q[idx] = qubit
        elif reg.name == "r0":
            anc_q[idx] = qubit
    return data_q, anc_q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    todo = [j for j in JOBS if not os.path.exists(os.path.join(args.out, f"{j}.npz"))]
    if not todo:
        print("all jobs already cached; nothing to fetch")
        return 0

    from qiskit_ibm_runtime import QiskitRuntimeService

    svc = QiskitRuntimeService()
    for jid in todo:
        print(f"[{jid}] {JOBS[jid]}: retrieving (free — no QPU charge)")
        job = svc.job(jid)
        circuit = job.inputs["pubs"][0][0]
        data_q, anc_q = extract_layout(circuit)

        res = job.result()[0].data
        rounds = sorted(
            (k for k in res.__dict__ if k.startswith("r") and k[1:].isdigit()),
            key=lambda k: int(k[1:]),
        )
        # BitArray.array is (shots, n_bytes); to_bool_array gives (shots, n_bits) MSB-first,
        # so reverse the bit axis to get register-index order.
        syn = np.stack(
            [getattr(res, r).to_bool_array()[:, ::-1] for r in rounds], axis=1
        ).astype(np.uint8)
        fin = res.fin.to_bool_array()[:, ::-1].astype(np.uint8)

        meta = {
            "job_id": jid,
            "label": JOBS[jid],
            "backend": job.backend().name,
            "created": str(job.creation_date),
            "shots": int(syn.shape[0]),
            "rounds": int(syn.shape[1]),
            "n_anc": int(syn.shape[2]),
            "n_data": int(fin.shape[1]),
            "data_qubits": data_q,
            "ancilla_qubits": anc_q,
            "qpu_charge_seconds": job.metrics().get("usage", {}).get(
                "qpu_charge_time_seconds"
            ),
        }
        print(
            f"  shots={meta['shots']} rounds={meta['rounds']} "
            f"n_anc={meta['n_anc']} n_data={meta['n_data']}"
        )
        print(f"  data qubits:    {data_q}")
        print(f"  ancilla qubits: {anc_q}")

        path = os.path.join(args.out, f"{jid}.npz")
        tmp = path + ".tmp"
        # np.savez_compressed appends '.npz' unless handed an open file object.
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, syn=syn, fin=fin, meta=json.dumps(meta))
        os.replace(tmp, path)
        print(f"  cached -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
