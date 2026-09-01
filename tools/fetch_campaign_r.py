#!/usr/bin/env python
"""Retrieve campaign R and cache it. Retrieval by job id is FREE -- no QPU charge.

Eight pubs of 25,000 shots alternating logical 0 / logical 1 (see `data/campaign_r_manifest.json`).
Cached as one .npz per pub so the time ordering is preserved: pub index IS collection order, and the
fit/evaluation split defined in `docs/R_PREREGISTRATION.md` step 3 is a split on that order.

Cache layout, one file per pub:
    syn   uint8 (25000, 50, 8)   ancilla measurement per round
    fin   uint8 (25000, 9)       final data-qubit readout
    meta  json                   pub index, prepared logical state, job id, layout

Usage:  python tools/fetch_campaign_r.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "data", "campaign_r_manifest.json")
OUTDIR = os.path.join(ROOT, "data", "campaign_r")


def main():
    with open(MANIFEST) as fh:
        man = json.load(fh)
    os.makedirs(OUTDIR, exist_ok=True)

    todo = [i for i in range(man["n_pubs"])
            if not os.path.exists(os.path.join(OUTDIR, f"pub{i}.npz"))]
    if not todo:
        print("every pub already cached; nothing to fetch")
        return 0

    from qiskit_ibm_runtime import QiskitRuntimeService

    svc = QiskitRuntimeService()
    job = svc.job(man["job_id"])
    print(f"job {man['job_id']}  status={job.status()}  "
          f"charge={job.metrics().get('usage', {}).get('qpu_charge_time_seconds')} s")
    res = job.result()

    for i in todo:
        data = res[i].data
        rounds = sorted((k for k in data.__dict__ if k.startswith("r") and k[1:].isdigit()),
                        key=lambda k: int(k[1:]))
        # BitArray.to_bool_array gives (shots, n_bits) MSB-first; reverse to register-index order.
        syn = np.stack([getattr(data, r).to_bool_array()[:, ::-1] for r in rounds],
                       axis=1).astype(np.uint8)
        fin = data.fin.to_bool_array()[:, ::-1].astype(np.uint8)
        y = man["pub_logical_states"][i]
        meta = dict(job_id=man["job_id"], pub=i, logical_state=y, shots=int(syn.shape[0]),
                    rounds=int(syn.shape[1]), n_anc=int(syn.shape[2]), n_data=int(fin.shape[1]),
                    data_qubits=man["data_qubits"], ancilla_qubits=man["ancilla_qubits"])
        print(f"  pub {i}  Y={y}  shots={meta['shots']}  rounds={meta['rounds']}  "
              f"mean fin={fin.mean():.4f}")
        path = os.path.join(OUTDIR, f"pub{i}.npz")
        tmp = path + ".tmp"
        # np.savez_compressed appends '.npz' unless handed an open file object.
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, syn=syn, fin=fin, meta=json.dumps(meta))
        os.replace(tmp, path)
    print(f"cached -> {OUTDIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
