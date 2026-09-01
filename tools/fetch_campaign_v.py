#!/usr/bin/env python
"""Fetch campaign V epoch E1 and cache each pub as an npz, tagged with its region and logical state.

Waits for the job if it is still running. Idempotent: pubs already cached are skipped.

Usage:  python tools/fetch_campaign_v.py
"""
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPOCH = os.environ.get("CAMPAIGN_V_EPOCH", "E1")
MANIFEST = os.path.join(ROOT, "data", f"campaign_v_{EPOCH.lower()}_manifest.json")
OUTDIR = os.path.join(ROOT, "data", f"campaign_v_{EPOCH.lower()}")


def main():
    with open(MANIFEST) as fh:
        man = json.load(fh)
    os.makedirs(OUTDIR, exist_ok=True)
    n_pubs = len(man["pub_order"])
    todo = [i for i in range(n_pubs)
            if not os.path.exists(os.path.join(OUTDIR, f"pub{i}.npz"))]
    if not todo:
        print("every pub already cached; nothing to fetch")
        return 0

    from qiskit_ibm_runtime import QiskitRuntimeService
    svc = QiskitRuntimeService()
    job = svc.job(man["job_id"])

    while True:
        st = str(job.status())
        if st in ("DONE", "COMPLETED"):
            break
        if st in ("ERROR", "CANCELLED", "FAILED"):
            print(f"job ended in {st}; nothing to fetch")
            try:
                print(job.error_message())
            except Exception:
                pass
            return 1
        print(f"  status={st} ...", flush=True)
        time.sleep(20)

    charge = job.metrics().get("usage", {}).get("qpu_charge_time_seconds")
    print(f"job {man['job_id']}  DONE  charged {charge} s  "
          f"(estimate was {man['estimate_s']:.1f} s)")
    res = job.result()

    for i in todo:
        data = res[i].data
        rounds = sorted((k for k in data.__dict__ if k.startswith("r") and k[1:].isdigit()),
                        key=lambda k: int(k[1:]))
        syn = np.stack([getattr(data, r).to_bool_array()[:, ::-1] for r in rounds],
                       axis=1).astype(np.uint8)
        fin = data.fin.to_bool_array()[:, ::-1].astype(np.uint8)
        tag = man["pub_order"][i]                       # e.g. "R1/L0"
        region, logical = tag.split("/")[0], int(tag.split("/L")[1])
        meta = dict(job_id=man["job_id"], pub=i, region=region, logical_state=logical,
                    qubits=man["regions"][region], shots=int(syn.shape[0]),
                    rounds=int(syn.shape[1]), n_anc=int(syn.shape[2]),
                    n_data=int(fin.shape[1]), epoch=man["epoch"],
                    charged_s=charge)
        print(f"  pub {i}  {tag}  shots={meta['shots']}  rounds={meta['rounds']}  "
              f"anc={meta['n_anc']}  mean syn={syn.mean():.4f}  mean fin={fin.mean():.4f}")
        path = os.path.join(OUTDIR, f"pub{i}.npz")
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, syn=syn, fin=fin, meta=json.dumps(meta))
        os.replace(tmp, path)

    man["charged_s"] = charge
    tmp = MANIFEST + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(man, fh, indent=1)
    os.replace(tmp, MANIFEST)
    print(f"cached -> {OUTDIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
