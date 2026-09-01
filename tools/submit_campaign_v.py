#!/usr/bin/env python
"""Campaign V epoch E1 — two disjoint regions on one device. 143 QPU-s.

🔴 SUBMISSION IS OPT-IN. Without `--submit` this builds every circuit, validates it against the
backend's ISA, prices the job and writes nothing to the device.

Pre-registered: `docs/expected.md`, "Campaign V", amended `b76c17c` before any shot was requested.
Frozen witness: `tools/frozen_witness.py`, sha256 verified below.
Simulator dry run of the whole analysis: `tools/campaign_dryrun.py`, admissible on 7/7 checks.

WHAT THIS TESTS, AND WHAT IT CANNOT

Two endpoints on the same shots -- `E1` operational (`Pr[Delta_b]` against a DEM fitted to that
context's own calibration split) and `E2` structural (the same-ancilla attenuation budget). Neither
implies the other; round 17 showed a shot-rate mixture violating `E2` while producing exactly zero
`E1`.

🔴 The account exposes ONE backend, so the second-processor axis -- the most valuable one -- cannot
be bought. This campaign closes the "one qubit path" and "one calibration" objections and leaves
"one processor" open. That is scope, and it goes in the paper's claim.

HOW THE SECOND REGION'S CIRCUIT IS BUILT

`R1` is campaign R's own line and its circuit is the frozen job's ISA circuit, byte for byte. `R2`
is a disjoint 17-qubit line with the identical topology, and its circuit is that same circuit
**relabelled** position by position onto `R2` -- not re-transpiled. Re-transpiling would make the
compiler an uncontrolled second variable. Region is then the only difference between the two
contexts.

🔴 `R2` is calibrated worse (median T1 231 vs 282 us, T2 107 vs 135 us, readout 0.0068 vs 0.0054),
declared in the pre-registration before submission. A magnitude difference between regions may not
be read as a region effect.

Usage:
    python tools/submit_campaign_v.py            # dry run: build, validate, price. No QPU.
    python tools/submit_campaign_v.py --submit   # actually submit. Requires --submit.
"""
import argparse
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frozen_witness import witness_hash  # noqa: E402

SOURCE_JOB = "da7miljsq5js73bk4vtg"
BACKEND = "ibm_cleveland"
WITNESS_SHA = "d1aed05de62d65b46e2ad18011ef5c7267be99e6d19475c650925f2cd2476ce2"

# both are connected 17-qubit lines, verified disjoint, all 16 internal edges present
R1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 19]
R2 = [16, 23, 22, 21, 36, 41, 42, 43, 44, 45, 37, 25, 26, 27, 28, 29, 30]
DATA_IDX = [0, 2, 4, 6, 8, 10, 12, 14, 16]      # positions along the line carrying data qubits
SHOTS_PER_PUB = 25_000
PUBS_PER_REGION = 4                              # alternating L0, L1, L0, L1 -> 100,000 shots
ROUNDS = 50
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPOCH = os.environ.get("CAMPAIGN_V_EPOCH", "E1")
MANIFEST = os.path.join(ROOT, "data", f"campaign_v_{EPOCH.lower()}_manifest.json")


def qpu_seconds(shots, rounds, pubs):
    """The project's measured model, priced with the fixed term PER PUB (campaign R showed it is)."""
    return pubs * 4.766 + 6.02e-5 * shots + 5.263e-6 * shots * rounds


def relabel(circ, src_line, dst_line):
    """Move a circuit from one physical line to another of identical topology, gate for gate."""
    from qiskit import QuantumCircuit
    m = dict(zip(src_line, dst_line))
    out = QuantumCircuit(*circ.qregs, *circ.cregs)
    for inst in circ.data:
        qs = [circ.find_bit(q).index for q in inst.qubits]
        if any(q not in m for q in qs):
            raise RuntimeError(f"{inst.operation.name} touches qubit(s) {qs} outside the source "
                               f"line; the frozen circuit is not confined to R1")
        out.append(inst.operation, [out.qubits[m[q]] for q in qs],
                   [out.clbits[circ.find_bit(c).index] for c in inst.clbits])
    return out


def build(service, backend):
    """Four circuits: (region, logical) for region in R1, R2 and logical in 0, 1."""
    from qiskit import QuantumCircuit
    c0 = service.job(SOURCE_JOB).inputs["pubs"][0][0]
    data_r1 = [R1[i] for i in DATA_IDX]

    def with_x(c, data):
        out = QuantumCircuit(*c.qregs, *c.cregs)
        for q in data:
            out.x(q)
        out.barrier(data)
        out.compose(c, inplace=True)
        assert out.size() == c.size() + len(data)
        return out

    r1_l0 = c0
    r1_l1 = with_x(c0, data_r1)
    r2_l0 = relabel(c0, R1, R2)
    r2_l1 = with_x(r2_l0, [R2[i] for i in DATA_IDX])
    # the relabelling must preserve everything but the physical qubits
    assert r2_l0.size() == c0.size() and r2_l0.depth() == c0.depth()
    assert [(r.name, r.size) for r in r2_l0.cregs] == [(r.name, r.size) for r in c0.cregs]
    return [("R1", 0, r1_l0), ("R1", 1, r1_l1), ("R2", 0, r2_l0), ("R2", 1, r2_l1)]


def validate(backend, circuits):
    bad = []
    for reg, lg, c in circuits:
        for inst in c.data:
            if inst.operation.name == "barrier":
                continue
            qs = tuple(c.find_bit(q).index for q in inst.qubits)
            if not backend.target.instruction_supported(inst.operation.name, qs):
                bad.append((f"{reg}/L{lg}", inst.operation.name, qs))
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true",
                    help="actually submit. Without it nothing touches the QPU.")
    args = ap.parse_args()

    if witness_hash() != WITNESS_SHA:
        print(f"** FROZEN WITNESS HASH MISMATCH\n   expected {WITNESS_SHA}\n   got      "
              f"{witness_hash()}\n   The campaign is void (kill condition 5). Not submitting.")
        return 1
    print(f"frozen witness sha256 verified: {WITNESS_SHA[:16]}...")

    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    svc = QiskitRuntimeService()
    backend = svc.backend(BACKEND)
    usage = svc.usage()
    remaining = usage["usage_remaining_seconds"]

    shots = SHOTS_PER_PUB * PUBS_PER_REGION * 2
    pubs = PUBS_PER_REGION * 2
    est = qpu_seconds(shots, ROUNDS, pubs)

    print(f"\nCAMPAIGN V, EPOCH E1 -- two disjoint regions on {BACKEND}")
    print(f"  backend      operational={backend.status().operational} "
          f"pending={backend.status().pending_jobs}")
    print(f"  allowance    {remaining} s remaining of {usage['usage_allocation_seconds']} s "
          f"(rolling 28-day window)")
    print(f"  plan         2 regions x {PUBS_PER_REGION} pubs x {SHOTS_PER_PUB:,} shots, "
          f"alternating L0/L1, {ROUNDS} rounds")
    print(f"               = {shots:,} shots total, {shots // 2:,} per region")
    print(f"  estimate     {est:.1f} s base, {est * 1.16:.1f} s corrected, "
          f"{est * 1.16 * 1.2:.1f} s with headroom  ({est * 1.16 * 1.2 / remaining:.1%} of "
          f"remaining)")
    print(f"  R1 = {R1}")
    print(f"  R2 = {R2}")

    print("\nbuilding (retrieving the frozen circuit is free)...")
    circs = build(svc, backend)
    for reg, lg, c in circs:
        print(f"  {reg}/L{lg}  qubits={c.num_qubits} depth={c.depth()} size={c.size()}")

    bad = validate(backend, circs)
    if bad:
        print("\n** ISA VALIDATION FAILED -- not submitting:")
        for lab, name, qs in bad[:20]:
            print(f"    {lab}: {name} on {qs}")
        return 1
    print("  ISA validation: OK, every instruction supported on its qubits")

    # 🔴 REGIONS ARE INTERLEAVED, not blocked. Collecting R1's four pubs and then R2's four would
    # put the two regions in contiguous blocks minutes apart, so any calibration drift between them
    # is indistinguishable from a region effect -- the exact confound campaign R's own amendment was
    # written to remove for the logical state. Both region and logical state alternate at pub
    # granularity here.
    order = []
    for i in range(PUBS_PER_REGION):
        for reg in ("R1", "R2"):
            order.append((reg, i % 2))
    lookup = {(r, l): c for r, l, c in circs}
    print(f"  pub order: {[f'{r}/L{l}' for r, l in order]}")

    if not args.submit:
        print("\nDRY RUN. Nothing was submitted. Re-run with --submit to spend "
              f"{est * 1.16 * 1.2:.0f} QPU-s.")
        return 0

    print("\nSUBMITTING ...", flush=True)
    sampler = SamplerV2(mode=backend)
    job = sampler.run([(lookup[(r, l)],) for r, l in order], shots=SHOTS_PER_PUB)
    import hashlib
    pr = backend.properties()
    fp = hashlib.sha256(repr([(q, round(pr.t1(q), 9), round(pr.t2(q), 9),
                              round(pr.readout_error(q), 9))
                              for q in sorted(set(R1) | set(R2))]).encode()).hexdigest()
    rec = dict(job_id=job.job_id(), backend=BACKEND, submitted_utc=datetime.datetime.utcnow()
               .isoformat() + "Z", epoch=EPOCH,
               calibration_last_update=str(pr.last_update_date),
               calibration_fingerprint=fp, regions=dict(R1=R1, R2=R2),
               pub_order=[f"{r}/L{l}" for r, l in order], shots_per_pub=SHOTS_PER_PUB,
               rounds=ROUNDS, source_job=SOURCE_JOB, witness_sha256=WITNESS_SHA,
               estimate_s=est * 1.16 * 1.2, remaining_before_s=remaining)
    tmp = MANIFEST + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(rec, fh, indent=1)
    os.replace(tmp, MANIFEST)
    print(f"  job id {job.job_id()}")
    print(f"  manifest written to {MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
