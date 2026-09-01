#!/usr/bin/env python
"""Campaign R — the pre-registered replication of `P5`, with randomized logical 0/1.

🔴 SUBMISSION IS OPT-IN. This script does nothing without `--submit`. Default is a dry run that
builds and validates every circuit, prices the campaign, and writes nothing to the device.

Pre-registration: `docs/R_PREREGISTRATION.md`, frozen before the shots exist. Estimate and staging:
`docs/HARDWARE_PLAN.md`. Owner approval recorded 2026-08-29.

HOW THE CIRCUITS ARE BUILT, AND WHY THIS WAY

The logical-0 circuit is **the frozen job's own ISA circuit, byte for byte** — retrieved from
`da7miljsq5js73bk4vtg`, not rebuilt. Rebuilding it would re-run a transpiler whose output is not
guaranteed identical, and campaign R is a REPLICATION: the circuit must not be a variable.

The logical-1 circuit is that same circuit with an `x` prepended on each of the nine data qubits.
`x` is native on Heron (`basis_gates = [cz, id, rz, sx, x]`), so the result is still ISA-valid and
nothing else about the circuit changes.

The logical state is invisible in the syndrome record, which is the entire point:
  - ancilla `j` measures the parity of data `j` and `j+1`; for |1...1> that parity is 0, exactly as
    for |0...0>, so `syn` is unchanged;
  - the final stabilizer `sfin[j] = fin[j] XOR fin[j+1]` is 0 for the all-ones readout too, so the
    detectors `D` are identical.
A decoder therefore cannot tell which state was prepared -- while the *truth* `Y` differs. That is
what makes ledger `B13` testable: an always-output-0 decoder and a decoder that has learned to defer
to the terminal readout both score perfectly on the cached all-|0...0> job, and both are exposed here.

🔴 DEVIATION FROM THE FROZEN PRE-REGISTRATION, DECLARED BEFORE SUBMISSION.
The pre-registration says two jobs of 100,000. This submits ONE job of eight 25,000-shot pubs
alternating L0, L1, L0, L1, ... The shot budget is unchanged (100,000 per logical state) and the
analysis is untouched. The reason is a confound the two-job design carries: L0 and L1 would be
collected in two contiguous blocks, possibly hours apart, so any calibration drift between them is
indistinguishable from the deference effect that R-5 exists to detect. Alternating blocks cut that
timescale by eight. Recorded in the manifest and in the pre-registration as an amendment.

Usage:
    python tools/submit_campaign_r.py                 # dry run: build, validate, price. No device.
    python tools/submit_campaign_r.py --submit        # actually submit. Requires --submit.
"""
import argparse
import datetime
import json
import os
import sys

SOURCE_JOB = "da7miljsq5js73bk4vtg"
BACKEND = "ibm_cleveland"
DATA_QUBITS = [0, 2, 4, 6, 8, 10, 12, 14, 19]
ANCILLA_QUBITS = [1, 3, 5, 7, 9, 11, 13, 15]
SHOTS_PER_PUB = 25_000
N_PUBS = 8                      # alternating L0, L1, ... -> 100,000 shots per logical state
ROUNDS = 50
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "data", "campaign_r_manifest.json")


def qpu_seconds(shots, rounds, jobs=1):
    """Three-parameter cost model from `tools/qpu_cost.py`, fitted on three measured points."""
    return jobs * 4.766 + 6.02e-5 * shots + 5.263e-6 * shots * rounds


def build(service):
    """Return (circ_L0, circ_L1). L0 is the frozen circuit itself; L1 prepends x on the data qubits."""
    from qiskit import QuantumCircuit

    job = service.job(SOURCE_JOB)
    c0 = job.inputs["pubs"][0][0]

    c1 = QuantumCircuit(*c0.qregs, *c0.cregs)
    for q in DATA_QUBITS:
        c1.x(q)
    c1.barrier(DATA_QUBITS)
    c1.compose(c0, inplace=True)

    assert c1.num_qubits == c0.num_qubits
    assert [(r.name, r.size) for r in c1.cregs] == [(r.name, r.size) for r in c0.cregs], \
        "classical registers must match or tools/fetch_jobs.py cannot read the result"
    assert c1.size() == c0.size() + len(DATA_QUBITS), "L1 must add exactly the nine x gates"
    return c0, c1


def validate(backend, circuits):
    """Every instruction must be supported on its qubits, or the job is rejected after being charged."""
    bad = []
    for label, c in circuits:
        for inst in c.data:
            name = inst.operation.name
            if name == "barrier":
                continue
            qs = tuple(c.find_bit(q).index for q in inst.qubits)
            if not backend.target.instruction_supported(name, qs):
                bad.append((label, name, qs))
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true",
                    help="actually submit. Without it this is a dry run and touches no QPU.")
    args = ap.parse_args()

    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

    svc = QiskitRuntimeService()
    usage = svc.usage()
    remaining = usage["usage_remaining_seconds"]
    backend = svc.backend(BACKEND)

    total_shots = SHOTS_PER_PUB * N_PUBS
    est_job = qpu_seconds(total_shots, ROUNDS, jobs=1)
    est_pub = qpu_seconds(total_shots, ROUNDS, jobs=N_PUBS)   # if the fixed cost is per pub, not per job

    print("CAMPAIGN R -- pre-registered replication of P5 with randomized logical 0/1")
    print(f"  backend           {BACKEND}  operational={backend.status().operational} "
          f"pending={backend.status().pending_jobs}")
    print(f"  usage period      {usage['usage_period']['start_time']} .. "
          f"{usage['usage_period']['end_time']}")
    print(f"  remaining         {remaining} s of {usage['usage_allocation_seconds']} s")
    print(f"  plan              1 job, {N_PUBS} pubs x {SHOTS_PER_PUB} shots, alternating L0/L1")
    print(f"                    = {total_shots} shots, {total_shots // 2} per logical state, "
          f"{ROUNDS} rounds")
    print(f"  estimated cost    {est_job:.1f} s if the fixed term is per JOB "
          f"({est_job / remaining:.2%} of remaining)")
    print(f"                    {est_pub:.1f} s if it is per PUB "
          f"({est_pub / remaining:.2%} of remaining)")

    print("\nbuilding circuits from the frozen job (retrieval is free)...")
    c0, c1 = build(svc)
    print(f"  L0  qubits={c0.num_qubits} depth={c0.depth()} size={c0.size()}  (frozen, unmodified)")
    print(f"  L1  qubits={c1.num_qubits} depth={c1.depth()} size={c1.size()}  "
          f"(+{len(DATA_QUBITS)} x gates on {DATA_QUBITS})")

    bad = validate(backend, [("L0", c0), ("L1", c1)])
    if bad:
        print("\n** ISA VALIDATION FAILED -- not submitting:")
        for label, name, qs in bad[:20]:
            print(f"    {label}: {name} on {qs}")
        return 1
    print("  ISA validation: OK, every instruction supported on its qubits")

    order = [(("L0", 0) if i % 2 == 0 else ("L1", 1)) for i in range(N_PUBS)]
    circs = {0: c0, 1: c1}
    print(f"  pub order: {[o[0] for o in order]}")

    if not args.submit:
        print("\nDRY RUN -- nothing submitted, 0 QPU seconds spent. "
              "Re-run with --submit to collect.")
        return 0

    if est_pub > remaining:
        print(f"\n** REFUSING: worst-case estimate {est_pub:.1f} s exceeds remaining {remaining} s.")
        return 1

    print("\nSUBMITTING...")
    sampler = SamplerV2(mode=backend)
    job = sampler.run([(circs[y],) for _, y in order], shots=SHOTS_PER_PUB)
    print(f"  job id: {job.job_id()}")

    manifest = dict(
        campaign="R",
        purpose="pre-registered replication of P5 + randomized logical 0/1 (B13) + Route A (P9/N10)",
        preregistration="docs/R_PREREGISTRATION.md",
        submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        job_id=job.job_id(),
        backend=BACKEND,
        source_circuit_job=SOURCE_JOB,
        shots_per_pub=SHOTS_PER_PUB,
        n_pubs=N_PUBS,
        total_shots=total_shots,
        rounds=ROUNDS,
        pub_logical_states=[y for _, y in order],
        data_qubits=DATA_QUBITS,
        ancilla_qubits=ANCILLA_QUBITS,
        estimated_qpu_seconds_per_job_model=est_job,
        estimated_qpu_seconds_per_pub_model=est_pub,
        usage_remaining_before=remaining,
        deviation_from_prereg=(
            "Pre-registration specifies two jobs of 100,000. Submitted as one job of eight "
            "25,000-shot pubs alternating L0/L1. Shot budget and analysis unchanged; the reason is "
            "that two contiguous blocks confound calibration drift with the deference effect R-5 "
            "exists to detect. Declared before submission."),
    )
    tmp = MANIFEST + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(manifest, fh, indent=1)
    os.replace(tmp, MANIFEST)
    print(f"  manifest -> {MANIFEST}")
    print("** Results are retrievable by job id for free. The id is now on disk AND above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
