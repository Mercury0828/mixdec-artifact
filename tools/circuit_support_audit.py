#!/usr/bin/env python
"""Does a single circuit fault flip at most two detectors, and does a circuit-level model satisfy
the same-stabiliser budget? 0 QPU.

Theorem~5 excludes independent-event models whose events flip at most two detectors. Whether that
class is the right null for a compiled circuit is a property of the circuit, not of the data, and it
is settled by enumeration rather than by comparison against a surrogate. This script does both
halves:

  SUPPORT   every elementary error mechanism of the circuit's detector error model, undecomposed,
            with the size of its detector set. A mechanism touching three or more detectors is a
            single fault the graphlike class cannot represent.

  BUDGET    G_same measured on shots drawn from that same circuit, by the frozen estimator. If a
            circuit whose faults are all support-two satisfies the budget while one carrying
            support-three mechanisms violates it, then the budget is reading circuit structure and
            not device noise, and the endpoint has to say so.

Three noise families are compared at a matched detector rate:

  phenomenological   data depolarisation between rounds, reset flips, measurement flips
  circuit-level      the same, plus depolarising noise after every Clifford, two-qubit gates
                     included
  submitted ISA      the frozen instruction-set circuit the campaign actually ran, translated gate
                     for gate by tools/isa_to_stim.py, with depolarising noise on every native
                     rz, sx and cz. Needs data/frozen_circuit/, written by
                     tools/fetch_frozen_circuit.py.

Usage:  python tools/circuit_support_audit.py [--shots 200000]
"""
import argparse
import collections
import json
import os
import sys

import numpy as np
import stim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frozen_witness import E2_graphlike_budget  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "circuit_support_audit.json")
DISTANCE = 9
ROUNDS = 50
N_ANC = 8
N_LAYERS = ROUNDS + 1


def isa_circuit(p):
    """The submitted instruction-set circuit, at one noise scale on every native operation."""
    from isa_to_stim import load_isa, translate
    global _ISA
    try:
        _ISA
    except NameError:
        _ISA = load_isa()
    return translate(_ISA, p1=p, p2=p, pr=p, pm=p, pidle=p)[0]


def circuit(p, clifford):
    """The memory circuit. `clifford=0` is the phenomenological family."""
    return stim.Circuit.generated(
        "repetition_code:memory",
        distance=DISTANCE,
        rounds=ROUNDS,
        after_clifford_depolarization=clifford,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p,
        before_round_data_depolarization=p,
    )


def support_histogram(circ):
    """Detector-set size of every elementary mechanism, undecomposed."""
    dem = circ.detector_error_model(decompose_errors=False)
    hist = collections.Counter()
    worst = []
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        dets = [t.val for t in inst.targets_copy() if t.is_relative_detector_id()]
        k = len(set(dets))
        hist[k] += 1
        if k >= 3 and len(worst) < 6:
            worst.append(sorted(set(dets)))
    return hist, worst


def detectors(circ, shots, seed):
    smp = circ.compile_detector_sampler(seed=seed)
    D = smp.sample(shots).astype(np.uint8)
    return D.reshape(shots, N_LAYERS, N_ANC)


def rate_for(target, build, shots=20000, seed=1):
    """Solve the noise scale so the mean detector rate matches the device's."""
    lo, hi = 1e-5, 0.05
    for _ in range(26):
        mid = 0.5 * (lo + hi)
        D = detectors(build(mid), shots, seed)
        if D.mean() < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=200000)
    ap.add_argument("--target-rate", type=float, default=0.0398,
                    help="the device's mean detector rate, from CLAIMS.md")
    args = ap.parse_args()

    out = {"distance": DISTANCE, "rounds": ROUNDS, "shots": args.shots,
           "target_detector_rate": args.target_rate, "families": []}

    families = [
        ("phenomenological", lambda p: circuit(p, 0.0)),
        ("circuit-level", lambda p: circuit(p, p)),
    ]
    if os.path.exists(os.path.join(ROOT, "data", "frozen_circuit", "r1_l0.qpy")):
        families.append(("submitted ISA", isa_circuit))
    else:
        print("data/frozen_circuit/ absent; run tools/fetch_frozen_circuit.py for the ISA row")

    for name, build in families:
        p = rate_for(args.target_rate, build)
        circ = build(p)
        hist, worst = support_histogram(circ)
        D = detectors(circ, args.shots, seed=7)
        e2 = E2_graphlike_budget(D)
        row = {
            "family": name,
            "noise_scale": p,
            "mean_detector_rate": float(D.mean()),
            "mechanisms": int(sum(hist.values())),
            "support_histogram": {str(k): int(v) for k, v in sorted(hist.items())},
            "max_support": int(max(hist)),
            "n_support_ge_3": int(sum(v for k, v in hist.items() if k >= 3)),
            "example_support_ge_3": worst,
            "G_same": e2["G_same"],
            "n_detectors_violating": e2["n_detectors_violating"],
            "B": e2["B"],
            "T_same": e2["T_same"],
        }
        out["families"].append(row)
        print(f"\n{name}")
        print(f"  noise scale                 {p:.5f}")
        print(f"  mean detector rate          {D.mean():.5f}")
        print(f"  elementary mechanisms       {row['mechanisms']}")
        print(f"  detector-support histogram  {row['support_histogram']}")
        print(f"  mechanisms with support>=3  {row['n_support_ge_3']}")
        print(f"  G_same                      {e2['G_same']:+.2f}   "
              f"(<= 0 for every graphlike model)")
        print(f"  detectors over budget       {e2['n_detectors_violating']}/{e2['n_detectors']}")

    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
