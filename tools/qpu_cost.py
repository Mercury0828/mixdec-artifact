#!/usr/bin/env python
"""QPU cost estimator — run this BEFORE any hardware submission (owner rule, 2026-08-29).

🔴 The single-rate model in the guide (shots x 5.97/20000 + shots x rounds x 5.47us) UNDER-predicts
the two frozen jobs by ~40% (3.64 s vs 6 s measured; 5.72 s vs 8 s measured). An under-predicting
budget model is the dangerous direction, so it is replaced here by a three-parameter fit:

    seconds = A + shots * B + shots * rounds * C

fitted to the three measured points on `ibm_cleveland` (2026-08-26):
    20,000-shot measurement circuit (~0 rounds)      = 5.97 s
    job da7mi6bsq5js73bk4veg, 12 rounds, 10k shots   = 6 s
    job da7miljsq5js73bk4vtg, 50 rounds, 10k shots   = 8 s

giving A = 4.77 s fixed per job, B = 6.0e-5 s/shot, C = 5.26e-6 s per shot-round. C agrees with the
independently quoted 5.47 us per repetition-code round, which is the one real consistency check
available.

🔴 **Only three calibration points, two of them at the same shot count, and the measurement circuit
is a different circuit type.** Treat A and B as rough. The large fixed A is the load-bearing finding:
many small campaigns are dominated by per-job overhead, so batching matters more than shot count.

Usage:  python tools/qpu_cost.py
"""
import argparse

A_FIXED = 4.766        # s, fixed per job
B_SHOT = 6.02e-5       # s per shot
C_SHOT_ROUND = 5.263e-6  # s per shot-round


def campaign_seconds(rounds, shots, n_jobs=1):
    return n_jobs * (A_FIXED + shots * B_SHOT + shots * rounds * C_SHOT_ROUND)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=7200.0)
    args = ap.parse_args()

    print("Calibration cross-check (the model is FITTED to these, so agreement is a sanity check")
    print("of the arithmetic, not independent validation)")
    for label, rounds, shots, measured in [
        ("measurement circuit", 0, 20_000, 5.97),
        ("da7mi6bsq5js73bk4veg", 12, 10_000, 6),
        ("da7miljsq5js73bk4vtg", 50, 10_000, 8),
    ]:
        print(f"  {label:<22} {rounds:>3} rounds x {shots:,} shots  "
              f"model {campaign_seconds(rounds, shots):5.2f} s  vs measured {measured} s")
    print(f"  independent check: guide quotes a 50-round / 20k-shot campaign at ~10.5 s; "
          f"model says {campaign_seconds(50, 20_000):.1f} s")

    print("\nPer-campaign cost")
    print(f"{'rounds':>7} {'shots':>9} {'seconds':>9} {'% of 28-day budget':>20}")
    for rounds in (40, 50, 100):
        for shots in (10_000, 20_000, 100_000, 200_000):
            s = campaign_seconds(rounds, shots)
            print(f"{rounds:>7} {shots:>9,} {s:>9.1f} {100 * s / args.budget:>19.2f}%")

    print("\nCandidate hardware designs")
    designs = [
        ("A. reuse the two cached jobs", 0, 0, 0,
         "retrievable by id; retrieval is FREE. Do this first, always."),
        ("B. deepen the existing point", 1, 50, 200_000,
         "same d=9 chain, 20x shots -> pushes the resolution floor from 5e-4 to ~2.5e-5"),
        ("C. round sweep", 3, 100, 100_000,
         "R in {40,50,100}: separates buffer effects from horizon effects"),
        ("D. distance sweep", 4, 50, 100_000,
         "d in {5,9,15,25}: tests whether b~L bites only at small d (risk R5)"),
        ("E. region x time replication", 15, 50, 100_000,
         "3 regions x 5 times: the stationarity/drift check A1 needs"),
    ]
    total = 0.0
    for name, n, rounds, shots, why in designs:
        s = campaign_seconds(rounds, shots, n) if n else 0.0
        overhead = n * A_FIXED
        total += s
        oh = f"[{overhead:.0f}s of that is per-job overhead]" if n > 1 else ""
        print(f"  {name:<30} {n:>2} jobs {s:>8.1f} s ({100 * s / args.budget:>5.2f}%) "
              f"{why} {oh}")
    print(f"  {'TOTAL B+C+D+E':<30} {'':>2}      {total:>8.1f} s "
          f"({100 * total / args.budget:.1f}% of one 28-day period)")

    print(f"\nBudget context: 7200 s / 28 days. As of 2026-08-29 the period had just reset.")
    print("Rule in force: nothing is submitted until the simulator pipeline is validated and this")
    print("estimate has been shown to the owner.")


if __name__ == "__main__":
    main()
