#!/usr/bin/env python
"""Head-to-head baselines and the quantified-benefit table (guide section 7.3), at 0 QPU.

Every method answers the SAME operational question: **choose a buffer width `b` for windowed
decoding, at a stated risk budget epsilon.** They differ only in what evidence they use.

  IND    Independence-certified. Choose the smallest b whose divergence bound is <= epsilon under an
         i.i.d. circuit-level noise model matched to the device's detector event rate. This is the
         assumption the whole field's buffer sizing rests on, in its most concrete form.
  SERIAL Fully serial (joint) decoding. Zero divergence by definition, zero concurrency. The
         conservative bound.
  NAIVE  Folklore parallel: b = d. No certificate.
  EMPLER Empirical selection: choose the smallest b whose *point estimate* of the divergence rate on
         training data is <= epsilon. No coverage guarantee.
         🔴 THIS ROW WAS MISLABELLED "(The ADaPT / Complementary-Gap style)" until 2026-08-29. It is
         not. ADaPT (arXiv:2605.01149) and STCG (arXiv:2605.14637) both escalate PER SHOT at runtime;
         EMPLER picks one fixed buffer offline from a point estimate and never escalates. Calling it
         their style made a strawman of the actual prior art, which is the single thing TQE lens 2
         forbids. The real methods are implemented and run in `tools/prior_art_triggers.py`
         (`data/prior_art_triggers.json`); EMPLER stays only as what it
         actually is -- the no-guarantee fixed-buffer heuristic.
  MIXDEC Ours: choose the smallest b whose one-sided Clopper-Pearson UPPER BOUND on training data,
         simultaneous over the grid, is <= epsilon. **Abstains** if no b qualifies.
  MAJ    Majority vote over the final data readout. 🔴 NOT a fault-tolerant decoder -- it needs a
         transversal readout of every data qubit, cannot run mid-computation, and uses no syndrome.
         Reported as a FLOOR on physical readout quality, never as a competitor. See ledger N8.

Every method is fitted on shots 0..4999 and evaluated on the held-out 5000..9999.

Usage:  python tools/baselines.py
"""
import json
import os
import sys
import time

import numpy as np
from scipy.stats import beta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detectors import build_detectors, load  # noqa: E402
from parallel_window import two_window  # noqa: E402
from weight_model_robustness import fit_weights_v2, sim_data  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B_GRID = [1, 2, 3, 4, 6, 8, 12, 16]
DELTA = 0.05
W1 = 25
N_LAYERS = 51
D_CODE = 9
EPSILONS = [1e-2, 3e-3, 1e-3]


def cp_upper(k, n, alpha):
    return 1.0 if k >= n else float(beta.ppf(1.0 - alpha, k + 1, n - k))


def divergence_curve(D, logical_j, wtab, ttab):
    """Divergence count (seam-repaired) and decode wall time at each buffer width."""
    out = {}
    for b in B_GRID:
        t0 = time.time()
        r = two_window(D, W1, b, logical_j=logical_j, eps=1e-3, seed=0, wtab=wtab, ttab=ttab)
        out[b] = (int(r["diverged_repaired"].sum()), (time.time() - t0) / D.shape[0])
    return out


def concurrency(b):
    """Wall-clock speedup of two parallel windows over serial joint decoding.

    Serial decodes all N_LAYERS; each window decodes its own commit span plus the buffer, and the
    two run concurrently, so the critical path is the longer window. Offline scheduling only —
    guide section 7.3 forbids packaging this as online microsecond throughput.
    """
    span = max(min(W1 + b, N_LAYERS), N_LAYERS - max(W1 - b, 0))
    return N_LAYERS / span


def main():
    syn, fin, _ = load(os.path.join(ROOT, "data", "da7miljsq5js73bk4vtg.npz"))
    Ddev = build_detectors(syn, fin)
    Dsim, _ = sim_data(Ddev.shape[0])
    fit, ev = slice(0, 5000), slice(5000, 10000)
    n_fit = n_ev = 5000
    alpha = DELTA / len(B_GRID)

    wd, td, _, _ = fit_weights_v2(Ddev[fit])
    ws, ts, _, _ = fit_weights_v2(Dsim[fit])

    dev_fit = divergence_curve(Ddev[fit], 0, wd, td)
    dev_ev = divergence_curve(Ddev[ev], 0, wd, td)
    sim_fit = divergence_curve(Dsim[fit], 7, ws, ts)

    f = fin[ev].astype(np.uint8)
    maj = float((f.sum(1) > 4.5).mean())

    print("Divergence counts per 5,000 shots (seam-repaired, held-out calibrated weights)")
    print(f"{'b':>3} {'DEVICE fit':>11} {'DEVICE eval':>12} {'SIM fit':>9} {'concurrency':>12}")
    for b in B_GRID:
        print(f"{b:>3} {dev_fit[b][0]:>11} {dev_ev[b][0]:>12} {sim_fit[b][0]:>9} "
              f"{concurrency(b):>11.2f}x")

    rows = []
    for eps in EPSILONS:
        print(f"\n{'='*104}\nRISK BUDGET epsilon = {eps:g}   (target: Pr[joint != windowed] <= eps)")
        print(f"{'method':<8} {'evidence used':<34} {'b':>4} {'certified':>11} "
              f"{'HELD-OUT actual':>16} {'held-out CP UB':>15} {'concurrency':>12} {'holds?':>7}")

        def emit(tag, why, b, cert):
            if b is None:
                print(f"{tag:<8} {why:<34} {'ABSTAIN':>4} {'-':>11} {'-':>16} {'-':>15} "
                      f"{'1.00x':>12} {'n/a':>7}")
                rows.append(dict(eps=eps, method=tag, b=None, abstained=True))
                return
            k = dev_ev[b][0]
            ub = cp_upper(k, n_ev, alpha)
            ok = "YES" if (cert is None or k / n_ev <= eps) else "NO"
            certs = "-" if cert is None else f"{cert:.2e}"
            print(f"{tag:<8} {why:<34} {b:>4} {certs:>11} {k/n_ev:>16.5f} {ub:>15.2e} "
                  f"{concurrency(b):>11.2f}x {ok:>7}")
            rows.append(dict(eps=eps, method=tag, b=b, certified=cert, held_out_rate=k / n_ev,
                             held_out_ub=ub, concurrency=concurrency(b), holds=ok))

        # IND: smallest b certified safe under the i.i.d. model (simulator), applied to the device
        b_ind = next((b for b in B_GRID
                      if cp_upper(sim_fit[b][0], n_fit, alpha) <= eps), None)
        emit("IND", "i.i.d. model, matched det. rate", b_ind,
             cp_upper(sim_fit[b_ind][0], n_fit, alpha) if b_ind else None)

        emit("SERIAL", "none: joint decoding, no windows", None, None)

        emit("NAIVE", f"folklore b = d = {D_CODE}", D_CODE if D_CODE in B_GRID else 8, None)

        b_emp = next((b for b in B_GRID if dev_fit[b][0] / n_fit <= eps), None)
        emit("EMPLER", "device point estimate (no bound)", b_emp, None)

        b_mix = next((b for b in B_GRID
                      if cp_upper(dev_fit[b][0], n_fit, alpha) <= eps), None)
        emit("MIXDEC", "device CP upper bound, simultaneous", b_mix,
             cp_upper(dev_fit[b_mix][0], n_fit, alpha) if b_mix else None)

    print(f"\n{'='*104}")
    print(f"MAJ (floor, NOT fault-tolerant): held-out logical error {maj:.5f}. Uses no syndrome and "
          f"needs transversal readout;\n    reported only to bound physical readout quality. Ledger N8.")
    print(f"\nDecode cost per shot, held-out device: "
          + "  ".join(f"b={b}: {dev_ev[b][1]*1e3:.2f} ms" for b in (1, 4, 16)))
    print("SERIAL (joint) decoding is the b -> infinity limit: concurrency 1.00x by construction.")

    out = os.path.join(ROOT, "data", "baselines.json")
    tmp = out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(dict(delta=DELTA, alpha_per_test=alpha, b_grid=B_GRID, n_fit=n_fit, n_ev=n_ev,
                       majority_vote_floor=maj,
                       dev_fit={str(b): dev_fit[b][0] for b in B_GRID},
                       dev_eval={str(b): dev_ev[b][0] for b in B_GRID},
                       sim_fit={str(b): sim_fit[b][0] for b in B_GRID},
                       rows=rows), fh, indent=1)
    os.replace(tmp, out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
