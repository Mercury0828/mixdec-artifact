#!/usr/bin/env python
"""One-command regeneration of every frozen number in `data/`, from the cached shots.

TQE lens 8: a reproducible artifact is an explicit acceptance signal, and this project had none.
This driver runs the analysis stages in dependency order, each writing its own JSON atomically.

🔴 IT NEVER TOUCHES THE QPU. The two fetch stages retrieve already-executed jobs BY ID, which is
free; every other stage reads the local `.npz` caches. There is no submit path anywhere in this file
or in anything it calls. Submission lives only in `tools/submit_campaign_r.py`, which additionally
refuses to act without `--submit`.

Stages, in order:

  fetch          the two 2026-08-26 jobs and the campaign-R job, by id (free, skipped if cached)
  detectors      reproduces the 13 frozen M1 correlation numbers as a self-check
  units          both scales of the same-ancilla correlation, and their identity  (`B14` item 3)
  surrogate      the fitted independent graph-edge surrogate and its divergence  (`P5`)
  familywise     device vs surrogate as ONE 5% familywise procedure               (`B14` item 4)
  heterogeneity  the shot-level-overdispersion control                            (self-check round 1)
  baselines      the head-to-head buffer-selection table
  route_a        the adaptive rule and the full placebo battery                   (`P9`, `N10`)
  prior_art      ADaPT and the complementary gap, implemented and run             (self-check round 2)
  campaign_r     the pre-registered campaign-R analysis, R-1 .. R-8
  resources      units, tie-break stability, seam sweep, measured latency         (self-check round 3)
  fetch_v        both campaign-V epochs, by id (free, skipped if cached)
  campaign_v     BOTH ENDPOINTS in all four contexts, under the frozen witness  (`E1`, `E2`)
  graphlike ..   the parity moments, the elimination ladder, the cross-prediction, the churn

Usage:
    python tools/reproduce_all.py                 # run every stage
    python tools/reproduce_all.py --list          # show stages and their outputs
    python tools/reproduce_all.py --only route_a prior_art
    python tools/reproduce_all.py --skip fetch    # offline: use the caches as they are
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

STAGES = [
    # 🔴 DEPENDENCY ORDER MATTERS and the previous list did not have it: `multiseam` ran before the
    # calibration it consumes, `parallel` cross-checked `resource_frontier` before it existed,
    # `block_inference` appeared twice, and `make_claims` -- which writes the authoritative
    # CLAIMS.md -- was absent entirely, so "one command regenerates every claim" was false.
    # Gate-5 finding 4.
    ("fetch", ["tools/fetch_jobs.py", "tools/fetch_campaign_r.py"],
     ["data/da7miljsq5js73bk4vtg.npz", "data/campaign_r/pub0.npz"],
     "retrieve the frozen jobs by id (free, no QPU charge)"),
    ("detectors", ["tools/detectors.py"], [],
     "reproduce the 13 frozen M1 correlation numbers"),
    ("units", ["tools/correlation_units.py"], ["data/correlation_units.json"],
     "both correlation scales and the identity between them"),
    ("surrogate", ["tools/independence_model.py"], ["data/independence_model.json"],
     "the fitted independent graph-edge surrogate"),
    ("familywise", ["tools/joint_surrogate_test.py"], ["data/joint_surrogate_test.json"],
     "device vs surrogate as one 5% familywise procedure"),
    ("heterogeneity", ["tools/heterogeneity_control.py"], ["data/heterogeneity_control.json"],
     "the shot-level overdispersion control; P5's surrogate counts"),
    ("blockinf", ["tools/block_inference.py"], ["data/block_inference.json"],
     "P5 without the i.i.d. assumption: circular within-pub bootstrap (needs heterogeneity)"),
    ("baselines", ["tools/baselines.py"], ["data/baselines.json"],
     "head-to-head buffer-selection table"),
    ("route_a", ["tools/route_a.py"], ["data/route_a.json"],
     "the adaptive rule and the placebo battery"),
    ("prior_art", ["tools/prior_art_triggers.py"], ["data/prior_art_triggers.json"],
     "ADaPT and the complementary gap; P9's deployable-rule rows"),
    ("campaign_r", ["tools/analyze_campaign_r.py"], ["data/campaign_r_results.json"],
     "the pre-registered campaign-R analysis, R-1 .. R-8"),
    ("audit", ["tools/audit_reanalysis.py"], ["data/audit_reanalysis.json"],
     "R-8 under one pooled Y-blind table, paired; N10 as McNemar"),
    ("tiebreak", ["tools/tiebreak_sensitivity.py"], ["data/tiebreak_sensitivity.json"],
     "every matched-escalation number under the online tie-break, 20 seeds, ADaPT included"),
    ("tiebreak_diag", ["tools/tiebreak_diagnostic.py"], ["data/tiebreak_diagnostic.json"],
     "the rate-matched re-check on F1's cell"),
    ("frontier", ["tools/resource_frontier.py"], ["data/resource_frontier.json"],
     "AUTHORITATIVE cost/certificate comparison: every decode charged, JOINT included"),
    ("eventblock", ["tools/event_block_inference.py"], ["data/event_block_inference.json"],
     "dependence-robust inference for P9's and P12's OWN event sequences (needs frontier's tau)"),
    ("resources", ["tools/resource_and_stability.py"], ["data/resource_and_stability.json"],
     "tie-break stability, seam sweep (its latency section is WITHDRAWN)"),
    ("noise", [["tools/multi_seam.py", "--calibrate"]], ["data/multi_seam_noise.json"],
     "the seam-rate-matched noise level the many-seam sweep runs at"),
    ("multiseam", ["tools/multi_seam.py"], ["data/multi_seam.json"],
     "P9 with S seams and an admissible local repair (needs `noise`)"),
    ("parallel", ["tools/parallel_runtime.py"], ["data/parallel_runtime.json"],
     "the REAL parallel decoder; cross-checks `frontier`, so must follow it. TIMED -- idle machine"),
    ("scaling", ["tools/scaling_sweep.py"], ["data/scaling_sweep.json"],
     "simulator scaling in rounds and distance; TIMED -- run with the machine idle"),
    # --- campaign V and the model ladder --------------------------------------------------
    # These were absent, and the paper's two headline endpoints live entirely inside them, so
    # "one command regenerates every claim" was false for a second time. `campaign_v_e1_results`
    # and `cross_prediction` each uniquely carry a number the paper prints.
    ("fetch_v", [{"argv": ["tools/fetch_campaign_v.py"], "env": {"CAMPAIGN_V_EPOCH": "E1"}},
                 {"argv": ["tools/fetch_campaign_v.py"], "env": {"CAMPAIGN_V_EPOCH": "E2"}}],
     ["data/campaign_v_e1/pub0.npz", "data/campaign_v_e2/pub0.npz"],
     "retrieve both campaign-V epochs by job id (free, no QPU charge)"),
    ("dryrun", ["tools/campaign_dryrun.py"], ["data/campaign_dryrun.json"],
     "the simulator qualification that ran before any campaign-V hardware time"),
    ("campaign_v", [{"argv": ["tools/analyze_campaign_v.py"], "env": {"CAMPAIGN_V_EPOCH": "E1"}},
                    {"argv": ["tools/analyze_campaign_v.py"], "env": {"CAMPAIGN_V_EPOCH": "E2"}}],
     ["data/campaign_v_e1_results.json", "data/campaign_v_e2_results.json"],
     "BOTH ENDPOINTS in all four contexts, scored by the frozen witness (needs fetch_v)"),
    ("graphlike", ["tools/graphlike_infeasibility.py"], ["data/graphlike_infeasibility.json"],
     "E2 on the retrospective records: the same-stabiliser budget and its surrogate"),
    ("parity", ["tools/parity_feasibility.py"], ["data/parity_feasibility.json"],
     "the parity-attenuation moments the graphlike characterisation is applied to"),
    ("persistent", ["tools/persistent_noise_model.py"], ["data/persistent_noise_model.json"],
     "the per-ancilla two-state persistence model, fitted"),
    ("refine", ["tools/persistent_refine.py"], ["data/persistent_refine.json"],
     "the refined persistence fit the ladder and the cross-prediction read (needs persistent)"),
    ("spatial", ["tools/spatial_persistent.py"], ["data/spatial_persistent.json"],
     "the spatial-dilation variant of the same class (needs refine)"),
    ("structure", ["tools/structure_learned_dem.py"], ["data/structure_learned_dem.json"],
     "the structure-learned detector error model, fitted from the device's own correlations"),
    ("ladder", ["tools/surrogate_ladder.py"], ["data/surrogate_ladder.json"],
     "the full elimination ladder, every class against the device's eight-width curve"),
    ("crosspred", ["tools/cross_prediction.py"],
     ["data/cross_prediction.json", "data/cross_prediction_selection.json"],
     "syndrome-selected against decoder-selected cells, and the transfer test (needs refine)"),
    ("syndrome", ["tools/syndrome_process.py"], ["data/syndrome_process.json"],
     "what the fitted model does reproduce: the syndrome process statistics (needs refine)"),
    ("nesting", ["tools/nesting_audit.py"], ["data/nesting_audit.json"],
     "every ordered pair of buffer widths against the joint decoder (needs refine)"),
    ("churn", ["tools/escalation_churn.py"], ["data/escalation_churn.json"],
     "the churn bound for a frozen trigger, and the escalation policies"),
    ("monotone", ["tools/escalation_monotonicity.py"], ["data/escalation_monotonicity.json"],
     "whether width decisions nest, measured shot by shot"),
    ("enlarge", ["tools/enlarge_policy.py"], ["data/enlarge_policy.json"],
     "the growing local repair and its admissibility check"),
    ("devsim", ["tools/device_vs_sim.py"], ["data/device_vs_sim.json"],
     "the device against its own matched simulator at equal detector rate"),
    ("wallclock", ["tools/wallclock_benchmark.py"], ["data/wallclock_benchmark.json"],
     "end-to-end wall-clock accounting. TIMED -- run with the machine idle"),
    ("span", [{"argv": ["tools/affine_span.py"], "env": {"CAMPAIGN_V_EPOCH": "E1"}},
              {"argv": ["tools/affine_span.py"], "env": {"CAMPAIGN_V_EPOCH": "E2"}}],
     ["data/affine_span.json"],
     "is the E1 feature outside the affine span of the E2 features (needs campaign_v)"),
    ("certificate", [{"argv": ["tools/span_certificate.py"], "env": {"CAMPAIGN_V_EPOCH": "E1"}},
                     {"argv": ["tools/span_certificate.py"], "env": {"CAMPAIGN_V_EPOCH": "E2"}}],
     ["data/span_certificate.json"],
     "the condition-aware lower bound on that residual, and both witnesses. SLOW, hours"),
    ("svd", [{"argv": ["tools/span_svd.py"], "env": {"CAMPAIGN_V_EPOCH": "E1"}},
             {"argv": ["tools/span_svd.py"], "env": {"CAMPAIGN_V_EPOCH": "E2"}}],
     ["data/span_svd.json"],
     "singular values per context, which the certificate's condition numbers come from. SLOW"),
    ("crosscheck", [{"argv": ["tools/span_crosscheck.py", "R1"], "env": {"CAMPAIGN_V_EPOCH": "E2"}}],
     ["data/span_crosscheck.json"],
     "the same residual from three rank cutoffs and a second LAPACK driver. SLOW"),
    ("coverage", [{"argv": ["tools/e1_coverage.py"], "env": {"CAMPAIGN_V_EPOCH": "E1"}},
                  {"argv": ["tools/e1_coverage.py"], "env": {"CAMPAIGN_V_EPOCH": "E2"}}],
     ["data/e1_coverage.json"],
     "the E1 bound at a valid 95% error split and under circular blocks. SLOW"),
    ("e2interval", [{"argv": ["tools/e2_device_interval.py"], "env": {"CAMPAIGN_V_EPOCH": "E1"}},
                    {"argv": ["tools/e2_device_interval.py"], "env": {"CAMPAIGN_V_EPOCH": "E2"}}],
     ["data/e2_device_interval.json"],
     "a sampling interval on the device side of the E2 margin, and the surrogate dispersion audit"),
    ("circuit", ["tools/fetch_frozen_circuit.py"],
     ["data/frozen_circuit/summary.json"],
     "the submitted ISA circuit, retrieved and archived. NETWORK + IBM credentials, 0 QPU"),
    ("support", ["tools/circuit_support_audit.py"],
     ["data/circuit_support_audit.json"],
     "the circuit premise behind E2, by enumerating every elementary fault mechanism"),
    ("risk", [{"argv": ["tools/logical_risk.py"], "env": {"CAMPAIGN_V_EPOCH": "E1"}}],
     ["data/logical_risk.json"],
     "the realised logical-risk difference against the certified envelope. SLOW"),
    ("baseline", [{"argv": ["tools/baseline_diagnostics.py"], "env": {"CAMPAIGN_V_EPOCH": "E1"}}],
     ["data/baseline_diagnostics.json"],
     "the conventional held-out diagnostics, device against its own fitted model. SLOW"),
    ("e2quantile", [{"argv": ["tools/e2_surrogate_quantile.py"], "env": {"CAMPAIGN_V_EPOCH": "E1"}}],
     ["data/e2_surrogate_quantile.json"],
     "the mixture reference as a Monte Carlo quantile of 200 draws. SLOW"),
    ("figures", ["tools/make_figures.py", "tools/make_figures_extra.py"],
     ["paper/figures/fig_curves.pdf", "paper/figures/fig_ladder.pdf",
      "paper/figures/fig_budget.pdf", "paper/figures/fig_autocorr.pdf",
      "paper/figures/fig_churn.pdf", "paper/figures/fig_regions.pdf",
      "paper/figures/fig_seams.pdf"],
     "the seven paper figures, rebuilt from the JSONs above into paper/figures/"),
    ("claims", ["tools/make_claims.py"], ["CLAIMS.md"],
     "the AUTHORITATIVE claim list, generated from data/*.json. Must run last."),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--skip", nargs="*", default=[])
    a = ap.parse_args()

    if a.list:
        print(f"{'stage':>14}  {'outputs':<44} what it does")
        for name, _, outs, why in STAGES:
            print(f"{name:>14}  {', '.join(outs) or '(self-check only)':<44} {why}")
        return 0

    todo = [s for s in STAGES
            if (a.only is None or s[0] in a.only) and s[0] not in a.skip]
    print(f"reproduce_all: {len(todo)} stage(s). 0 QPU seconds -- retrieval by job id is free "
          f"and no stage has a submit path.\n")
    t_all = time.time()
    failed = []
    for name, scripts, outs, why in todo:
        print("=" * 100)
        print(f"[{name}] {why}")
        for script in scripts:
            # a stage entry may be "tools/x.py" or ["tools/x.py", "--flag"]. The previous version
            # stored the flag inside the FILENAME, so the stage failed with "can't open file
            # 'tools/multi_seam.py --calibrate'". Gate-5 finding 4.
            # a stage entry may also be a dict {"argv": [...], "env": {...}} so that a stage
            # parameterised by an environment variable -- campaign V is, by epoch -- runs once
            # per value without duplicating the script.
            env = None
            if isinstance(script, dict):
                parts = list(script["argv"])
                env = dict(os.environ, **script.get("env", {}))
            else:
                parts = [script] if isinstance(script, str) else list(script)
            argv = [PY, "-u", os.path.join(ROOT, parts[0])] + parts[1:]
            t0 = time.time()
            rc = subprocess.call(argv, cwd=ROOT, env=env)
            print(f"  {' '.join(parts)}: exit {rc} in {time.time() - t0:.0f}s")
            if rc != 0:
                failed.append(f"{name}:{parts[0]}")
        for o in outs:
            p = os.path.join(ROOT, o)
            print(f"  {'OK ' if os.path.exists(p) else 'MISSING'} {o}")
            if not os.path.exists(p):
                failed.append(f"{name}:{o}")

    print("\n" + "=" * 100)
    print(f"total {time.time() - t_all:.0f}s")
    if failed:
        print("FAILED: " + ", ".join(failed))
        return 1
    print("every stage completed and every output present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
