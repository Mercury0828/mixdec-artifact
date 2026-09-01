# Reproducing every number

> **`CLAIMS.md` is the authoritative list of results.** This file regenerates them. Where the two
> disagree, `CLAIMS.md` is right, because it is generated from `data/*.json` and this file is not.

One driver regenerates every frozen result from the cached shots.

```bash
pip install -r requirements.txt
python tools/reproduce_all.py --list          # every stage, its outputs, and what it does
python tools/reproduce_all.py --skip fetch    # everything, offline, from the local caches
python tools/reproduce_all.py --only support risk baseline
```

## Nothing here can spend processor time

The fetch stages retrieve **already-executed** jobs **by id**, which is not charged. Every other
stage reads local `.npz` caches. Submission exists in exactly two files, `tools/submit_campaign_r.py`
and `tools/submit_campaign_v.py`. Neither does anything without an explicit `--submit`, both verify
the frozen witness hash first, and both refuse if the worst-case estimate exceeds the remaining
allocation. `tools/reproduce_all.py` never calls either.

## The data

Five jobs on `ibm_cleveland`, an IBM Heron r2 processor with 156 qubits, running a distance-9
bit-flip repetition code. Region `R1` is the physical chain with data qubits
`[0,2,4,6,8,10,12,14,19]` and ancillas `[1,3,5,7,9,11,13,15]`; region `R2` is the disjoint chain
`[16,23,22,21,36,41,42,43,44,45,37,25,26,27,28,29,30]` in the same order, carrying the same circuit
with its labels rewritten rather than a fresh transpilation.

| role | job id | rounds | shots | charged | cache |
|---|---|---|---|---|---|
| pilot | `da7mi6bsq5js73bk4veg` | 12 | 20,000 | 6 s | `data/da7mi6bsq5js73bk4veg.npz` |
| frozen source | `da7miljsq5js73bk4vtg` | 50 | 100,000 | 8 s | `data/da7miljsq5js73bk4vtg.npz` |
| replication | `da9h4herbfbs73chl6tg` | 50 | 200,000 | 119 s | `data/campaign_r/pub*.npz` |
| campaign V, epoch 1 | `daaaee4jbipc73ffn220` | 50 | 200,000 | 119 s | `data/campaign_v_e1/pub*.npz` |
| campaign V, epoch 2 | `daabe9urbfbs73cihfp0` | 50 | 200,000 | 119 s | `data/campaign_v_e2/pub*.npz` |

Total processor time across the project: **371 seconds**. Every simulator result costs none.

Each campaign job submits eight process units of 25,000 shots, alternating region and logical state
at process-unit granularity so that drift within a job stays separable from a region effect. A pub's
`meta` string records its job id, pub index, region, logical state, qubit list, shot count, round
count and charge. The first logical-zero pub of a context is its calibration split and the second is
its evaluation split, so the two share no shot.

Every analysis is confined to the logical-zero arm. On the logical-one arm the final readout mean is
0.2175 where a noiseless memory would return one, so the symmetric model the decoder is built on does
not describe that arm, and the sign of the decoder difference reverses there. The two arms are never
pooled and no arm-averaged rate is reported.

## Environment

Python 3.11.4 with the versions pinned in `requirements.txt`. The fetch stages need a configured
`QiskitRuntimeService` account; every other stage runs offline once `data/` is populated, which it is
in this repository.

Several stages are slow. `certificate` runs a rank-revealing solve on a 24,000 x 10,609 design and
takes hours; `risk`, `baseline`, `coverage`, `e2interval` and `e2quantile` decode or resample tens of
thousands of shots and take minutes each. `--list` marks them.

## Stage map

`python tools/reproduce_all.py --list` prints the current list with each stage's outputs. The stages
behind the paper's two endpoints are:

| stage | writes | what it establishes |
|---|---|---|
| `fetch` | `data/**/*.npz` | the shots, by job id |
| `campaign_v` | `campaign_v_e*_results.json` | the twelve pre-registered criteria, E1 and E2 |
| `coverage` | `e1_coverage.json` | the E1 bound at an even 95% error split and under circular blocks |
| `e2interval` | `e2_device_interval.json` | a sampling interval on the device side of the E2 margin |
| `e2quantile` | `e2_surrogate_quantile.json` | the mixture control as a Monte Carlo quantile of 200 draws |
| `support` | `circuit_support_audit.json` | every elementary Pauli mechanism of the submitted circuit, with its detector support |
| `risk` | `logical_risk.json` | the realised logical-risk difference against the certified envelope |
| `baseline` | `baseline_diagnostics.json` | the ordinary held-out diagnostics, device against its own fitted model |
| `certificate` | `span_certificate.json` | the condition-aware bound behind the non-nesting argument |
| `claims` | `CLAIMS.md` | the authoritative claim list. Must run last |

## What to check first

**Stale artifacts.** Any change to `fit_weights_v2`, to `WindowGraph` edge construction, or to the
detector convention in `tools/detectors.py` invalidates every JSON downstream of it. Re-run the
driver after touching those.

**The generator derives its conclusions.** `python tools/make_claims.py --check` feeds the generator
inputs asserting the opposite of each reported conclusion and requires the corresponding output token
to appear and its opposite to vanish. Twelve cases, including the sign of the realised risk
difference and the circuit's largest detector support.

**The frozen witness.** `tools/frozen_witness.py` holds the parameter-free definitions of both
endpoints. Its SHA-256 is verified at the start of every scoring run and a mismatch stops the run.

## Data and code availability statement

> The syndrome data supporting this work were collected on the IBM Quantum `ibm_cleveland` processor
> and are retrievable by the job identifiers listed above. Cached copies, the submitted
> instruction-set circuit, all analysis code and the pre-registered criteria with their falsifiers
> are available at this repository, archived at the DOI cited in the paper. Every reported number is
> regenerated by a single driver, `tools/reproduce_all.py`.
