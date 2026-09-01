# Two qualification tests for a fitted detector error model — artifact

Code, cached hardware records and frozen results for

> J. Shen and H. Zhong, *A Fitted Detector Error Model Fails Two Qualification Tests on a Heron
> Processor.*

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22217806.svg)](https://doi.org/10.5281/zenodo.22217806)

Archived at [10.5281/zenodo.22217806](https://doi.org/10.5281/zenodo.22217806). That version is the
one the numbers in the paper were produced from; cite it rather than the branch.

Everything here regenerates from the cached shots. **No step in this repository spends processor
time**, and the two endpoints the paper reports can be recomputed end to end offline.

## What the paper measures

A decoder is handed a detector error model before it decodes anything, and that model is normally
judged on how well it reproduces the syndrome statistics it was fitted to. Two things it is not
usually asked for are measured here, on `ibm_cleveland`, a 156-qubit Heron r2 processor running a
distance-9 repetition memory for 50 rounds, in four contexts spanning two disjoint qubit lines and
two sides of a recalibration.

**E1, decision conservativeness.** A joint decode of the whole record and a windowed decode that cuts
it at one seam disagree on some shots. That disagreement needs no knowledge of the prepared logical
state, and its probability upper-bounds the change in logical risk the substitution causes. E1 asks
whether a model fitted from the device's own syndromes predicts it.

**E2, class realizability.** Singleton and same-stabiliser pair parity attenuations satisfy an exact
inequality under every independent-event model whose faults flip at most two detectors. E2 asks
whether the measured moments satisfy it.

## Layout

```
CLAIMS.md      the authoritative list of every result, generated from data/*.json
REPRODUCE.md   how to regenerate all of it, and what each stage costs
tools/         the analysis, one concern per file
data/          cached process-unit records, frozen results, the submitted circuit
```

`CLAIMS.md` is generated, not written. `tools/make_claims.py --check` feeds the generator inputs
asserting the opposite of each reported conclusion and requires the corresponding output token to
appear and its opposite to vanish, so a conclusion that is asserted rather than derived fails the
check.

## Hardware records

Five jobs, 371 seconds of processor time in total.

| role | job id | shots | charged |
|---|---|---|---|
| pilot | `da7mi6bsq5js73bk4veg` | 20,000 | 6 s |
| frozen source | `da7miljsq5js73bk4vtg` | 100,000 | 8 s |
| replication | `da9h4herbfbs73chl6tg` | 200,000 | 119 s |
| campaign V, epoch 1 | `daaaee4jbipc73ffn220` | 200,000 | 119 s |
| campaign V, epoch 2 | `daabe9urbfbs73cihfp0` | 200,000 | 119 s |

`data/campaign_r/`, `data/campaign_v_e1/` and `data/campaign_v_e2/` hold the per-pub syndrome and
final-readout arrays as `.npz`, each with a metadata string recording its job id, region, logical
state, shot count and round count. `data/frozen_circuit/` holds the submitted instruction-set circuit
in QPY with the SHA-256 of each serialised form; it is Clifford throughout, and
`tools/isa_to_stim.py` translates it gate for gate for the fault enumeration.

## Quick start

```bash
pip install -r requirements.txt
python tools/reproduce_all.py --list        # every stage, its outputs, and what it does
python tools/reproduce_all.py --skip fetch  # regenerate everything from the local caches
```

The `fetch` stages retrieve already-executed jobs by id and need IBM Quantum credentials. Skipping
them uses the caches in `data/`, which is the intended path. See `REPRODUCE.md`.

## Pre-registration

The two criteria, the tolerance, the eight buffer widths, the seam position, the tie-breaking rule,
the block-length grid, the per-context surrogate construction and the SHA-256 of the module that
computes both endpoints were committed before either campaign V job was submitted.
`tools/frozen_witness.py` carries that module and its hash is verified at the start of every scoring
run. Analyses added after that freeze are labelled post-registration wherever they appear, in the
paper and in `CLAIMS.md`.

## Licence

Code is MIT (see `LICENSE`). The cached measurement records in `data/` are released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Please cite the paper.

## Contact

Jiachen Shen, University of Houston, <jshen28@cougarnet.uh.edu>,
[0000-0002-7233-497X](https://orcid.org/0000-0002-7233-497X).
Hui Zhong, Miami University, <zhongh7@miamioh.edu>,
[0009-0005-7952-007X](https://orcid.org/0009-0005-7952-007X) (corresponding author).
