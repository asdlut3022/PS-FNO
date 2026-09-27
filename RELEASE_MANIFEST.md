# RELEASE MANIFEST — PS-FNO code release

This manifest maps the archived result records and the executable entry points
shipped in this repository to what they support in the manuscript, and states
exactly what is reproduced from scratch versus what is reproduced from archived
results. It is the authoritative disclosure for the public code release. **`REPRODUCIBILITY_STATUS_MATRIX.md` is the one-page
per-artifact companion**: it lists, entry point by entry point, which seeds the script
trains versus which the table reports, where the script writes versus where the table is
read from, and which manuscript experiments are out of scope. The two files state the same
scope and agree by construction.

**Scope of this release.** This repository contains the executable main programs
for the five primary benchmarks (Tables 4, 5, 6, 7, 8) and the archived result
records shipped with this release for those benchmarks, together with the
cross-benchmark statistics of Section 4.10, the records behind the equal-parameter controls
discussed in the benchmark and ablation sections, and the two raw cost records behind
Table 15 and Supplementary Table S2 — the per-step inference timing of the two
spectral kernels (`results/inference_benchmark.json`) and the external-baseline
wall-clock, memory and inference measurements
(`results/baseline_cost_results.json`). Everything else in the manuscript is **outside this release** — neither
its programs nor its result records are included. That covers: geometry
generalization (Supplementary Section S21), zero-shot resolution transfer (Table 9 and Supplementary Section S23), the
per-layer physics-ratio analysis (Table 10), the kernel and rank ablations
(Table 11), the component ablation (Table 12), the information-matched controls
(Table 13), the training-stage ablation (Table 14), the training-time and
peak-memory profiling runs behind Table 15, and the S-series supplementary experiments (long-horizon extrapolation, noise
robustness, resolution scalability, rank analysis, data efficiency, 1D Burgers,
kernel mismatch, proofs of the main-text results, Proposition 2 numerical
verification, empirical checks of the main-text theorem's and proposition's
premises, wave kernel template error, and the InfoMatched controls). This is a **scope boundary**: those
experiments are reported in the manuscript from the authors' own runs and are not part of
this package. The division of material is described in the manuscript's Data
availability statement.

**Manuscript table numbering.** The main text is the revision of
NEUNET-D-26-02460; its main-text tables are numbered Table 1--15 (Table 1: method
positioning; Table 2: parameter comparison; Table 3: implementation details;
Table 4: 2D wavy-fin heat; Table 5: convection-diffusion; Table 6:
reaction-diffusion; Table 7: damped wave; Table 8: 3D LTNE; Table 9: 2D
super-resolution; Table 10: physics-ratio; Table 11: kernel ablation; Table
12: component ablation; Table 13: InfoMatched; Table 14: training-stage
ablation; Table 15: computational cost). Geometry generalization and 3D
super-resolution are supplementary sections (S21 and S23) rather than
main-text tables. The table numbers below follow the manuscript.

## 1. Five main benchmarks (executable entry points)

Each script is self-contained in data generation and model definition. The scripts
provide the minimal entry points for the corresponding main-text tables. Seed coverage
is not uniform across the five entry points and is stated per row below: two of the
five scripts are single-seed entry points, so the three-seed figures reported in the
manuscript for those benchmarks come from the archived per-seed JSON, not from one run
of the entry point. Two of the five pipelines resume from a saved SS-stage checkpoint
(see the checkpoint note in `reproducibility.md`); the other three train from scratch
in a single process. Re-running these entry points produces a new seeded realization,
not byte-identical values.

| Benchmark (main-text table) | Executable script | Seed coverage | Per-seed results JSON |
|---|---|---|---|
| 2D diffusion / heat sink (Table 4) | `scripts/run_heat_ar_gridsearch.py` (SS+AR) | 42/123/456 | `results/heat_ar_gridsearch_results.json` |
| 2D convection-diffusion (Table 5) | `scripts/run_conv_diff.py` | **42 (single-seed)** | `results/conv_diff_multi_seed_results.json` |
| 2D reaction-diffusion (Table 6) | `scripts/run_reaction_diff_ar_valfix.py` | 42/123/456 | `results/reaction_diff_ar_valfix_results.json` |
| 2D damped wave (Table 7) | `scripts/run_wave_with_ar.py` | **789 (single-seed AR)** | `results/wave_ar_results_3seed_backup.json` |
| 3D LTNE turbine (Table 8) | `3d/run_3d_multi_seed.py` | 42/123/456 | `results/3d_multi_seed_results.json` |

A sixth script, `scripts/run_statistical_tests.py`, is included because it produces the
cross-benchmark statistics of Section 4.10 for exactly these five benchmarks. It reads
the archived per-seed JSON and the archived wave log; it does not train.

**Wave seed provenance (Table 7 and the statistics).** The damped-wave benchmark uses three
seeds in the paper. Two are archived in `results/wave_ar_results_3seed_backup.json` (its
`seeds` field reads `[42, 789]`); the third has **no JSON artifact** — in the original
pipeline it lives only in a training log, `results_wave/training_nt5_seed314_777.log`,
lines 156–158 (values 0.1674 / 0.1026). A copy of that log is included at
`results_wave/training_nt5_seed314_777.log` (repo root; the script at
`scripts/run_statistical_tests.py` parses it) so the statistical script reads actual
provenance instead of a hardcoded constant. The three-seed statistics are as reported in
the paper; the third seed's pair is log-level provenance, not a re-runnable JSON.

**Reaction-diffusion row coverage (Table 6).** The **SS-stage rows** (`Standard FNO+SS`
2.582±0.224, `Physics FNO+SS` 2.013±0.056) are backed per seed by the `standard_ss` /
`physics_ss` blocks of `results/reaction_diff_ar_results.json`; the **AR rows**
(`Standard FNO+SS+AR` 2.503±0.319, `Physics FNO+SS+AR` 1.876±0.152) by the `standard_ar` /
`physics_ar` blocks of `results/reaction_diff_ar_valfix_results.json`. The table's stds are
those blocks' ddof=0 values (the valfix file's plain `stdev` would give 0.390 / 0.186).
All of these rows come from runs that construct the model with `pde_type='heat'`: the
reaction-diffusion benchmark is executed through the diffusion branch of the spectral
template, and no reaction coefficient is instantiated or trained in it.

**Heat AR-protocol footnote (Table 4).** The footnote value for the superseded original
AR protocol, Physics FNO+SS+AR 0.789±.054, is backed by
`results/heat_ar_seed789_results.json`, whose `three_seed_summary` block carries
`ar_wm_per_seed` = [0.848, 0.718, 0.8005] (seeds 42/123/789; ddof=0). That file covers
the **physics arm** of the footnote protocol.

**Note on seed selection.** For the five benchmarks in this release, the manuscript
states that the seeds used in the paper were fixed as part of each benchmark's training
protocol rather than selected on the basis of observed outcomes, and that no run of these
five benchmarks was withheld or re-seeded for an unfavorable result. That is an authors'
statement of protocol. The statistical script reads the two archived seeds from the backup
JSON and the third from the archived log, exactly as the paper reports.

## 2. Archived result records and the manuscript locations they support

Every file under `results/` (and the wave log) is listed here with the manuscript
location it supports. A record is included only if it backs a cell of Tables 4, 5, 6, 7, 8,
the cross-benchmark statistics of Section 4.10, the cost material behind Table 15 and
Supplementary Table S2, or a statement in the corresponding benchmark or ablation
section (including the equal-parameter controls).

| File | Supports |
|---|---|
| `results/heat_ar_gridsearch_results.json` | Table 4 — heat, per-seed AR grid record: the grid outcome (`grid`, `best_lr`, `best_patience`), the AR-stage validation score (`final_val`) and the AR-stage test score (`final_test`) |
| `results/heat_ar_seed789_results.json` | Table 4 footnote — original AR protocol, physics arm (`three_seed_summary`) |
| `results/ffno_heat_results.json` | Table 4 — F-FNO row (three seeds) |
| `results/baselines_heat_pino_podrbf_results.json` | Table 4 — POD-RBF row (3.427) |
| `results/baseline_results.json` | Table 4 — DeepONet row (2.171) and the untuned-PINO value behind the Table 4 footnote (>76 K); see Section 4 note 1 |
| `results/pino_final_results.json` | Table 4 — tuned PINO row (1.648) |
| `results/podrbf_deterministic_results.json` | Tables 4, 5, 6 — POD-RBF rows (`m = 50`, thin-plate spline) |
| `results/conv_diff_multi_seed_results.json` | Table 5 — per-seed main rows |
| `results/ffno_convdiff_results.json` | Table 5 — F-FNO row (three seeds) |
| `results/baselines_convdiff_results.json` | Table 5 — external baseline rows |
| `results/reaction_diff_ar_valfix_results.json` | Table 6 — AR rows |
| `results/reaction_diff_ar_results.json` | Table 6 — SS rows |
| `results/ffno_reactiondiff_results.json` | Table 6 — F-FNO row (three seeds) |
| `results/baselines_reactiondiff_results.json` | Table 6 — external baseline rows |
| `results/wave_ar_results_3seed_backup.json` | Table 7 — seeds 42 and 789, per-seed SS and AR (`standard_ss`, `standard_ss_ar`, `physics_ss`, `physics_ss_ar`) |
| `results/ffno_wave_results.json` | Table 7 — F-FNO row (three seeds) |
| `results/baselines_wave_results.json` | Table 7 — external baseline rows |
| `results/3d_multi_seed_results.json` | Table 8 — Standard/Physics FNO, per seed, ID and OOD |
| `results/deeponet_3d_eqp_seed42_results.json` | Table 8 — DeepONet-3D row |
| `results/ffno_3d_w320_seed42_results.json` | Table 8 — F-FNO (W=320) row |
| `results/equal_params_2d_results.json` | Heat equal-parameter control — cited in the parameter-fairness discussion of Section 4.11 rather than in a standalone paragraph of Section 4.2; tabulated in Supplementary Section S7 |
| `results/equal_params_convdiff_results.json` | Section 4.4 parameter-fairness text |
| `results/equal_params_reactdiff_results.json` | Section 4.5 parameter-fairness text |
| `results/equal_params_wave_results.json` | Section 4.6 parameter-fairness text |
| `results/friedman_results.json` | Section 4.10 — Friedman ranking; written by `scripts/run_statistical_tests.py` |
| `results/inference_benchmark.json` | Table 15 — per-step inference timing on the 2D grid (the raw measurement behind the 4.9\,ms physics / 3.2\,ms standard per-step figures quoted in the main text); the timing protocol is stated in the Table 15 caption |
| `results/baseline_cost_results.json` | Supplementary Table S2 — external-baseline wall-clock, peak GPU memory and per-step inference time (the raw measurements behind the PINO, DeepONet and POD-RBF rows of that supplementary table, profiled under the same protocol as main-text Table 15); the profiling protocol is stated in that table's caption, and POD-RBF memory is CPU process RSS during fit. The file also records the accuracy of that same single-seed profiling run; those figures are **not the source of the reported accuracies** — the archived accuracy records for the external baselines are the `results/baselines_*_results.json` and `results/baseline_results.json` entries above, and `results/podrbf_deterministic_results.json` / `results/pino_final_results.json` for the selected POD-RBF and tuned-PINO configurations. DeepONet's and PINO's profiling errors differ from their reported rows; the POD-RBF entry is the exception, agreeing with the reported Table 4 POD-RBF row at the reported precision, although that row is read from `results/baselines_heat_pino_podrbf_results.json` (Section 2) |
| `results_wave/training_nt5_seed314_777.log` | Table 7 — third wave seed (314) provenance; parsed by `scripts/run_statistical_tests.py` |

## 3. Not published (scope statement)

Not included in this release, by design:

- **All programs and records for experiments outside the five benchmarks** — geometry
  generalization, zero-shot resolution transfer, the per-layer physics-ratio analysis,
  the kernel/rank/component/training-stage ablations, the information-matched controls,
  the training-time and peak-memory profiling runs behind Table 15, and the S-series supplementary experiments
  (long-horizon extrapolation, noise robustness, resolution scalability, rank analysis,
  data efficiency, 1D Burgers, kernel mismatch, proofs of the main-text results,
  Proposition 2 numerical verification, empirical checks of the main-text theorem's and
  proposition's premises, wave kernel template error, and the InfoMatched controls). See
  the scope statement at the top of this file.
- **The equal-parameter control harness.** Its four per-seed records are archived as
  `results/equal_params_*_results.json` and back the parameter-fairness statements in the
  benchmark and ablation sections; the harness that produced them is not part of this release.
- **Legacy / failed experiment directories** and expired checkpoints.
- **Checkpoints** (`*.pt`, `*.pth`) — too large (multiple GB); figures/scripts read them
  only after a training run; the numbers are in the archived JSON.
- **Raw training data snapshots** — generated on the fly by the deterministic data
  generators; no files are committed.
- **Internal working directories / old `run_*` versions** — the author's private
  iteration artifacts; none of the published numbers require them beyond the archived
  JSON listed in Section 2.

## 4. Archived-record notes (read this before verifying)

Section 2 maps each archived file to the manuscript location it backs. Two conventions
need a word of explanation before they are checked:

0. **`corr_init_scale` is not wired in the released code.** The spectral convolution in
   `scripts/physics_fno.py` accepts a `corr_init_scale` keyword for call compatibility but
   never reads it; the correction scale is set by `corr_scale_param` instead. Arms built
   with different `corr_init_scale` values at a fixed seed therefore have byte-identical
   initial `state_dict`s, and a sweep over that argument is not initialization-sensitivity
   evidence. See the README note "Parameter-initialization note", which also records
   that the `corr_init_scale` docstring in `scripts/physics_fno.py` was corrected, and
   that the module's executable behaviour under the released entry points' configuration
   is the archived original's; it also lists the measured correction-to-base ratio for
   each released spectral-convolution class.

1. **`results/baseline_results.json` holds a table row and a footnote value.** `DeepONet`
   (`test.worst_max` 2.171) is the **Table 4 DeepONet row** and the source of the Friedman
   matrix's heat DeepONet cell in `scripts/run_statistical_tests.py`. `PINO`
   (`test.worst_max` 76.885) is the **untuned** run behind the Table 4 footnote — the
   manuscript's PINO row is the tuned configuration (1.648), archived separately in
   `results/pino_final_results.json`, so the two PINO numbers are two different
   configurations rather than two versions of one reported cell.

2. **`results/baselines_heat_pino_podrbf_results.json` is named for two baselines but holds
   one.** The file now contains **only the POD-RBF block** (`test.worst_max` 3.4268, the
   Table 4 POD-RBF row); the PINO block was removed. The PINO row is in
   `results/pino_final_results.json` (tuned) and `results/baseline_results.json` (untuned), as
   above. The filename is the original harness's and is kept so that the archived record and
   the manuscript's provenance trail stay consistent. `results/podrbf_deterministic_results.json`
   holds the deterministic POD-RBF records behind the reported rows of Tables 4, 5 and 6
   (`m = 50`, thin-plate spline); the full hyperparameter search history is not part of this
   release.

## 5. External-baseline records: seed labeling

The external-baseline JSONs (`results/baselines_heat_pino_podrbf_results.json`,
`baselines_convdiff_results.json`, `baselines_reactiondiff_results.json`,
`baselines_wave_results.json`, `pino_final_results.json`,
`podrbf_deterministic_results.json`) are **single-seed records that do not carry an
explicit `seed` field**. The manuscript labels them "single-seed"; because the specific
seed value is not recorded in the file it is treated as *unknown* rather than asserted.
What the archived file establishes is the absence of a per-seed key, not the absence of
other runs: it holds one result set per benchmark and says nothing further about how
many runs produced it. The manuscript makes no seed-to-seed variability claim for these
methods. Two exceptions: the 3D DeepONet record
`results/deeponet_3d_eqp_seed42_results.json` **does** carry `seed = 42`, and the F-FNO
records (`results/ffno_*.json`) are three-seed and carry per-seed keys, with one exception: `results/ffno_3d_w320_seed42_results.json` is the single-seed (seed 42) record behind the 3D F-FNO row, as its own section below notes.

**One archived metadata string predates this wording.** The shipped copy of
`results/friedman_results.json` — the output of `scripts/run_statistical_tests.py`, and the
file that script overwrites when re-run — carries a `cell_policy` field describing how each
cell of the ranking matrix was read. The archived copy's wording asserts a specific seed
value for the external baselines; the script now emits the wording above instead, because
that value is not recorded in those files and is therefore not asserted. Only that
descriptive string differs. The archived numeric content is identical to what the current
script produces: re-running it inside a copy of the repository (see
`REPRODUCIBILITY_STATUS_MATRIX.md` §4c) reproduces every numeric field exactly, with
`cell_policy` the single metadata divergence.

## 6. Cross-benchmark statistics (Section 4.10)

`scripts/run_statistical_tests.py` is included and needs no training. Run from the
repository root, it reads the archived per-seed records listed in Section 2 and prints
the paired Wilcoxon significance (p = 0.0625), the Friedman ranking (χ² = 17.5714,
p = 0.0035), the geometric-mean error ratio (1.3301) and the Hodges–Lehmann estimator
(0.4388) reported in Section 4.10. Three caveats: it parses the wave log at
`results_wave/training_nt5_seed314_777.log`, so that file must be present; it
**overwrites** `results/friedman_results.json`, and it takes no command-line arguments at
all -- even `--help` runs the full pass and rewrites the file -- so an exact re-run should
be performed on a copy if the archived copy is to be preserved; and its `provenance` block
records source paths **repository-relative**, so a re-run inside a copy reproduces the
archived `provenance` block exactly and the single metadata divergence is the descriptive
`cell_policy` string (Section 5).

The archived `results/friedman_results.json` also carries a `cell_policy` string that
described the DeepONet/PINO/POD-RBF cells as "seed-42 single value". Those records carry
no `seed` field, so the seed is not recoverable from them; the released script now says
so instead, and the archived string is superseded. The archived file itself is left
untouched, since it is a delivered record and not a document.
