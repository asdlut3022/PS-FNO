# PS-FNO: Physics-Structured Fourier Neural Operators via Spectral Semigroup Embedding

Code for the manuscript *"Physics-Structured Fourier Neural Operators via Spectral
Semigroup Embedding"* (Neural Networks, NEUNET-D-26-02460).

This repository contains the data-generation scripts, model implementations, and the
**five benchmark pipelines** that produce Tables 4, 5, 6, 7 and 8 of the manuscript,
together with the archived result records shipped with this release for those benchmarks,
the records behind the equal-parameter controls discussed in the benchmark and ablation sections,
the cross-benchmark statistics of Section 4.10, and the two raw cost records behind
Table 15 and Supplementary Table S2 — the per-step inference timing of the two spectral
kernels (`results/inference_benchmark.json`) and the external-baseline wall-clock, memory
and inference measurements (`results/baseline_cost_results.json`). Everything else in the manuscript — geometry
generalization (Supplementary Section S21), zero-shot resolution transfer (Table 9 and Supplementary Section S23), the per-layer
physics-ratio analysis (Table 10), the kernel/rank/component/training-stage ablations
(Tables 11, 12, 14), the information-matched controls (Table 13), the training-time and
peak-memory profiling runs behind Table 15, and the S-series supplementary experiments — is outside this release:
neither its programs nor its result records are included. That is a scope boundary — a
statement of what this package contains. **`REPRODUCIBILITY_STATUS_MATRIX.md` is the single per-artifact
statement of this boundary** (what is executable, what is archived-only, what is out of
scope); `RELEASE_MANIFEST.md` states it formally and maps every file in `results/` to the
manuscript location it supports.

**What this package supports.** Two things: **pipeline execution** — the five entry points
run end to end as executable fresh runs (real data generation, training, evaluation and a
result file on the machine that runs them) — and **archived-record verification** — the archived per-seed
records shipped with this release for the five benchmarks can be checked as shipped, and the
cross-benchmark statistics of Section 4.10 are re-derivable from those records, with no training at all. Those records are result JSON files, together with one
log-level provenance record (the third damped-wave seed) identified under
*Scope of the reproduction claim* below. It does **not** support reproducing every reported number value-for-value
from a clean checkout: the training runs are stochastic, so a re-run yields a new
realization, and the saved SS-stage checkpoints behind the heat and reaction-diffusion
AR-stage values are too large to host and are **not** in this repository.

**Scope of the reproduction claim.** The five main benchmark *entry points* are
executable, but this does **not** mean the five tables can be recomputed value-for-value
from a clean checkout in one process. Two pipelines resume from a saved SS-stage
checkpoint (heat, reaction-diffusion) that is too large to host and is not in this
repository; the reported AR-stage values for them are archived in `results/`. Two entry
points are single-seed (`run_conv_diff.py` trains seed 42 only; `run_wave_with_ar.py`
trains seed 789 only), so the three-seed figures in Tables 5 and 7 come from the archived
per-seed JSON, not from one run of those entry points. Only the data-generation scripts
are deterministic for a given seed; the training runs are stochastic, so a fresh run
yields a new realization rather than byte-identical values. The archived records shipped with this release for the five benchmarks are the
per-seed result JSON in `results/` together with one training log, from which the third
damped-wave seed's two values come rather than from a result JSON (log-level
provenance; see `REPRODUCIBILITY_STATUS_MATRIX.md` §(f)). The archived records listed in
`RELEASE_MANIFEST.md` Section 2 can be checked, and the cross-benchmark statistics of
Section 4.10 re-derived, with no training at all. The Data availability statement
describes this two-tier structure.

**Record coverage.** `results/` holds the archived per-seed records shipped with this
release for the five main benchmarks (Tables 4, 5, 6, 7, 8) and the cross-benchmark
statistics of Section 4.10, which can be re-derived with no training at all.
`RELEASE_MANIFEST.md` Section 2 maps every archived file to the manuscript location it
backs. Training runs are stochastic, so re-running an
entry point yields a new realization rather than byte-identical values; the archived
records are the authoritative source for the reported means. One file is laid out in a way
that can be misread during verification, and it is described in `RELEASE_MANIFEST.md`
Section 4:

- **`results/baseline_results.json` holds a table row and a footnote value.** Its
  `DeepONet` block (`worst_max` 2.171) is the Table 4 DeepONet row and the source of the
  Friedman matrix's heat DeepONet cell; its `PINO` block (76.885) is the **untuned** run
  the Table 4 footnote describes, whereas the table's PINO row is the tuned configuration
  (1.648), archived in `results/pino_final_results.json`. The two PINO numbers are two
  configurations, not two versions of one cell.

## Layout

```
code_release/
├── scripts/     # 2D model, solvers, data generation, and 2D experiments
├── 3d/          # 3D LTNE (turbine-blade) model and experiments
├── results/     # archived per-seed result records (see RELEASE_MANIFEST.md Section 2)
├── results_wave/ # damped-wave training log holding the third seed's provenance
├── requirements.txt
├── README.md
├── RELEASE_MANIFEST.md
├── REPRODUCIBILITY_STATUS_MATRIX.md  # per-artifact scope: executable / archived / out of scope
└── reproducibility.md
```

## What is in this repository

| Component | Files |
|---|---|
| Model: physics-structured kernel | `scripts/physics_fno.py`, `scripts/fno.py` (standard FNO) |
| Irregular-domain support | `scripts/irregular_domain.py` (SDF masks) |
| Solvers / data generation | `scripts/solve_wave.py` (damped wave, Kelvin–Voigt), `scripts/solve_conv_diff.py`, `scripts/solve_reaction_diff.py`, `scripts/train_v10_clean.py` (heat data generation) |
| Diffusion (heat) benchmark | `scripts/run_heat_ar_gridsearch.py` (symmetric AR grid, validation selection) |
| Convection-diffusion | `scripts/run_conv_diff.py` |
| Reaction-diffusion | `scripts/run_reaction_diff_ar_valfix.py` |
| Damped wave | `scripts/run_wave_with_ar.py` (SS + AR training), `scripts/run_wave_experiment.py` (SS-only older loop, used as a library by the AR script) |
| 3D LTNE (turbine) | `3d/run_3d_multi_seed.py`, `3d/train_causal_pino_3d.py` (data gen), `3d/physics_fno_3d.py`, `3d/run_comparison_3d.py`, `3d/run_ood_experiment.py` |
| Statistical tests | `scripts/run_statistical_tests.py` |
| Archived-result evidence | the 27 JSON records in `results/` plus `results_wave/training_nt5_seed314_777.log`; `RELEASE_MANIFEST.md` Section 2 maps each one to the manuscript location it supports |

The external-baseline results, the POD-RBF / PINO tuning records, the equal-parameter
controls and the F-FNO results are **archived result records**: their generating
harnesses for the external methods are not part of this release. `RELEASE_MANIFEST.md`
maps every archived record to the manuscript location it supports. The manuscript's
Data availability statement describes this two-tier structure.

**Auxiliary scripts — not entry points.** The following modules are imported as libraries by
the entry points, but are **not** among the five recommended entry points and are **not**
covered by the output-path scope in the notes below:
`scripts/fno.py`, `scripts/irregular_domain.py`, `scripts/physics_fno.py`,
`scripts/solve_wave.py`, `scripts/train_v10_clean.py`, `scripts/run_wave_experiment.py`,
`3d/run_ood_experiment.py`, `3d/train_causal_pino_3d.py`. Each carries `__main__` or demo
code; running one directly can write inside the package (see the exceptions listed under
"Output-path scope" below), and some of those paths are historical or demonstration code
rather than part of any reported run.

## Data partitioning and seeds

- **2D diffusion (heat)** and **reaction-diffusion**: **150 training / 20 validation /
  20 test** trajectories per seed, seeds `{42, 123, 456}`; the validation and test
  generators are offset by `seed+2000` and `seed+1000` respectively, so the three
  splits are disjoint. Heat and reaction-diffusion use the **validation** split for
  AR grid/checkpoint selection.
- **Convection-diffusion**: **150 training / 20 test** trajectories (seeds
  `{42, 123, 456}` in the archived records; the released entry point
  `scripts/run_conv_diff.py` trains seed 42 only). **No validation split is
  generated for this benchmark**: a pre-specified `lr = 1e-4` is used and the AR
  checkpoint is selected on the **training** worst-case error (the protocol of
  Section 3.4.2 of the manuscript).
- Damped wave: **250 training** / 20 test trajectories (no validation split; the
  AR checkpoint is selected on training worst-case error, the protocol stated in
  Section 3.4 of the manuscript and summarized in Supplementary Section S1). The released
  entry point `scripts/run_wave_with_ar.py`
  trains seed 789 only; the other two seeds come from the archived per-seed JSON
  (seed 42) and, for seed 314, from the archived training log rather than a result
  JSON (log-level provenance, matching the Data availability statement's wording).
- 3D LTNE: **150 training / 20 test** trajectories (scheduled sampling only).
- AR checkpoint selection and grid selection are the same for the standard and
  physics models; the test split is used for reporting only, never for selection:
  it is evaluated at each reported stage, in every benchmark (see the
  checkpoint-selection remarks in Sections 3.4.2 and
  4.4 of the manuscript).
- AR grid: `lr ∈ {1e-5, 3e-5, 1e-4} × patience {40, 80}`, identical for the standard
  and physics models. This grid is used by the **heat** benchmark only
  (`scripts/run_heat_ar_gridsearch.py`). Reaction-diffusion uses a single fixed
  setting (`lr = 3e-5`, `patience = 80`) rather than a grid; convection-diffusion and
  damped wave use the pre-specified `lr = 1e-4` noted above.
- **Reaction-diffusion runs through the diffusion branch.** The released entry point
  `scripts/run_reaction_diff_ar_valfix.py` constructs the model with `pde_type='heat'`,
  so the reaction-diffusion data are propagated by the diffusion branch of the spectral
  template, and no reaction coefficient is instantiated or trained in that run. This
  benchmark is best read as a **diffusion spectral template applied to
  reaction-diffusion data**. The general implementation does carry a learnable,
  possibly-zero reaction rate on the reaction-diffusion branch, but this entry point
  does not exercise it.
- For the five benchmarks in this release, the seeds were fixed as part of each
  benchmark's training protocol rather than selected on the basis of observed
  outcomes.

## Parameter-initialization note (`corr_init_scale`)

`PhysicsStructuredPINO`'s spectral convolution accepts a keyword argument named
`corr_init_scale`. **This argument is not read:** it is retained in the signature for call
compatibility only, and the correction scale is set by a *different* argument,
`corr_scale_param` (see the comment at the assignment in
`scripts/physics_fno.py`). Pass `corr_scale_param` to change the correction
initialization. Consequently, any historical sweep that varied `corr_init_scale` did
**not** vary the initialization: arms built with different `corr_init_scale` values at a
fixed seed produce byte-identical initial `state_dict`s. Such a sweep is not
initialization-sensitivity evidence, and the manuscript does not present it as such.

**Semantics at the released configuration.** An earlier revision of `scripts/physics_fno.py`
described `corr_init_scale` as setting the correction magnitude. That description was
**incorrect**; the docstring now states that the argument is not read and points to
`corr_scale_param`. Under the configuration used by the released entry points, the
module's executable behaviour is that of the archived original which produced the
reported results: the correction-initialization constant and the training-loop batch
rule are the reported values, and `corr_scale_param` is retained only as a hook that no
entry point passes. The two statements below remain the authoritative description.

**Actual correction scales in the released code.** `corr_scale_param` is passed by none
of the five benchmark entry points (only the gate-control scripts pass it). With the
default `corr_scale_param=None` the correction is drawn at `0.01x` the base weight scale
in the 2D class `PhysicsStructuredSpectralConv2d` (measured ratio `corr/base = 0.0100`;
the value used for the reported 2D results), at `0.01x` in the 3D multi-seed class `MultiFieldSpectralConv3d`
(measured `0.0100`), and at `1/(c*r)` in the 3D out-of-distribution class
`SingleFieldSpectralConv3d` (measured `8.007x`).


## Environment used for the reported results

**This is a description of the environment in which the reported numbers were produced, not a
pinned environment specification.** `requirements.txt` lists the packages with loose lower
bounds; it does not pin exact versions or a full transitive dependency set, so it is not
sufficient to reconstruct a byte-identical environment, and the reproducibility it supports is
at the level of the pipeline rather than of the numerical outputs. The versions actually used
are recorded below for reference. No dependency upgrade is required to read the archived
records.

| Component | Version |
|---|---|
| Python | 3.9.23 (Anaconda) |
| PyTorch | 2.5.1+cu121 |
| CUDA (runtime bundled with PyTorch) | 12.1 |
| cuDNN | 9.1.0 |
| GPU | NVIDIA GeForce RTX 4070 **Laptop** GPU, 8 GB VRAM, single device |

The GPU is the 8 GB laptop variant of the RTX 4070 (not the 12 GB desktop
card); the 8 GB figure quoted in the manuscript's implementation section refers
to this device.

## Reproducing the main results

**What the commands below do, and what they do not.** They support two things: **pipeline
execution** — the five entry points run end to end — and **archived-record verification** — the
archived per-seed records shipped with this release for the five benchmarks can be checked as shipped, and the
cross-benchmark statistics can be re-derived from them, with no training at all. They do **not** reproduce every reported number value-for-value from a
clean checkout: the training runs are stochastic, and two of the five pipelines (heat and
reaction-diffusion) resume from a saved SS-stage checkpoint that is too large to host and is
**not** in this repository. See "What this package supports" and "Scope of the reproduction
claim" above.

From a Python environment with the packages in `requirements.txt`:

```bash
# Run every command from the repository root.
# Each of the five benchmark entry points listed below protects its archived result file:
# three of them have that record as their default destination and stop rather than overwrite
# it, whereas the convection-diffusion and damped-wave entry points default to a path that is
# not an archived record, where a first run writes a new file inside the package
# (REPRODUCIBILITY_STATUS_MATRIX.md, Sections 4(b) and 4(c)). The forms below write the run to
# runs/<benchmark>/ and leave the archive as it is.
# 2D diffusion / heat (SS + symmetric AR grid search, validation selection)
python scripts/run_heat_ar_gridsearch.py --output-dir runs/heat
# 2D convection-diffusion (single seed; AR checkpoint selected on training worst-case)
python scripts/run_conv_diff.py --output-dir runs/conv_diff
# 2D reaction-diffusion (AR, val-fix protocol; fixed lr=3e-5, patience=80;
# runs through the heat/diffusion branch, pde_type='heat')
python scripts/run_reaction_diff_ar_valfix.py --output-dir runs/react_diff
# 2D damped wave (SS + AR; single seed)
python scripts/run_wave_with_ar.py --output-dir runs/wave
# 3D turbine LTNE (ID + OOD, three seeds) -- either of these two forms works:
python 3d/run_3d_multi_seed.py --output-dir runs/3d
#   cd 3d && python run_3d_multi_seed.py --output-dir ../runs/3d
# Post-processing on the archived records -- NOT one of the five entry points above.
# It has no --help and no --output-dir, and any invocation rewrites
# results/friedman_results.json in place. Run it inside a copy of the repository.
python scripts/run_statistical_tests.py
```

Heat and reaction-diffusion resume from a saved SS-stage checkpoint (see the
`reproducibility.md` checkpoint note); convection-diffusion, wave, and 3D train from
scratch in one process. The five entry points write their per-seed JSON under the output root — `results/` by default, or the
`--output-dir` given above — and the path is script-relative, so the `python scripts/...`
form above is required (the scripts are not discoverable by bare filename from the repository root). The archived per-seed JSON for the five main
benchmarks is in `results/`. That archived JSON is included in this
repository, so `run_statistical_tests.py` can be run with no training — but see the
warning below before running it.

**`scripts/run_statistical_tests.py` is a post-processing script, not one of the five
entry points.** It does not go through the shared command line: it accepts neither a real
`--help` (it ignores the flag and executes the full pass) nor `--output-dir`, and it
**overwrites `results/friedman_results.json` in place**, replacing the archived copy of
that file. The supported way to run it is inside a **copy** of the repository, so the
shipped archive is never the thing being rewritten:

```bash
cp -r code_release psfno_scratch
cd psfno_scratch && python scripts/run_statistical_tests.py
```

The shell examples in this document are POSIX shell (`bash`), and are not a statement that
they work unchanged in every shell. On Windows PowerShell the two copy-based recipes above
and below become

```powershell
Copy-Item -Recurse code_release psfno_scratch
Set-Location psfno_scratch; python scripts/run_statistical_tests.py
```

The `python …` invocations themselves are identical in both shells; only the copy, change-
directory and diff commands differ.

**Output-path scope, and what it does not cover.** In the five benchmark entry points listed
above, the artifacts the shared command line (`_release_cli.py`) routes — the result JSON, the
intermediate `*_partial.json`, the figures and the checkpoint directory — are resolved from a
single **output root**, which defaults to the script's own `results/` directory. `--output-dir D`
moves those CLI-managed artifacts off the package and into `D`. This is why the commands above
invoke the scripts as `python scripts/<name>.py` from the repository root rather than by bare
filename.

That scope has two known exceptions, both visible in this repository:

- **An import-time write inside the package.** `scripts/run_wave_experiment.py` creates
  `<repo>/results/results_wave/` in a **module-level** statement (lines 41-42:
  `OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "results",
  "results_wave")`, followed by `os.makedirs`). The wave entry point imports this module, so a
  wave run aimed elsewhere with `--output-dir` still creates that directory inside the package.
- **Auxiliary scripts.** `scripts/fno.py`, `scripts/irregular_domain.py`,
  `scripts/physics_fno.py`, `scripts/solve_wave.py`, `scripts/train_v10_clean.py`,
  `scripts/run_wave_experiment.py`, `3d/run_ood_experiment.py` and
  `3d/train_causal_pino_3d.py` carry `__main__` or demo paths that write inside the package.
  They are not routed through the shared command line and are not covered by the scope above.

Because of this, the shared command line re-walks the package after a run and **reports** any
file that appeared inside it: an unintended in-package write is meant to be **observable**,
rather than assumed away. The repository is intended to be self-contained for
the data-generation scripts, which are deterministic given the fixed seeds above; the
training runs (and therefore the reproduced numbers) are stochastic and depend on the
environment.

**`--tiny` (CPU self-test).** Every entry point accepts `--tiny`, a CPU pipeline
self-test preset — a few trajectories, three time steps, two epochs. It drives the
whole path (data generation, training, checkpoint save and reload, evaluation, result
JSON) and is the quickest way to check an installation without a GPU. It is far too
short to reproduce anything: it produces **no manuscript value**, and the result files
it writes carry a top-level `"_preset"` stamp recording that, so a self-test output
cannot be mistaken for an archived record.

## Baseline tuning disclosure (per the Action Editor request)

The **selected** configurations of the external baselines, the search protocols that
produced them, and the outcomes archived for those selected configurations are
disclosed as follows: the search protocols and the searched values are described in the
Supplementary Material (Sections S1, S14, S15, S17), and the selected configurations'
outcomes are archived in `results/`. The full tuning grid, the per-configuration
validation values, and the external-baseline search harness are **not** part of this
release (each entry below states what is and is not archived):

- **POD-RBF**: validation-selected search over `n_modes ∈ {10,20,30,50,70,100} × kernel
  {thin-plate spline, multiquadric, Gaussian, linear}` (two kernel families failed to fit
  and were excluded). The reported configuration (50 modes, thin-plate spline) was
  confirmed best. The archived `results/podrbf_deterministic_results.json` records the
  POD-RBF results behind the reported rows of Tables 4, 5 and 6 (`m = 50`, thin-plate
  spline) — the full grid, its per-configuration validation values, and the per-seed
  selection are not archived in this repository (the search is described in Supplementary
  Section S15).
- **PINO**: search over the PDE-residual weight `λ_pde ∈ {1e-1,1e-3,1e-4}` and a
  delayed-ramp schedule (`results/pino_final_results.json`; this file likewise records
  **only the best configuration**, not the full search).
- **DeepONet and F-FNO**: a two-stage validation-only hyperparameter search
  (Supplementary Material, "DeepONet and F-FNO hyperparameter search"),
  conducted on the 2D heat benchmark only: stage 1 screens six architectural
  configurations per baseline over 600 epochs on the validation split, and stage 2
  trains the selected configuration for the full 2000-epoch budget (a single test
  evaluation after training). The search confirms the reported configurations:
  DeepONet's 10.6M branch–trunk architecture (heat and convection-diffusion) and
  23.6M matched-parameter variant (reaction-diffusion and wave), and F-FNO's 23.6M
  factorized configuration. Validation selection is used throughout; no test-set
  selection is performed.
- **External-baseline seed labeling**: the `baselines_*.json`,
  `pino_final_results.json` and `podrbf_deterministic_results.json` records are
  single-seed and do not carry a `seed` field, so the specific seed value is treated as
  unknown rather than asserted. What the archived file shows is one result set per
  benchmark; it does not record how many runs produced that record.
  `results/deeponet_3d_eqp_seed42_results.json` is the
  exception and does record `seed = 42`. See `RELEASE_MANIFEST.md` Section 5.
