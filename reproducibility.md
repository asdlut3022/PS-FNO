# Reproduction Notes

This repository accompanies
*Physics-Structured Fourier Neural Operators via Spectral Semigroup Embedding*
(Neural Networks, NEUNET-D-26-02460). It contains the executable main programs for
the five primary benchmarks together with the archived result records shipped with
this release for those benchmarks, the records behind the equal-parameter controls
discussed in the benchmark and ablation sections, the Section 4.10 statistics, and the
two raw cost records behind Table 15 and Supplementary Table S2 (per-step inference
timing, and the external-baseline wall-clock, memory and inference measurements).
Experiments outside the five benchmarks (geometry generalization, resolution transfer,
the kernel/rank/component/training-stage ablations, the information-matched controls,
the training-time and peak-memory
profiling runs behind Table 15, and the S-series
supplementary experiments) are **not** part
of this release — that is a statement of what this package contains. **The
one-page per-artifact statement of this boundary is `REPRODUCIBILITY_STATUS_MATRIX.md`;
the file-by-file mapping is `RELEASE_MANIFEST.md`.**

## Environment used for the reported results

Python 3.9.23 (Anaconda); PyTorch 2.5.1+cu121 (CUDA 12.1 runtime, cuDNN 9.1.0);
single NVIDIA GeForce RTX 4070 **Laptop** GPU with 8 GB VRAM. This is the 8 GB
laptop variant, not the 12 GB desktop card; the 8 GB device is the one referred
to in the manuscript's implementation section. `requirements.txt`
carries loose lower bounds for the packages; the versions above are those used
to produce the archived records. No dependency upgrade is needed to read the
archived JSONs.

## One-shot verification (no GPU training required)

The archived result JSONs for the five main benchmarks are in
`results/`. `RELEASE_MANIFEST.md` Section 2 maps each of the 27 files to the
manuscript location it supports. The damped-wave seed 314 values are in
`results_wave/training_nt5_seed314_777.log` (log-level provenance — no JSON
artifact). The archived records are therefore verifiable without training. **One file
needs a word of explanation** and is described in
`RELEASE_MANIFEST.md` §4: `results/baseline_results.json` holds the Table 4 DeepONet row
together with the untuned-PINO value behind the Table 4 footnote (the table's PINO row is
the tuned configuration, archived in `results/pino_final_results.json`). After

```bash
pip install -r requirements.txt
python scripts/run_statistical_tests.py
```

the printed Wilcoxon / Friedman statistics reproduce the manuscript values
(p = 0.0625, Friedman χ² = 17.5714 p = 0.0035, geometric-mean ratio 1.3301,
Hodges–Lehmann 0.4388) and each per-benchmark mean WorstMax, exactly.
(Note: this overwrites the archived `results/friedman_results.json`; run it on a
copy if the archived file is to be preserved. It also reads
`results_wave/training_nt5_seed314_777.log`, so that file must be present.)

## Training runs (GPU required)

The seed coverage differs by entry point, and it is important to distinguish
the seeds a script trains from the seeds actually reported in the manuscript:

- `run_heat_ar_gridsearch.py` — `SEEDS = [42, 123, 456]`.
- `run_conv_diff.py` — **`SEED = 42` only** (a single-seed entry point). The
  three-seed convection-diffusion results in the manuscript (Table 5) come from
  the archived per-seed JSON `results/conv_diff_multi_seed_results.json`, not
  from this script in one run.
- `run_reaction_diff_ar_valfix.py` — `SEEDS = [42, 123, 456]`. This entry point
  constructs the model with `pde_type='heat'`: the reaction-diffusion data are
  propagated by the diffusion branch of the spectral template, and no reaction
  coefficient is instantiated or trained in this run.
- `run_wave_with_ar.py` — **`SEEDS = [789]` only** (single-seed AR entry point
  that also drives figure generation). The manuscript wave table (Table 7)
  reports seeds `{42, 789, 314}`; seeds 42/789 are in
  `results/wave_ar_results_3seed_backup.json` and seed 314 is log-level
  provenance (`results_wave/training_nt5_seed314_777.log`).
- `run_wave_experiment.py` — `SEEDS = [42, 789, 456]`, **scheduled-sampling
  (SS) only** (no AR stage), and is an older SS loop rather than the final
  benchmark protocol; it is **not** the source of Table 7. It is required as a
  library: `run_wave_with_ar.py` imports `train_wave_scheduled_sampling`,
  `eval_wave_full` and `eval_wave_model` from it.
- `3d/run_3d_multi_seed.py` — `SEEDS = [42, 123, 456]`.
- `run_statistical_tests.py` — reads the archived per-seed JSON above; does not
  train.

A first (cheap) point of reference: a single-seed run of `run_heat_ar_gridsearch.py`
or `run_conv_diff.py` takes ~1–2 h on an RTX 4070; the full three-seed 2D
suite is ~6–10 h, and the 3D LTNE suite several hours per seed. The archived
per-seed JSON in `results/` are the authoritative source for the reported
three-seed means; re-running the entry points produces a new seeded
realization, not byte-identical values.

Order (run each from the repository root; the scripts are script-relative, so the
`scripts/` prefix is required. Add `--output-dir <dir>` to write a fresh run outside the
package — three of the five default to the archived record and stop without it):

1. `python scripts/run_heat_ar_gridsearch.py --output-dir runs/heat`   — diffusion (heat), SS + AR grid
2. `python scripts/run_conv_diff.py --output-dir runs/conv_diff`             — convection-diffusion
3. `python scripts/run_reaction_diff_ar_valfix.py --output-dir runs/react_diff` — reaction-diffusion
4. `python scripts/run_wave_with_ar.py --output-dir runs/wave`          — damped wave
5. `python 3d/run_3d_multi_seed.py --output-dir runs/3d`              — 3D LTNE (ID + OOD)
6. `python scripts/run_statistical_tests.py`     — statistics

**Where each script writes, versus where the table is read from.** The two are not
always the same file. For three benchmarks the entry point's default destination is the very
JSON the manuscript table is read from — that file is the archived record, so the default run
stops rather than write it, and `--force` is required to overwrite it:
`results/heat_ar_gridsearch_results.json`, `results/reaction_diff_ar_valfix_results.json`
and `results/3d_multi_seed_results.json`. For the other two the entry point writes a
*new* file under a *different* name from the archived record:

| Benchmark | The script writes | The table is read from |
|---|---|---|
| convection-diffusion | `results/conv_diff_results.json` | `results/conv_diff_multi_seed_results.json` (three seeds, archived) |
| damped wave | `results/results_wave/wave_ar_results.json` (plus `..._partial.json` and two figures under `results/results_wave/figures/`) | `results/wave_ar_results_3seed_backup.json` (seeds 42, 789) + `results_wave/training_nt5_seed314_777.log` (seed 314) |

`run_statistical_tests.py` likewise **overwrites** `results/friedman_results.json`. Copy
the repository before an exact re-run if the archived copies are to be preserved
(`cp -r code_release psfno_scratch` in POSIX shell; `Copy-Item -Recurse code_release
psfno_scratch` in Windows PowerShell — the shell examples here are POSIX and are not a
cross-platform guarantee). The full
per-artifact table is in `REPRODUCIBILITY_STATUS_MATRIX.md` §2.

**Checkpoint note.** Two of the five pipelines resume from a saved SS-stage
checkpoint because that is how the paper's protocol is defined (the SS-stage
model selection was already done before the AR stage):

- `run_heat_ar_gridsearch.py` expects `checkpoints/phys_heat_valfix_ss_seed{42,123}.pth` and
  `checkpoints/std_heat_valfix_ss_seed{42,123}.pth` (seeds 42/123 from the heat
  validation-fix pipeline), and for seed 456 the non-valfix names
  `checkpoints/phys_heat_ss_seed456.pth` / `checkpoints/std_heat_ss_seed456.pth`
  (the AR grid search then trains from them).
- `run_reaction_diff_ar_valfix.py` expects
  `checkpoints/{std,phys}_reactdiff_ss_seed{42,123,456}.pth`.

These checkpoint files are too large to host (several GB) and are not included in
this repository. If a checkpoint is absent, the entry point trains that SS stage
fresh with the same protocol (2000 epochs, p_teacher 0.9 -> 0.3, warmup 500,
anneal 1500) and uses the result; a checkpoint obtained this way yields a fresh
realization, not the archived per-seed values. The per-seed results that the
manuscript reports from these pipelines are archived in
`results/heat_ar_gridsearch_results.json` (AR stage),
`results/reaction_diff_ar_results.json` (reaction-diffusion SS stage) and
`results/reaction_diff_ar_valfix_results.json` (reaction-diffusion AR stage), so
those published numbers can be checked without the checkpoints.

The remaining pipelines (convection-diffusion, wave, 3D) train from scratch in a
single process and need no pre-existing files.

## 3D external-dependency boundary

`3d/run_3d_multi_seed.py` imports `physics_fno_3d`, `train_causal_pino_3d`
where all data generation is self-contained, and `run_comparison_3d.py` (the
original run used `run_ood_experiment.generate_ood_test`; the copy in `3d/`
is the same module, path-adjusted). `3d/` has no dependency outside this
repository; the full 3D training pipeline and OOD data generation are
included.

## Output-path policy

All `sys.path` insertions point to the script's own directory
(`os.path.dirname(os.path.abspath(__file__))`). In the five benchmark entry points, the
artifacts the shared command line (`_release_cli.py`) routes — the result JSON, the
intermediate `*_partial.json`, the figures and the checkpoint directory — are resolved from a
single **output root** that defaults to the script's own `results/` directory. `--output-dir D`
rebinds that root to `D`, which moves those CLI-managed artifacts outside the package; the
shared command line reports at exit any file that nevertheless appeared inside it.

**Not covered by this policy** — both write inside the package independently of `--output-dir`:

- `scripts/run_wave_experiment.py` creates `<repo>/results/results_wave/` in a **module-level**
  `os.makedirs` (lines 41-42 of that file). `scripts/run_wave_with_ar.py` imports that module,
  so a wave run creates this directory inside the package even when aimed elsewhere with
  `--output-dir`.
- The auxiliary modules whose `__main__`/demo paths write inside the package:
  `scripts/fno.py`, `scripts/irregular_domain.py`, `scripts/physics_fno.py`,
  `scripts/solve_wave.py`, `scripts/train_v10_clean.py`, `scripts/run_wave_experiment.py`,
  `3d/run_ood_experiment.py`, `3d/train_causal_pino_3d.py`.

`scripts/run_statistical_tests.py` is a **post-processing** script rather than an entry point:
it takes no `--output-dir` and no real `--help`, and any invocation rewrites
`results/friedman_results.json` in place, so run it in a copy.
