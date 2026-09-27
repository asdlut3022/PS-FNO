# REPRODUCIBILITY STATUS MATRIX — PS-FNO code release

**Manuscript**: *Physics-Structured Fourier Neural Operators via Spectral Semigroup
Embedding* (Neural Networks, NEUNET-D-26-02460).
**Purpose**: one page that states, per artifact, **what is executable in this release and
what is not**, so that neither a reviewer nor an author has to infer the boundary from
scattered paragraphs. This file is the single scope statement referenced by `README.md`,
`RELEASE_MANIFEST.md` and `reproducibility.md`.

---

## 1. Scope in one paragraph

This release is the **five-benchmark reproducibility package**: it ships the executable main
programs for the five primary benchmarks (Tables 4, 5, 6, 7, 8), the archived per-seed result
records shipped with this release for those benchmarks, the records behind the
equal-parameter controls discussed in the benchmark and ablation sections, the
cross-benchmark statistics of Section 4.10, and the two raw cost records behind
Table 15 and Supplementary Table S2 (`results/inference_benchmark.json`,
`results/baseline_cost_results.json`).
Every other experiment in the manuscript — geometry generalization (Supplementary Section S21), zero-shot
resolution transfer (Table 9 and Supplementary Section S23), the per-layer physics-ratio analysis (Table 10), the
kernel/rank/component/training-stage ablations (Tables 11, 12, 14), the information-matched
controls (Table 13), the training-time and peak-memory profiling runs behind Table 15, and the S-series supplementary
experiments — is **outside this release**: neither its programs nor its result records are
included. The list above defines what this package contains; it is a statement of scope
only.

---

## 2. Executable scope — the five main benchmarks

Legend for column *Reproducible as*:

- **clean re-run** — the script trains from scratch in one process and needs no external file;
  a fresh run yields a new seeded realization, not byte-identical values.
- **checkpoint-gated re-run** — the script resumes an SS-stage checkpoint that is too large to
  host (several GB) and is **not** in this repository. If the checkpoint is absent the script
  trains that SS stage fresh with the same documented protocol, so it still runs end-to-end,
  but the result is a fresh realization, not the archived per-seed values.
- **archived-check** — reported numbers can be verified against the shipped records with no
  training at all. The release provides archived verification records for the benchmark
  values and metrics identified in the per-artifact mapping below; those records are of two
  kinds (per-seed result JSON, and training-log provenance for one damped-wave seed), and
  the mapping names the file and the manuscript location each one supports.

| # | Benchmark (table) | Entry point (run from repo root) | Seeds the entry point **trains** | Seeds the **table** reports | Split (train/val/test) | Script writes | Archived record that the table is read from | Reproducible as |
|---|---|---|---|---|---|---|---|---|
| 1 | 2D diffusion / heat (Table 4) | `python scripts/run_heat_ar_gridsearch.py` | 42, 123, 456 | 42, 123, 456 | 150 / 20 / 20 | `results/heat_ar_gridsearch_results.json` (`--force` needed to write this file) | same file | checkpoint-gated re-run + archived-check |
| 2 | 2D convection-diffusion (Table 5) | `python scripts/run_conv_diff.py` | **42 only** | 42, 123, 456 | 150 / — / 20 | `results/conv_diff_results.json` — **not** a shipped file (see §4b) | `results/conv_diff_multi_seed_results.json` | clean re-run (seed 42 only) + archived-check |
| 3 | 2D reaction-diffusion (Table 6) | `python scripts/run_reaction_diff_ar_valfix.py` | 42, 123, 456 | 42, 123, 456 | 150 / 20 / 20 | `results/reaction_diff_ar_valfix_results.json` (`--force` needed to write this file) | same file (AR rows) + `results/reaction_diff_ar_results.json` (SS rows) | checkpoint-gated re-run + archived-check |
| 4 | 2D damped wave (Table 7) | `python scripts/run_wave_with_ar.py` | **789 only** | 42, 789, 314 | 250 / — / 20 | `results/results_wave/wave_ar_results.json` + `..._partial.json` + two figure files — **not** shipped files (see §4b) | `results/wave_ar_results_3seed_backup.json` (42, 789) + `results_wave/training_nt5_seed314_777.log` (314) | clean re-run (seed 789 only) + archived-check |
| 5 | 3D LTNE turbine (Table 8) | `python 3d/run_3d_multi_seed.py` | 42, 123, 456 | 42, 123, 456 | 150 / — / 20 (ID) + 20 (OOD) | `results/3d_multi_seed_results.json` (`--force` needed to write this file) | same file | clean re-run + archived-check |

A sixth script, `python scripts/run_statistical_tests.py`, produces the Section 4.10 statistics
for exactly these five benchmarks. It reads the archived per-seed JSON and the archived wave log;
it trains nothing, and it takes no command-line arguments. **It overwrites
`results/friedman_results.json`** — including when it is invoked with `--help`.

**Val/test generation offsets** (2D benchmarks): validation `seed+2000`, test `seed+1000`,
training `seed` — the three splits are disjoint. 3D: ID test `seed+1000`, OOD test `seed+2000`.

**Selection protocol.** 2D diffusion and reaction-diffusion select the AR configuration /
checkpoint on the **validation** split. Convection-diffusion and damped wave have **no
validation split** and select on **training** worst-case error (the protocol stated in the
manuscript). 3D LTNE uses scheduled sampling only. The test split is used for reporting only:
it is evaluated at each reported stage, never for selection.

**Reaction-diffusion branch.** The reaction-diffusion entry point constructs the model with
`pde_type='heat'`: its data are propagated by the diffusion branch of the spectral template,
and no reaction coefficient is instantiated or trained. The benchmark is a diffusion
spectral template applied to reaction-diffusion data.

---

## 3. Out of scope — auxiliary experiments (not in this release)

Programs **and** result records both absent:

| Manuscript location | Experiment |
|---|---|
| Supplementary S21 | geometry generalization |
| Table 9 and Supplementary S23 | zero-shot resolution transfer (2D, 3D) |
| Table 10 | per-layer physics-ratio analysis |
| Table 11 | kernel ablation |
| Table 12 | component ablation |
| Table 13 | information-matched controls (2D and 3D) |
| Table 14 | training-stage ablation |
| Table 15 | the training-time and peak-memory profiling runs — their programs and their measurements are both absent; what *is* shipped from the cost material is the per-step inference record (`results/inference_benchmark.json`) and the external-baseline cost record of Supplementary Table S2 (`results/baseline_cost_results.json`) |
| S2, S3, S5, S6, S8, S9, S12, S13, S16, S18–S20, S22 | long-horizon temporal extrapolation, physical-parameter noise robustness, resolution scalability, effective correction rank analysis, data efficiency, 1D Burgers benchmark, kernel mismatch ablation, proofs of the main-text results, numerical verification of Proposition 2, wave InfoMatched control, InfoMatched PINO control, wave kernel template error, empirical checks of the main-text theorem's and proposition's premises |

These are reported in the manuscript from the authors' own runs. Their absence here is the
division of material described above and nothing more.

---

## 4. Known asymmetries and pitfalls (read before verifying)

**(a) Two entry points are single-seed.** `run_conv_diff.py` trains seed 42 only;
`run_wave_with_ar.py` trains seed 789 only. The three-seed figures in Tables 5 and 7 come from
the archived per-seed JSON, not from one run of those entry points.

**(b) The script's output file is not always the archived file.** For convection-diffusion and
damped wave, the script writes a *new* file under a *different* name in a *different* location
from the archived record the table is read from:

| Benchmark | Script writes | Table is read from |
|---|---|---|
| convection-diffusion | `results/conv_diff_results.json` | `results/conv_diff_multi_seed_results.json` (3 seeds, archived) |
| damped wave | `results/results_wave/wave_ar_results.json` | `results/wave_ar_results_3seed_backup.json` (2 seeds) + the seed-314 log |

The archived file is **never** written by the entry point. A run of the entry point therefore
does not modify the archived record for these two benchmarks.

**(c) Where a re-run writes, and how the archive is protected.** In the five benchmark entry
points, the artifacts the shared command line (`_release_cli.py`) routes — the result JSON, the
intermediate `*_partial.json`, the figures and the checkpoint directory — are resolved from a
single **output root**. That root defaults to the location of the archived record and is
**script-relative**, so a copy of the repository isolates those CLI-managed artifacts; they are
not resolved from the process working directory.

This scope has **two known exceptions**, both of which write inside the package independently of
`--output-dir`:

- `scripts/run_wave_experiment.py` performs a **module-level** `os.makedirs` on
  `<repo>/results/results_wave/` (lines 41-42 of that file). `scripts/run_wave_with_ar.py`
  imports this module, so **every wave run creates that directory inside the package**, even
  when the run is aimed elsewhere with `--output-dir`.
- Eight auxiliary modules — `scripts/fno.py`, `scripts/irregular_domain.py`,
  `scripts/physics_fno.py`, `scripts/solve_wave.py`, `scripts/train_v10_clean.py`,
  `scripts/run_wave_experiment.py`, `3d/run_ood_experiment.py`, `3d/train_causal_pino_3d.py` —
  carry `__main__` or demo paths that write inside the package. They are not routed through the
  shared command line.

The shared command line therefore re-walks the package after a run and **reports** any file that
appeared inside it, so an unintended in-package write is **observable** rather than assumed
absent.

In addition, the five training entry
points share one command line (`_release_cli.py`) and **refuse by default to overwrite a file
that already exists in the shipped `results/`** — the run stops before any training with
`REFUSING to overwrite the archived result file ... Pass --output-dir <dir> to write
elsewhere, or --force to overwrite it.`

| Entry point | Default destination | Reads the same file first? | Effect on the shipped archive |
|---|---|---|---|
| `run_conv_diff.py` | `results/conv_diff_results.json` (**not** a shipped record) | no | a first run writes a **new** file at that path; later runs refuse without `--force`. No shipped record is overwritten |
| `run_heat_ar_gridsearch.py` | `results/heat_ar_gridsearch_results.json` | **yes** (resume store) | refuses without `--force`; with `--output-dir D` the run writes a **new** realization to `D`, leaving the archive untouched. It does **not** resume from the archived per-seed blocks: rebinding the destination also rebinds the resume store and the SS-stage checkpoints, both of which are then looked for under `D`, so the run trains fresh unless `--ckpt-dir` names an existing checkpoint set |
| `run_reaction_diff_ar_valfix.py` | `results/reaction_diff_ar_valfix_results.json` | no | refuses without `--force` |
| `run_wave_with_ar.py` | `results/results_wave/wave_ar_results.json` + `..._partial.json` + two figures under `results/results_wave/figures/` (**not** shipped records) | no | a first run writes **new** files at those paths; later runs refuse without `--force`. No shipped record is overwritten |
| `3d/run_3d_multi_seed.py` | `results/3d_multi_seed_results.json` | no | refuses without `--force` |
| `run_statistical_tests.py` **(post-processing; not one of the five entry points)** | `results/friedman_results.json` | no | **overwrites outright** — it has no `--output-dir` and no `--help`: any invocation, including `--help`, runs the full pass and rewrites the file. Use a copy (recipe below). Re-running it reproduces every numeric and provenance field; the single divergence from the shipped copy is the descriptive `cell_policy` string, which the archive took from an earlier revision of the script (see `RELEASE_MANIFEST.md` §5) |

**Running the package without disturbing the shipped archive.** In each of the five entry
points, the CLI-managed output paths are resolved from the output root rather than from the
working directory, and `--output-dir D` moves the result JSON, the partial file, the figures
**and** the checkpoints into `D`. For those five, either of these therefore keeps the shipped
archive untouched — run inside a **copy** of the repository, or point `--output-dir` outside
the package with no copy at all. This does **not** cover the import-time write or the auxiliary
scripts listed in (c) above, which create files inside the package regardless of
`--output-dir`; the package-watch report at exit is what makes those visible:

```bash
cp -r code_release psfno_scratch                       # one-time
cd psfno_scratch && python scripts/run_conv_diff.py    # and so on

# or, without a copy:
python scripts/run_conv_diff.py --output-dir /tmp/out  # shipped results/ untouched
```

If neither is wanted, snapshot first and diff afterwards:

```bash
cp -r results results_orig                # before any run
diff -rq results results_orig             # after
```

These examples are POSIX shell (`bash`), not a cross-platform guarantee. Windows PowerShell
equivalents:

```powershell
Copy-Item -Recurse code_release psfno_scratch          # one-time
Set-Location psfno_scratch; python scripts/run_conv_diff.py

# or, without a copy:
python scripts/run_conv_diff.py --output-dir C:\out    # shipped results/ untouched

# snapshot first, compare afterwards
Copy-Item -Recurse results results_orig
Compare-Object (Get-ChildItem -Recurse results | Get-FileHash) `
               (Get-ChildItem -Recurse results_orig | Get-FileHash)
```

Only the copy / change-directory / diff commands differ between shells; the `python …`
invocations are identical.

Do not run `run_statistical_tests.py` directly in an unpacked archive you intend to keep.

**(d) Fresh-SS checkpoint naming (fixed in both checkpoint-gated pipelines).** When no SS
checkpoint is present, the entry point trains the SS stage fresh. Earlier revisions saved it
under a `_gridnew_` name while looking it up under the plain name — `run_reaction_diff_ar_valfix.py`
wrote `{arm}_reactdiff_gridnew_ss_seed{seed}.pth` but looked for `{arm}_reactdiff_ss_seed{seed}.pth`,
and `run_heat_ar_gridsearch.py` wrote `{arm}_heat_gridnew_ss_seed{seed}.pth` but looked for
`{arm}_heat_ss_seed{seed}.pth` (or the `_valfix_` name for seeds 42/123). A checkpoint written by
one run was therefore invisible to the next, and the SS stage was retrained every time. Both
scripts now look up the plain name first, accept the legacy `_gridnew_` name as a fallback, and
**write the fresh checkpoint under the plain name they looked for**. The `_valfix_` name is never
written by the fallback, so a fresh-protocol checkpoint cannot be mistaken for the validation-fix
pipeline's checkpoint that the reported numbers use. The training protocol is unchanged. This
affects wall-clock only and no archived value; the earlier revision is described in git history
rather than silently rewritten.

**(e) The 3D entry point trains exactly the seeds it is given.** `3d/run_3d_multi_seed.py` has no
path that reads a previously computed result file back in: every seed in `SEEDS`, or in
`--seed`/`--seeds`, is trained fresh, and each per-seed block of
`results/3d_multi_seed_results.json` comes from that run. The module header states the same.

**(f) The damped-wave third seed is log-level provenance.** Seed 314 has no JSON artifact; its
two values live in `results_wave/training_nt5_seed314_777.log` (lines 156–158) and are parsed by
`run_statistical_tests.py`, which asserts they equal 0.1674 / 0.1026 and fails loudly otherwise.

**(g) One file holds two different quantities.** `results/baseline_results.json` carries the
Table 4 DeepONet row (2.171) *and* the untuned-PINO value behind the Table 4 footnote (76.885).
The table's PINO row is the *tuned* configuration (1.648) in `results/pino_final_results.json`.
These are two configurations, not two versions of one cell. See `RELEASE_MANIFEST.md` §4.

**(h) First-path filenames are legacy.** `results/baselines_heat_pino_podrbf_results.json` is
named for both baselines but holds **only the POD-RBF block**; the PINO block was removed.
`results/podrbf_deterministic_results.json` holds the deterministic POD-RBF records behind the
reported rows of Tables 4, 5 and 6. This is stated in `RELEASE_MANIFEST.md` §4 note 2.

---

## 5. What this release does **not** claim

- It does **not** claim that every manuscript table can be recomputed value-for-value from a
  clean checkout. Two pipelines are checkpoint-gated; two entry points are single-seed; training
  is stochastic. The Data availability statement describes this two-tier structure.
- The release includes archived result records for the five primary benchmarks; for each
  shipped record `RELEASE_MANIFEST.md` §2 identifies the file and the manuscript location it
  supports.
- It makes **no** claim about training-time efficiency (see the manuscript's cost discussion).

---

## 6. Verification status of this package

| Check | Method | Result |
|---|---|---|
| Every `.py` parses | `ast.parse` over all 19 release Python files (including `_release_cli.py` and the five in `3d/`) | pass |
| Every top-level import resolves | every `Import` / `ImportFrom` module name in those files, checked against the environment | 0 unresolved |
| One complete non-training pipeline executes | `python scripts/run_statistical_tests.py`, run **inside a copy** of the repository | exit 0; reproduces the four Section 4.10 statistics (0.0625, 17.5714, 1.3301, 0.4388) |
| The copy recipe in §4c isolates the archive | the same run rewrote the copy's `results/friedman_results.json` (hash changed) while the original archive stayed byte-identical | confirmed |
| The five entry points execute end-to-end | each of the five, on CPU with a tiny data/epoch preset (`--tiny`), inside a copy: real data generation → short training → checkpoint save → load → evaluation → result JSON | exit 0 for all five; a result JSON was written in each case |
| The two checkpoint-gated pipelines reuse a freshly trained SS checkpoint | each pipeline run twice in the same copy: first run trains the SS stage fresh, second run must find it | both second runs found the checkpoint and did **not** retrain the SS stage |
| A run cannot silently overwrite the shipped archive | `run_heat_ar_gridsearch.py` run in a copy with no `--output-dir`, where the archived JSON exists | exits 1 before training with `REFUSING to overwrite the archived result file …`; the same command with `--output-dir` runs to completion |
| The archive stays untouched across all of the above | SHA-256 of all 28 archived files before and after the whole battery | 0 files changed |
| Documentation and code behavior agree | `--help` on the five entry points and on the post-processing script; **the output-isolation scope stated in the four `.md` files checked against the code** — the five entry points' CLI-managed paths, plus the two named exceptions (`run_wave_experiment.py`'s module-level `os.makedirs`, and the eight auxiliary modules), which are documented rather than claimed away; **no text in these files asserts a package-wide output guarantee**; every script path quoted in the four `.md` files resolved; `RELEASE_MANIFEST.md` §2 cross-checked against `results/` on disk — all run inside a temporary copy | all pass |
| That documentation check can itself fail | the pre-fix `_release_cli.py` path bug injected into the copy: the check must go red, and green again after a byte-for-byte restore | red with the bug present, green after restore |

**What the CPU runs are, and are not.** They use `--tiny` (a few trajectories, three time
steps, two epochs) and an 8-way-shrunk 3D grid. They establish that each entry point runs,
saves, reloads, evaluates and writes a result record through the real code path — not that any
manuscript number is reproduced, and no manuscript number is produced by them. Because a
self-test writes the same file name and the same per-arm schema as an archived record, every
result file produced under the preset also carries a top-level `"_preset"` stamp, and the
preset prints a notice when it starts: the output identifies itself, so it does not rely on a
reader having seen this paragraph.

**Not verified.** The five entry points have **not** been executed at the manuscript's own
training budget (2000 SS / 300–800 AR epochs) or on a GPU in this package's own history; that
is what a clean-checkout reproduction would require, and it is why §5 distinguishes
"archived record" from "reproduced from a clean checkout". The two checkpoint-gated pipelines
additionally require the unhosted SS-stage checkpoints to reach the reported values, or they
train a fresh realization instead.

---

## 7. Environment used for the reported results

Python 3.9.23 (Anaconda); PyTorch 2.5.1+cu121 (CUDA 12.1 runtime, cuDNN 9.1.0); single NVIDIA
GeForce RTX 4070 **Laptop** GPU, 8 GB VRAM. `requirements.txt` carries loose lower bounds. No
dependency upgrade is needed to read the archived records.
