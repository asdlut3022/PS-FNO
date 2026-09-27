"""
Statistical significance testing for PS-FNO vs Standard FNO across 5 benchmarks.
Written for Neural Networks major revision (AE requirement: "Statistical tests
such as Wilcoxon, Friedman, etc. can be included.")

Two complementary analyses, following standard practice for comparing two
methods across multiple datasets/tasks (Demsar 2006, JMLR):

1. PRIMARY: Cross-benchmark paired Wilcoxon signed-rank test. Each of the 5
   benchmarks contributes one paired observation (Standard-FNO mean vs
   PS-FNO mean worst-case error, 3-seed average per benchmark). This is the
   standard way to test "does the physics kernel help systematically across
   tasks" and has adequate statistical power (n=5).

2. SECONDARY (supplementary, per-benchmark): paired Wilcoxon within each
   benchmark's 3 seeds. Reported for transparency, but flagged: with n=3,
   the minimum achievable two-sided p-value for a paired Wilcoxon
   signed-rank test is 0.25 (2^-3 * 2), so these can NEVER reach the
   conventional p<0.05 threshold regardless of effect size. We report them
   honestly with this caveat rather than omit them or overstate them.

Data sources (worst-case error, Physics FNO+SS+AR vs Standard FNO+SS+AR,
2D main-table configuration, post val-split-fix where available; all
per-seed numbers are read from JSON files for 42/789 and from the archived
training log for seed 314 -- see the wave note below, no hardcoded result
values are used):
  - diffusion (heat):     heat_ar_gridsearch_results.json (symmetric grid
                          search, seeds {42,123,456}, val-selection, test once)
  - conv-diff:            conv_diff_multi_seed_results.json
  - reaction-diff:        reaction_diff_ar_valfix_results.json (valfix-caliber
                          AR, seeds {42,123,456}, val-selection; honest caliber
                          for the main paper, supersedes old 2.241/1.852)
  - damped wave:          results/wave_ar_results_3seed_backup.json
                          (seeds {42,789}; seed 314 from the training log at
                          results_wave/training_nt5_seed314_777.log)
  - 3D LTNE (OOD, no AR): three seeds JSON (results/3d_multi_seed_results.json)
"""
import json
import os
import numpy as np
from scipy import stats

# ------------------------------------------------------------------
# Per-seed worst-case error, Standard vs Physics -- ALL from JSON files
# ------------------------------------------------------------------

benchmarks = {}
sources = {}  # per-benchmark JSON path + key provenance, for the report

# --- conv-diff (no test-set leakage in AR selection; train-based selection) ---
P1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "results", "conv_diff_multi_seed_results.json")
with open(P1) as f:
    d = json.load(f)
std = [d[f'seed_{s}']['standard_ss_ar']['worst_max'] for s in [42, 123, 456]]
phys = [d[f'seed_{s}']['physics_ss_ar']['worst_max'] for s in [42, 123, 456]]
benchmarks['conv-diff'] = {'standard': std, 'physics': phys}
sources['conv-diff'] = [f'{P1} -> seed_{s}.{{standard,physics}}_ss_ar.worst_max'
                        for s in [42, 123, 456]]

# --- heat (diffusion): symmetric grid search (val-selected, test-once) ---
P2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "results", "heat_ar_gridsearch_results.json")
with open(P2) as f:
    d = json.load(f)
std = [d['standard'][str(s)]['final_test']['worst_max'] for s in [42, 123, 456]]
phys = [d['physics'][str(s)]['final_test']['worst_max'] for s in [42, 123, 456]]
benchmarks['diffusion-heat'] = {'standard': std, 'physics': phys}
sources['diffusion-heat'] = [f'{P2} -> {m}.{s}.final_test.worst_max'
                             for m in ['standard', 'physics'] for s in [42, 123, 456]]

# --- reaction-diff: valfix-caliber AR (val-selected, test-once), main-paper caliber ---
P3 = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "results", "reaction_diff_ar_valfix_results.json")
with open(P3) as f:
    d = json.load(f)
std = [d['standard_ar'][str(s)]['worst_max'] for s in [42, 123, 456]]
phys = [d['physics_ar'][str(s)]['worst_max'] for s in [42, 123, 456]]
benchmarks['reaction-diff'] = {'standard': std, 'physics': phys}
sources['reaction-diff'] = [f'{P3} -> {m}_ar.{s}.worst_max'
                            for m in ['standard', 'physics'] for s in [42, 123, 456]]

# --- damped wave (paper values use seeds {42,789,314}) ---
# 42/789 read from the backup JSON (verified identical to paper, seed-by-seed);
# 314 has no JSON artifact -- its pair (std 0.1674 / phys 0.1026) is documented
# in results_wave/training_nt5_seed314_777.log L156/L158 (log-level provenance),
# and the log is archived in this repository so the provenance is inspectable.
# The parse below reads the values from the log; the constants 0.1674/0.1026
# are kept ONLY as a consistency assertion (they are not the result source --
# the JSON/log paths are -- and they merely guard against an unexpected change
# in the archived log output).
P4 = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "results", "wave_ar_results_3seed_backup.json")
with open(P4) as f:
    d = json.load(f)
P4_log = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "results_wave", "training_nt5_seed314_777.log")
with open(P4_log) as f:
    log_text = f.read()
std_314 = 0.1674
phys_314 = 0.1026
_ok_std = False
_ok_phys = False
# The log contains TWO summary blocks: a seed-314 block (lines ~155--158,
# "  Std SS+AR: ..." with a single space before the colon) and a THREE-SEED
# block (lines ~319--323, "  Std SS+AR   : ..." with extra spaces before the
# colon). The paper value is the seed-314 block, so we must match the single-
# space format and therefore take the FIRST occurrence of each line kind.
for line in log_text.splitlines():
    if 'Std SS+AR' in line and 'WorstMax=' in line and not line.strip().startswith('Std SS+AR   '):
        std_314 = float(line.split('WorstMax=')[-1].strip())
        _ok_std = True
        break
for line in log_text.splitlines():
    if 'Phys SS+AR' in line and 'WorstMax=' in line and not line.strip().startswith('Phys SS+AR   '):
        phys_314 = float(line.split('WorstMax=')[-1].strip())
        _ok_phys = True
        break
if not (_ok_std and _ok_phys):
    raise RuntimeError("seed-314 log parse failed; expected 'Std SS+AR' / 'Phys SS+AR' with WorstMax= in results_wave/training_nt5_seed314_777.log")
if abs(std_314 - 0.1674) > 1e-6 or abs(phys_314 - 0.1026) > 1e-6:
    raise RuntimeError(f"seed-314 log values mismatch: got std={std_314}, phys={phys_314}; expected 0.1674 / 0.1026 (paper values)")
std = [d['standard_ss_ar']['42']['worst_max'], d['standard_ss_ar']['789']['worst_max'],
       std_314]  # seed 314, parsed from results_wave/training_nt5_seed314_777.log
phys = [d['physics_ss_ar']['42']['worst_max'], d['physics_ss_ar']['789']['worst_max'],
        phys_314]  # seed 314, parsed from results_wave/training_nt5_seed314_777.log
benchmarks['damped-wave'] = {'standard': std, 'physics': phys}
sources['damped-wave'] = [
    f'{P4} -> {{standard,physics}}_ss_ar.42.worst_max (verified == paper)',
    f'{P4} -> {{standard,physics}}_ss_ar.789.worst_max (verified == paper)',
    f'{P4_log} -> Std/Phys SS+AR WorstMax (seed 314, log-level provenance, '
    'no JSON artifact; log archived in this repository)',
]

# --- 3D LTNE (OOD worst-case; no AR stage, SS-only, no leakage) ---
P5 = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "results", "3d_multi_seed_results.json")
with open(P5) as f:
    d = json.load(f)
seeds_3d = list(d['per_seed'].keys())
std = [d['per_seed'][s]['standard']['ood']['worst_max'] for s in seeds_3d]
phys = [d['per_seed'][s]['physics']['ood']['worst_max'] for s in seeds_3d]
benchmarks['3D-LTNE-OOD'] = {'standard': std, 'physics': phys}
sources['3D-LTNE-OOD'] = [f'{P5} -> per_seed.{s}.{{standard,physics}}.ood.worst_max'
                          for s in seeds_3d]

# ------------------------------------------------------------------
# Per-benchmark (secondary, n=3) paired Wilcoxon signed-rank test
# ------------------------------------------------------------------
print('\n' + '=' * 70)
print('SECONDARY: Per-benchmark paired Wilcoxon signed-rank test (n=3 seeds)')
print('NOTE: minimum achievable two-sided p-value at n=3 is 0.25 regardless')
print('of effect size -- these tests are UNDERPOWERED by construction and')
print('are reported for transparency only, not as evidence of significance.')
print('=' * 70)

per_benchmark_rows = []
for name, v in benchmarks.items():
    std, phys = v['standard'], v['physics']
    diffs = np.array(std) - np.array(phys)
    n_better = int(np.sum(diffs > 0))
    try:
        res = stats.wilcoxon(std, phys)
        pval = res.pvalue
    except ValueError as e:
        pval = float('nan')
    mean_std, mean_phys = np.mean(std), np.mean(phys)
    rel_improve = (mean_std - mean_phys) / mean_std * 100
    print(f'{name:20s}: Std={mean_std:.4f} Phys={mean_phys:.4f} '
          f'improve={rel_improve:+.1f}%  physics-better-in {n_better}/{len(std)} seeds  '
          f'Wilcoxon p={pval:.4f}')
    per_benchmark_rows.append((name, mean_std, mean_phys, rel_improve, n_better, len(std), pval))

# ------------------------------------------------------------------
# PRIMARY: Cross-benchmark paired Wilcoxon signed-rank test (n=benchmarks)
# ------------------------------------------------------------------
print('\n' + '=' * 70)
print('PRIMARY: Cross-benchmark paired Wilcoxon signed-rank test')
print('Each benchmark contributes one paired observation (3-seed mean).')
print('=' * 70)

bench_names = list(benchmarks.keys())
std_means = np.array([np.mean(benchmarks[b]['standard']) for b in bench_names])
phys_means = np.array([np.mean(benchmarks[b]['physics']) for b in bench_names])

print(f'n = {len(bench_names)} benchmarks: {bench_names}')
print(f'Standard FNO means: {std_means}')
print(f'Physics FNO means:  {phys_means}')
n_better = int(np.sum(std_means > phys_means))
print(f'Physics FNO better in {n_better}/{len(bench_names)} benchmarks')

if len(bench_names) >= 4:
    res = stats.wilcoxon(std_means, phys_means)
    print(f'Wilcoxon signed-rank: statistic={res.statistic}, p={res.pvalue:.4f}')
else:
    print('n<4: Wilcoxon signed-rank cannot achieve p<0.05 even in the best '
          'case; reporting paired t-test as a supplementary check only.')

res_t = stats.ttest_rel(std_means, phys_means)
print(f'Paired t-test (supplementary): t={res_t.statistic:.3f}, p={res_t.pvalue:.4f}')

# ------------------------------------------------------------------
# Effect sizes & sensitivity
# ------------------------------------------------------------------
print('\n' + '=' * 70)
print('EFFECT SIZES & SENSITIVITY')
print('=' * 70)

# Geometric-mean error ratio across benchmarks (ratio >1 => physics better)
ratio = np.exp(np.mean(np.log(std_means / phys_means)))
pct_std_higher = (ratio - 1) * 100
pct_phys_lower = (1 - 1 / ratio) * 100
print(f'Geometric-mean error ratio (std/phys): {ratio:.4f}')
print(f'  => Standard FNO errors are {pct_std_higher:.1f}% higher '
      f'(geometric-mean sense; dimensionless)')
print(f'  <=> equivalently, Physics FNO errors are {pct_phys_lower:.1f}% lower')
log_ratios = np.log(std_means / phys_means)
print(f'Paired log-ratio: median={np.median(log_ratios):.4f} '
      f'(exp={np.exp(np.median(log_ratios)):.4f}); '
      f'mean={np.mean(log_ratios):.4f} (exp={np.exp(np.mean(log_ratios)):.4f})')

# Hodges-Lehmann estimator: median of pairwise diffs (std - phys).
# NOTE: the five benchmarks are not all in Kelvin
# (wave is normalized displacement), so the cross-benchmark absolute difference
# is reported in benchmark-native error units and is kept only as a footnote,
# NOT as a primary effect size and NOT with a K unit.
all_diffs = np.array([])
for b in bench_names:
    sd = np.array(benchmarks[b]['standard'])
    ph = np.array(benchmarks[b]['physics'])
    all_diffs = np.concatenate([all_diffs, sd - ph])
pairwise = np.array([(all_diffs[i] + all_diffs[j]) / 2
                     for i in range(len(all_diffs))
                     for j in range(i, len(all_diffs))])
hl = float(np.median(pairwise))
print(f'Hodges-Lehmann estimator of (std - phys): {hl:.4f} '
      f'benchmark-native error units (footnote only; NO K unit -- the wave '
      f'benchmark is normalized displacement, so the cross-benchmark '
      f'absolute difference is not in Kelvin)')

# Leave-one-benchmark-out sensitivity of the primary Wilcoxon
print('\nLeave-one-benchmark-out sensitivity (primary paired Wilcoxon):')
for b in bench_names:
    idx = [i for i, x in enumerate(bench_names) if x != b]
    sm = std_means[idx]
    pm = phys_means[idx]
    try:
        p_loo = stats.wilcoxon(sm, pm).pvalue
    except ValueError:
        p_loo = float('nan')
    print(f'  drop {b:20s}: n={len(idx)} p={p_loo:.4f}')

# Provenance: every number -> JSON path + key (seed 314 from the log only)
print('\n' + '=' * 70)
print('PROVENANCE (JSON paths; seed 314 from results_wave/training_nt5_seed314_777.log)')
print('=' * 70)
for b in bench_names:
    print(f'[{b}]')
    for src in sources[b]:
        print(f'  {src}')

print('\nDONE. Wave entry uses paper data {42,789,314}: 42/789 from backup '
      'JSON (verified), 314 parsed from results_wave/training_nt5_seed314_777.log '
      '(log-level provenance, no JSON artifact; log archived in this repository).')

# ==================================================================
# FRIEDMAN TEST: multi-method x multi-benchmark
# ranking over the four 2D benchmarks. The AE request names "Wilcoxon,
# Friedman, etc."; the primary paired Wilcoxon above is a 2-method
# comparison, whereas Friedman ranks >=2 methods per benchmark.
#
# Matrix: 4 (2D benchmarks) x 6 (methods), one value per cell.
# Data policy: Standard/PS-FNO/F-FNO cells
# use the 3-seed mean WorstMax (identical to the main-text tables);
# DeepONet/PINO/POD-RBF cells use the single archived external-baseline run
# per benchmark. Those JSONs carry no `seed` field, so the seed behind them is
# NOT recorded and is not asserted here; PINO on
# heat uses the tuned delayed-ramp configuration reported in the main
# text (pino_final_results.json), because the untuned PINO diverges.
# ==================================================================

def _mean_wm(rows):
    return float(np.mean([r['worst_max'] for r in rows]))

def _cell(path, key):
    with open(path) as f:
        d = json.load(f)
    if isinstance(d[key], dict) and 'test' in d[key]:
        return float(d[key]['test']['worst_max'])
    if isinstance(d[key], dict) and 'worst_max' in d[key]:
        return float(d[key]['worst_max'])
    raise KeyError(f'{path} {key}: unexpected structure')

f2d_names = ['diffusion-heat', 'conv-diff', 'reaction-diff', 'damped-wave']

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))


def _rel(p):
    """Repo-relative POSIX path, for anything written into the JSON.

    Reads stay script-relative so the whole tree can be copied anywhere; the
    recorded provenance is emitted repo-relative so that a re-run in a
    different directory yields identical output and no absolute path of
    whoever ran it leaks into the public archive.
    """
    return os.path.relpath(p, _REPO_ROOT).replace(os.sep, '/')

# Standard FNO / PS-FNO cells: reuse the benchmarks dict (3-seed means,
# identical to the main-text tables tab:2d_results / tab:conv_diff_results
# / tab:reaction_diff_results / tab:wave_results).
std_cells = {b: float(np.mean(benchmarks[b]['standard'])) for b in f2d_names}
phys_cells = {b: float(np.mean(benchmarks[b]['physics'])) for b in f2d_names}

# F-FNO: 3-seed mean WorstMax from ffno_*.json (matches main-text tables).
ffno_sources = {
    'diffusion-heat': os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "results", "ffno_heat_results.json"),
    'conv-diff':      os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "results", "ffno_convdiff_results.json"),
    'reaction-diff':  os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "results", "ffno_reactiondiff_results.json"),
    'damped-wave':    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "results", "ffno_wave_results.json"),
}
ffno_cells = {}
for b in f2d_names:
    with open(ffno_sources[b]) as f:
        ffno_cells[b] = _mean_wm(json.load(f)['results'])

# DeepONet / PINO / POD-RBF: the single archived external-baseline run per
# benchmark. These JSONs carry no `seed` field, so no seed value is asserted.
# conv-diff POD-RBF appears only in podrbf_deterministic_results.json
# (the main-text conv-diff table reports the same 4.785 value).
do_paths = {
    'diffusion-heat': (os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "results", "baseline_results.json"), 'DeepONet'),
    'conv-diff':      (os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "results", "baselines_convdiff_results.json"), 'deeponet'),
    'reaction-diff':  (os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "results", "baselines_reactiondiff_results.json"), 'deeponet'),
    'damped-wave':    (os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "results", "baselines_wave_results.json"), 'deeponet'),
}
pino_paths = {
    'diffusion-heat': (os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "results", "pino_final_results.json"), 'PINO_selfnorm'),
    'conv-diff':      (os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "results", "baselines_convdiff_results.json"), 'pino'),
    'reaction-diff':  (os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "results", "baselines_reactiondiff_results.json"), 'pino'),
    'damped-wave':    (os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "results", "baselines_wave_results.json"), 'pino'),
}
podrbf_paths = {
    'diffusion-heat': (os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "results", "baselines_heat_pino_podrbf_results.json"), 'pod_rbf'),
    'conv-diff':      (os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "results", "podrbf_deterministic_results.json"), 'Conv-Diff'),
    'reaction-diff':  (os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "results", "baselines_reactiondiff_results.json"), 'pod_rbf'),
    'damped-wave':    (os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "results", "baselines_wave_results.json"), 'pod_rbf'),
}
ext_cells = {}
for meth, paths in [('DeepONet', do_paths), ('PINO', pino_paths),
                    ('POD-RBF', podrbf_paths)]:
    ext_cells[meth] = {b: _cell(*paths[b]) for b in f2d_names}

methods = ['Standard FNO', 'PS-FNO', 'DeepONet', 'PINO', 'POD-RBF', 'F-FNO']
matrix = np.array([
    [std_cells[b], phys_cells[b]] +
    [ext_cells[m][b] for m in ['DeepONet', 'PINO', 'POD-RBF']] +
    [ffno_cells[b]]
    for b in f2d_names
])  # shape (4 benchmarks, 6 methods)

print('\n' + '=' * 70)
print('FRIEDMAN TEST: 4 (2D benchmarks) x 6 (methods), one value per cell')
print('Cell policy: Standard/PS-FNO/F-FNO = 3-seed mean WorstMax (main-text')
print('values); DeepONet/PINO/POD-RBF = the single archived baseline run')
print('(no seed field, seed not asserted); PINO-heat =')
print('tuned config (pino_final_results.json). POD-RBF conv-diff cell from')
print('podrbf_deterministic_results.json (main-text table reports 4.785).')
print('=' * 70)
for i, b in enumerate(f2d_names):
    cells = '  '.join(f'{matrix[i][j]:8.4f}' for j in range(len(methods)))
    print(f'[{b:14s}] {cells}')

if matrix.shape[0] >= 3 and matrix.shape[1] >= 3:
    res_f = stats.friedmanchisquare(*[matrix[:, j] for j in range(matrix.shape[1])])
    chi2, p_f = float(res_f.statistic), float(res_f.pvalue)
    ranks = np.argsort(np.argsort(matrix, axis=1), axis=1) + 1  # 1 = best in benchmark
    mean_ranks = ranks.mean(axis=0)
    print(f'\nFriedman chi2 = {chi2:.4f}, p = {p_f:.4f} (df = {matrix.shape[1] - 1})')
    print('Average rank per method (1 = best within a benchmark):')
    for j, m in enumerate(methods):
        print(f'  {m:14s}: {mean_ranks[j]:.2f}')
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "results", "friedman_results.json"), 'w') as f:
        json.dump({
            'benchmarks': f2d_names,
            'methods': methods,
            'matrix': {b: {m: matrix[i][j] for j, m in enumerate(methods)}
                       for i, b in enumerate(f2d_names)},
            'chi2': chi2, 'p': p_f, 'df': matrix.shape[1] - 1,
            'mean_ranks': {m: float(mean_ranks[j]) for j, m in enumerate(methods)},
            'cell_policy': ('Standard/PS-FNO/F-FNO = 3-seed mean WorstMax; '
                            'DeepONet/PINO/POD-RBF = single archived baseline run, '
                            'no seed field in the file so the seed is not asserted; '
                            'PINO-heat = tuned config (pino_final_results.json); '
                            'POD-RBF conv-diff from podrbf_deterministic_results.json'),
            'provenance': {
                'standard_physics': 'benchmarks dict (same sources as primary Wilcoxon)',
                'ffno': {b: _rel(ffno_sources[b]) for b in f2d_names},
                'deeponet': {b: _rel(do_paths[b][0]) for b in f2d_names},
                'pino': {b: _rel(pino_paths[b][0]) for b in f2d_names},
                'podrbf': {b: _rel(podrbf_paths[b][0]) for b in f2d_names},
            },
        }, f, indent=1)
    print('\nFRIEDMAN results saved to ../results/friedman_results.json')
else:
    print('\nMatrix too small for Friedman; skipped.')
