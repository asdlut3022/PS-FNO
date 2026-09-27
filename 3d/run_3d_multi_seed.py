# 3D Multi-Seed Experiment
# Physics SingleStream vs Standard FNO (equal params), 3 seeds
# 3D LTNE results across multiple seeds
# Every seed in SEEDS is trained fresh: no result file is ever read back in, so
# the seed set on the command line is exactly the seed set trained.

import sys, os, time, json, torch, numpy as np
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))          # release root, for _release_cli

from physics_fno_3d import PhysicsSingleStreamPINO3D, StandardPINO3D
from run_comparison_3d import train_3d_model, eval_full
from train_causal_pino_3d import (TurbineBladeGeometry, LTNEPorousSolver,
                                      Normalizer3D, generate_data_3d)
from run_ood_experiment import generate_ood_test
from _release_cli import apply_cli, stamp_preset

SEEDS = [42, 123, 456]
N_EP = 3000
NX, NY, NZ, NT = 28, 14, 28, 10
N_TRAIN, N_TEST = 150, 20
_DEFAULT_RESULTS = os.path.join(_HERE, os.pardir, "results", "3d_multi_seed_results.json")
# This pipeline trains from scratch and saves no checkpoints, so it takes no
# checkpoint directory -- the third return value is discarded on purpose.
# Supplying one would only create an empty directory under the output root.
SEEDS, RESULTS_FILE, _, _CFG = apply_cli(
    SEEDS, _DEFAULT_RESULTS, device='cuda',
    description='3D multi-seed experiment: Physics SingleStream vs Standard FNO.')
device = _CFG['device']
NX = _CFG.get('nx', NX)
NY = _CFG.get('ny', NY)
NZ = _CFG.get('nz', NZ)
NT = _CFG.get('nt', NT)
N_TRAIN = _CFG.get('n_train', N_TRAIN)
N_TEST = _CFG.get('n_test', N_TEST)
N_EP = _CFG.get('epochs', N_EP)

def set_seed(s):
    torch.manual_seed(s)
    np.random.seed(s)
    if device.startswith('cuda') and torch.cuda.is_available():
        torch.cuda.manual_seed(s)
        torch.cuda.manual_seed_all(s)


def empty_cache():
    if device.startswith('cuda'):
        torch.cuda.empty_cache()


def ts():
    from datetime import datetime
    return datetime.now().strftime('%H:%M:%S')

if __name__ == '__main__':
    print(f'{"=" * 70}')
    print(f'3D MULTI-SEED EXPERIMENT')
    print(f'  Physics SS (w=64, rank=2) ~22.3M vs Standard (w=66) ~22.3M')
    print(f'  Seeds: {SEEDS}')
    print(f'Started: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'{"=" * 70}')

    geometry = TurbineBladeGeometry(device=device)

    all_results = {'seeds': SEEDS, 'per_seed': {}}
    stamp_preset(all_results, _CFG)

    for seed in SEEDS:
        print(f'\n{"=" * 70}')
        print(f'[{ts()}] SEED {seed}')
        print(f'{"=" * 70}')

        # --------------------------------------------------------
        # Step 1: Shared normalizer (same procedure every seed)
        # --------------------------------------------------------
        print(f'\n[{ts()}] Computing shared normalizer (seed={seed})...', flush=True)
        set_seed(seed)
        solver_hot = LTNEPorousSolver(geometry, nx=NX, ny=NY, nz=NZ, device=device,
                                       power_density=0.50e9, h_conv=80, T_ambient=310,
                                       h_sf_a_sf=200000.0, v_darcy=0.15, T_f_inlet=360.0)
        T_hot, _, _ = solver_hot.solve(t_total=0.2, nt_save=5)
        T_max = T_hot.max()
        del solver_hot; empty_cache()

        solver_cold = LTNEPorousSolver(geometry, nx=NX, ny=NY, nz=NZ, device=device,
                                        power_density=0.35e9, h_conv=180, T_ambient=290,
                                        h_sf_a_sf=1000000.0, v_darcy=0.85, T_f_inlet=330.0)
        T_cold, _, _ = solver_cold.solve(t_total=0.2, nt_save=5)
        T_min = T_cold.min()
        del solver_cold; empty_cache()

        shared_normalizer = Normalizer3D(T_min=T_min - 10, T_max=T_max + 30)
        print(f'  Normalizer range: [{shared_normalizer.T_min:.0f}, {shared_normalizer.T_max:.0f}] K')

        # --------------------------------------------------------
        # Step 2: Training data
        # --------------------------------------------------------
        print(f'\n[{ts()}] Generating TRAINING data (seed={seed})...', flush=True)
        set_seed(seed)
        train_ranges = {
            'power': (0.35e9, 0.50e9),
            'h_sf_a_sf': (400000, 700000),
            'v_darcy': (0.30, 0.60),
        }
        train_data = generate_data_3d(geometry, nx=NX, ny=NY, nz=NZ, nt=NT,
                                       n_traj=N_TRAIN, device=device,
                                       param_ranges=train_ranges,
                                       ext_normalizer=shared_normalizer)

        # --------------------------------------------------------
        # Step 3: ID test
        # --------------------------------------------------------
        print(f'\n[{ts()}] Generating ID test data (seed={seed})...', flush=True)
        set_seed(seed + 1000)
        id_test = generate_data_3d(geometry, nx=NX, ny=NY, nz=NZ, nt=NT,
                                    n_traj=N_TEST, device=device,
                                    param_ranges=train_ranges,
                                    ext_normalizer=shared_normalizer)

        # --------------------------------------------------------
        # Step 4: OOD test
        # --------------------------------------------------------
        print(f'\n[{ts()}] Generating OOD test data (seed={seed})...', flush=True)
        set_seed(seed + 2000)
        ood_test = generate_ood_test(geometry, shared_normalizer, nx=28, ny=14, nz=28,
                                      nt=10, n_traj=20, device=device)

        seed_result = {}

        # ========================================================
        # Exp A: Physics SingleStream (w=64, rank=2, ~22.3M)
        # ========================================================
        print(f'\n[{ts()}] Training Physics SingleStream (seed={seed})...')
        set_seed(seed)
        m_phys = PhysicsSingleStreamPINO3D(modes=8, width=64, n_layers=5, corr_rank=2, device=device)
        n_p_phys = sum(p.numel() for p in m_phys.parameters())
        print(f'  params: {n_p_phys:,}')

        t0 = time.time()
        m_phys = train_3d_model(m_phys, train_data, n_epochs=N_EP, device=device, mode='scheduled_sampling')
        t_phys = time.time() - t0

        r_phys_id = eval_full(m_phys, id_test, f'PhysSS [ID, seed={seed}]')
        r_phys_ood = eval_full(m_phys, ood_test, f'PhysSS [OOD, seed={seed}]')

        # Physics analysis
        t_array = train_data['t']
        dt_val = ((t_array[1] - t_array[0]) / t_array.max()).item()
        pa = m_phys.analyze_physics_contribution(dt_val)

        seed_result['physics'] = {
            'id': r_phys_id, 'ood': r_phys_ood,
            'params': n_p_phys, 'time': t_phys,
        }
        seed_result['physics_analysis'] = pa
        del m_phys; empty_cache()

        # ========================================================
        # Exp B: Standard FNO (w=66, ~22.3M)
        # ========================================================
        print(f'\n[{ts()}] Training Standard FNO w=66 (seed={seed})...')
        set_seed(seed)
        m_std = StandardPINO3D(modes=8, width=66, n_layers=5, device=device)
        n_p_std = sum(p.numel() for p in m_std.parameters())
        print(f'  params: {n_p_std:,}')

        t0 = time.time()
        m_std = train_3d_model(m_std, train_data, n_epochs=N_EP, device=device, mode='scheduled_sampling')
        t_std = time.time() - t0

        r_std_id = eval_full(m_std, id_test, f'Std66 [ID, seed={seed}]')
        r_std_ood = eval_full(m_std, ood_test, f'Std66 [OOD, seed={seed}]')

        seed_result['standard'] = {
            'id': r_std_id, 'ood': r_std_ood,
            'params': n_p_std, 'time': t_std,
        }
        del m_std; empty_cache()

        all_results['per_seed'][str(seed)] = seed_result

        # Intermediate save
        with open(RESULTS_FILE, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f'  [Saved intermediate results]', flush=True)

    # ========================================================
    # Summary statistics
    # ========================================================
    print(f'\n{"=" * 70}')
    print(f'3D MULTI-SEED SUMMARY')
    print(f'{"=" * 70}')

    for method in ['physics', 'standard']:
        for split in ['id', 'ood']:
            maes = [all_results['per_seed'][str(s)][method][split]['mae'] for s in SEEDS]
            wms = [all_results['per_seed'][str(s)][method][split]['worst_max'] for s in SEEDS]
            print(f'{method:>10} [{split:>3}]: MAE={np.mean(maes):.4f}±{np.std(maes):.4f}K  '
                  f'WorstMax={np.mean(wms):.4f}±{np.std(wms):.4f}K')

    # Physics vs Standard improvement
    for split in ['id', 'ood']:
        std_wm = np.mean([all_results['per_seed'][str(s)]['standard'][split]['worst_max'] for s in SEEDS])
        phys_wm = np.mean([all_results['per_seed'][str(s)]['physics'][split]['worst_max'] for s in SEEDS])
        print(f'\nPhysics vs Standard [{split}] WorstMax: {(1 - phys_wm/std_wm)*100:+.1f}%')

    # Save final
    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f'\nSaved: {RESULTS_FILE}')
    print(f'Completed: {time.strftime("%Y-%m-%d %H:%M:%S")}')
