"""
Case 3: OOD Generalization Test
================================
Train on NARROW LTNE ranges, test on both ID and OOD.
Physics FNO with per-sample conditioning should generalize to OOD;
Standard FNO without parameter info should degrade.

Training:  h_sf_a_sf [400k, 700k], v_darcy [0.30, 0.60]
ID test:   h_sf_a_sf [400k, 700k], v_darcy [0.30, 0.60]
OOD test:  h_sf_a_sf [200k, 400k]∪[700k, 1M], v_darcy [0.15, 0.30]∪[0.60, 0.85]
"""
import sys, os, time, json, torch, numpy as np
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from physics_fno_3d import PhysicsStructuredPINO3D, StandardPINO3D
from run_comparison_3d import train_3d_model, eval_full
from train_causal_pino_3d import (TurbineBladeGeometry, LTNEPorousSolver,
                                      Normalizer3D, generate_data_3d)

device = 'cuda'
BASE_SEED = 42
N_EP = 3000

def set_seed(s):
    torch.manual_seed(s)
    np.random.seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(s)
        torch.cuda.manual_seed_all(s)

def ts():
    from datetime import datetime
    return datetime.now().strftime('%H:%M:%S')

def generate_ood_test(geometry, normalizer, nx=28, ny=14, nz=28, nt=10,
                      n_traj=20, device='cuda'):
    """Generate OOD test data: parameters OUTSIDE training range"""
    # For each trajectory, sample h_sf_a_sf and v_darcy from OOD regions
    param_log = []
    all_T_solid = []
    all_T_f = []
    t_array = None

    for i in range(n_traj):
        power = 0.35e9 + 0.15e9 * np.random.rand()  # same power range
        h_conv = 80 + 120 * np.random.rand()
        T_amb = 290 + 20 * np.random.rand()
        T_inlet = 330 + 40 * np.random.rand()

        # OOD h_sf_a_sf: [200k, 400k] or [700k, 1M]
        if np.random.rand() < 0.5:
            h_sf_a_sf = 200000 + 200000 * np.random.rand()  # [200k, 400k]
        else:
            h_sf_a_sf = 700000 + 300000 * np.random.rand()  # [700k, 1M]

        # OOD v_darcy: [0.15, 0.30] or [0.60, 0.85]
        if np.random.rand() < 0.5:
            v_darcy = 0.15 + 0.15 * np.random.rand()  # [0.15, 0.30]
        else:
            v_darcy = 0.60 + 0.25 * np.random.rand()  # [0.60, 0.85]

        solver = LTNEPorousSolver(geometry, nx=nx, ny=ny, nz=nz, device=device,
                                   power_density=power, h_conv=h_conv, T_ambient=T_amb,
                                   h_sf_a_sf=h_sf_a_sf, v_darcy=v_darcy, T_f_inlet=T_inlet)
        T_hist, Tf_hist, t_arr = solver.solve(t_total=0.2, nt_save=nt)
        del solver
        if i % 10 == 0:
            torch.cuda.empty_cache()
        if t_array is None:
            t_array = t_arr

        T_s_norm = normalizer.normalize(T_hist)
        T_f_norm = normalizer.normalize(Tf_hist)
        all_T_solid.append(torch.from_numpy(T_s_norm).float().to(device))
        all_T_f.append(torch.from_numpy(T_f_norm).float().to(device))
        param_log.append({'power': power, 'h_conv': h_conv, 'T_amb': T_amb,
                          'h_sf_a_sf': h_sf_a_sf, 'v_darcy': v_darcy, 'T_inlet': T_inlet})

    # Reuse grid from a dummy generation (same geometry)
    from train_causal_pino_3d import create_3d_grid
    bounds = [(0, geometry.chord), (-0.1, 0.1), (0, geometry.span)]
    X, Y, Z = create_3d_grid(nx, ny, nz, bounds, device)
    mask = geometry.create_mask(X, Y, Z)
    dist_surface = geometry.get_surface_distance(X, Y, Z)
    X_norm = (X - bounds[0][0]) / (bounds[0][1] - bounds[0][0])
    Y_norm = (Y - bounds[1][0]) / (bounds[1][1] - bounds[1][0])
    Z_norm = (Z - bounds[2][0]) / (bounds[2][1] - bounds[2][0])
    grid = torch.stack([X_norm, Y_norm, Z_norm], dim=-1).unsqueeze(0)

    h_list = [p['h_sf_a_sf'] for p in param_log]
    v_list = [p['v_darcy'] for p in param_log]
    print(f"  OOD params: h_sf_a_sf [{min(h_list):.0f}, {max(h_list):.0f}]  "
          f"v_darcy [{min(v_list):.2f}, {max(v_list):.2f}]", flush=True)

    return {
        'T': torch.stack(all_T_solid, dim=0),
        'T_f': torch.stack(all_T_f, dim=0),
        'mask': mask, 'grid': grid,
        't': torch.from_numpy(t_array).float().to(device),
        'normalizer': normalizer,
        'X': X, 'Y': Y, 'Z': Z,
        'dist_surface': dist_surface,
        'h_conv_array': np.array(h_list),
        'v_cool_array': np.array(v_list),
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--physics-only', action='store_true',
                        help='Skip Standard training, only run Physics')
    args = parser.parse_args()

    print(f'{"=" * 70}')
    print(f'Case 3: OOD Generalization Test')
    if args.physics_only:
        print(f'  MODE: Physics-only (skipping Standard)')
    print(f'Started: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'{"=" * 70}')

    geometry = TurbineBladeGeometry()

    # --------------------------------------------------------
    # Step 1: Create shared normalizer from OOD extreme probes
    # --------------------------------------------------------
    print(f'\n[{ts()}] Computing shared normalizer from OOD extremes...', flush=True)
    set_seed(BASE_SEED)
    # Hot: max power, weakest coupling, slowest flow
    solver_hot = LTNEPorousSolver(geometry, nx=28, ny=14, nz=28, device=device,
                                   power_density=0.50e9, h_conv=80, T_ambient=310,
                                   h_sf_a_sf=200000.0, v_darcy=0.15, T_f_inlet=360.0)
    T_hot, _, _ = solver_hot.solve(t_total=0.2, nt_save=5)
    T_max = T_hot.max()
    del solver_hot; torch.cuda.empty_cache()

    # Cold: min power, strongest coupling, fastest flow
    solver_cold = LTNEPorousSolver(geometry, nx=28, ny=14, nz=28, device=device,
                                    power_density=0.35e9, h_conv=180, T_ambient=290,
                                    h_sf_a_sf=1000000.0, v_darcy=0.85, T_f_inlet=330.0)
    T_cold, _, _ = solver_cold.solve(t_total=0.2, nt_save=5)
    T_min = T_cold.min()
    del solver_cold; torch.cuda.empty_cache()

    shared_normalizer = Normalizer3D(T_min=T_min - 10, T_max=T_max + 30)
    print(f'  Normalizer range: [{shared_normalizer.T_min:.0f}, {shared_normalizer.T_max:.0f}] K')

    # --------------------------------------------------------
    # Step 2: Generate training data (NARROW ranges)
    # --------------------------------------------------------
    print(f'\n[{ts()}] Generating TRAINING data (narrow LTNE ranges)...', flush=True)
    set_seed(BASE_SEED)
    train_ranges = {
        'power': (0.35e9, 0.50e9),
        'h_sf_a_sf': (400000, 700000),  # NARROW
        'v_darcy': (0.30, 0.60),         # NARROW
    }
    train_data = generate_data_3d(geometry, nx=28, ny=14, nz=28, nt=10,
                                   n_traj=150, device=device,
                                   param_ranges=train_ranges,
                                   ext_normalizer=shared_normalizer)

    # --------------------------------------------------------
    # Step 3: Generate ID test data (same narrow ranges)
    # --------------------------------------------------------
    print(f'\n[{ts()}] Generating ID test data (same narrow ranges)...', flush=True)
    set_seed(BASE_SEED + 1000)
    id_test = generate_data_3d(geometry, nx=28, ny=14, nz=28, nt=10,
                                n_traj=20, device=device,
                                param_ranges=train_ranges,
                                ext_normalizer=shared_normalizer)

    # --------------------------------------------------------
    # Step 4: Generate OOD test data (OUTSIDE training ranges)
    # --------------------------------------------------------
    print(f'\n[{ts()}] Generating OOD test data (outside training ranges)...', flush=True)
    set_seed(BASE_SEED + 2000)
    ood_test = generate_ood_test(geometry, shared_normalizer, nx=28, ny=14, nz=28,
                                  nt=10, n_traj=20, device=device)

    print(f'\n  Train h_sf_a_sf: [{min(train_data["h_conv_array"]):.0f}, {max(train_data["h_conv_array"]):.0f}]'
          f'  v_darcy: [{min(train_data["v_cool_array"]):.2f}, {max(train_data["v_cool_array"]):.2f}]')

    results = {}

    # --------------------------------------------------------
    # Step 5: Standard FNO (width=64, 21M)
    # --------------------------------------------------------
    if args.physics_only:
        # Use previously verified results (same seed = same data = same results)
        print(f'\n[{ts()}] SKIPPING Standard (using previous results)...', flush=True)
        results['Standard'] = {
            'params': 20997058, 'time': 2748.0,
            'id':  {'mae': 1.7085, 'avg_max': 3.7204, 'worst_max': 6.4072},
            'ood': {'mae': 1.8639, 'avg_max': 4.2696, 'worst_max': 7.1449},
        }
        print(f'  Std [ID]:  MAE={1.7085:.4f}K  WorstMax={6.4072:.4f}K')
        print(f'  Std [OOD]: MAE={1.8639:.4f}K  WorstMax={7.1449:.4f}K')
    else:
        print(f'\n[{ts()}] Training Standard 3D FNO (width=64)...', flush=True)
        set_seed(BASE_SEED)
        m_std = StandardPINO3D(modes=8, width=64, n_layers=5, device=device)
        n_p = sum(p.numel() for p in m_std.parameters())
        print(f'  params={n_p:,}')
        t0 = time.time()
        m_std = train_3d_model(m_std, train_data, n_epochs=N_EP, device=device, mode='scheduled_sampling')
        t_std = time.time() - t0

        r_id = eval_full(m_std, id_test, 'Std [ID Test]')
        r_ood = eval_full(m_std, ood_test, 'Std [OOD Test]')
        results['Standard'] = {'params': n_p, 'time': t_std, 'id': r_id, 'ood': r_ood}
        del m_std; torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Step 6: Physics Dual-Stream (width=64, 13M)
    # --------------------------------------------------------
    print(f'\n[{ts()}] Training Physics Dual-Stream (width=64)...', flush=True)
    set_seed(BASE_SEED)
    m_phy = PhysicsStructuredPINO3D(modes=8, width=64, n_layers=5, device=device)
    n_p2 = sum(p.numel() for p in m_phy.parameters())
    print(f'  params={n_p2:,}')
    t0 = time.time()
    m_phy = train_3d_model(m_phy, train_data, n_epochs=N_EP, device=device, mode='scheduled_sampling')
    t_phy = time.time() - t0

    r_id2 = eval_full(m_phy, id_test, 'Physics [ID Test]')
    r_ood2 = eval_full(m_phy, ood_test, 'Physics [OOD Test]')

    # Physics analysis
    t_array = train_data['t']
    dt_val = ((t_array[1] - t_array[0]) / t_array.max()).item()
    pa = m_phy.analyze_physics_contribution(dt_val)

    results['Physics'] = {'params': n_p2, 'time': t_phy,
                          'id': r_id2, 'ood': r_ood2, 'physics_analysis': pa}
    del m_phy; torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    print(f'\n{"=" * 95}')
    print(f'{"CASE 3: ID vs OOD GENERALIZATION RESULTS":^95}')
    print(f'{"=" * 95}')
    print(f'{"Method":<30} {"Params":>12} {"Test":>6} {"MAE(K)":>8} {"AvgMax(K)":>10} {"WorstMax(K)":>12}')
    print(f'{"-" * 95}')
    for name, r in results.items():
        for split, label in [('id', 'ID'), ('ood', 'OOD')]:
            t = r[split]
            print(f'{name:<30} {r["params"]:>12,} {label:>6} {t["mae"]:>8.4f} {t["avg_max"]:>10.4f} {t["worst_max"]:>12.4f}')

    # Compute improvements
    std_id = results['Standard']['id']
    std_ood = results['Standard']['ood']
    phy_id = results['Physics']['id']
    phy_ood = results['Physics']['ood']

    print(f'\n{"=" * 60}')
    print(f'Physics vs Standard Improvement:')
    print(f'  ID  MAE: {(1-phy_id["mae"]/std_id["mae"])*100:+.1f}%  '
          f'WorstMax: {(1-phy_id["worst_max"]/std_id["worst_max"])*100:+.1f}%')
    print(f'  OOD MAE: {(1-phy_ood["mae"]/std_ood["mae"])*100:+.1f}%  '
          f'WorstMax: {(1-phy_ood["worst_max"]/std_ood["worst_max"])*100:+.1f}%')
    print(f'{"=" * 60}')

    # Save
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "results", "ood_results.json"), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'  [Saved: ood_results.json]')

    print(f'\nCompleted: {time.strftime("%Y-%m-%d %H:%M:%S")}')
