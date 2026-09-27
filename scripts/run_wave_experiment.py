# Wave equation experiment: Standard vs Physics FNO
# Engineering setting: 2D damped elastic membrane vibration (plate with a circular cutout defect)
# PDE: d2u/dt2 + 2*gamma*du/dt = c^2 * laplacian(u)
# Spectral kernel: D(k,dt) = exp(-gamma*dt) * exp(i*c*|k|*dt) — oscillation + decay (hyperbolic)
#
# Essential difference from the heat-conduction family:
#   heat: exp(-alpha*|k|^2*dt) — quadratic, pure decay
#   wave: exp(-gamma*dt)*exp(i*c*|k|*dt) — linear, oscillatory

import torch
import numpy as np
import sys
import time
import json
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from irregular_domain import LShapedDomain
from physics_fno import PhysicsStructuredPINO, StandardAutoregressivePINO
from solve_wave import generate_wave_data, WaveNormalizer

# ============================================================
# Configuration
# ============================================================
SEEDS = [42, 789, 456]
device = 'cuda'

WIDTH = 96
MODES = 16
N_LAYERS = 5
NX, NY = 32, 32
NT = 11      # 11 snapshots => 9 prediction steps (the first two steps are the initial condition)
N_TRAIN = 150
N_TEST = 20
N_EPOCHS = 2000

# Use an L-shaped domain — a classic irregular domain, not a heat sink
DOMAIN = LShapedDomain(width=1.0, height=1.0, cut_size=0.5)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "results", "results_wave")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Training function dedicated to the wave equation
# ============================================================
def train_wave_scheduled_sampling(model, data, n_epochs=2000, device='cuda',
                                   p_start=0.9, p_end=0.3, warmup=500, anneal_end=1500,
                                   is_physics=False, feed_gamma=False):
    """Scheduled sampling training for the wave equation

    Key difference: the autoregressive loop needs two time steps as input [u(t), u(t-dt)]

    Args:
        model: PhysicsStructuredPINO or StandardAutoregressivePINO
        data: dict returned by generate_wave_data
        is_physics: whether this is the Physics model (controls per-sample parameter passing)
        feed_gamma: whether to additionally feed the per-sample gamma (False = default behavior unchanged)
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=500, T_mult=2, eta_min=1e-6
    )

    mask = data['mask']
    U_all = data['U']         # (n_traj, nt, nx, ny)
    t_array = data['t']       # (nt,)
    normalizer = data['normalizer']
    n_traj, nt = U_all.shape[0], U_all.shape[1]
    X, Y = data['X'], data['Y']
    grid = torch.stack([X, Y], dim=-1).unsqueeze(0)
    valid = mask > 0.5
    dt_array = t_array[1:] - t_array[:-1]

    # per-sample physical parameters
    c_arr = data['c_array']
    c_mean = c_arr.mean()
    c_norm = torch.tensor(c_arr / c_mean, dtype=torch.float32, device=device)

    # Per-sample structural damping (not computed when feed_gamma=False)
    g_norm = None
    if feed_gamma:
        g_arr = data['gamma_array']
        g_mean = g_arr.mean()
        g_norm = torch.tensor(g_arr / g_mean, dtype=torch.float32, device=device)

    sample_errors = np.ones(n_traj) * 10.0
    best_max, best_state = float('inf'), None
    p_teacher = p_start

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        # Scheduled Sampling probability
        if epoch < warmup:
            p_teacher = p_start
        elif epoch < anneal_end:
            p_teacher = p_start - (p_start - p_end) * (epoch - warmup) / (anneal_end - warmup)
        else:
            p_teacher = p_end

        # Sampling strategy
        if epoch < 200:
            batch_idx = np.random.choice(n_traj, min(8, n_traj), replace=False)
        else:
            hard_idx = np.argsort(sample_errors)[-5:]
            random_idx = np.random.choice(n_traj, 3, replace=False)
            batch_idx = np.unique(np.concatenate([hard_idx, random_idx]))

        traj_losses = []
        batch_errors = {}

        for idx in batch_idx:
            U_traj = U_all[idx]   # (nt, nx, ny)
            traj_max_err = 0
            traj_loss = 0

            # Wave equation: the first two time steps are the initial condition
            u_prev = U_traj[0]    # u(t_0)
            u_curr = U_traj[1]    # u(t_1)

            # Predict starting from t=1 (both t=0 and t=1 are needed)
            for t_idx in range(1, nt - 1):
                u_next_true = U_traj[t_idx + 1]  # ground truth at t_{idx+1}
                dt = dt_array[t_idx]

                # Scheduled Sampling: decide whether to use the ground truth
                if t_idx > 1:  # the ground truth is always used as input when t_idx=1
                    use_teacher = torch.rand(1).item() < p_teacher
                    if use_teacher:
                        u_curr = U_traj[t_idx]
                        u_prev = U_traj[t_idx - 1]

                # Forward prediction
                sc = c_norm[idx:idx+1] if is_physics else None
                sgamma = g_norm[idx:idx+1] if (is_physics and feed_gamma) else None
                u_next_pred = model(
                    u_curr.unsqueeze(0), dt.unsqueeze(0), grid,
                    u_prev=u_prev.unsqueeze(0), sample_c=sc, sample_gamma_w=sgamma
                ).squeeze(0)
                u_next_pred = u_next_pred * mask  # domain mask

                # Loss computation
                diff = (u_next_pred - u_next_true)[valid]
                loss_mse = (diff ** 2).mean()

                abs_diff = diff.abs()
                max_err = abs_diff.max()
                loss_max = max_err ** 2 * 0.8

                k = max(1, int(abs_diff.numel() * 0.01))
                top_k = torch.topk(abs_diff, k).values
                loss_topk = (top_k ** 2).mean() * 0.5

                traj_loss += loss_mse + loss_max + loss_topk
                traj_max_err = max(traj_max_err, (max_err * normalizer.scale).item())

                # Update the time step (scheduled sampling: keep the predicted value)
                u_prev = u_curr.detach()
                u_curr = u_next_pred.detach()

            n_pred_steps = nt - 2  # start at t=1, predict through t=nt-1
            traj_losses.append(traj_loss / n_pred_steps)
            batch_errors[idx] = traj_max_err

        for idx, err in batch_errors.items():
            sample_errors[idx] = 0.6 * sample_errors[idx] + 0.4 * err

        avg_loss = torch.stack(traj_losses).mean()
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 100 == 0 or epoch == 0:
            mae, max_err = eval_wave_model(model, data, grid, is_physics=is_physics, feed_gamma=feed_gamma)
            lr = optimizer.param_groups[0]['lr']
            print(f'[WAVE_SS] Epoch {epoch+1}/{n_epochs} | Loss: {avg_loss.item():.6f} '
                  f'| MAE: {mae:.6f} | Max: {max_err:.6f} | LR: {lr:.2e} | p: {p_teacher:.2f}', flush=True)

            if max_err < best_max:
                best_max = max_err
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                print(f'  -> New best! Max: {best_max:.6f}', flush=True)

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print(f'[WAVE_SS] Best Max Error: {best_max:.6f}', flush=True)
    return model


@torch.no_grad()
def eval_wave_model(model, data, grid, is_physics=False, feed_gamma=False):
    """Evaluate the wave equation model (autoregressive rollout)"""
    model.eval()
    mask = data['mask']
    U_all = data['U']
    t_array = data['t']
    normalizer = data['normalizer']
    valid = mask > 0.5
    nt = U_all.shape[1]
    dt_array = t_array[1:] - t_array[:-1]

    c_arr = data['c_array']
    c_mean = c_arr.mean()
    c_norm = torch.tensor(c_arr / c_mean, dtype=torch.float32, device=data['X'].device)

    g_norm = None
    if feed_gamma:
        g_arr = data['gamma_array']
        g_mean = g_arr.mean()
        g_norm = torch.tensor(g_arr / g_mean, dtype=torch.float32, device=data['X'].device)

    all_max = []
    for traj_idx in range(U_all.shape[0]):
        U_traj = U_all[traj_idx]
        u_prev = U_traj[0].clone()
        u_curr = U_traj[1].clone()

        traj_max = 0
        for t_idx in range(1, nt - 1):
            dt = dt_array[t_idx]
            u_true = U_traj[t_idx + 1]

            sc = c_norm[traj_idx:traj_idx+1] if is_physics else None
            sgamma = g_norm[traj_idx:traj_idx+1] if (is_physics and feed_gamma) else None
            u_next = model(
                u_curr.unsqueeze(0), dt.unsqueeze(0), grid,
                u_prev=u_prev.unsqueeze(0), sample_c=sc, sample_gamma_w=sgamma
            ).squeeze(0)
            u_next = u_next * mask

            # Compute the error after denormalization
            err = torch.abs(
                u_next * normalizer.scale - u_true * normalizer.scale
            )[valid].max().item()
            traj_max = max(traj_max, err)

            u_prev = u_curr
            u_curr = u_next

        all_max.append(traj_max)

    model.train()
    return float(np.mean(all_max)), float(np.max(all_max))


def eval_wave_full(model, data, name='', is_physics=False, feed_gamma=False):
    """Full evaluation: MAE, AvgMax, WorstMax"""
    model.eval()
    mask = data['mask']
    U_all = data['U']
    t_array = data['t']
    normalizer = data['normalizer']
    valid = mask > 0.5
    valid_np = mask.cpu().numpy() > 0.5
    X, Y = data['X'], data['Y']
    grid = torch.stack([X, Y], dim=-1).unsqueeze(0)
    nt = U_all.shape[1]
    dt_array = t_array[1:] - t_array[:-1]

    c_arr = data['c_array']
    c_mean = c_arr.mean()
    c_norm = torch.tensor(c_arr / c_mean, dtype=torch.float32, device=X.device)

    g_norm = None
    if feed_gamma:
        g_arr = data['gamma_array']
        g_mean = g_arr.mean()
        g_norm = torch.tensor(g_arr / g_mean, dtype=torch.float32, device=X.device)

    all_mae, all_max = [], []

    with torch.no_grad():
        for traj_idx in range(U_all.shape[0]):
            U_traj = U_all[traj_idx]
            u_prev = U_traj[0].clone()
            u_curr = U_traj[1].clone()

            preds = [U_traj[0].cpu().numpy(), U_traj[1].cpu().numpy()]
            for t_idx in range(1, nt - 1):
                dt = dt_array[t_idx]

                sc = c_norm[traj_idx:traj_idx+1] if is_physics else None
                sgamma = g_norm[traj_idx:traj_idx+1] if (is_physics and feed_gamma) else None
                u_next = model(
                    u_curr.unsqueeze(0), dt.unsqueeze(0), grid,
                    u_prev=u_prev.unsqueeze(0), sample_c=sc, sample_gamma_w=sgamma
                ).squeeze(0)
                u_next = u_next * mask
                preds.append(u_next.cpu().numpy())

                u_prev = u_curr
                u_curr = u_next

            # Denormalize
            preds_phys = np.array(preds) * normalizer.scale
            trues_phys = U_traj.cpu().numpy() * normalizer.scale

            # Compare only predictions at t >= 2 (t=0 and t=1 are the initial condition)
            errors = np.abs(preds_phys[2:] - trues_phys[2:])[:, valid_np]
            all_mae.append(errors.mean())
            all_max.append(errors.max())

    print(f'\n{name}:', flush=True)
    print(f'  MAE:      {np.mean(all_mae):.6f} (physical units)', flush=True)
    print(f'  Avg Max:  {np.mean(all_max):.6f}', flush=True)
    print(f'  Worst Max: {np.max(all_max):.6f}', flush=True)
    return {
        'mae': float(np.mean(all_mae)),
        'avg_max': float(np.mean(all_max)),
        'worst_max': float(np.max(all_max)),
    }


# ============================================================
# Main experiment
# ============================================================
if __name__ == '__main__':
    print('=' * 70, flush=True)
    print('Wave Equation Experiment: Standard vs Physics FNO', flush=True)
    print(f'PDE: d2u/dt2 + 2*gamma*du/dt = c^2 * laplacian(u)', flush=True)
    print(f'Domain: L-shaped (classic irregular domain)', flush=True)
    print(f'Seeds: {SEEDS}', flush=True)
    print(f'Config: width={WIDTH}, modes={MODES}, layers={N_LAYERS}', flush=True)
    print(f'Training: {N_EPOCHS} epochs SS', flush=True)
    print('=' * 70, flush=True)

    all_results = {'seeds': SEEDS, 'standard': {}, 'physics': {}}

    for seed_i, seed in enumerate(SEEDS):
        print(f'\n{"="*70}', flush=True)
        print(f'SEED {seed} ({seed_i+1}/{len(SEEDS)})', flush=True)
        print(f'{"="*70}', flush=True)

        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Generate data
        print('Generating training data...', flush=True)
        train_data = generate_wave_data(DOMAIN, nx=NX, ny=NY, nt=NT, n_traj=N_TRAIN,
                                         device=device, seed=seed)
        test_data = generate_wave_data(DOMAIN, nx=NX, ny=NY, nt=NT, n_traj=N_TEST,
                                        device=device, seed=seed + 1000)

        X, Y = test_data['X'], test_data['Y']
        grid = torch.stack([X, Y], dim=-1).unsqueeze(0)

        # --- Standard FNO ---
        print(f'\nTraining Standard FNO+SS (seed {seed})...', flush=True)
        torch.manual_seed(seed + 10000)
        torch.cuda.manual_seed_all(seed + 10000)
        model_std = StandardAutoregressivePINO(
            modes=MODES, width=WIDTH, n_layers=N_LAYERS, device=device, wave_mode=True
        )
        n_params_std = sum(p.numel() for p in model_std.parameters())
        print(f'  Standard params: {n_params_std/1e6:.2f}M', flush=True)

        t0 = time.time()
        model_std = train_wave_scheduled_sampling(
            model_std, train_data, n_epochs=N_EPOCHS, device=device,
            is_physics=False
        )
        t_std = time.time() - t0

        res_std = eval_wave_full(model_std, test_data, f'Standard FNO (seed {seed})', is_physics=False)
        res_std['time'] = t_std
        res_std['params'] = n_params_std
        all_results['standard'][seed] = res_std

        del model_std
        torch.cuda.empty_cache()

        # --- Physics FNO ---
        print(f'\nTraining Physics FNO+SS (seed {seed})...', flush=True)
        torch.manual_seed(seed + 10000)
        torch.cuda.manual_seed_all(seed + 10000)
        model_phys = PhysicsStructuredPINO(
            modes=MODES, width=WIDTH, n_layers=N_LAYERS,
            device=device, corr_rank=8, pde_type='wave'
        )
        n_params_phys = sum(p.numel() for p in model_phys.parameters())
        print(f'  Physics params: {n_params_phys/1e6:.2f}M (+{(n_params_phys-n_params_std)/n_params_std*100:.1f}%)', flush=True)

        t0 = time.time()
        model_phys = train_wave_scheduled_sampling(
            model_phys, train_data, n_epochs=N_EPOCHS, device=device,
            is_physics=True
        )
        t_phys = time.time() - t0

        res_phys = eval_wave_full(model_phys, test_data, f'Physics FNO (seed {seed})', is_physics=True)
        res_phys['time'] = t_phys
        res_phys['params'] = n_params_phys

        # Physics ratio
        dt_avg = float((test_data['t'][2] - test_data['t'][1]).item())
        phys_analysis = model_phys.analyze_physics_contribution(dt_avg)
        avg_phys_ratio = np.mean([r['physics_ratio'] for r in phys_analysis])
        res_phys['physics_ratio'] = float(avg_phys_ratio)
        print(f'  Physics ratio: {avg_phys_ratio:.3f}', flush=True)

        all_results['physics'][seed] = res_phys

        del model_phys
        torch.cuda.empty_cache()

        # Per-seed improvement
        for k in ['mae', 'avg_max', 'worst_max']:
            imp = (1 - res_phys[k] / res_std[k]) * 100
            sign = '+' if imp > 0 else ''
            print(f'  {k} improvement: {sign}{imp:.1f}%', flush=True)

        # Save intermediate results
        partial_path = os.path.join(OUTPUT_DIR, 'wave_results_partial.json')
        with open(partial_path, 'w') as f:
            json.dump(all_results, f, indent=2)

    # ============================================================
    # Summary statistics
    # ============================================================
    print('\n' + '=' * 70, flush=True)
    print('WAVE EXPERIMENT RESULTS SUMMARY', flush=True)
    print('=' * 70, flush=True)

    summary = {}
    for method in ['standard', 'physics']:
        results = all_results[method]
        for k in ['mae', 'avg_max', 'worst_max']:
            vals = [results[seed][k] for seed in SEEDS]
            summary[f'{method}_{k}_mean'] = float(np.mean(vals))
            summary[f'{method}_{k}_std'] = float(np.std(vals))

    for k_label, k in [('MAE', 'mae'), ('AvgMax', 'avg_max'), ('WorstMax', 'worst_max')]:
        s_mean = summary[f'standard_{k}_mean']
        s_std = summary[f'standard_{k}_std']
        p_mean = summary[f'physics_{k}_mean']
        p_std = summary[f'physics_{k}_std']
        imp = (1 - p_mean / s_mean) * 100
        print(f'  {k_label}: Standard {s_mean:.6f}+/-{s_std:.6f}  '
              f'Physics {p_mean:.6f}+/-{p_std:.6f}  '
              f'Improvement: {imp:+.1f}%', flush=True)

    if 'physics_ratio' in all_results['physics'][SEEDS[0]]:
        avg_rho = np.mean([all_results['physics'][s]['physics_ratio'] for s in SEEDS])
        print(f'  Avg physics ratio: {avg_rho:.3f}', flush=True)

    summary['improvement_pct'] = {
        k: float((1 - summary[f'physics_{k}_mean'] / summary[f'standard_{k}_mean']) * 100)
        for k in ['mae', 'avg_max', 'worst_max']
    }

    all_results['summary'] = summary

    # Save final results
    final_path = os.path.join(OUTPUT_DIR, 'wave_results.json')
    with open(final_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nResults saved to: {final_path}', flush=True)

    print('\n' + '=' * 70, flush=True)
    print('DONE.', flush=True)
    print('=' * 70, flush=True)
