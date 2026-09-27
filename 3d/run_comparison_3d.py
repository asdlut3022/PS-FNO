# 3D Physics-Structured FNO vs Standard FNO comparison experiment
# Turbine blade CHT problem (two fields: T_solid + T_coolant)

import torch
import os
import torch.nn.functional as F
import numpy as np
import sys
import time
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_causal_pino_3d import (
    TurbineBladeGeometry, generate_data_3d,
    apply_boundary_conditions, evaluate_causal,
)
from physics_fno_3d import PhysicsStructuredPINO3D, StandardPINO3D

device = 'cuda'
torch.cuda.empty_cache()

# ============================================================
# LTNE Physical Constants (fluid field — constant properties)
# ============================================================
_POROSITY = 0.3
_EPS_F = _POROSITY                        # fluid volume fraction
_RHO_F = 998.0                            # water density [kg/m³]
_CP_F = 4182.0                            # water specific heat [J/kgK]
_K_F = 0.6                                # water thermal conductivity [W/mK]
_ALPHA_F = _K_F / (_RHO_F * _CP_F)       # ≈ 1.437e-7 m²/s


def compute_fluid_pde_residual(T_s_cur, T_f_cur, T_f_next, mask,
                                dt_phys, dx, dy, dz,
                                h_sf_a_sf, v_darcy):
    """
    Fluid-field PDE residual (normalized temperature space, Forward Euler scheme).

    Fluid PDE:
      ∂T_f/∂t = α_f·∇²T_f - v_darcy·∂T_f/∂z + β_f·(T_s - T_f)

    Residual (multiplied by dt to remove the dimensional difference):
      R = ΔT_f_pred - dt·[α_f·∇²T_f + (−v)·∂T_f/∂z + β_f·(T_s − T_f)]
    where ΔT_f_pred = T_f_next − T_f_cur

    Spatial derivatives use finite differences, consistent with the PDE solver.
    The normalized temperature T_norm = (T-T_min)/scale is a linear transform, so the PDE form is unchanged.

    Args:
        T_s_cur:    [nx, ny, nz] current solid temperature (normalized)
        T_f_cur:    [nx, ny, nz] current fluid temperature (normalized)
        T_f_next:   [nx, ny, nz] predicted fluid temperature (normalized, differentiable)
        mask:       [nx, ny, nz] in-domain mask
        dt_phys:    physical time step [s]
        dx, dy, dz: physical grid spacing [m]
        h_sf_a_sf:  volumetric heat transfer coefficient [W/m³K] (scalar)
        v_darcy:    Darcy velocity [m/s] (scalar)

    Returns:
        scalar: mean squared residual over the valid domain
    """
    valid = mask > 0.5

    # Predicted temperature change
    delta_Tf = T_f_next - T_f_cur

    # Laplacian (central difference, consistent with solver._laplacian_3d)
    lap = torch.zeros_like(T_f_cur)
    lap[1:-1, :, :] += (T_f_cur[2:, :, :] - 2 * T_f_cur[1:-1, :, :] + T_f_cur[:-2, :, :]) / (dx ** 2)
    lap[:, 1:-1, :] += (T_f_cur[:, 2:, :] - 2 * T_f_cur[:, 1:-1, :] + T_f_cur[:, :-2, :]) / (dy ** 2)
    lap[:, :, 1:-1] += (T_f_cur[:, :, 2:] - 2 * T_f_cur[:, :, 1:-1] + T_f_cur[:, :, :-2]) / (dz ** 2)

    # Upwind scheme ∂T_f/∂z (consistent with solver._upwind_dTdz)
    dTf_dz = torch.zeros_like(T_f_cur)
    if v_darcy >= 0:
        dTf_dz[:, :, 1:] = (T_f_cur[:, :, 1:] - T_f_cur[:, :, :-1]) / dz
    else:
        dTf_dz[:, :, :-1] = (T_f_cur[:, :, 1:] - T_f_cur[:, :, :-1]) / dz

    # Coupling coefficient β_f = h_sf_a_sf / (ε_f · ρ_f · cp_f)
    beta_f = h_sf_a_sf / (_EPS_F * _RHO_F * _CP_F)

    # PDE right-hand side × dt
    pde_change = dt_phys * (_ALPHA_F * lap - v_darcy * dTf_dz + beta_f * (T_s_cur - T_f_cur))

    # Residual: predicted change vs. PDE-predicted change
    R_f = delta_Tf - pde_change

    return (R_f[valid] ** 2).mean()

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)


def train_3d_model(model, data, n_epochs=3000, device='cuda',
                   mode='scheduled_sampling',
                   p_start=0.9, p_end=0.3, warmup=500, anneal_end=2000,
                   eval_every=200, init_batch=8, max_batch=None,
                   pde_weight=0.0, pde_warmup=500, pde_anneal_end=1500):
    """Unified 3D training function (two fields, supports per-sample physical parameters + PDE residual loss)

    PDE residual loss parameters (disabled by default, pde_weight=0):
      pde_weight:     maximum PDE loss weight λ_max
      pde_warmup:     epoch at which the PDE loss starts to be introduced (basic patterns are learned first)
      pde_anneal_end: epoch at which the PDE loss reaches its maximum weight
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=800, T_mult=2, eta_min=1e-6
    )

    mask = data['mask']
    T_all = data['T']
    Tf_all = data['T_f']
    t_array = data['t']
    grid = data['grid']
    normalizer = data['normalizer']
    dist_surface = data['dist_surface']

    n_traj, nt = T_all.shape[0], len(t_array)
    valid = mask > 0.5
    dt = (t_array[1] - t_array[0]) / t_array.max()

    # per-sample PDE parameters (effective for the Physics model only)
    has_sample_alpha = ('h_conv_array' in data and hasattr(model, 'pde_type'))
    has_sample_vx = ('v_cool_array' in data and hasattr(model, 'pde_type'))
    if has_sample_alpha:
        h_arr = data['h_conv_array']
        h_mean = h_arr.mean()
        alpha_norm = torch.tensor(h_arr / h_mean, dtype=torch.float32, device=device)
    if has_sample_vx:
        v_arr = data['v_cool_array']
        v_mean = v_arr.mean()
        vx_norm = torch.tensor(v_arr / v_mean, dtype=torch.float32, device=device)

    # PDE residual loss: physical grid parameters
    use_pde = pde_weight > 0 and 'h_conv_array' in data and 'v_cool_array' in data
    if use_pde:
        dt_phys = (t_array[1] - t_array[0]).item()   # physical time step [s]
        dx = (data['X'][1, 0, 0] - data['X'][0, 0, 0]).item()
        dy = (data['Y'][0, 1, 0] - data['Y'][0, 0, 0]).item()
        dz = (data['Z'][0, 0, 1] - data['Z'][0, 0, 0]).item()
        h_conv_arr = data['h_conv_array']   # [n_traj] h_sf_a_sf values
        v_cool_arr = data['v_cool_array']   # [n_traj] v_darcy values
        print(f'  PDE loss: λ_max={pde_weight}, warmup={pde_warmup}, '
              f'anneal_end={pde_anneal_end}', flush=True)
        print(f'  Grid: dx={dx:.4f}m, dy={dy:.4f}m, dz={dz:.4f}m, '
              f'dt_phys={dt_phys:.6f}s', flush=True)

    sample_errors = np.ones(n_traj) * 10
    best_max, best_state = float('inf'), None

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        # Scheduled sampling probability
        if mode == 'teacher_forcing':
            teacher_prob = 1.0
        elif mode == 'scheduled_sampling':
            if epoch < warmup:
                teacher_prob = p_start
            elif epoch < anneal_end:
                teacher_prob = p_start - (p_start - p_end) * (epoch - warmup) / (anneal_end - warmup)
            else:
                teacher_prob = p_end

        # PDE loss weight schedule: 0 → ramp → pde_weight
        if use_pde and epoch >= pde_warmup:
            if epoch < pde_anneal_end:
                lam_pde = pde_weight * (epoch - pde_warmup) / (pde_anneal_end - pde_warmup)
            else:
                lam_pde = pde_weight
        else:
            lam_pde = 0.0

        # Batch selection
        if epoch < 200:
            batch_idx = np.random.choice(n_traj, min(init_batch, n_traj), replace=False)
        else:
            hard_idx = np.argsort(sample_errors)[-6:]
            random_idx = np.random.choice(n_traj, 2, replace=False)
            batch_idx = np.unique(np.concatenate([hard_idx, random_idx]))

        # Cap batch size for memory-constrained models (e.g. dual-stream Physics FNO)
        if max_batch and len(batch_idx) > max_batch:
            batch_idx = batch_idx[:max_batch]

        traj_losses = []
        batch_errors = {}

        for idx in batch_idx:
            T_traj_s = T_all[idx]
            T_traj_f = Tf_all[idx]
            T_cur_s = T_traj_s[0].unsqueeze(0)
            T_cur_f = T_traj_f[0].unsqueeze(0)
            traj_loss = 0
            traj_max_err = 0

            sa = alpha_norm[idx:idx+1] if has_sample_alpha else None
            sv = vx_norm[idx:idx+1] if has_sample_vx else None

            # per-sample PDE coefficients
            if use_pde and lam_pde > 0:
                h_val = float(h_conv_arr[idx])
                v_val = float(v_cool_arr[idx])

            for t_idx in range(1, nt):
                T_true_s = T_traj_s[t_idx]

                T_pred_s, T_pred_f = model(T_cur_s, T_cur_f, dt, grid,
                                            sample_alpha=sa, sample_vx=sv)
                T_pred_s, T_pred_f = apply_boundary_conditions(
                    T_pred_s.squeeze(0), T_pred_f.squeeze(0),
                    mask, dist_surface,
                    T_cur_s.squeeze(0), T_cur_f.squeeze(0)
                )

                diff = (T_pred_s - T_true_s)[valid]
                abs_diff = diff.abs()

                loss_mse = (diff ** 2).mean()
                loss_l1 = abs_diff.mean()
                max_err = abs_diff.max()
                loss_max = max_err ** 2 * 3.0

                k5 = max(1, int(abs_diff.numel() * 0.05))
                loss_top5 = (torch.topk(abs_diff.flatten(), k5).values ** 2).mean() * 0.5

                k1 = max(1, int(abs_diff.numel() * 0.01))
                loss_top1 = (torch.topk(abs_diff.flatten(), k1).values ** 2).mean() * 1.0

                step_loss = loss_mse + 0.3 * loss_l1 + loss_max + loss_top5 + loss_top1

                # PDE residual loss (fluid field)
                if use_pde and lam_pde > 0:
                    pde_res = compute_fluid_pde_residual(
                        T_cur_s.squeeze(0), T_cur_f.squeeze(0),
                        T_pred_f, mask,
                        dt_phys, dx, dy, dz,
                        h_val, v_val)
                    step_loss = step_loss + lam_pde * pde_res

                accumulation_weight = 1.0 + 0.3 * (t_idx / nt)
                traj_loss += accumulation_weight * step_loss

                traj_max_err = max(traj_max_err, (max_err * normalizer.scale).item())

                # Input for next step
                if mode == 'teacher_forcing' or np.random.rand() < teacher_prob:
                    T_cur_s = T_traj_s[t_idx].unsqueeze(0)
                    T_cur_f = T_traj_f[t_idx].unsqueeze(0)
                else:
                    T_cur_s = T_pred_s.unsqueeze(0).detach()
                    T_cur_f = T_pred_f.unsqueeze(0).detach()

            traj_losses.append(traj_loss / (nt - 1))
            batch_errors[idx] = traj_max_err

        for idx, err in batch_errors.items():
            sample_errors[idx] = 0.7 * sample_errors[idx] + 0.3 * err

        avg_loss = torch.stack(traj_losses).mean()
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if epoch % 50 == 0:
            torch.cuda.empty_cache()

        if (epoch + 1) % eval_every == 0 or epoch == 0:
            mae, max_err = evaluate_causal(model, data, grid, dt)
            lr = optimizer.param_groups[0]['lr']
            p_str = f' | p: {teacher_prob:.2f}' if mode == 'scheduled_sampling' else ''
            pde_str = f' | λ_pde: {lam_pde:.4f}' if use_pde else ''
            print(f'[{mode.upper()}] Epoch {epoch+1}/{n_epochs} | Loss: {avg_loss.item():.6f} | '
                  f'MAE: {mae:.4f}K | Max: {max_err:.4f}K | LR: {lr:.2e}{p_str}{pde_str}', flush=True)

            if max_err < best_max:
                best_max = max_err
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                print(f'  -> New best! Max: {best_max:.4f}K', flush=True)

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print(f'Best Max Error: {best_max:.4f}K', flush=True)
    return model


def eval_full(model, data, name=''):
    """Full evaluation (two fields, supports per-sample physical parameters) — last-step error only"""
    model.eval()
    mask = data['mask']
    T_all = data['T']
    Tf_all = data['T_f']
    t_array = data['t']
    grid = data['grid']
    normalizer = data['normalizer']
    dist_surface = data['dist_surface']
    nt = T_all.shape[1]
    valid = mask > 0.5
    dt = (t_array[1] - t_array[0]) / t_array.max()

    # per-sample PDE parameters
    has_sample_alpha = ('h_conv_array' in data and hasattr(model, 'pde_type'))
    has_sample_vx = ('v_cool_array' in data and hasattr(model, 'pde_type'))
    device = T_all.device
    if has_sample_alpha:
        h_arr = data['h_conv_array']
        h_mean = h_arr.mean()
        alpha_norm = torch.tensor(h_arr / h_mean, dtype=torch.float32, device=device)
    if has_sample_vx:
        v_arr = data['v_cool_array']
        v_mean = v_arr.mean()
        vx_norm = torch.tensor(v_arr / v_mean, dtype=torch.float32, device=device)

    all_mae, all_max = [], []
    with torch.no_grad():
        for traj_idx in range(T_all.shape[0]):
            T_traj_s = T_all[traj_idx]
            T_traj_f = Tf_all[traj_idx]
            T_cur_s = T_traj_s[0].unsqueeze(0)
            T_cur_f = T_traj_f[0].unsqueeze(0)

            sa = alpha_norm[traj_idx:traj_idx+1] if has_sample_alpha else None
            sv = vx_norm[traj_idx:traj_idx+1] if has_sample_vx else None

            for t_idx in range(1, nt):
                T_pred_s, T_pred_f = model(T_cur_s, T_cur_f, dt, grid,
                                            sample_alpha=sa, sample_vx=sv)
                T_pred_s, T_pred_f = apply_boundary_conditions(
                    T_pred_s.squeeze(0), T_pred_f.squeeze(0),
                    mask, dist_surface,
                    T_cur_s.squeeze(0), T_cur_f.squeeze(0)
                )
                T_cur_s = T_pred_s.unsqueeze(0)
                T_cur_f = T_pred_f.unsqueeze(0)

            # Last-step error only (consistent with the 2D eval_2d)
            T_pred_K = normalizer.denormalize(T_pred_s)
            T_true_K = normalizer.denormalize(T_traj_s[-1])
            err = torch.abs(T_pred_K - T_true_K)[valid]
            all_mae.append(err.mean().item())
            all_max.append(err.max().item())

    if name:
        print(f'{name}: MAE={np.mean(all_mae):.4f}K AvgMax={np.mean(all_max):.4f}K '
              f'WorstMax={np.max(all_max):.4f}K', flush=True)
    return {'mae': float(np.mean(all_mae)), 'avg_max': float(np.mean(all_max)),
            'worst_max': float(np.max(all_max))}


# ============================================================
# This module is a library, not a runnable entry point: it provides
# train_3d_model() and eval_full(), imported by run_3d_multi_seed.py
# (the Table 9 entry point) and by run_ood_experiment.py. The 3D LTNE
# benchmark uses scheduled sampling only -- there is no AR stage for 3D.
# ============================================================
