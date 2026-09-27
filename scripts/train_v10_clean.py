# Causal-PINO V10: Autoregressive + Numba accelerated (Clean version)
# Target: Max Error < 0.8K
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from numba import njit, prange

from irregular_domain import IrregularFinsDomain, DomainUtils
from fno import FNOBlock2d

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = 'cuda'
torch.cuda.empty_cache()


class AutoregressivePINO(nn.Module):
    def __init__(self, modes=16, width=96, n_layers=5, device='cuda'):
        super().__init__()
        self.fc0 = nn.Linear(4, width)
        self.backbone = nn.ModuleList()
        for i in range(n_layers):
            self.backbone.append(FNOBlock2d(width, width, modes, modes, 'gelu'))
        self.fc1 = nn.Linear(width, width // 2)
        self.fc2 = nn.Linear(width // 2, 1)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        self.to(device)

    def forward(self, T_current, dt, grid):
        batch_size, nx, ny = T_current.shape
        if isinstance(dt, (int, float)):
            dt = torch.tensor([dt], device=T_current.device).expand(batch_size)
        dt_exp = dt.view(-1, 1, 1, 1).expand(batch_size, nx, ny, 1)
        x = torch.cat([T_current.unsqueeze(-1), grid.expand(batch_size, -1, -1, -1), dt_exp], dim=-1)
        x = self.fc0(x).permute(0, 3, 1, 2)
        for block in self.backbone:
            x = x + block(x)
        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        delta_T = self.fc2(x).squeeze(-1)
        return T_current + self.residual_scale * delta_T


class Normalizer:
    def __init__(self, T_min=280.0, T_max=380.0):
        self.T_min, self.T_max = T_min, T_max
        self.scale = T_max - T_min

    def normalize(self, T):
        return (T - self.T_min) / self.scale

    def denormalize(self, T):
        return T * self.scale + self.T_min


@njit(parallel=True, cache=True)
def solve_heat_numba(nx, ny, T_hot, T_amb, alpha, Bi, t_total, nt_save, ramp_fraction=0.7, ic_fraction=1.0):
    """Numba-accelerated heat conduction solver - time-varying ramp heating BC
    ramp_fraction: fraction of the total time over which the bottom heating ramps linearly from T_amb to T_hot
    ic_fraction: percentage of the steady-state gradient used as the initial condition (1.0=full gradient, 0.2=20% gradient, 0.0=pure cold start)
    BC at the bottom: T_hot(t) = T_amb + (T_hot-T_amb)*min(t/t_ramp, 1)
    """
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    dt_max = min(dx, dy)**2 / (4 * alpha)
    dt = min(0.0001, dt_max * 0.5)
    n_steps = int(t_total / dt)
    save_every = max(1, n_steps // (nt_save - 1))
    t_ramp = ramp_fraction * t_total

    # IC: ic_fraction of steady-state gradient
    T = np.ones((nx, ny)) * T_amb
    if ic_fraction > 0.0:
        for i in range(nx):
            for j in range(ny):
                y_val = j * dy
                T[i, j] = T_amb + ic_fraction * (T_hot - T_amb) * (1.0 - y_val)

    T_history = np.zeros((nt_save, nx, ny))
    t_history = np.zeros(nt_save)
    T_history[0] = T.copy()
    save_idx = 1

    for step in range(1, n_steps + 1):
        t_current = step * dt
        ramp_factor = min(t_current / t_ramp, 1.0)
        T_hot_t = T_amb + (T_hot - T_amb) * ramp_factor

        T_new = T.copy()
        for i in prange(1, nx - 1):
            for j in range(1, ny - 1):
                d2Tdx2 = (T[i+1, j] - 2*T[i, j] + T[i-1, j]) / (dx*dx)
                d2Tdy2 = (T[i, j+1] - 2*T[i, j] + T[i, j-1]) / (dy*dy)
                T_new[i, j] = T[i, j] + alpha * dt * (d2Tdx2 + d2Tdy2)

        for i in range(nx):
            T_new[i, 0] = T_hot_t
            T_new[i, ny-1] = (T_new[i, ny-2] + Bi * dy * T_amb) / (1 + Bi * dy)
        for j in range(ny):
            T_new[0, j] = T_new[1, j]
            T_new[nx-1, j] = T_new[nx-2, j]
        T = T_new

        if (step % save_every == 0 or step == n_steps) and save_idx < nt_save:
            T_history[save_idx] = T.copy()
            t_history[save_idx] = step * dt
            save_idx += 1

    return T_history, t_history


def generate_data(domain, nx=32, ny=32, nt=10, n_traj=100, device='cuda', seed=None, ramp_fraction=0.7, ic_fraction=1.0, noise_sigma=0.0):
    if seed is not None:
        np.random.seed(seed)
    print(f'Generating {n_traj} trajectories (ramp_fraction={ramp_fraction}, ic_fraction={ic_fraction})...', flush=True)
    normalizer = Normalizer()
    X, Y = DomainUtils.create_grid(nx, ny, (0, 1.0), (0, 1.0), device)
    mask = domain.mask(X, Y)
    all_T = []
    alpha_list = []

    # Warmup numba
    _ = solve_heat_numba(nx, ny, 350.0, 305.0, 0.02, 0.6, 1.0, nt, ramp_fraction, ic_fraction)

    # Noise RNG: independent stream so the training noise
    # never perturbs the seed chain that controls data/split/batch selection.
    noise_rng = np.random.default_rng((seed if seed is not None else 0) + 777)

    t_array = None
    for i in range(n_traj):
        T_hot = 345 + np.random.rand() * 10
        T_amb = 300 + np.random.rand() * 10
        alpha = 0.016 + np.random.rand() * 0.008
        Bi = 0.4 + np.random.rand() * 0.4

        T_hist, t_arr = solve_heat_numba(nx, ny, T_hot, T_amb, alpha, Bi, 1.0, nt, ramp_fraction, ic_fraction)
        if t_array is None:
            t_array = t_arr
        T_norm = normalizer.normalize(T_hist)
        if noise_sigma > 0.0:
            # Multiplicative Gaussian noise on the normalized training fields,
            # i.i.d. per point; the clean (sigma=0) arm is the in-script baseline.
            T_norm = T_norm * (1.0 + noise_sigma * noise_rng.standard_normal(T_norm.shape))
        all_T.append(torch.from_numpy(T_norm).float().to(device))
        alpha_list.append(alpha)

    print(f'Generated {n_traj} trajectories', flush=True)
    print(f'  alpha range: [{min(alpha_list):.4f}, {max(alpha_list):.4f}]')
    return {'T': torch.stack(all_T, dim=0), 'mask': mask, 'X': X, 'Y': Y,
            't': torch.from_numpy(t_array).float().to(device), 'normalizer': normalizer,
            'alpha_array': np.array(alpha_list)}


def apply_bc(T, X, Y, mask, T_hot_norm):
    T_out = T.clone()
    valid = mask > 0.5
    hot = (Y < 0.1) & valid
    if hot.any():
        w = torch.clamp((0.1 - Y) / 0.08, 0, 1)
        T_out = torch.where(hot, T_out * (1-w) + T_hot_norm * w, T_out)
    return T_out * mask


def compute_boundary_weight(Y, mask):
    valid = mask > 0.5
    weight = torch.ones_like(Y)
    hot_boundary = (Y < 0.15) & valid
    weight = torch.where(hot_boundary, weight * 3.0, weight)
    cold_boundary = (Y > 0.85) & valid
    weight = torch.where(cold_boundary, weight * 2.0, weight)
    return weight * mask


def train_autoregressive(model, data, n_epochs=2000, device='cuda'):
    """Training - does not use AMP (incompatible with complex FFT)"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=500, T_mult=2, eta_min=1e-6)

    mask, X, Y = data['mask'], data['X'], data['Y']
    T_all, t_array = data['T'], data['t']
    normalizer = data['normalizer']
    n_traj, nt = T_all.shape[0], len(t_array)
    grid = torch.stack([X, Y], dim=-1).unsqueeze(0)
    valid = mask > 0.5
    boundary_weight = compute_boundary_weight(Y, mask)
    dt_array = t_array[1:] - t_array[:-1]

    sample_errors = np.ones(n_traj) * 10
    best_max, best_state = float('inf'), None

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        if epoch < 200:
            batch_idx = np.random.choice(n_traj, min(8, n_traj), replace=False)
        else:
            hard_idx = np.argsort(sample_errors)[-5:]
            random_idx = np.random.choice(n_traj, 3, replace=False)
            batch_idx = np.unique(np.concatenate([hard_idx, random_idx]))

        total_loss = 0
        batch_errors = {}

        for idx in batch_idx:
            T_traj = T_all[idx]
            traj_max_err = 0
            traj_loss = 0

            for t_idx in range(nt - 1):
                T_current = T_traj[t_idx]
                T_next_true = T_traj[t_idx + 1]
                dt = dt_array[t_idx]
                # Extract the BC step by step from the ground truth (the ramp BC changes at every step)
                T_hot_norm = T_next_true[(Y < 0.1) & valid].mean()

                T_next_pred = model(T_current.unsqueeze(0), dt.unsqueeze(0), grid).squeeze(0)
                T_next_pred = apply_bc(T_next_pred, X, Y, mask, T_hot_norm)

                diff = (T_next_pred - T_next_true)[valid]
                weight = boundary_weight[valid]
                loss_mse = ((diff ** 2) * weight).mean()

                abs_diff = diff.abs()
                max_err = abs_diff.max()
                loss_max = max_err ** 2 * 0.8

                k = max(1, int(abs_diff.numel() * 0.01))
                top_k = torch.topk(abs_diff, k).values
                loss_topk = (top_k ** 2).mean() * 0.5

                traj_loss += loss_mse + loss_max + loss_topk
                traj_max_err = max(traj_max_err, (max_err * normalizer.scale).item())

            total_loss += traj_loss / (nt - 1)
            batch_errors[idx] = traj_max_err

        for idx, err in batch_errors.items():
            sample_errors[idx] = 0.6 * sample_errors[idx] + 0.4 * err

        total_loss = total_loss / len(batch_idx)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 100 == 0 or epoch == 0:
            mae, max_err = eval_model(model, data, grid)
            lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch+1}/{n_epochs} | Loss: {total_loss.item():.6f} | MAE: {mae:.4f}K | Max: {max_err:.4f}K | LR: {lr:.2e}', flush=True)

            if max_err < best_max:
                best_max = max_err
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                print(f'  -> New best! Max: {best_max:.4f}K', flush=True)

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print(f'Restored best (Max: {best_max:.4f}K)', flush=True)
    return model


def eval_model(model, data, grid):
    model.eval()
    mask, X, Y = data['mask'], data['X'], data['Y']
    T_all, t_array = data['T'], data['t']
    normalizer = data['normalizer']
    valid = mask > 0.5
    nt = len(t_array)
    dt_array = t_array[1:] - t_array[:-1]

    all_max = []
    with torch.no_grad():
        for traj_idx in range(T_all.shape[0]):
            T_traj = T_all[traj_idx]
            T_current = T_traj[0].clone()

            traj_max = 0
            for t_idx in range(nt - 1):
                dt = dt_array[t_idx]
                T_true = T_traj[t_idx + 1]
                T_hot_norm = T_true[(Y < 0.1) & valid].mean()
                T_next = model(T_current.unsqueeze(0), dt.unsqueeze(0), grid).squeeze(0)
                T_next = apply_bc(T_next, X, Y, mask, T_hot_norm)
                err = torch.abs(normalizer.denormalize(T_next) - normalizer.denormalize(T_true))[valid].max().item()
                traj_max = max(traj_max, err)
                T_current = T_next
            all_max.append(traj_max)

    model.train()
    return np.mean(all_max), np.max(all_max)


def eval_full(model, data, name=''):
    model.eval()
    mask, X, Y = data['mask'], data['X'], data['Y']
    t_array, normalizer = data['t'], data['normalizer']
    mask_np = mask.cpu().numpy()
    valid_np = mask_np > 0.5
    valid = mask > 0.5
    grid = torch.stack([X, Y], dim=-1).unsqueeze(0)
    nt = len(t_array)
    dt_array = t_array[1:] - t_array[:-1]

    all_mae, all_max = [], []
    T_pred_save, T_true_save = None, None

    for traj_idx in range(data['T'].shape[0]):
        T_traj = data['T'][traj_idx]
        T_current = T_traj[0].clone()

        with torch.no_grad():
            preds = [T_current.cpu().numpy()]
            for t_idx in range(nt - 1):
                dt = dt_array[t_idx]
                T_hot_norm = T_traj[t_idx + 1][(Y < 0.1) & valid].mean()
                T_next = model(T_current.unsqueeze(0), dt.unsqueeze(0), grid).squeeze(0)
                T_next = apply_bc(T_next, X, Y, mask, T_hot_norm)
                preds.append(T_next.cpu().numpy())
                T_current = T_next

        preds_K = normalizer.denormalize(np.array(preds))
        trues_K = normalizer.denormalize(T_traj.cpu().numpy())
        errors = np.abs(preds_K - trues_K)[:, valid_np]

        all_mae.append(errors.mean())
        all_max.append(errors.max())
        if traj_idx == 0:
            T_pred_save, T_true_save = preds_K, trues_K

    print(f'\n{name}:', flush=True)
    print(f'  MAE: {np.mean(all_mae):.4f}K', flush=True)
    print(f'  Avg Max: {np.mean(all_max):.4f}K', flush=True)
    print(f'  Worst Max: {np.max(all_max):.4f}K', flush=True)
    return np.mean(all_mae), np.max(all_max), T_pred_save, T_true_save, mask_np


if __name__ == '__main__':
    domain = IrregularFinsDomain(domain_size=(1.0, 1.0), fin_type='wavy', n_fins=3, base_height=0.1)

    print('=' * 60, flush=True)
    print('Causal-PINO V10: Autoregressive + Numba Accelerated', flush=True)
    print('Target: Max Error < 0.8K', flush=True)
    print('=' * 60, flush=True)

    print('\n[Phase 1] Training at 32x32...', flush=True)
    train_data = generate_data(domain, nx=32, ny=32, nt=10, n_traj=80, device=device, seed=SEED)
    test_data = generate_data(domain, nx=32, ny=32, nt=10, n_traj=20, device=device, seed=SEED+1000)

    model = AutoregressivePINO(modes=16, width=96, n_layers=5, device=device)
    print(f'Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M', flush=True)

    model = train_autoregressive(model, train_data, n_epochs=2000, device=device)

    print('\n' + '=' * 60, flush=True)
    eval_full(model, train_data, 'Train Set (32x32)')
    mae, max_err, T_pred, T_true, mask_np = eval_full(model, test_data, 'Test Set (32x32)')

    torch.save(model.state_dict(), os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "checkpoints", "causal_pino_v10_clean.pth"))

    X_np, Y_np = test_data['X'].cpu().numpy(), test_data['Y'].cpu().numpy()
    valid = mask_np > 0.5

    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    for col, t_idx in enumerate(range(0, 10, 2)):
        vmin, vmax = T_true[:, valid].min(), T_true[:, valid].max()
        for row, (arr, title) in enumerate([(T_true[t_idx], 'True'), (T_pred[t_idx], 'Pred'), (np.abs(T_pred[t_idx] - T_true[t_idx]), 'Error')]):
            ax = axes[row, col]
            masked = np.ma.masked_where(mask_np < 0.5, arr)
            if row < 2:
                im = ax.contourf(X_np, Y_np, masked.T, levels=20, cmap='jet', vmin=vmin, vmax=vmax)
            else:
                im = ax.contourf(X_np, Y_np, masked.T, levels=20, cmap='hot')
            ax.set_title(f't={t_idx} {title}')
            ax.set_aspect('equal')
            plt.colorbar(im, ax=ax)

    plt.suptitle(f'V10 Autoregressive (Numba)\nMAE:{mae:.4f}K Max:{max_err:.4f}K', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "figures", "v10_clean_results.png"), dpi=150)
    print(f'\nSaved: v10_clean_results.png', flush=True)
    plt.close()

    print('\n' + '=' * 60, flush=True)
    status = 'OK' if max_err < 0.8 else '--' if max_err < 1.0 else 'XX'
    print(f'[{status}] Max Error: {max_err:.4f}K (Target: <0.8K)', flush=True)
    print('=' * 60, flush=True)
