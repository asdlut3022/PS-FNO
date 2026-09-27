# Convection-diffusion equation solver + data generation
# PDE: dT/dt = alpha * laplacian(T) - vx * dT/dx - vy * dT/dy
#
# Compared with solve_heat_numba in train_v10_clean.py:
#   - adds the advection term (upwind scheme)
#   - stricter CFL condition (constrained by both diffusion and advection)
#   - output format is fully compatible

import numpy as np
import torch
from numba import njit, prange


@njit(parallel=True, cache=True)
def solve_conv_diff_numba(nx, ny, T_hot, T_amb, alpha, Bi, vx, vy, t_total, nt_save, ramp_fraction=0.7):
    """Numba-accelerated convection-diffusion solver (time-dependent ramp heating BC)

    PDE: dT/dt = alpha * (d2T/dx2 + d2T/dy2) - vx * dT/dx - vy * dT/dy

    Spatial discretization:
      - diffusion term: second-order central difference
      - advection term: first-order upwind scheme

    Boundary conditions (ramp BC):
      - bottom (j=0): 3 discrete heat patches, T_hot(t) = T_amb + (T_hot-T_amb)*min(t/t_ramp, 1.0)
      - top (j=ny-1): Robin convection
      - left/right: Neumann zero flux

    Initial condition: uniform cold state T_amb (simulating chip cold start + convective purge)

    Args:
        nx, ny: grid resolution
        T_hot: final heat source temperature (K)
        T_amb: ambient temperature (K)
        alpha: thermal diffusion coefficient
        Bi: Biot number
        vx, vy: advection velocity components
        t_total: total simulation time
        nt_save: number of saved time snapshots
        ramp_fraction: fraction of the total time taken by the BC ramp-up phase (0~1)
    """
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)

    # CFL condition: constrained by both diffusion and advection
    dt_diff = min(dx, dy) ** 2 / (4 * alpha)
    dt_advx = dx / (abs(vx) + 1e-10)
    dt_advy = dy / (abs(vy) + 1e-10)
    dt_max = min(dt_diff, dt_advx, dt_advy)
    dt = min(0.0001, dt_max * 0.4)  # safety factor 0.4

    n_steps = int(t_total / dt)
    save_every = max(1, n_steps // (nt_save - 1))
    t_ramp = ramp_fraction * t_total

    # Heat patch positions (3 patches, simulating chip heat sources)
    patch_centers = (0.2, 0.5, 0.8)
    patch_half_width = 0.06

    # Initial condition: uniform cold state
    T = np.ones((nx, ny)) * T_amb

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
                # Diffusion term (central difference)
                d2Tdx2 = (T[i + 1, j] - 2 * T[i, j] + T[i - 1, j]) / (dx * dx)
                d2Tdy2 = (T[i, j + 1] - 2 * T[i, j] + T[i, j - 1]) / (dy * dy)

                # Advection term (upwind scheme)
                if vx >= 0:
                    dTdx = (T[i, j] - T[i - 1, j]) / dx
                else:
                    dTdx = (T[i + 1, j] - T[i, j]) / dx

                if vy >= 0:
                    dTdy = (T[i, j] - T[i, j - 1]) / dy
                else:
                    dTdy = (T[i, j + 1] - T[i, j]) / dy

                T_new[i, j] = T[i, j] + dt * (
                    alpha * (d2Tdx2 + d2Tdy2) - vx * dTdx - vy * dTdy
                )

        # Bottom BC: discrete heat patches (ramp heating)
        for i in range(nx):
            x_val = i * dx
            is_patch = False
            for pc in patch_centers:
                if abs(x_val - pc) < patch_half_width:
                    is_patch = True
            if is_patch:
                T_new[i, 0] = T_hot_t
            else:
                T_new[i, 0] = T_amb + (T_hot_t - T_amb) * 0.2  # low-temperature background also follows the ramp
            T_new[i, ny - 1] = (T_new[i, ny - 2] + Bi * dy * T_amb) / (1 + Bi * dy)  # top: Robin
        for j in range(ny):
            T_new[0, j] = T_new[1, j]          # left: Neumann
            T_new[nx - 1, j] = T_new[nx - 2, j]  # right: Neumann
        T = T_new

        if (step % save_every == 0 or step == n_steps) and save_idx < nt_save:
            T_history[save_idx] = T.copy()
            t_history[save_idx] = step * dt
            save_idx += 1

    return T_history, t_history


def generate_conv_diff_data(domain, nx=32, ny=32, nt=10, n_traj=100,
                            device='cuda', seed=None, ramp_fraction=0.7):
    """Generate convection-diffusion training/test data (ramp BC)

    Parameter ranges:
      - T_hot: U(345, 355) K
      - T_amb: U(300, 310) K
      - alpha: U(0.016, 0.024)
      - Bi: U(0.4, 0.8)
      - vx: U(0.5, 2.0)  (horizontal advection, Pe ~ 20-125)
      - vy: 0  (simplification: horizontal advection only)

    Output format matches train_v10_clean.generate_data, additionally including vx_array
    """
    from train_v10_clean import Normalizer
    from irregular_domain import DomainUtils

    if seed is not None:
        np.random.seed(seed)

    print(f'Generating {n_traj} conv-diff trajectories (ramp BC, numba)...', flush=True)
    normalizer = Normalizer()
    X, Y = DomainUtils.create_grid(nx, ny, (0, 1.0), (0, 1.0), device)
    mask = domain.mask(X, Y)
    all_T = []
    vx_list = []
    alpha_list = []

    # Warmup numba
    _ = solve_conv_diff_numba(nx, ny, 350.0, 305.0, 0.02, 0.6, 1.0, 0.0, 1.0, nt, ramp_fraction)

    t_array = None
    for i in range(n_traj):
        T_hot = 345 + np.random.rand() * 10
        T_amb = 300 + np.random.rand() * 10
        alpha = 0.016 + np.random.rand() * 0.008
        Bi = 0.4 + np.random.rand() * 0.4
        vx = 0.5 + np.random.rand() * 1.5   # U(0.5, 2.0)
        vy = 0.0

        T_hist, t_arr = solve_conv_diff_numba(nx, ny, T_hot, T_amb, alpha, Bi, vx, vy, 1.0, nt, ramp_fraction)
        if t_array is None:
            t_array = t_arr
        T_norm = normalizer.normalize(T_hist)
        all_T.append(torch.from_numpy(T_norm).float().to(device))
        vx_list.append(vx)
        alpha_list.append(alpha)

    print(f'Generated {n_traj} conv-diff trajectories', flush=True)
    print(f'  vx range: [{min(vx_list):.3f}, {max(vx_list):.3f}]')
    print(f'  alpha range: [{min(alpha_list):.4f}, {max(alpha_list):.4f}]')
    print(f'  Pe range: [{min(vx_list)/max(alpha_list):.1f}, {max(vx_list)/min(alpha_list):.1f}]')

    return {
        'T': torch.stack(all_T, dim=0),
        'mask': mask, 'X': X, 'Y': Y,
        't': torch.from_numpy(t_array).float().to(device),
        'normalizer': normalizer,
        'vx_array': np.array(vx_list),
        'alpha_array': np.array(alpha_list),
    }
