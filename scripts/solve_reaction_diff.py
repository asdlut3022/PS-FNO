# Reaction-diffusion equation solver + data generation
# PDE: dT/dt = alpha * laplacian(T) - beta * (T - T_env)
#
# Physical scenario: heat sink fin + surface convective/radiative cooling
# - alpha: thermal diffusion coefficient (material property)
# - beta: surface heat loss coefficient (linearized equivalent of convection + radiation)
#
# Spectral kernel: D(k,dt) = exp(-(alpha*|k|^2 + beta)*dt)
#   - alpha*|k|^2: frequency-dependent diffusion decay
#   - beta: uniform volumetric cooling decay

import numpy as np
import torch
from numba import njit, prange


@njit(parallel=True, cache=True)
def solve_reaction_diff_numba(nx, ny, T_hot, T_amb, alpha, Bi, beta, t_total, nt_save, ramp_fraction=0.7):
    """Numba-accelerated reaction-diffusion solver (time-dependent ramp heating BC)

    PDE: dT/dt = alpha * (d2T/dx2 + d2T/dy2) - beta * (T - T_amb)

    Spatial discretization:
      - diffusion term: second-order central difference
      - reaction term: treated implicitly (analytic integrating factor)

    Boundary conditions (same ramp BC as conv_diff):
      - bottom (j=0): 3 discrete heat patches, T_hot(t) = T_amb + (T_hot-T_amb)*min(t/t_ramp, 1.0)
      - top (j=ny-1): Robin convection
      - left/right: Neumann zero flux

    Args:
        nx, ny: grid resolution
        T_hot: final heat source temperature (K)
        T_amb: ambient temperature (K)
        alpha: thermal diffusion coefficient
        Bi: Biot number
        beta: reaction (cooling) coefficient (s^-1)
        t_total: total simulation time
        nt_save: number of saved time snapshots
        ramp_fraction: fraction of the total time taken by the BC ramp-up phase (0~1)
    """
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)

    # CFL condition: diffusion stability + reaction term stability
    dt_diff = min(dx, dy) ** 2 / (4 * alpha)
    dt_react = 1.0 / (beta + 1e-10)  # characteristic time of the reaction term
    dt_max = min(dt_diff, dt_react * 0.5)
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

                # Reaction term: -beta * (T - T_amb)
                reaction = -beta * (T[i, j] - T_amb)

                T_new[i, j] = T[i, j] + dt * (
                    alpha * (d2Tdx2 + d2Tdy2) + reaction
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
                T_new[i, 0] = T_amb + (T_hot_t - T_amb) * 0.2
            T_new[i, ny - 1] = (T_new[i, ny - 2] + Bi * dy * T_amb) / (1 + Bi * dy)
        for j in range(ny):
            T_new[0, j] = T_new[1, j]
            T_new[nx - 1, j] = T_new[nx - 2, j]
        T = T_new

        if (step % save_every == 0 or step == n_steps) and save_idx < nt_save:
            T_history[save_idx] = T.copy()
            t_history[save_idx] = step * dt
            save_idx += 1

    return T_history, t_history


def generate_reaction_diff_data(domain, nx=32, ny=32, nt=10, n_traj=100,
                                 device='cuda', seed=None, ramp_fraction=0.7):
    """Generate reaction-diffusion training/test data (ramp BC)

    Parameter ranges:
      - T_hot: U(345, 355) K
      - T_amb: U(300, 310) K
      - alpha: U(0.016, 0.024)
      - Bi: U(0.4, 0.8)
      - beta: U(0.5, 2.0) — surface heat loss coefficient

    Output format matches generate_conv_diff_data, including alpha_array and beta_array
    """
    from train_v10_clean import Normalizer
    from irregular_domain import DomainUtils

    if seed is not None:
        np.random.seed(seed)

    print(f'Generating {n_traj} reaction-diff trajectories (ramp BC, numba)...', flush=True)
    normalizer = Normalizer()
    X, Y = DomainUtils.create_grid(nx, ny, (0, 1.0), (0, 1.0), device)
    mask = domain.mask(X, Y)
    all_T = []
    alpha_list = []
    beta_list = []

    # Warmup numba
    _ = solve_reaction_diff_numba(nx, ny, 350.0, 305.0, 0.02, 0.6, 1.0, 1.0, nt, ramp_fraction)

    t_array = None
    for i in range(n_traj):
        T_hot = 345 + np.random.rand() * 10
        T_amb = 300 + np.random.rand() * 10
        alpha = 0.016 + np.random.rand() * 0.008
        Bi = 0.4 + np.random.rand() * 0.4
        beta = 0.5 + np.random.rand() * 1.5   # U(0.5, 2.0)

        T_hist, t_arr = solve_reaction_diff_numba(nx, ny, T_hot, T_amb, alpha, Bi, beta,
                                                    1.0, nt, ramp_fraction)
        if t_array is None:
            t_array = t_arr
        T_norm = normalizer.normalize(T_hist)
        all_T.append(torch.from_numpy(T_norm).float().to(device))
        alpha_list.append(alpha)
        beta_list.append(beta)

    print(f'Generated {n_traj} reaction-diff trajectories', flush=True)
    print(f'  alpha range: [{min(alpha_list):.4f}, {max(alpha_list):.4f}]')
    print(f'  beta range: [{min(beta_list):.3f}, {max(beta_list):.3f}]')

    return {
        'T': torch.stack(all_T, dim=0),
        'mask': mask, 'X': X, 'Y': Y,
        't': torch.from_numpy(t_array).float().to(device),
        'normalizer': normalizer,
        'alpha_array': np.array(alpha_list),
        'beta_array': np.array(beta_list),
    }
