# 2D viscoelastic wave equation solver + data generation (Kelvin-Voigt damping)
# PDE: d2u/dt2 + 2*gamma*du/dt = c^2 * laplacian(u) + eta * laplacian(du/dt)
#
# Physical setting: stress wave propagation in viscoelastic materials / polymer plate vibration
# - c: wave speed (elastic material property)
# - gamma: structural damping coefficient (frequency-independent uniform attenuation)
# - eta: viscoelastic damping coefficient (Kelvin-Voigt viscosity, frequency-dependent attenuation)
#
# Spectral kernel: D(k,dt) = exp(-(gamma + eta*|k|^2)*dt) * exp(i*c*|k|*dt)
#   - exp(-gamma*dt): uniform damping attenuation (independent of the mode)
#   - exp(-eta*|k|^2*dt): viscoelastic attenuation (high-frequency modes decay quickly, diffusion-like)
#   - exp(i*c*|k|*dt): oscillation phase
#
# Key differences from other PDEs:
#   - Heat: exp(-alpha*|k|^2*dt) — pure attenuation, no oscillation
#   - Pure wave: exp(-gamma*dt)*exp(i*c*|k|*dt) — uniform attenuation + oscillation
#   - Viscoelastic wave: exp(-(gamma+eta|k|^2)*dt)*exp(i*c*|k|*dt) — frequency-dependent attenuation + oscillation
#     (naturally mixes diffusive and wave behavior)
#
# Time discretization: modified Störmer-Verlet (central difference, second-order accurate)
#   u^{n+1} = [2*u^n - (1-gamma*dt)*u^{n-1} + c^2*dt^2*lap(u^n)
#              + eta*dt*(lap(u^n) - lap(u^{n-1}))] / (1+gamma*dt)
#
# CFL condition: min(c*dt/dx <= 1/sqrt(2), eta*dt/dx^2 <= 0.5) (2D)

import numpy as np
import os
import torch
from numba import njit, prange


@njit(parallel=True, cache=True)
def solve_wave_numba(nx, ny, c, gamma, amp, x0, y0, sigma,
                     t_total, nt_save,
                     bc_left='fixed', bc_right='fixed',
                     bc_bottom='fixed', bc_top='free',
                     cut_i=-1, cut_j=-1, eta=0.0):
    """Numba-accelerated 2D viscoelastic wave equation solver (Kelvin-Voigt, supports L-shaped domains)

    PDE: d2u/dt2 + 2*gamma*du/dt = c^2 * laplacian(u) + eta * laplacian(du/dt)

    Spatial discretization: second-order central difference (Laplacian)
    Time discretization: modified Störmer-Verlet with Kelvin-Voigt damping
      u^{n+1} = [2*u^n - (1-gamma*dt)*u^{n-1} + c^2*dt^2*lap(u^n)
                 + eta*dt*(lap(u^n) - lap(u^{n-1}))] / (1+gamma*dt)

    Initial conditions:
      u(x,y,0) = amp * exp(-((x-x0)^2 + (y-y0)^2) / (2*sigma^2))  (Gaussian pulse)
      du/dt(x,y,0) = 0  (zero initial velocity)

    Boundary conditions:
      fixed (Dirichlet): u = 0
      free (Neumann): du/dn = 0
      L-shaped domain interior boundary: Dirichlet u = 0 (enforced by zeroing the cut region at each step)

    Args:
        nx, ny: grid resolution
        c: wave speed
        gamma: structural damping coefficient
        amp: initial pulse amplitude
        x0, y0: pulse center position
        sigma: pulse width (Gaussian std)
        t_total: total simulation time
        nt_save: number of time snapshots to save
        bc_left/right/bottom/top: boundary condition type ('fixed' or 'free')
        cut_i: starting row index of the L-shaped domain cut (-1 means no cut)
        cut_j: starting column index of the L-shaped domain cut
        eta: Kelvin-Voigt viscoelastic damping coefficient (>=0, 0 means the pure wave equation)

    Returns:
        u_history: (nt_save, nx, ny) time series of the displacement field
        t_history: (nt_save,) corresponding time points
    """
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)

    # CFL condition: c*dt/h <= 1/sqrt(2) for 2D
    h_min = min(dx, dy)
    dt_cfl = h_min / (c * 1.4142135623730951)  # sqrt(2) ≈ 1.414
    # Damping stability
    dt_damp = 1.0 / (gamma + 1e-10)
    # Viscoelastic stability: in the two-step scheme eta*dt/((1+gamma*dt)*dx^2) < 0.25 (2D)
    # Conservative estimate: dt < 0.125 * dx^2 / eta
    dt_visco = 0.125 * h_min * h_min / (eta + 1e-10) if eta > 0 else 1e10
    dt = min(0.5 * dt_cfl, 0.5 * dt_damp, dt_visco)  # safety factor 0.5 (CFL/damp)

    n_steps = int(t_total / dt) + 1
    save_every = max(1, n_steps // (nt_save - 1))

    has_cut = (cut_i > 0 and cut_j > 0)

    # Initial condition: Gaussian pulse, zero velocity
    u_curr = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            xi = i * dx
            yj = j * dy
            r2 = (xi - x0) ** 2 + (yj - y0) ** 2
            u_curr[i, j] = amp * np.exp(-r2 / (2.0 * sigma ** 2))

    # The L-shaped domain constraint must also be applied to the initial condition
    if has_cut:
        for i in range(cut_i, nx):
            for j in range(cut_j, ny):
                u_curr[i, j] = 0.0

    # Zero initial velocity => u^{-1} ≈ u^0 - dt*v^0 + 0.5*dt^2*a^0
    # v^0 = 0, a^0 = c^2*lap(u^0)
    u_prev = u_curr.copy()

    c2dt2 = c * c * dt * dt
    denom = 1.0 + gamma * dt
    coeff_curr = 2.0 / denom
    coeff_prev = -(1.0 - gamma * dt) / denom
    coeff_lap = c2dt2 / denom
    coeff_visco = eta * dt / denom  # coefficient of the Kelvin-Voigt viscoelastic term

    # Save history
    u_history = np.zeros((nt_save, nx, ny))
    t_history = np.zeros(nt_save)
    u_history[0] = u_curr.copy()
    t_history[0] = 0.0
    save_idx = 1

    # Special treatment of the first step (zero initial velocity)
    u_next = np.zeros((nx, ny))
    for i in prange(1, nx - 1):
        for j in range(1, ny - 1):
            lap = (u_curr[i + 1, j] - 2 * u_curr[i, j] + u_curr[i - 1, j]) / (dx * dx) \
                + (u_curr[i, j + 1] - 2 * u_curr[i, j] + u_curr[i, j - 1]) / (dy * dy)
            # Zero initial velocity: u^1 = u^0 + 0.5*c^2*dt^2*lap(u^0)
            # (from the Taylor expansion, since v^0=0)
            u_next[i, j] = u_curr[i, j] + 0.5 * c2dt2 * lap

    # Apply the boundary conditions
    # Left (i=0)
    if bc_left == 'fixed':
        for j in range(ny):
            u_next[0, j] = 0.0
    else:
        for j in range(ny):
            u_next[0, j] = u_next[1, j]
    # Right (i=nx-1)
    if bc_right == 'fixed':
        for j in range(ny):
            u_next[nx - 1, j] = 0.0
    else:
        for j in range(ny):
            u_next[nx - 1, j] = u_next[nx - 2, j]
    # Bottom (j=0)
    if bc_bottom == 'fixed':
        for i in range(nx):
            u_next[i, 0] = 0.0
    else:
        for i in range(nx):
            u_next[i, 0] = u_next[i, 1]
    # Top (j=ny-1)
    if bc_top == 'fixed':
        for i in range(nx):
            u_next[i, ny - 1] = 0.0
    else:
        for i in range(nx):
            u_next[i, ny - 1] = u_next[i, ny - 2]

    # L-shaped domain constraint: force u=0 in the cut region (must be applied at every step!)
    if has_cut:
        for i in range(cut_i, nx):
            for j in range(cut_j, ny):
                u_next[i, j] = 0.0

    u_prev = u_curr.copy()
    u_curr = u_next.copy()

    if save_every == 1 and save_idx < nt_save:
        u_history[save_idx] = u_curr.copy()
        t_history[save_idx] = dt
        save_idx += 1

    # Main loop (step >= 2)
    for step in range(2, n_steps + 1):
        u_next = np.zeros((nx, ny))
        for i in prange(1, nx - 1):
            for j in range(1, ny - 1):
                lap_curr = (u_curr[i + 1, j] - 2 * u_curr[i, j] + u_curr[i - 1, j]) / (dx * dx) \
                    + (u_curr[i, j + 1] - 2 * u_curr[i, j] + u_curr[i, j - 1]) / (dy * dy)
                lap_prev = (u_prev[i + 1, j] - 2 * u_prev[i, j] + u_prev[i - 1, j]) / (dx * dx) \
                    + (u_prev[i, j + 1] - 2 * u_prev[i, j] + u_prev[i, j - 1]) / (dy * dy)
                u_next[i, j] = coeff_curr * u_curr[i, j] + coeff_prev * u_prev[i, j] \
                    + coeff_lap * lap_curr + coeff_visco * (lap_curr - lap_prev)

        # Boundary conditions
        if bc_left == 'fixed':
            for j in range(ny):
                u_next[0, j] = 0.0
        else:
            for j in range(ny):
                u_next[0, j] = u_next[1, j]
        if bc_right == 'fixed':
            for j in range(ny):
                u_next[nx - 1, j] = 0.0
        else:
            for j in range(ny):
                u_next[nx - 1, j] = u_next[nx - 2, j]
        if bc_bottom == 'fixed':
            for i in range(nx):
                u_next[i, 0] = 0.0
        else:
            for i in range(nx):
                u_next[i, 0] = u_next[i, 1]
        if bc_top == 'fixed':
            for i in range(nx):
                u_next[i, ny - 1] = 0.0
        else:
            for i in range(nx):
                u_next[i, ny - 1] = u_next[i, ny - 2]

        # L-shaped domain constraint: force u=0 in the cut region (must be applied at every step!)
        if has_cut:
            for i in range(cut_i, nx):
                for j in range(cut_j, ny):
                    u_next[i, j] = 0.0

        u_prev = u_curr.copy()
        u_curr = u_next.copy()

        if (step % save_every == 0 or step == n_steps) and save_idx < nt_save:
            u_history[save_idx] = u_curr.copy()
            t_history[save_idx] = step * dt
            save_idx += 1

    return u_history, t_history


class WaveNormalizer:
    """Wave equation normalizer

    Unlike heat conduction, the displacement of the wave equation can take positive and negative values.
    Uses symmetric normalization: u_norm = u / scale, so that the data roughly lie in [-1, 1].
    """
    def __init__(self, scale=1.0):
        self.scale = scale

    def normalize(self, u):
        return u / self.scale

    def denormalize(self, u):
        return u * self.scale

    def fit(self, u_all):
        """Determine the scale adaptively from the data"""
        max_abs = np.max(np.abs(u_all))
        self.scale = max_abs if max_abs > 1e-10 else 1.0
        return self


def generate_wave_data(domain, nx=32, ny=32, nt=11, n_traj=100,
                       device='cuda', seed=None,
                       t_total=1.0, c_range=(0.8, 1.5),
                       gamma_range=(0.1, 0.5), amp_range=(0.5, 1.5),
                       sigma_range=(0.08, 0.15),
                       cut_frac_x=0.5, cut_frac_y=0.5,
                       eta=0.0):
    """Generate training/test data for the viscoelastic wave equation

    Physical setting: stress wave propagation in viscoelastic materials / polymer plate vibration

    Note: nt=11 because the wave equation needs two steps, u(t) and u(t-dt), as input,
    so 11 snapshots produce 9 prediction steps (consistent with 10 snapshots -> 9 steps for the heat equation)

    Parameter ranges (configurable):
      - c: U(c_range[0], c_range[1])  wave speed
      - gamma: U(gamma_range[0], gamma_range[1])  structural damping coefficient
      - eta: fixed Kelvin-Voigt viscoelastic coefficient (shared by all trajectories)
      - amp: U(amp_range[0], amp_range[1])  initial pulse amplitude
      - x0, y0: random position (avoiding the boundaries and obstacles)
      - sigma: U(sigma_range[0], sigma_range[1])  pulse width

    Domain: uses the passed-in domain object (should support an SDF mask)
    Boundary conditions: left/right fixed (Dirichlet u=0), bottom/top free (Neumann du/dn=0)
    L-shaped domain: cut_frac_x/y specify the cut position (the upper-right quadrant is removed)

    Output format:
      'U': (n_traj, nt, nx, ny) normalized displacement field (including the t0 step)
      'mask': (nx, ny) domain mask
      'X', 'Y': (nx, ny) grid coordinates
      't': (nt,) time array
      'normalizer': WaveNormalizer
      'c_array': (n_traj,) wave speed
      'gamma_array': (n_traj,) damping coefficient
    """
    from irregular_domain import DomainUtils

    if seed is not None:
        np.random.seed(seed)

    print(f'Generating {n_traj} wave trajectories...', flush=True)

    X, Y = DomainUtils.create_grid(nx, ny, (0, 1.0), (0, 1.0), device)
    mask = domain.mask(X, Y)
    mask_np = mask.cpu().numpy()

    # Compute the L-shaped domain cut indices (auto-detected from the mask)
    # For a standard L-shaped domain (cut_frac=0.5): cut_i = cut_j = 16 (when nx=32)
    cut_i = int(np.ceil(cut_frac_x * (nx - 1)))
    cut_j = int(np.ceil(cut_frac_y * (ny - 1)))
    # Verify: make sure the mask is indeed 0 in the cut region
    if mask_np[min(cut_i, nx-1), min(cut_j, ny-1)] > 0.5:
        print(f'  WARNING: cut region ({cut_i},{cut_j}) is inside domain, disabling cut')
        cut_i = -1
        cut_j = -1
    else:
        print(f'  L-shaped domain: cut region i>={cut_i}, j>={cut_j}')

    # Warm up numba (with the cut parameters and eta)
    _ = solve_wave_numba(nx, ny, 1.0, 0.3, 1.0, 0.5, 0.5, 0.1, 0.5, nt,
                         cut_i=cut_i, cut_j=cut_j, eta=eta)

    all_u_raw = []
    c_list = []
    gamma_list = []
    t_array = None

    for i in range(n_traj):
        c = c_range[0] + np.random.rand() * (c_range[1] - c_range[0])
        gamma_val = gamma_range[0] + np.random.rand() * (gamma_range[1] - gamma_range[0])
        amp = amp_range[0] + np.random.rand() * (amp_range[1] - amp_range[0])
        sigma = sigma_range[0] + np.random.rand() * (sigma_range[1] - sigma_range[0])

        # Pulse position: chosen randomly inside the domain, away from the boundaries
        x0 = 0.5  # default value
        y0 = 0.5
        for _ in range(50):
            x0_cand = 0.2 + np.random.rand() * 0.6   # [0.2, 0.8]
            y0_cand = 0.2 + np.random.rand() * 0.6   # [0.2, 0.8]
            # Check whether it is inside the domain (discrete check)
            ix = int(x0_cand * (nx - 1))
            iy = int(y0_cand * (ny - 1))
            ix = min(max(ix, 0), nx - 1)
            iy = min(max(iy, 0), ny - 1)
            if mask_np[ix, iy] > 0.5:
                x0 = x0_cand
                y0 = y0_cand
                break

        u_hist, t_arr = solve_wave_numba(
            nx, ny, c, gamma_val, amp, x0, y0, sigma,
            t_total=t_total, nt_save=nt,
            bc_left='fixed', bc_right='fixed',
            bc_bottom='free', bc_top='free',
            cut_i=cut_i, cut_j=cut_j, eta=eta
        )

        # Safety mask: ensure regions outside the domain are exactly zero (in theory the solver already guarantees this)
        for t_idx in range(nt):
            u_hist[t_idx] = u_hist[t_idx] * mask_np

        all_u_raw.append(u_hist)
        c_list.append(c)
        gamma_list.append(gamma_val)
        if t_array is None:
            t_array = t_arr

    # Adaptive normalization
    all_u_stacked = np.stack(all_u_raw, axis=0)  # (n_traj, nt, nx, ny)
    normalizer = WaveNormalizer()
    normalizer.fit(all_u_stacked)

    all_U = []
    for i in range(n_traj):
        u_norm = normalizer.normalize(all_u_raw[i])
        all_U.append(torch.from_numpy(u_norm).float().to(device))

    print(f'Generated {n_traj} wave trajectories', flush=True)
    print(f'  c range: [{min(c_list):.3f}, {max(c_list):.3f}]')
    print(f'  gamma range: [{min(gamma_list):.3f}, {max(gamma_list):.3f}]')
    print(f'  eta (Kelvin-Voigt): {eta:.4f}')
    print(f'  normalizer scale: {normalizer.scale:.4f}')

    return {
        'U': torch.stack(all_U, dim=0),       # (n_traj, nt, nx, ny)
        'mask': mask,                           # (nx, ny)
        'X': X, 'Y': Y,                        # (nx, ny)
        't': torch.from_numpy(t_array).float().to(device),  # (nt,)
        'normalizer': normalizer,
        'c_array': np.array(c_list),
        'gamma_array': np.array(gamma_list),
    }


if __name__ == '__main__':
    """Verify the solver: visualization of a single wave propagation"""
    import matplotlib.pyplot as plt

    print('Wave equation solver verification...')

    # Simple rectangular domain test
    nx, ny, nt = 64, 64, 20
    c, gamma = 1.2, 0.2
    amp, sigma = 1.0, 0.1

    u_hist, t_hist = solve_wave_numba(
        nx, ny, c, gamma, amp, 0.5, 0.5, sigma,
        t_total=1.0, nt_save=nt
    )

    print(f'  Time steps saved: {nt}')
    print(f'  u range: [{u_hist.min():.4f}, {u_hist.max():.4f}]')
    print(f'  Energy at t=0: {np.sum(u_hist[0]**2):.4f}')
    print(f'  Energy at t=final: {np.sum(u_hist[-1]**2):.4f}')
    print(f'  Energy ratio: {np.sum(u_hist[-1]**2) / (np.sum(u_hist[0]**2) + 1e-10):.4f}')

    # Visualization
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for col, t_idx in enumerate([0, 2, 5, 10, nt-1]):
        vmax = np.abs(u_hist).max()
        for row, (t_i, title) in enumerate([
            (t_idx, f't={t_hist[t_idx]:.3f}'),
            (t_idx, f'|u| at t={t_hist[t_idx]:.3f}')
        ]):
            ax = axes[row, col]
            if row == 0:
                im = ax.imshow(u_hist[t_i].T, origin='lower', cmap='RdBu_r',
                              vmin=-vmax, vmax=vmax, extent=[0,1,0,1])
            else:
                im = ax.imshow(np.abs(u_hist[t_i]).T, origin='lower', cmap='hot',
                              vmin=0, vmax=vmax, extent=[0,1,0,1])
            ax.set_title(title)
            plt.colorbar(im, ax=ax)

    plt.suptitle(f'Damped Wave: c={c}, gamma={gamma}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "figures", "wave_solver_verification.png"), dpi=150)
    plt.close()
    print('Saved: wave_solver_verification.png')
