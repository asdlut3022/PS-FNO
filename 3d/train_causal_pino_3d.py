"""
3D Causal-PINO: LTNE Porous Media Multi-Physics
====================================================
Turbine blade porous-media cooling — LTNE (Local Thermal Non-Equilibrium) model

Physical fields (two complete 3D fields, coupled over the whole volume):
- Solid temperature T_s: 3D nonlinear heat conduction + EM volumetric heat source + surface convection/radiation + volumetric coupling β_s(T_f-T_s)
- Fluid temperature T_f: 3D diffusion + z-direction Darcy advection + volumetric coupling β_f(T_s-T_f)

PDE:
  ∂T_s/∂t = ∇·(α_s(T_s)∇T_s) + Q_em/(ρcp)_s + β_s·(T_f - T_s)    [reaction-diffusion type]
  ∂T_f/∂t = α_f·∇²T_f - v_z·∂T_f/∂z + β_f·(T_s - T_f)              [advection-diffusion type]

  where β_s = h_sf·a_sf / (ε_s·ρ_s·cp_s),  β_f = h_sf·a_sf / (ε_f·ρ_f·cp_f)
  h_sf·a_sf is the volumetric heat transfer coefficient [W/m³K], acting over the whole volume

Nonlinearities:
- k_s(T) = 11 + 0.015*(T-300) W/mK  (Inconel 718)
- cp_s(T) = 435 + 0.22*(T-300) J/kgK
- σ(T) = σ_0 / (1 + 0.005*(T-300))
- Radiation: q_rad = ε·σ_SB·(T⁴ - T_amb⁴), ε=0.5

FNO input: T_solid(1) + T_fluid(1) + grid(3) + dt(1) = 6 channels
FNO output: ΔT_solid(1) + ΔT_fluid(1) = 2 channels
"""

import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.empty_cache()


# ============================================================
# Part 1: 3D Turbine Blade Geometry
# ============================================================

class TurbineBladeGeometry:
    """3D Turbine Blade"""
    def __init__(self, chord=1.0, span=2.0, naca='0012', twist_angle=45.0,
                 n_cooling_holes=3, hole_radius=0.025, device='cuda'):
        self.chord = chord
        self.span = span
        self.twist_angle = np.radians(twist_angle)
        self.t = int(naca[2:]) / 100
        self.hole_radius = hole_radius
        self.hole_positions = [0.3, 0.5, 0.7]
        self.device = device

    def naca_thickness(self, x):
        x = np.clip(x, 0, 1)
        t = self.t
        return 5 * t * (0.2969 * np.sqrt(x + 1e-10) - 0.1260 * x
                        - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)

    def get_twist_angle(self, z):
        return self.twist_angle * (z / self.span)

    def is_inside_blade(self, x, y, z):
        z_valid = (z >= 0) & (z <= self.span)
        theta = self.get_twist_angle(np.clip(z, 0, self.span))
        cx = self.chord / 2
        x_rot = (x - cx) * np.cos(-theta) - y * np.sin(-theta) + cx
        y_rot = (x - cx) * np.sin(-theta) + y * np.cos(-theta)
        x_norm = np.clip(x_rot / self.chord, 0, 1)
        yt = self.naca_thickness(x_norm) * self.chord
        in_airfoil = (np.abs(y_rot) <= yt) & (x_rot >= 0) & (x_rot <= self.chord)

        in_hole = np.zeros_like(x, dtype=bool)
        for x_pos in self.hole_positions:
            hole_x = x_pos * self.chord
            hx = (hole_x - cx) * np.cos(theta) + cx
            hy = (hole_x - cx) * np.sin(theta)
            dist = np.sqrt((x - hx)**2 + (y - hy)**2)
            in_hole = in_hole | (dist < self.hole_radius)

        return z_valid & in_airfoil & ~in_hole

    def create_mask(self, X, Y, Z):
        if isinstance(X, torch.Tensor):
            X_np, Y_np, Z_np = X.cpu().numpy(), Y.cpu().numpy(), Z.cpu().numpy()
        else:
            X_np, Y_np, Z_np = X, Y, Z
        mask = self.is_inside_blade(X_np, Y_np, Z_np).astype(np.float32)
        return torch.from_numpy(mask).float().to(self.device)

    def get_surface_distance(self, X, Y, Z):
        if isinstance(X, torch.Tensor):
            X_np, Y_np, Z_np = X.cpu().numpy(), Y.cpu().numpy(), Z.cpu().numpy()
        else:
            X_np, Y_np, Z_np = X, Y, Z

        theta = self.get_twist_angle(np.clip(Z_np, 0, self.span))
        cx = self.chord / 2
        x_rot = (X_np - cx) * np.cos(-theta) - Y_np * np.sin(-theta) + cx
        y_rot = (X_np - cx) * np.sin(-theta) + Y_np * np.cos(-theta)
        x_norm = np.clip(x_rot / self.chord, 0.01, 0.99)
        yt = self.naca_thickness(x_norm) * self.chord
        dist_surf = yt - np.abs(y_rot)
        dist_surf = np.maximum(dist_surf, 0.001)
        dist_z = np.minimum(Z_np, self.span - Z_np)
        dist_z = np.maximum(dist_z, 0.001)
        dist = np.minimum(dist_surf, dist_z)
        return torch.from_numpy(dist.astype(np.float32)).to(self.device)


def create_3d_grid(nx, ny, nz, bounds, device='cuda'):
    x = torch.linspace(bounds[0][0], bounds[0][1], nx, device=device)
    y = torch.linspace(bounds[1][0], bounds[1][1], ny, device=device)
    z = torch.linspace(bounds[2][0], bounds[2][1], nz, device=device)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    return X, Y, Z


# ============================================================
# Part 2: FNO Block
# ============================================================

class FNOBlock3d(nn.Module):
    """3D FNO Block"""
    def __init__(self, channels, modes=8):
        super().__init__()
        self.channels = channels
        self.modes = modes

        self.scale = 1 / (channels * channels)
        self.weights1 = nn.Parameter(self.scale * torch.rand(channels, channels, modes, modes, modes, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(channels, channels, modes, modes, modes, dtype=torch.cfloat))

        self.conv = nn.Conv3d(channels, channels, 1)
        self.norm = nn.InstanceNorm3d(channels)

    def compl_mul3d(self, x, weights):
        return torch.einsum("bixyz,ioxyz->boxyz", x, weights)

    def forward(self, x):
        batch_size = x.shape[0]
        residual = x

        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        out_ft = torch.zeros(batch_size, self.channels, x.size(-3), x.size(-2),
                            x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)

        out_ft[:, :, :self.modes, :self.modes, :self.modes] = self.compl_mul3d(
            x_ft[:, :, :self.modes, :self.modes, :self.modes], self.weights1)
        out_ft[:, :, -self.modes:, :self.modes, :self.modes] = self.compl_mul3d(
            x_ft[:, :, -self.modes:, :self.modes, :self.modes], self.weights2)

        x_ft_out = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        x_conv = self.conv(residual)

        return F.gelu(self.norm(x_ft_out + x_conv)) + residual


# ============================================================
# Part 3: Causal Prediction Model (2-channel: T_solid + T_coolant)
# ============================================================

class CausalPINO3D(nn.Module):
    """
    Causal prediction model (multi-physics CHT):
    Input: T_solid + T_f_embed + grid + dt = 6 channels
    Output: ΔT_solid + ΔT_f_embed = 2 channels
    """
    def __init__(self, modes=8, width=64, n_layers=5, device='cuda'):
        super().__init__()
        self.width = width
        self.device = device

        # Input: T_solid(1) + T_f_embed(1) + grid(3) + dt(1) = 6 channels
        self.fc0 = nn.Linear(6, width)

        self.fno_blocks = nn.ModuleList([
            FNOBlock3d(width, modes=modes)
            for _ in range(n_layers)
        ])

        # Output: ΔT_solid + ΔT_f_embed = 2 channels
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, 2)

        self.to(device)

    def forward(self, T_solid, T_f_embed, dt, grid):
        """
        Args:
            T_solid: [batch, nx, ny, nz]
            T_f_embed: [batch, nx, ny, nz]
            dt: scalar or [batch]
            grid: [1, nx, ny, nz, 3]
        Returns:
            T_solid_next, T_f_next: each [batch, nx, ny, nz]
        """
        batch_size = T_solid.shape[0]
        nx, ny, nz = T_solid.shape[1], T_solid.shape[2], T_solid.shape[3]

        if dt.dim() == 0:
            dt = dt.unsqueeze(0)
        dt_exp = dt.view(-1, 1, 1, 1, 1).expand(batch_size, nx, ny, nz, 1)

        x = torch.cat([
            T_solid.unsqueeze(-1),
            T_f_embed.unsqueeze(-1),
            grid.expand(batch_size, -1, -1, -1, -1),
            dt_exp
        ], dim=-1)  # [batch, nx, ny, nz, 6]

        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)

        for block in self.fno_blocks:
            x = block(x)

        x = x.permute(0, 2, 3, 4, 1)
        x = F.gelu(self.fc1(x))
        out = self.fc2(x)  # [batch, nx, ny, nz, 2]

        T_solid_next = T_solid + out[..., 0]
        T_f_next = T_f_embed + out[..., 1]

        return T_solid_next, T_f_next

    def rollout(self, T_s0, T_f0, n_steps, dt, grid, mask=None):
        T_s_hist, T_f_hist = [T_s0], [T_f0]
        T_s, T_f = T_s0, T_f0
        for _ in range(n_steps):
            T_s, T_f = self.forward(T_s, T_f, dt, grid)
            if mask is not None:
                T_s = T_s * mask
            T_s_hist.append(T_s)
            T_f_hist.append(T_f)
        return torch.stack(T_s_hist, dim=1), torch.stack(T_f_hist, dim=1)


# ============================================================
# Part 4: LTNE Porous Media Solver
# ============================================================

class LTNEPorousSolver:
    """
    3D LTNE (Local Thermal Non-Equilibrium) Porous Media Solver
    ============================================================
    Turbine blade porous cooling structure: solid skeleton + fluid seepage, coupled over the whole volume

    Solid PDE (reaction-diffusion type):
      ε_s·ρ_s·cp_s · ∂T_s/∂t = ∇·(k_eff_s·∇T_s) + Q_em + h_sf·a_sf·(T_f - T_s)
                                              + surface convection + radiation

    Fluid PDE (advection-diffusion type):
      ε_f·ρ_f·cp_f · ∂T_f/∂t = k_eff_f·∇²T_f - ε_f·ρ_f·cp_f·v_z·∂T_f/∂z
                                              + h_sf·a_sf·(T_s - T_f)

    Key: both β_s and β_f are volumetric coupling terms acting at every grid point in the whole blade domain
    """
    def __init__(self, geometry, nx=28, ny=14, nz=28, device='cuda',
                 power_density=1e9, h_conv=100, T_ambient=300,
                 h_sf_a_sf=500000.0, v_darcy=0.5, T_f_inlet=350.0,
                 porosity=0.3):
        self.geo = geometry
        self.nx, self.ny, self.nz = nx, ny, nz
        self.device = device

        self.bounds = [(0, geometry.chord), (-0.1, 0.1), (0, geometry.span)]
        self.X, self.Y, self.Z = create_3d_grid(nx, ny, nz, self.bounds, device)
        self.mask = geometry.create_mask(self.X, self.Y, self.Z)
        self.dist_surface = geometry.get_surface_distance(self.X, self.Y, self.Z)

        self.dx = (self.bounds[0][1] - self.bounds[0][0]) / (nx - 1)
        self.dy = (self.bounds[1][1] - self.bounds[1][0]) / (ny - 1)
        self.dz = (self.bounds[2][1] - self.bounds[2][0]) / (nz - 1)

        # Solid material (Inconel 718-like, T-dependent)
        self.rho_s = 8200.0
        self.skin_depth = 0.05

        # Porous media parameters
        self.porosity = porosity                   # ε_f (fluid fraction)
        self.eps_s = 1.0 - porosity                # ε_s (solid fraction)
        self.eps_f = porosity

        # Fluid properties (water as coolant — high thermal inertia for meaningful LTNE)
        self.rho_f = 998.0                         # water density [kg/m³]
        self.cp_f = 4182.0                         # water specific heat [J/kgK]
        self.k_f = 0.6                             # thermal conductivity of water [W/mK]
        self.k_eff_f = self.eps_f * self.k_f       # effective fluid conductivity
        self.alpha_f = self.k_eff_f / (self.eps_f * self.rho_f * self.cp_f)

        # Volumetric interphase heat transfer coefficient [W/m³K]
        self.h_sf_a_sf = h_sf_a_sf
        # β_s = h_sf·a_sf / (ε_s·ρ_s·cp_s)  — volumetric, full domain
        # β_f = h_sf·a_sf / (ε_f·ρ_f·cp_f)  — volumetric, full domain
        # (computed dynamically for solid due to cp_s(T))

        # Darcy velocity (z-direction, through-thickness cooling)
        self.v_darcy = v_darcy
        self.T_f_inlet = T_f_inlet

        # Configurable parameters
        self.power_density = power_density
        self.h_conv = h_conv        # external surface convection
        self.T_ambient = T_ambient

        # Radiation
        self.emissivity = 0.5
        self.sigma_sb = 5.67e-8

    @staticmethod
    def k_of_T(T):
        """Thermal conductivity [W/mK]: Inconel 718"""
        return 11.0 + 0.015 * (T - 300.0)

    @staticmethod
    def cp_of_T(T):
        """Specific heat [J/kgK]"""
        return 435.0 + 0.22 * (T - 300.0)

    def compute_em_heat_source(self, T, power_density):
        """EM volumetric heating (skin effect + geometry)"""
        sigma_ratio = 1.0 / (1 + 0.005 * (T - 300))
        sigma_ratio = torch.clamp(sigma_ratio, 0.1, 1.0)
        skin_factor = torch.exp(-self.dist_surface / self.skin_depth)
        x_norm = self.X / self.geo.chord
        le_factor = 1.0 + 0.5 * torch.exp(-x_norm / 0.2)
        return power_density * skin_factor * sigma_ratio * le_factor * self.mask

    def _laplacian_3d(self, T):
        """Compute 3D Laplacian with zero-flux BC at boundaries"""
        d2Tdx2 = torch.zeros_like(T)
        d2Tdy2 = torch.zeros_like(T)
        d2Tdz2 = torch.zeros_like(T)
        d2Tdx2[1:-1, :, :] = (T[2:, :, :] - 2 * T[1:-1, :, :] + T[:-2, :, :]) / self.dx ** 2
        d2Tdy2[:, 1:-1, :] = (T[:, 2:, :] - 2 * T[:, 1:-1, :] + T[:, :-2, :]) / self.dy ** 2
        d2Tdz2[:, :, 1:-1] = (T[:, :, 2:] - 2 * T[:, :, 1:-1] + T[:, :, :-2]) / self.dz ** 2
        return d2Tdx2 + d2Tdy2 + d2Tdz2

    def _upwind_dTdz(self, T):
        """Upwind derivative ∂T/∂z for v_darcy > 0 (backward difference)"""
        dTdz = torch.zeros_like(T)
        if self.v_darcy >= 0:
            dTdz[:, :, 1:] = (T[:, :, 1:] - T[:, :, :-1]) / self.dz
        else:
            dTdz[:, :, :-1] = (T[:, :, 1:] - T[:, :, :-1]) / self.dz
        return dTdz

    def solve_one_step(self, T_s, T_f, Q, dt):
        """
        One timestep: semi-implicit for coupling stability

        Solid: T_s_new = T_s + dt*(α_s·∇²T_s + Q/(ε_s·ρ_s·cp_s) + β_s·(T_f - T_s))
             = T_s + dt*(α_s·∇²T_s + Q/(ε_s·ρ_s·cp_s) + β_s·T_f - β_s·T_s)
             Semi-implicit: T_s_new·(1 + dt·β_s) = T_s + dt*(α_s·∇²T_s + Q/(ε_s·ρ_s·cp_s) + β_s·T_f)

        Fluid: T_f_new = T_f + dt*(α_f·∇²T_f - v_z·∂T_f/∂z + β_f·(T_s - T_f))
             Semi-implicit: T_f_new·(1 + dt·β_f) = T_f + dt*(α_f·∇²T_f - v_z·∂T_f/∂z + β_f·T_s)
        """
        # --- Solid field ---
        k_s = self.k_of_T(T_s)
        cp_s = self.cp_of_T(T_s)
        k_eff_s = self.eps_s * k_s
        alpha_s = k_eff_s / (self.eps_s * self.rho_s * cp_s)   # = k_s / (ρ_s·cp_s)
        beta_s = self.h_sf_a_sf / (self.eps_s * self.rho_s * cp_s)  # volumetric coupling

        lap_Ts = self._laplacian_3d(T_s)
        Q_norm = Q / (self.eps_s * self.rho_s * cp_s)

        # Semi-implicit: T_s_new = (T_s + dt*(diffusion + source + β_s·T_f)) / (1 + dt·β_s)
        T_s_new = (T_s + dt * (alpha_s * lap_Ts + Q_norm + beta_s * T_f)) / (1.0 + dt * beta_s)

        # Surface convection (external)
        surf = (self.mask > 0.5) & (self.dist_surface < 2 * self.dx)
        h_eff = self.h_conv / (self.eps_s * self.rho_s * cp_s * self.dx)
        T_s_new = torch.where(surf, T_s_new - dt * h_eff * (T_s - self.T_ambient), T_s_new)

        # Surface radiation: ε σ (T⁴ - T_amb⁴)
        rad = self.emissivity * self.sigma_sb * (T_s ** 4 - self.T_ambient ** 4) \
              / (self.eps_s * self.rho_s * cp_s * self.dx)
        T_s_new = torch.where(surf, T_s_new - dt * rad, T_s_new)

        # Mask: outside blade → T_ambient
        T_s_new = torch.where(self.mask > 0.5, T_s_new, torch.full_like(T_s_new, self.T_ambient))

        # --- Fluid field ---
        beta_f = self.h_sf_a_sf / (self.eps_f * self.rho_f * self.cp_f)  # volumetric coupling

        lap_Tf = self._laplacian_3d(T_f)
        dTf_dz = self._upwind_dTdz(T_f)

        # Semi-implicit: T_f_new = (T_f + dt*(diffusion - advection + β_f·T_s)) / (1 + dt·β_f)
        T_f_new = (T_f + dt * (self.alpha_f * lap_Tf
                                - self.v_darcy * dTf_dz
                                + beta_f * T_s)) / (1.0 + dt * beta_f)

        # Fluid inlet BC: z=0 plane → T_f_inlet
        T_f_new[:, :, 0] = self.T_f_inlet

        # Outside blade mask → T_ambient
        T_f_new = torch.where(self.mask > 0.5, T_f_new, torch.full_like(T_f_new, self.T_ambient))

        return T_s_new, T_f_new

    def solve(self, t_total=0.2, nt_save=10):
        """Solve LTNE porous media problem, return (T_solid, T_fluid, t_array)"""
        # Stability: dt < min(diffusion CFL, advection CFL)
        alpha_s_ref = 11.0 / (self.rho_s * 435.0)  # reference at 300K
        dt_diff_s = 0.2 * min(self.dx, self.dy, self.dz) ** 2 / alpha_s_ref
        dt_diff_f = 0.2 * min(self.dx, self.dy, self.dz) ** 2 / max(self.alpha_f, 1e-10)
        dt_adv = 0.5 * self.dz / max(abs(self.v_darcy), 0.01)
        dt = min(dt_diff_s, dt_diff_f, dt_adv, 0.0005)
        n_steps = int(t_total / dt)
        save_every = max(1, n_steps // (nt_save - 1))

        # Initial conditions: uniform T_ambient
        T_s = torch.full((self.nx, self.ny, self.nz), self.T_ambient,
                         dtype=torch.float32, device=self.device) * self.mask + \
              self.T_ambient * (1 - self.mask)
        T_f = torch.full((self.nx, self.ny, self.nz), self.T_f_inlet,
                         dtype=torch.float32, device=self.device) * self.mask + \
              self.T_ambient * (1 - self.mask)

        T_s_history = [T_s.cpu().numpy().copy()]
        T_f_history = [T_f.cpu().numpy().copy()]
        t_history = [0.0]

        for step in range(1, n_steps + 1):
            Q = self.compute_em_heat_source(T_s, self.power_density)
            T_s, T_f = self.solve_one_step(T_s, T_f, Q, dt)

            if step % save_every == 0 or step == n_steps:
                T_s_history.append(T_s.cpu().numpy().copy())
                T_f_history.append(T_f.cpu().numpy().copy())
                t_history.append(step * dt)

        return (np.array(T_s_history[:nt_save]),
                np.array(T_f_history[:nt_save]),
                np.array(t_history[:nt_save]))


# ============================================================
# Part 5: Normalizer
# ============================================================

class Normalizer3D:
    def __init__(self, T_min=290.0, T_max=420.0):
        self.T_min = T_min
        self.T_max = T_max
        self.scale = T_max - T_min

    def normalize(self, T):
        return (T - self.T_min) / self.scale

    def denormalize(self, T):
        return T * self.scale + self.T_min


# ============================================================
# Part 6: Data Generation (parameter diversity + CHT)
# ============================================================

def generate_data_3d(geometry, nx=28, ny=14, nz=28, nt=10, n_traj=150, device='cuda',
                     param_ranges=None, ext_normalizer=None):
    """Data generation: parameter diversity for the LTNE porous medium

    Args:
        param_ranges: dict with optional keys 'power', 'h_sf_a_sf', 'v_darcy'
                      each a (min, max) tuple. Defaults used if not provided.
        ext_normalizer: external Normalizer3D instance. If provided, skip probe step.
    """
    print(f"Generating {n_traj} LTNE porous trajectories with parameter diversity...", flush=True)

    # Default parameter ranges
    pr = {
        'power': (0.35e9, 0.50e9),
        'h_sf_a_sf': (200000, 900000),
        'v_darcy': (0.15, 0.85),
    }
    if param_ranges:
        pr.update(param_ranges)

    if ext_normalizer is not None:
        normalizer = ext_normalizer
        print(f"  Using external normalizer: [{normalizer.T_min:.0f}, {normalizer.T_max:.0f}] K", flush=True)
    else:
        # Probe runs at extreme corners to estimate full temperature range
        solver_hot = LTNEPorousSolver(geometry, nx=nx, ny=ny, nz=nz, device=device,
                                       power_density=0.50e9, h_conv=80, T_ambient=310,
                                       h_sf_a_sf=200000.0, v_darcy=0.15, T_f_inlet=360.0)
        T_hot, _, _ = solver_hot.solve(t_total=0.2, nt_save=5)
        T_max_data = T_hot.max()
        del solver_hot
        torch.cuda.empty_cache()

        solver_cold = LTNEPorousSolver(geometry, nx=nx, ny=ny, nz=nz, device=device,
                                        power_density=0.35e9, h_conv=180, T_ambient=290,
                                        h_sf_a_sf=900000.0, v_darcy=0.85, T_f_inlet=330.0)
        T_cold, _, _ = solver_cold.solve(t_total=0.2, nt_save=5)
        T_min_data = T_cold.min()
        del solver_cold
        torch.cuda.empty_cache()

        normalizer = Normalizer3D(T_min=T_min_data - 10, T_max=T_max_data + 30)
        print(f"  Temperature range: [{normalizer.T_min:.0f}, {normalizer.T_max:.0f}] K", flush=True)

    all_T_solid = []
    all_T_f = []
    t_array = None
    param_log = []

    p_lo, p_hi = pr['power']
    h_lo, h_hi = pr['h_sf_a_sf']
    v_lo, v_hi = pr['v_darcy']

    for i in range(n_traj):
        power = p_lo + (p_hi - p_lo) * np.random.rand()
        h_conv = 80 + 120 * np.random.rand()                      # [80, 200] W/m²K
        T_amb = 290 + 20 * np.random.rand()                       # [290, 310] K
        T_inlet = 330 + 40 * np.random.rand()                     # [330, 370] K

        h_sf_a_sf = h_lo + (h_hi - h_lo) * np.random.rand()
        v_darcy = v_lo + (v_hi - v_lo) * np.random.rand()

        solver = LTNEPorousSolver(geometry, nx=nx, ny=ny, nz=nz, device=device,
                                   power_density=power, h_conv=h_conv, T_ambient=T_amb,
                                   h_sf_a_sf=h_sf_a_sf, v_darcy=v_darcy, T_f_inlet=T_inlet)

        T_hist, Tf_hist, t_arr = solver.solve(t_total=0.2, nt_save=nt)
        del solver
        if i % 30 == 0:
            torch.cuda.empty_cache()

        if t_array is None:
            t_array = t_arr

        T_s_norm = normalizer.normalize(T_hist)
        T_f_norm = normalizer.normalize(Tf_hist)
        all_T_solid.append(torch.from_numpy(T_s_norm).float().to(device))
        all_T_f.append(torch.from_numpy(T_f_norm).float().to(device))
        param_log.append({'power': power, 'h_conv': h_conv, 'T_amb': T_amb,
                          'h_sf_a_sf': h_sf_a_sf, 'v_darcy': v_darcy, 'T_inlet': T_inlet})

        if (i + 1) % 30 == 0:
            print(f"  Trajectory {i+1}/{n_traj}", flush=True)

    # Build grid (use last solver's bounds — they're all identical)
    bounds = [(0, geometry.chord), (-0.1, 0.1), (0, geometry.span)]
    X, Y, Z = create_3d_grid(nx, ny, nz, bounds, device)
    mask = geometry.create_mask(X, Y, Z)
    dist_surface = geometry.get_surface_distance(X, Y, Z)

    X_norm = (X - bounds[0][0]) / (bounds[0][1] - bounds[0][0])
    Y_norm = (Y - bounds[1][0]) / (bounds[1][1] - bounds[1][0])
    Z_norm = (Z - bounds[2][0]) / (bounds[2][1] - bounds[2][0])
    grid = torch.stack([X_norm, Y_norm, Z_norm], dim=-1).unsqueeze(0)

    powers = [p['power'] for p in param_log]
    h_sf_list = [p['h_sf_a_sf'] for p in param_log]
    v_darcy_list = [p['v_darcy'] for p in param_log]
    print(f"  Parameter diversity summary:")
    print(f"    Power:     [{min(powers)/1e9:.2f}, {max(powers)/1e9:.2f}] GW/m^3")
    print(f"    h_sf·a_sf: [{min(h_sf_list):.0f}, {max(h_sf_list):.0f}] W/m^3K")
    print(f"    v_darcy:   [{min(v_darcy_list):.2f}, {max(v_darcy_list):.2f}] m/s")

    return {
        'T': torch.stack(all_T_solid, dim=0),       # [n_traj, nt, nx, ny, nz]
        'T_f': torch.stack(all_T_f, dim=0),          # [n_traj, nt, nx, ny, nz]
        'mask': mask,
        'grid': grid,
        't': torch.from_numpy(t_array).float().to(device),
        'normalizer': normalizer,
        'X': X, 'Y': Y, 'Z': Z,
        'dist_surface': dist_surface,
        # Per-sample parameters for physics kernel conditioning:
        # h_conv_array → maps to sample_alpha (β modulation) in physics_fno_3d.py
        # v_cool_array → maps to sample_vx (vz modulation) in physics_fno_3d.py
        'h_conv_array': np.array([p['h_sf_a_sf'] for p in param_log]),
        'v_cool_array': np.array([p['v_darcy'] for p in param_log]),
    }


# ============================================================
# Part 7: Boundary Conditions
# ============================================================

def apply_boundary_conditions(T_pred_s, T_pred_f, mask, dist_surface, T_prev_s, T_prev_f):
    """Enforce boundary conditions for both solid and coolant fields"""
    T_out_s = T_pred_s.clone()

    surface_weight = torch.exp(-dist_surface / 0.02)
    surface_weight = torch.clamp(surface_weight, 0, 0.2)
    T_out_s = T_out_s * (1 - surface_weight) + T_prev_s * surface_weight
    T_out_s = T_out_s * mask

    # Coolant embedding doesn't need surface BC
    return T_out_s, T_pred_f


# ============================================================
# Part 8: CHT Training
# ============================================================

def train_causal(model, data, n_epochs=5000, device='cuda'):
    """Causal training: two channels (T_solid + T_f_embed)"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=800, T_mult=2, eta_min=1e-6
    )

    mask = data['mask']
    T_all = data['T']        # [n_traj, nt, nx, ny, nz]
    Tf_all = data['T_f']     # [n_traj, nt, nx, ny, nz]
    t_array = data['t']
    grid = data['grid']
    normalizer = data['normalizer']
    dist_surface = data['dist_surface']

    n_traj, nt = T_all.shape[0], len(t_array)
    valid = mask > 0.5

    dt = (t_array[1] - t_array[0]) / t_array.max()

    sample_errors = np.ones(n_traj) * 10
    best_max = float('inf')
    best_state = None

    print(f"\nTraining Causal-PINO 3D CHT ({n_epochs} epochs)...", flush=True)
    print(f"  Trajectories: {n_traj}, Time steps: {nt}", flush=True)
    print(f"  Grid: {T_all.shape[2]}x{T_all.shape[3]}x{T_all.shape[4]}", flush=True)
    print(f"  Mode: CAUSAL + CHT (2-channel: T_solid + T_coolant)", flush=True)

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        # Scheduled Sampling
        if epoch < 500:
            teacher_prob = 0.9
        elif epoch < 2000:
            teacher_prob = 0.9 - 0.6 * (epoch - 500) / 1500
        else:
            teacher_prob = 0.3

        # Hard example mining
        if epoch < 200:
            batch_idx = np.random.choice(n_traj, min(8, n_traj), replace=False)
        else:
            n_hard = 6
            n_random = 2
            hard_idx = np.argsort(sample_errors)[-n_hard:]
            random_idx = np.random.choice(n_traj, n_random, replace=False)
            batch_idx = np.unique(np.concatenate([hard_idx, random_idx]))

        total_loss = 0
        batch_errors = {}

        for idx in batch_idx:
            T_traj = T_all[idx]       # [nt, nx, ny, nz]
            Tf_traj = Tf_all[idx]     # [nt, nx, ny, nz]

            traj_loss = 0
            traj_max_err = 0

            T_cur_s = T_traj[0].unsqueeze(0)   # [1, nx, ny, nz]
            T_cur_f = Tf_traj[0].unsqueeze(0)

            for t_idx in range(1, nt):
                T_true_s = T_traj[t_idx]
                T_true_f = Tf_traj[t_idx]

                T_pred_s, T_pred_f = model(T_cur_s, T_cur_f, dt, grid)
                T_pred_s = T_pred_s.squeeze(0)
                T_pred_f = T_pred_f.squeeze(0)

                T_pred_s, T_pred_f = apply_boundary_conditions(
                    T_pred_s, T_pred_f, mask, dist_surface,
                    T_cur_s.squeeze(0), T_cur_f.squeeze(0)
                )

                # Loss on solid (primary field)
                diff_s = (T_pred_s - T_true_s)[valid]
                abs_diff = diff_s.abs()
                loss_mse = (diff_s ** 2).mean()
                loss_l1 = abs_diff.mean()
                max_err = abs_diff.max()
                loss_max = max_err ** 2 * 1.0

                k5 = max(1, int(abs_diff.numel() * 0.05))
                loss_top5 = (torch.topk(abs_diff.flatten(), k5).values ** 2).mean() * 0.4
                k1 = max(1, int(abs_diff.numel() * 0.01))
                loss_top1 = (torch.topk(abs_diff.flatten(), k1).values ** 2).mean() * 0.3

                # Loss on coolant field (secondary, lower weight)
                diff_f = (T_pred_f - T_true_f)[valid]
                loss_f = (diff_f ** 2).mean() * 0.3

                step_loss = loss_mse + 0.3 * loss_l1 + loss_max + loss_top5 + loss_top1 + loss_f

                accumulation_weight = 1.0 + 0.3 * (t_idx / nt)
                traj_loss += accumulation_weight * step_loss

                err_K = (max_err * normalizer.scale).item()
                traj_max_err = max(traj_max_err, err_K)

                # Scheduled sampling
                if np.random.rand() < teacher_prob:
                    T_cur_s = T_true_s.unsqueeze(0)
                    T_cur_f = T_true_f.unsqueeze(0)
                else:
                    T_cur_s = T_pred_s.unsqueeze(0).detach()
                    T_cur_f = T_pred_f.unsqueeze(0).detach()

            total_loss += traj_loss / (nt - 1)
            batch_errors[idx] = traj_max_err

        for idx, err in batch_errors.items():
            sample_errors[idx] = 0.7 * sample_errors[idx] + 0.3 * err

        total_loss = total_loss / len(batch_idx)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if epoch % 50 == 0:
            torch.cuda.empty_cache()

        if (epoch + 1) % 100 == 0 or epoch == 0:
            mae, max_err = evaluate_causal(model, data, grid, dt)
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{n_epochs} | Loss: {total_loss.item():.6f} | "
                  f"MAE: {mae:.4f}K | Max: {max_err:.4f}K | LR: {lr:.2e} | "
                  f"TeacherP: {teacher_prob:.2f}", flush=True)

            if max_err < best_max:
                best_max = max_err
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                print(f"  -> New best! Max: {best_max:.4f}K", flush=True)

            if best_max < 0.5:
                print(f"\n  TARGET ACHIEVED: Max Error < 0.5K", flush=True)
                break

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print(f"\nRestored best model (Max: {best_max:.4f}K)", flush=True)

    return model


def evaluate_causal(model, data, grid, dt):
    """Evaluation: fully autoregressive, reports the solid temperature field error (supports per-sample conditioning)"""
    model.eval()

    mask = data['mask']
    T_all = data['T']
    Tf_all = data['T_f']
    normalizer = data['normalizer']
    dist_surface = data['dist_surface']
    nt = T_all.shape[1]

    valid = mask > 0.5
    all_max = []

    # per-sample PDE parameters (effective for the Physics model only)
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

    with torch.no_grad():
        for traj_idx in range(min(T_all.shape[0], 30)):
            T_traj = T_all[traj_idx]
            Tf_traj = Tf_all[traj_idx]
            traj_max = 0

            T_cur_s = T_traj[0].unsqueeze(0)
            T_cur_f = Tf_traj[0].unsqueeze(0)

            sa = alpha_norm[traj_idx:traj_idx+1] if has_sample_alpha else None
            sv = vx_norm[traj_idx:traj_idx+1] if has_sample_vx else None

            for t_idx in range(1, nt):
                T_pred_s, T_pred_f = model(T_cur_s, T_cur_f, dt, grid,
                                            sample_alpha=sa, sample_vx=sv)
                T_pred_s = T_pred_s.squeeze(0)
                T_pred_f = T_pred_f.squeeze(0)

                T_pred_s, T_pred_f = apply_boundary_conditions(
                    T_pred_s, T_pred_f, mask, dist_surface,
                    T_cur_s.squeeze(0), T_cur_f.squeeze(0)
                )

                T_pred_K = normalizer.denormalize(T_pred_s)
                T_true_K = normalizer.denormalize(T_traj[t_idx])
                err = torch.abs(T_pred_K - T_true_K)[valid].max().item()
                traj_max = max(traj_max, err)

                T_cur_s = T_pred_s.unsqueeze(0)
                T_cur_f = T_pred_f.unsqueeze(0)

            all_max.append(traj_max)

    model.train()
    return np.mean(all_max), np.max(all_max)


# ============================================================
# Part 9: Visualization
# ============================================================

_FIG_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               os.pardir, 'figures', 'blade_causal_results.png')


def visualize_causal_results(model, data, save_path=_FIG_DEFAULT):
    """Visualize CHT prediction results (solid + coolant)"""
    # The default is anchored to this script's directory. A bare filename would
    # be written into whatever directory the module happened to be run from.
    save_path = os.path.abspath(save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.eval()

    mask = data['mask']
    T_all = data['T']
    Tf_all = data['T_f']
    t_array = data['t']
    grid = data['grid']
    normalizer = data['normalizer']
    X, Y, Z = data['X'], data['Y'], data['Z']
    dist_surface = data['dist_surface']

    dt = (t_array[1] - t_array[0]) / t_array.max()

    T_traj = T_all[0]
    Tf_traj = Tf_all[0]

    T_preds_s = [normalizer.denormalize(T_traj[0]).cpu().numpy()]
    T_trues_s = [normalizer.denormalize(T_traj[0]).cpu().numpy()]

    T_cur_s = T_traj[0].unsqueeze(0)
    T_cur_f = Tf_traj[0].unsqueeze(0)

    with torch.no_grad():
        for t_idx in range(1, len(t_array)):
            T_pred_s, T_pred_f = model(T_cur_s, T_cur_f, dt, grid)
            T_pred_s = T_pred_s.squeeze(0)
            T_pred_f = T_pred_f.squeeze(0)

            T_pred_s, T_pred_f = apply_boundary_conditions(
                T_pred_s, T_pred_f, mask, dist_surface,
                T_cur_s.squeeze(0), T_cur_f.squeeze(0)
            )

            T_preds_s.append(normalizer.denormalize(T_pred_s).cpu().numpy())
            T_trues_s.append(normalizer.denormalize(T_traj[t_idx]).cpu().numpy())

            T_cur_s = T_pred_s.unsqueeze(0)
            T_cur_f = T_pred_f.unsqueeze(0)

    mask_np = mask.cpu().numpy()
    nz = mask_np.shape[2]
    z_mid = nz // 2

    X_np = X.cpu().numpy()
    Y_np = Y.cpu().numpy()

    fig, axes = plt.subplots(3, 5, figsize=(22, 13))

    n_plots = 5
    t_indices = np.linspace(0, len(t_array) - 1, n_plots, dtype=int)

    for col, t_idx in enumerate(t_indices):
        T_true = T_trues_s[t_idx][:, :, z_mid]
        T_pred = T_preds_s[t_idx][:, :, z_mid]
        mask_slice = mask_np[:, :, z_mid]
        X_slice = X_np[:, :, z_mid]
        Y_slice = Y_np[:, :, z_mid]

        vmin = T_true[mask_slice > 0.5].min()
        vmax = T_true[mask_slice > 0.5].max()

        ax = axes[0, col]
        T_masked = np.ma.masked_where(mask_slice < 0.5, T_true)
        cf = ax.contourf(X_slice, Y_slice, T_masked, levels=20, cmap='jet', vmin=vmin, vmax=vmax)
        ax.contour(X_slice, Y_slice, mask_slice, levels=[0.5], colors='k', linewidths=1)
        ax.set_title(f't={t_array[t_idx].item():.3f}s - True', fontsize=10)
        ax.set_aspect('equal')
        plt.colorbar(cf, ax=ax, label='T(K)')

        ax = axes[1, col]
        T_masked = np.ma.masked_where(mask_slice < 0.5, T_pred)
        cf = ax.contourf(X_slice, Y_slice, T_masked, levels=20, cmap='jet', vmin=vmin, vmax=vmax)
        ax.contour(X_slice, Y_slice, mask_slice, levels=[0.5], colors='k', linewidths=1)
        ax.set_title(f'CHT Predicted', fontsize=10)
        ax.set_aspect('equal')
        plt.colorbar(cf, ax=ax, label='T(K)')

        ax = axes[2, col]
        error = np.abs(T_pred - T_true)
        err_masked = np.ma.masked_where(mask_slice < 0.5, error)
        cf = ax.contourf(X_slice, Y_slice, err_masked, levels=20, cmap='jet')
        ax.contour(X_slice, Y_slice, mask_slice, levels=[0.5], colors='k', linewidths=1)
        max_err = error[mask_slice > 0.5].max()
        ax.set_title(f'Error (max={max_err:.2f}K)', fontsize=10)
        ax.set_aspect('equal')
        plt.colorbar(cf, ax=ax, label='|Error|(K)')

    plt.suptitle('3D Causal-PINO CHT: Turbine Blade Multi-Physics\n'
                 '(Nonlinear k(T) + Radiation + Coolant Coupling)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}", flush=True)
    plt.close()


# ============================================================
# Part 10: Main
# ============================================================

if __name__ == '__main__':
    print("=" * 70)
    print("3D Causal-PINO: Multi-Physics CHT")
    print("=" * 70)
    print("\nPhysics:")
    print("  - Nonlinear k(T), cp(T) (Inconel 718)")
    print("  - EM heating with σ(T) (stronger T-dependence)")
    print("  - Surface radiation ε·σ(T⁴-T_amb⁴)")
    print("  - External convection h·(T-T_amb)")
    print("  - Internal coolant channels (3 holes, 1D advection)")
    print("  - Bidirectional coupling: solid ↔ coolant")
    print("\nFNO Architecture:")
    print("  - Input:  T_solid + T_f_embed + grid + dt = 6 channels")
    print("  - Output: ΔT_solid + ΔT_f_embed = 2 channels")
    print("\nParameter Diversity:")
    print("  - Power: [0.50, 2.00] GW/m³")
    print("  - h_conv: [30, 500] W/m²K")
    print("  - T_ambient: [280, 320] K")
    print("  - v_coolant: [5, 60] m/s")
    print("  - T_coolant_inlet: [320, 380] K")
    print("=" * 70)

    if torch.cuda.is_available():
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\nGPU Memory: {total_mem:.1f} GB")

    geometry = TurbineBladeGeometry(
        chord=1.0, span=2.0, naca='0012',
        twist_angle=45.0, n_cooling_holes=3,
        hole_radius=0.025, device=device
    )

    nx, ny, nz = 28, 14, 28

    print(f"\n[1] Generating training data (grid: {nx}x{ny}x{nz})...", flush=True)
    train_data = generate_data_3d(geometry, nx=nx, ny=ny, nz=nz, nt=10,
                                   n_traj=150, device=device)

    print("\n[2] Generating test data...", flush=True)
    test_data = generate_data_3d(geometry, nx=nx, ny=ny, nz=nz, nt=10,
                                  n_traj=20, device=device)

    print("\n[3] Creating Causal-PINO 3D CHT model...", flush=True)
    model = CausalPINO3D(modes=8, width=64, n_layers=5, device=device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params/1e6:.2f}M", flush=True)

    print("\n[4] CHT Training (with Scheduled Sampling)...", flush=True)
    model = train_causal(model, train_data, n_epochs=5000, device=device)

    print("\n[5] Final evaluation (full autoregressive)...", flush=True)
    t_array_train = train_data['t']
    t_array_test = test_data['t']
    dt_train = (t_array_train[1] - t_array_train[0]) / t_array_train.max()
    dt_test = (t_array_test[1] - t_array_test[0]) / t_array_test.max()

    train_mae, train_max = evaluate_causal(model, train_data, train_data['grid'], dt_train)
    test_mae, test_max = evaluate_causal(model, test_data, test_data['grid'], dt_test)

    print(f"\n  Train: MAE={train_mae:.4f}K, Max={train_max:.4f}K")
    print(f"  Test:  MAE={test_mae:.4f}K, Max={test_max:.4f}K")

    print("\n[6] Visualizing results...", flush=True)
    visualize_causal_results(model, test_data,
                            save_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "figures", "blade_causal_results.png"))

    torch.save(model.state_dict(), os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "checkpoints", "causal_pino_3d.pth"))
    print("\nModel saved: causal_pino_3d.pth", flush=True)

    print("\n" + "=" * 70)
    print("Multi-Physics CHT Results:")
    print(f"  Train: MAE={train_mae:.4f}K, WorstMax={train_max:.4f}K")
    print(f"  Test:  MAE={test_mae:.4f}K, WorstMax={test_max:.4f}K")
    print("=" * 70)
