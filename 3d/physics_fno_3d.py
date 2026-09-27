# 3D Physics-Structured FNO for Multi-Physics Heat Transfer
# Dual-Stream Architecture: solid stream (solid decay) + fluid stream (fluid decay) + gated coupling
#
# Design rationale:
#   The spectral weights (64×64) of a single-stream architecture mix solid/fluid channels,
#   which destroys the semantics of the physics kernel
#   Dual-Stream lets each stream independently maintain pure physics features:
#     solid stream (32ch): spectral conv (32×32) + exp(-(α_s|k|²+β_s)dt) + bypass(32→32)
#     fluid stream (32ch): spectral conv (32×32) + exp(-(α_f|k|²+β_f)dt - iv·k·dt) + bypass(32→32)
#   Gated coupling: h_s += gate * Conv1x1(h_f), corresponding to the β(T_s-T_f) term in the PDE
#
# Per-sample conditioning:
#   sample_alpha → modulates β_s (h_sf_a_sf: volumetric coupling)
#   sample_vx → modulates vz (v_darcy: Darcy velocity)
#
# FiLM conditioning (optional, use_film=True):
#   The physics kernel only modulates the spectral decay (20-35% of output); the correction
#   network and the bypass are completely parameter-blind.
#   FiLM makes the entire network aware of the physical parameters:
#     gamma, beta = MLP(sample_alpha, sample_vx)
#     h = gamma * h + beta   (Feature-wise Linear Modulation)
#   This way 100% of the output responds to parameter changes, improving OOD generalization.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FiLMConditioner(nn.Module):
    """Feature-wise Linear Modulation conditioned on physical parameters.

    Maps (sample_alpha, sample_vx) → per-channel (gamma, beta)
    then applies: out = gamma * x + beta

    Initialized to identity transform (gamma=1, beta=0) so that
    FiLM starts as a no-op and gradually learns to modulate.
    """

    def __init__(self, channels, n_params=2, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(channels // 4, 16)
        self.channels = channels
        self.net = nn.Sequential(
            nn.Linear(n_params, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, channels * 2),
        )
        # Initialize near-identity: gamma≈1, beta≈0
        # Use small random weights (not zero) so gradients flow from step 1
        nn.init.normal_(self.net[-1].weight, std=1e-4)
        nn.init.zeros_(self.net[-1].bias)
        self.net[-1].bias.data[:channels] = 1.0  # gamma init = 1

    def forward(self, x, params):
        """
        x:      [B, C, nx, ny, nz]
        params: [B, 2]  (sample_alpha, sample_vx)
        Returns: modulated x with same shape
        """
        mod = self.net(params)  # [B, 2*C]
        gamma = mod[:, :self.channels].view(-1, self.channels, 1, 1, 1)
        beta = mod[:, self.channels:].view(-1, self.channels, 1, 1, 1)
        return gamma * x + beta


class SingleFieldSpectralConv3d(nn.Module):
    """
    Single-field 3D Fourier spectral convolution + physics decay kernel

    Supports two modes:
      'solid':  D(k,dt) = exp(-(α|k|² + β)·dt)                  real decay
      'fluid':  D(k,dt) = exp(-(α|k|² + β)·dt - i·vz·kz·dt)    complex decay

    + low-rank correction R_corr = U×V
    """

    def __init__(self, channels, modes=8, corr_rank=4, field_type='solid'):
        super().__init__()
        self.channels = channels
        self.modes = modes
        self.corr_rank = corr_rank
        self.field_type = field_type

        scale = 1 / (channels * channels)
        corr_scale = 1 / (channels * corr_rank)

        # Main weights (channels × channels, within-stream only, no cross-stream mixing)
        self.weights1 = nn.Parameter(scale * torch.rand(
            channels, channels, modes, modes, modes, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.rand(
            channels, channels, modes, modes, modes, dtype=torch.cfloat))

        # Low-rank correction
        self.corr_U1 = nn.Parameter(corr_scale * torch.rand(
            channels, corr_rank, modes, modes, modes, dtype=torch.cfloat))
        self.corr_V1 = nn.Parameter(corr_scale * torch.rand(
            corr_rank, channels, modes, modes, modes, dtype=torch.cfloat))
        self.corr_U2 = nn.Parameter(corr_scale * torch.rand(
            channels, corr_rank, modes, modes, modes, dtype=torch.cfloat))
        self.corr_V2 = nn.Parameter(corr_scale * torch.rand(
            corr_rank, channels, modes, modes, modes, dtype=torch.cfloat))

        # Physics parameters
        self.log_alpha = nn.Parameter(torch.zeros(1))
        self.log_beta = nn.Parameter(torch.full((1,), -5.0))

        if field_type == 'fluid':
            self.log_vz = nn.Parameter(torch.zeros(1))
            self.v_sign_z = nn.Parameter(torch.ones(1))

        # Wavenumber buffers
        k1 = torch.arange(modes).float()
        k2 = torch.arange(modes).float()
        k3 = torch.arange(modes).float()
        K1p, K2p, K3p = torch.meshgrid(k1, k2, k3, indexing='ij')
        self.register_buffer('k_sq_pos', K1p**2 + K2p**2 + K3p**2)
        # clone(): meshgrid returns expanded views with zero stride, and a
        # zero-stride buffer cannot be the target of load_state_dict's copy_.
        self.register_buffer('kz_pos', K3p.clone())

        k1_neg = torch.arange(modes, 0, -1).float()
        K1n, K2n, K3n = torch.meshgrid(k1_neg, k2, k3, indexing='ij')
        self.register_buffer('k_sq_neg', K1n**2 + K2n**2 + K3n**2)
        self.register_buffer('kz_neg', K3n.clone())

    def _compute_decay(self, k_sq, kz, dt, sample_alpha=None, sample_vx=None):
        """Compute physics decay kernel.

        Conditioning (independent):
          sample_alpha (h_sf_a_sf) -> modulates beta (coupling rate)
          sample_vx (v_darcy) -> modulates vz (advection, fluid only)
        """
        alpha = self.log_alpha.exp()
        beta = self.log_beta.exp()

        per_sample = (sample_alpha is not None) or (sample_vx is not None)

        # Diffusion: beta modulated by sample_alpha (h_sf_a_sf)
        if sample_alpha is not None:
            beta_eff = beta * sample_alpha.view(-1, 1, 1, 1)
            diffusion = -(alpha * k_sq.unsqueeze(0) + beta_eff) * dt
        else:
            diffusion = -(alpha * k_sq + beta) * dt

        if self.field_type == 'fluid':
            vz = torch.tanh(self.v_sign_z) * self.log_vz.exp()
            # Advection: vz modulated by sample_vx (v_darcy)
            if sample_vx is not None:
                vz_eff = vz * sample_vx.view(-1, 1, 1, 1)
                advection = -1j * vz_eff * kz.unsqueeze(0) * dt
            elif per_sample:
                advection = -1j * vz * kz.unsqueeze(0) * dt
            else:
                advection = -1j * vz * kz * dt
            return torch.exp(diffusion + advection)

        return torch.exp(diffusion)

    def compl_mul3d(self, x, weights):
        return torch.einsum("bixyz,ioxyz->boxyz", x, weights)

    def _low_rank_mul3d(self, x_ft, U, V):
        temp = torch.einsum("bixyz,irxyz->brxyz", x_ft, U)
        return torch.einsum("brxyz,roxyz->boxyz", temp, V)

    def forward(self, x, dt=None, sample_alpha=None, sample_vx=None):
        batch_size = x.shape[0]
        nx, ny, nz = x.size(-3), x.size(-2), x.size(-1)
        m = self.modes

        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        out_ft = torch.zeros(batch_size, self.channels, nx, ny,
                             nz // 2 + 1, dtype=torch.cfloat, device=x.device)

        x_pos = x_ft[:, :, :m, :m, :m]
        x_neg = x_ft[:, :, -m:, :m, :m]

        main_pos = self.compl_mul3d(x_pos, self.weights1)
        main_neg = self.compl_mul3d(x_neg, self.weights2)

        if dt is not None and dt > 0:
            d_pos = self._compute_decay(self.k_sq_pos, self.kz_pos, dt, sample_alpha, sample_vx)
            d_neg = self._compute_decay(self.k_sq_neg, self.kz_neg, dt, sample_alpha, sample_vx)

            if d_pos.dim() == 3:
                d_pos = d_pos.unsqueeze(0).unsqueeze(0)
                d_neg = d_neg.unsqueeze(0).unsqueeze(0)
            else:
                d_pos = d_pos.unsqueeze(1)
                d_neg = d_neg.unsqueeze(1)

            main_pos = main_pos * d_pos
            main_neg = main_neg * d_neg

        corr_pos = self._low_rank_mul3d(x_pos, self.corr_U1, self.corr_V1)
        corr_neg = self._low_rank_mul3d(x_neg, self.corr_U2, self.corr_V2)

        out_ft[:, :, :m, :m, :m] = main_pos + corr_pos
        out_ft[:, :, -m:, :m, :m] = main_neg + corr_neg

        return torch.fft.irfftn(out_ft, s=(nx, ny, nz))

    def get_physics_info(self, dt):
        with torch.no_grad():
            alpha = self.log_alpha.exp().item()
            beta = self.log_beta.exp().item()
            info = {'alpha': alpha, 'beta': beta}
            if self.field_type == 'fluid':
                info['vz'] = (torch.tanh(self.v_sign_z) * self.log_vz.exp()).item()

            decay = torch.exp(-(alpha * self.k_sq_pos + beta) * dt)
            w_norm = (self.weights1 * decay.unsqueeze(0).unsqueeze(0)).abs().mean().item()
            corr_norm = (self.corr_U1.abs().mean() * self.corr_V1.abs().mean() *
                         self.corr_rank).item()
            total = w_norm + corr_norm + 1e-12
            info['physics_ratio'] = w_norm / total
            info['correction_ratio'] = corr_norm / total
            return info


class SingleFieldFNOBlock3d(nn.Module):
    """Single-field FNO block: spectral conv (physics decay) + bypass + norm + optional FiLM"""
    def __init__(self, channels, modes=8, corr_rank=4, field_type='solid',
                 use_film=False):
        super().__init__()
        self.spectral_conv = SingleFieldSpectralConv3d(
            channels, modes=modes, corr_rank=corr_rank, field_type=field_type
        )
        self.conv = nn.Conv3d(channels, channels, 1)
        self.norm = nn.InstanceNorm3d(channels)
        self.film = FiLMConditioner(channels) if use_film else None

    def forward(self, x, dt=None, sample_alpha=None, sample_vx=None,
                film_params=None):
        residual = x
        x_ft = self.spectral_conv(x, dt=dt, sample_alpha=sample_alpha, sample_vx=sample_vx)
        x_conv = self.conv(residual)
        out = F.gelu(self.norm(x_ft + x_conv))
        if self.film is not None and film_params is not None:
            out = self.film(out, film_params)
        return out + residual


class PhysicsStructuredPINO3D(nn.Module):
    """
    Dual-Stream Physics-Structured PINO 3D
    ========================================
    solid stream: SingleFieldFNOBlock(solid) × n_layers — exp(-(α_s|k|²+β_s)dt)
    fluid stream: SingleFieldFNOBlock(fluid) × n_layers — exp(-(α_f|k|²+β_f)dt - iv·k·dt)
    Gated coupling: gated cross-coupling between every pair of layers

    Each stream independently maintains pure physics channel semantics;
    the physics kernel acts directly on 32ch pure-solid / pure-fluid features
    (not corrupted by cross-field mixing)
    """
    def __init__(self, modes=8, width=64, n_layers=5, device='cuda', corr_rank=4,
                 use_film=False):
        super().__init__()
        self.width = width
        self.device = device
        self.pde_type = 'cht_3d'
        self.use_film = use_film
        ch = width // 2  # 32 per stream

        # Split Encoder
        self.fc0_s = nn.Linear(5, ch)  # T_s + grid(3) + dt → 32ch
        self.fc0_f = nn.Linear(5, ch)  # T_f + grid(3) + dt → 32ch

        # Dual streams
        self.solid_blocks = nn.ModuleList([
            SingleFieldFNOBlock3d(ch, modes=modes, corr_rank=corr_rank,
                                  field_type='solid', use_film=use_film)
            for _ in range(n_layers)
        ])
        self.fluid_blocks = nn.ModuleList([
            SingleFieldFNOBlock3d(ch, modes=modes, corr_rank=corr_rank,
                                  field_type='fluid', use_film=use_film)
            for _ in range(n_layers)
        ])

        # Cross-coupling: gated 1x1 conv at each layer
        self.couple_f2s = nn.ModuleList([nn.Conv3d(ch, ch, 1) for _ in range(n_layers)])
        self.couple_s2f = nn.ModuleList([nn.Conv3d(ch, ch, 1) for _ in range(n_layers)])
        # Learnable gate per layer (initialized small → coupling grows during training)
        self.coupling_gate = nn.Parameter(torch.full((n_layers,), -2.0))  # sigmoid(-2)≈0.12

        # Split Decoder
        self.fc1_s = nn.Linear(ch, ch)
        self.fc1_f = nn.Linear(ch, ch)
        self.fc2_s = nn.Linear(ch, 1)
        self.fc2_f = nn.Linear(ch, 1)

        self.to(device)

    def forward(self, T_solid, T_f_embed, dt, grid, sample_alpha=None, sample_vx=None):
        batch_size = T_solid.shape[0]
        nx, ny, nz = T_solid.shape[1], T_solid.shape[2], T_solid.shape[3]

        if isinstance(dt, (int, float)):
            dt_val = dt
            dt = torch.tensor([dt], device=T_solid.device)
        else:
            dt_val = dt.item() if dt.numel() == 1 else dt[0].item()

        if dt.dim() == 0:
            dt = dt.unsqueeze(0)
        dt_exp = dt.view(-1, 1, 1, 1, 1).expand(batch_size, nx, ny, nz, 1)

        # Build FiLM params: [B, 2] from sample_alpha and sample_vx
        film_params = None
        if self.use_film and (sample_alpha is not None or sample_vx is not None):
            a_val = sample_alpha if sample_alpha is not None else torch.ones(
                batch_size, device=T_solid.device)
            v_val = sample_vx if sample_vx is not None else torch.ones(
                batch_size, device=T_solid.device)
            # Ensure [B] shape
            if a_val.dim() == 0:
                a_val = a_val.unsqueeze(0)
            if v_val.dim() == 0:
                v_val = v_val.unsqueeze(0)
            film_params = torch.stack([a_val.view(-1), v_val.view(-1)], dim=-1)

        # Shared spatial features
        shared = torch.cat([
            grid.expand(batch_size, -1, -1, -1, -1),
            dt_exp
        ], dim=-1)  # [batch, nx, ny, nz, 4]

        # Split Encode
        x_s_in = torch.cat([T_solid.unsqueeze(-1), shared], dim=-1)    # [B, nx, ny, nz, 5]
        x_f_in = torch.cat([T_f_embed.unsqueeze(-1), shared], dim=-1)  # [B, nx, ny, nz, 5]

        h_s = self.fc0_s(x_s_in).permute(0, 4, 1, 2, 3)  # [B, ch, nx, ny, nz]
        h_f = self.fc0_f(x_f_in).permute(0, 4, 1, 2, 3)  # [B, ch, nx, ny, nz]

        # Dual-stream processing with cross-coupling
        for i in range(len(self.solid_blocks)):
            # Independent physics evolution
            h_s_new = self.solid_blocks[i](
                h_s, dt=dt_val, sample_alpha=sample_alpha,
                film_params=film_params)
            h_f_new = self.fluid_blocks[i](
                h_f, dt=dt_val, sample_alpha=sample_alpha,
                sample_vx=sample_vx, film_params=film_params)

            # Gated cross-coupling
            gate = torch.sigmoid(self.coupling_gate[i])
            h_s = h_s_new + gate * self.couple_f2s[i](h_f_new)
            h_f = h_f_new + gate * self.couple_s2f[i](h_s_new)

        # Split Decode
        h_s = h_s.permute(0, 2, 3, 4, 1)  # [B, nx, ny, nz, ch]
        h_f = h_f.permute(0, 2, 3, 4, 1)

        dT_s = self.fc2_s(F.gelu(self.fc1_s(h_s))).squeeze(-1)  # [B, nx, ny, nz]
        dT_f = self.fc2_f(F.gelu(self.fc1_f(h_f))).squeeze(-1)

        return T_solid + dT_s, T_f_embed + dT_f

    def analyze_physics_contribution(self, dt):
        results = []
        for i, (sb, fb) in enumerate(zip(self.solid_blocks, self.fluid_blocks)):
            s_info = sb.spectral_conv.get_physics_info(dt)
            f_info = fb.spectral_conv.get_physics_info(dt)
            gate = torch.sigmoid(self.coupling_gate[i]).item()
            results.append({
                'layer': i,
                'solid': s_info,
                'fluid': f_info,
                'coupling_gate': gate,
                # Backward-compatible keys
                'physics_ratio': (s_info['physics_ratio'] + f_info['physics_ratio']) / 2,
                'alpha_s': s_info['alpha'], 'beta_s': s_info['beta'],
                'alpha_c': f_info['alpha'], 'beta_c': f_info['beta'],
                'vz': f_info.get('vz', 0),
            })
        return results


class StandardFNOBlock3d(nn.Module):
    """Standard 3D FNO block (baseline)"""
    def __init__(self, channels, modes=8):
        super().__init__()
        self.channels = channels
        self.modes = modes
        scale = 1 / (channels * channels)
        self.weights1 = nn.Parameter(scale * torch.rand(
            channels, channels, modes, modes, modes, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.rand(
            channels, channels, modes, modes, modes, dtype=torch.cfloat))
        self.conv = nn.Conv3d(channels, channels, 1)
        self.norm = nn.InstanceNorm3d(channels)

    def compl_mul3d(self, x, weights):
        return torch.einsum("bixyz,ioxyz->boxyz", x, weights)

    def forward(self, x, dt=None):
        batch_size = x.shape[0]
        residual = x
        m = self.modes
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        out_ft = torch.zeros(batch_size, self.channels, x.size(-3), x.size(-2),
                             x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :m, :m, :m] = self.compl_mul3d(
            x_ft[:, :, :m, :m, :m], self.weights1)
        out_ft[:, :, -m:, :m, :m] = self.compl_mul3d(
            x_ft[:, :, -m:, :m, :m], self.weights2)
        x_ft_out = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        x_conv = self.conv(residual)
        return F.gelu(self.norm(x_ft_out + x_conv)) + residual


class StandardPINO3D(nn.Module):
    """Standard 3D PINO (baseline) - two fields: T_solid + T_coolant"""
    def __init__(self, modes=8, width=64, n_layers=5, device='cuda'):
        super().__init__()
        self.width = width
        self.device = device
        # Input: T_solid(1) + T_f_embed(1) + grid(3) + dt(1) = 6 channels
        self.fc0 = nn.Linear(6, width)
        self.fno_blocks = nn.ModuleList([
            StandardFNOBlock3d(width, modes=modes) for _ in range(n_layers)
        ])
        self.fc1 = nn.Linear(width, width)
        # Output: dT_solid + dT_f = 2 channels
        self.fc2 = nn.Linear(width, 2)
        self.to(device)

    def forward(self, T_solid, T_f_embed, dt, grid, **kwargs):
        """kwargs absorbs sample_alpha/sample_vx (ignored for standard FNO)"""
        batch_size = T_solid.shape[0]
        nx, ny, nz = T_solid.shape[1], T_solid.shape[2], T_solid.shape[3]

        if isinstance(dt, (int, float)):
            dt = torch.tensor([dt], device=T_solid.device)
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


# ============================================================
# Hybrid Physics-Structured FNO (split encoder + full spectrum + per-channel phase gating)
# ============================================================
# Core design:
#   1. Split encoder: fc0_s(T_s) -> 32ch, fc0_f(T_f) -> 32ch, concat -> 64ch
#      - channels 0-31 start as solid features, 32-63 start as fluid features
#   2. Full spectral capacity: 64x64 spectral weights (not the 32x32 of the dual stream)
#   3. Per-channel phase gating:
#        D_c(k,dt) = exp(-(alpha|k|^2+beta)dt) * exp(-i*sigma(g_c)*vz*kz*dt)
#      - g_c initialization: channels 0-31 -> -2 (sigma~0.12, almost no advection)
#      -                     channels 32-63 -> +2 (sigma~0.88, almost full advection)
#   4. Conditioning: beta is modulated by sample_alpha (h_sf_a_sf),
#      vz is modulated by sample_vx (v_darcy) (independent, not conflated)
# ============================================================

class MultiFieldSpectralConv3d(nn.Module):
    """
    Hybrid 3D spectral convolution: full spectral weights + per-channel phase gating

    D_c(k,dt) = exp(-(alpha|k|^2+beta)dt) * exp(-i*sigma(g_c)*vz*kz*dt)
      - alpha: learnable diffusion coefficient (shared, init = 1.0)
      - beta: learnable coupling/loss rate (shared, init ~0.007)
      - vz: learnable advection velocity (shared, z-direction Darcy flow)
      - g_c: per-channel advection gate (64 values, initialized to match the encoder split)

    Forward computation:
      y_c = D_c * (W @ x)_c + R_corr @ x_c
      D_c is per-channel: solid channels have gate ~0 (no advection),
      fluid channels have gate ~1 (full advection)

    Per-sample conditioning:
      sample_alpha (h_sf_a_sf-normalized) -> independently modulates beta (coupling strength)
      sample_vx (v_darcy-normalized) -> independently modulates vz (advection velocity)

    + low-rank correction R_corr = U*V (across all channels, applied to the original input)
    """

    def __init__(self, channels, modes=8, corr_rank=4):
        super().__init__()
        self.channels = channels
        self.modes = modes
        self.corr_rank = corr_rank

        scale = 1 / (channels * channels)
        corr_scale = scale * 0.01

        # Main weights: all-channel interaction (channels x channels)
        self.weights1 = nn.Parameter(scale * torch.rand(
            channels, channels, modes, modes, modes, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.rand(
            channels, channels, modes, modes, modes, dtype=torch.cfloat))

        # Low-rank correction: across all channels
        self.corr_U1 = nn.Parameter(corr_scale * torch.rand(
            channels, corr_rank, modes, modes, modes, dtype=torch.cfloat))
        self.corr_V1 = nn.Parameter(corr_scale * torch.rand(
            corr_rank, channels, modes, modes, modes, dtype=torch.cfloat))
        self.corr_U2 = nn.Parameter(corr_scale * torch.rand(
            channels, corr_rank, modes, modes, modes, dtype=torch.cfloat))
        self.corr_V2 = nn.Parameter(corr_scale * torch.rand(
            corr_rank, channels, modes, modes, modes, dtype=torch.cfloat))

        # Unified physics parameters (shared, not split per channel)
        self.log_alpha = nn.Parameter(torch.zeros(1))            # diffusion coefficient alpha, init = 1.0
        self.log_beta = nn.Parameter(torch.full((1,), -5.0))     # coupling/loss rate beta, init ~0.007
        self.log_vz = nn.Parameter(torch.zeros(1))               # |vz|, init = 1.0
        self.v_sign_z = nn.Parameter(torch.ones(1))              # tanh -> sign

        # Per-channel advection gate (aligned with encoder split)
        # channels 0..ch//2-1: solid-like -> sigma(-2)~0.12, minimal advection
        # channels ch//2..ch-1: fluid-like -> sigma(+2)~0.88, near-full advection
        gate_init = torch.empty(channels)
        gate_init[:channels // 2] = -2.0
        gate_init[channels // 2:] = 2.0
        self.adv_gate = nn.Parameter(gate_init)

        # Wavenumber buffers
        k1 = torch.arange(modes).float()
        k2 = torch.arange(modes).float()
        k3 = torch.arange(modes).float()
        K1p, K2p, K3p = torch.meshgrid(k1, k2, k3, indexing='ij')
        self.register_buffer('k_sq_pos', K1p**2 + K2p**2 + K3p**2)
        # clone(): meshgrid returns expanded views with zero stride, and a
        # zero-stride buffer cannot be the target of load_state_dict's copy_.
        self.register_buffer('kz_pos', K3p.clone())

        k1_neg = torch.arange(modes, 0, -1).float()
        K1n, K2n, K3n = torch.meshgrid(k1_neg, k2, k3, indexing='ij')
        self.register_buffer('k_sq_neg', K1n**2 + K2n**2 + K3n**2)
        self.register_buffer('kz_neg', K3n.clone())

    def _compute_decay(self, k_sq, kz, dt, sample_alpha=None, sample_vx=None):
        """Compute per-channel spectral decay with gated advection.

        D_c(k,dt) = exp(-(alpha|k|^2+beta)dt) * exp(-i*sigma(g_c)*vz*kz*dt)

        Conditioning (independent):
          - sample_alpha -> modulates beta (h_sf_a_sf)
          - sample_vx -> modulates vz (v_darcy)

        Returns:
          (channels, m, m, m) without per-sample conditioning
          (batch, channels, m, m, m) with per-sample conditioning
        """
        alpha = self.log_alpha.exp()
        beta = self.log_beta.exp()
        vz = torch.tanh(self.v_sign_z) * self.log_vz.exp()
        gate = torch.sigmoid(self.adv_gate)  # (channels,)

        per_sample = (sample_alpha is not None) or (sample_vx is not None)

        # --- Diffusion exponent: -(alpha|k|^2 + beta) * dt ---
        if sample_alpha is not None:
            sa = sample_alpha.view(-1, 1, 1, 1)        # (batch,1,1,1)
            beta_eff = beta * sa
            diff_exp = -(alpha * k_sq.unsqueeze(0) + beta_eff) * dt  # (batch,m,m,m)
        else:
            diff_exp = -(alpha * k_sq + beta) * dt      # (m,m,m)

        # --- Advection phase: -i * vz * kz * dt ---
        if sample_vx is not None:
            sv = sample_vx.view(-1, 1, 1, 1)            # (batch,1,1,1)
            vz_eff = vz * sv
            phase = -1j * vz_eff * kz.unsqueeze(0) * dt # (batch,m,m,m)
        elif per_sample:
            phase = -1j * vz * kz.unsqueeze(0) * dt     # (1,m,m,m)
        else:
            phase = -1j * vz * kz * dt                  # (m,m,m)

        # --- Per-channel gating ---
        if per_sample:
            # diff_exp: (batch,m,m,m), phase: (batch,m,m,m) or (1,m,m,m)
            # gate: (channels,) -> (1, channels, 1, 1, 1)
            D = torch.exp(
                diff_exp.unsqueeze(1)
                + gate.view(1, -1, 1, 1, 1) * phase.unsqueeze(1)
            )  # (batch, channels, m, m, m)
        else:
            # diff_exp: (m,m,m), phase: (m,m,m)
            # gate: (channels,) -> (channels, 1, 1, 1)
            D = torch.exp(
                diff_exp.unsqueeze(0)
                + gate.view(-1, 1, 1, 1) * phase.unsqueeze(0)
            )  # (channels, m, m, m)

        return D

    def compl_mul3d(self, x, weights):
        return torch.einsum("bixyz,ioxyz->boxyz", x, weights)

    def _low_rank_mul3d(self, x_ft, U, V):
        temp = torch.einsum("bixyz,irxyz->brxyz", x_ft, U)
        return torch.einsum("brxyz,roxyz->boxyz", temp, V)

    def forward(self, x, dt=None, sample_alpha=None, sample_vx=None):
        batch_size = x.shape[0]
        nx, ny, nz = x.size(-3), x.size(-2), x.size(-1)
        m = self.modes

        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        out_ft = torch.zeros(batch_size, self.channels, nx, ny,
                             nz // 2 + 1, dtype=torch.cfloat, device=x.device)

        x_pos = x_ft[:, :, :m, :m, :m]
        x_neg = x_ft[:, :, -m:, :m, :m]

        # Step 1: Standard channel mixing W @ x
        main_pos = self.compl_mul3d(x_pos, self.weights1)
        main_neg = self.compl_mul3d(x_neg, self.weights2)

        # Step 2: Apply per-channel gated decay on OUTPUT
        if dt is not None and dt > 0:
            dp = self._compute_decay(
                self.k_sq_pos, self.kz_pos, dt, sample_alpha, sample_vx)
            dn = self._compute_decay(
                self.k_sq_neg, self.kz_neg, dt, sample_alpha, sample_vx)

            # dp/dn: (channels,m,m,m) or (batch,channels,m,m,m)
            if dp.dim() == 4:
                # No per-sample: (channels,m,m,m) -> (1,channels,m,m,m)
                dp = dp.unsqueeze(0)
                dn = dn.unsqueeze(0)
            # else: already (batch,channels,m,m,m)

            main_pos = main_pos * dp
            main_neg = main_neg * dn

        # Step 3: Low-rank correction on original input (no physics, learnable residual)
        corr_pos = self._low_rank_mul3d(x_pos, self.corr_U1, self.corr_V1)
        corr_neg = self._low_rank_mul3d(x_neg, self.corr_U2, self.corr_V2)

        out_ft[:, :, :m, :m, :m] = main_pos + corr_pos
        out_ft[:, :, -m:, :m, :m] = main_neg + corr_neg

        return torch.fft.irfftn(out_ft, s=(nx, ny, nz))

    def get_physics_info(self, dt):
        """Analyze physics contribution: per-channel gated decay vs correction."""
        with torch.no_grad():
            alpha = self.log_alpha.exp().item()
            beta = self.log_beta.exp().item()
            vz = (torch.tanh(self.v_sign_z) * self.log_vz.exp()).item()

            gate = torch.sigmoid(self.adv_gate)  # (channels,)
            ch = self.channels
            gate_solid = gate[:ch // 2].mean().item()
            gate_fluid = gate[ch // 2:].mean().item()

            # Compute per-channel decay at positive modes
            decay = self._compute_decay(self.k_sq_pos, self.kz_pos, dt)
            # decay: (channels, m, m, m)
            decay_mean = decay.abs().mean().item()

            # Physics norm: ||D * W|| ~ ||W|| * mean(|D|)
            w_norm = self.weights1.abs().mean().item()
            physics_norm = w_norm * decay_mean

            # Correction norm: ||U @ V||
            corr_full = torch.einsum("irxyz,roxyz->ioxyz", self.corr_U1, self.corr_V1)
            corr_norm = corr_full.abs().mean().item()

            total = physics_norm + corr_norm + 1e-12

            return {
                'alpha': alpha,
                'beta': beta,
                'vz': vz,
                'gate_solid_mean': gate_solid,
                'gate_fluid_mean': gate_fluid,
                'decay_mean': decay_mean,
                'physics_ratio': physics_norm / total,
                'correction_ratio': corr_norm / total,
            }


class MultiFieldFNOBlock3d(nn.Module):
    """Physics-structured FNO block: per-channel gated spectral conv + bypass + norm + residual"""
    def __init__(self, channels, modes=8, corr_rank=4):
        super().__init__()
        self.spectral_conv = MultiFieldSpectralConv3d(
            channels, modes=modes, corr_rank=corr_rank
        )
        self.conv = nn.Conv3d(channels, channels, 1)
        self.norm = nn.InstanceNorm3d(channels)

    def forward(self, x, dt=None, sample_alpha=None, sample_vx=None):
        residual = x
        x_ft = self.spectral_conv(x, dt=dt, sample_alpha=sample_alpha, sample_vx=sample_vx)
        x_conv = self.conv(residual)
        return F.gelu(self.norm(x_ft + x_conv)) + residual


class PhysicsSingleStreamPINO3D(nn.Module):
    """
    Hybrid physics-structured 3D PINO: split encoder + full spectrum + per-channel phase gating

    Architecture:
      Encode: fc0_s([T_s, grid, dt]) -> 32ch (solid)
              fc0_f([T_f, grid, dt]) -> 32ch (fluid)
              concat -> 64ch (channels 0-31 = solid, 32-63 = fluid)
      Spectral layers: 64x64 full spectral weights + per-channel phase gating (5 layers)
      Decode: fc1(64->64) + fc2(64->2) -> [dT_s, dT_f]

    Physics kernel (per-channel):
      D_c(k,dt) = exp(-(alpha|k|^2+beta)dt) * exp(-i*sigma(g_c)*vz*kz*dt)

    Per-sample conditioning:
      sample_alpha (h_sf_a_sf) -> independently modulates beta
      sample_vx (v_darcy) -> independently modulates vz
    """

    def __init__(self, modes=8, width=64, n_layers=5, corr_rank=4, device='cuda'):
        super().__init__()
        self.width = width
        self.device = device
        self.pde_type = 'ltne_single_stream'  # enables per-sample conditioning

        ch_half = width // 2

        # Split field-aware encoder
        # Input: field_value(1) + grid(3) + dt(1) = 5
        self.fc0_s = nn.Linear(5, ch_half)  # T_s + grid + dt -> 32ch (solid)
        self.fc0_f = nn.Linear(5, ch_half)  # T_f + grid + dt -> 32ch (fluid)

        # Full-width spectral layers + per-channel physics gating
        self.fno_blocks = nn.ModuleList([
            MultiFieldFNOBlock3d(width, modes=modes, corr_rank=corr_rank)
            for _ in range(n_layers)
        ])

        # Decoder
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, 2)  # dT_solid + dT_f

        # Residual scaling factor
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

        self.to(device)

    def forward(self, T_solid, T_f_embed, dt, grid, sample_alpha=None, sample_vx=None):
        batch_size = T_solid.shape[0]
        nx, ny, nz = T_solid.shape[1], T_solid.shape[2], T_solid.shape[3]

        if isinstance(dt, (int, float)):
            dt = torch.tensor([dt], device=T_solid.device)
        if dt.dim() == 0:
            dt = dt.unsqueeze(0)
        dt_exp = dt.view(-1, 1, 1, 1, 1).expand(batch_size, nx, ny, nz, 1)
        dt_val = (dt[0] if dt.numel() > 0 else dt).item()

        # Shared spatial features: grid(3) + dt(1) = 4
        shared = torch.cat([
            grid.expand(batch_size, -1, -1, -1, -1),
            dt_exp
        ], dim=-1)  # [B, nx, ny, nz, 4]

        # Split field-aware encoding
        x_s_in = torch.cat([T_solid.unsqueeze(-1), shared], dim=-1)    # [B, nx, ny, nz, 5]
        x_f_in = torch.cat([T_f_embed.unsqueeze(-1), shared], dim=-1)  # [B, nx, ny, nz, 5]

        x_s = self.fc0_s(x_s_in)  # [B, nx, ny, nz, 32]
        x_f = self.fc0_f(x_f_in)  # [B, nx, ny, nz, 32]

        # Concat: channels 0-31 = solid features, 32-63 = fluid features
        # (aligned with adv_gate initialization in MultiFieldSpectralConv3d)
        x = torch.cat([x_s, x_f], dim=-1)  # [B, nx, ny, nz, 64]
        x = x.permute(0, 4, 1, 2, 3)       # [B, 64, nx, ny, nz]

        for block in self.fno_blocks:
            x = block(x, dt=dt_val, sample_alpha=sample_alpha, sample_vx=sample_vx)

        x = x.permute(0, 2, 3, 4, 1)  # [B, nx, ny, nz, 64]
        x = F.gelu(self.fc1(x))
        out = self.fc2(x)  # [B, nx, ny, nz, 2]

        T_solid_next = T_solid + self.residual_scale * out[..., 0]
        T_f_next = T_f_embed + self.residual_scale * out[..., 1]
        return T_solid_next, T_f_next

    def analyze_physics_contribution(self, dt):
        results = []
        for i, block in enumerate(self.fno_blocks):
            info = block.spectral_conv.get_physics_info(dt)
            info['layer'] = i
            results.append(info)
        return results
