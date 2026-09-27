# Physics-Structured FNO for Heat Transfer
# Embeds the spectral decay structure of the PDE Green's function into the FNO spectral convolution kernel
#
# Standard FNO:  out = R_free · v_ft                              (freely learned)
# This method:   out = [decay(k,dt) ⊙ R_free + R_corr] · v_ft    (physics prior + correction)
#
# Unified physics kernel: D(k,dt) = exp(-(α|k|²+ β + iv·k)dt)
#   - α: learnable diffusion coefficient (spatial-frequency-dependent decay)
#   - β: learnable convective loss rate (uniform decay, captures the surface convective cooling effect)
#   - v: learnable advection velocity (phase rotation, conv_diff mode only)
#   - low-rank correction corr_rank=8 (parameter overhead ~17%)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PhysicsStructuredSpectralConv2d(nn.Module):
    """
    Physics-structured 2D Fourier spectral convolution layer

    R_total(k, dt) = exp(-(α|k|²+β)·dt) ⊙ R_base(k) + R_correction(k)

    Unified physics kernel: D(k,dt; α,β,v) = exp(-(α|k|² + β + iv·k)·dt)
    - α: learnable diffusion coefficient (scalar)
    - β: learnable convective loss rate (scalar, captures the surface convective cooling effect)
    - v: learnable advection velocity (conv_diff mode only)
    - R_correction: low-rank factorization U×V (rank=8)
    """

    def __init__(self, in_channels, out_channels, modes1, modes2,
                 corr_rank=8, pde_type='heat', use_correction=True,
                 corr_init_scale=0.01, use_modulation=True, use_learned_vx=True,
                 corr_scale_param=None):
        """
        Args:
            in_channels, out_channels: number of input/output channels
            modes1, modes2: number of retained Fourier modes
            corr_rank: rank of the correction term (r << min(in_ch, out_ch))
            pde_type: PDE type. 'heat' instantiates the diffusion kernel with no reaction
                coefficient; other values ('reaction_diff', 'conv_diff', 'wave') add the
                corresponding learnable coefficients. The released reaction-diffusion
                benchmark entry point (scripts/run_reaction_diff_ar_valfix.py) passes
                pde_type='heat', i.e. it runs through the diffusion branch.
            use_correction: whether to use the low-rank correction R_corr (False = pure physics kernel test)
            corr_init_scale: NOT WIRED. Retained in the signature for call compatibility,
                but this argument is never read: the correction scale is set by
                ``corr_scale_param`` (see the comment at its assignment below), which is
                the argument that actually changes the initialization. ``None`` draws the
                correction factors at the same per-element scale as the base weights; a
                value ``p`` scales them by ``p`` relative to that. Historical sweeps that
                varied ``corr_init_scale`` therefore did not vary the initialization.
            use_modulation: whether to use physics modulation (False = disable physics modulation, keep only the low-rank correction)
            use_learned_vx: whether the conv_diff advection term uses a learnable vx_corr multiplier
                            (False = use vx from the data directly)
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.corr_rank = corr_rank
        self.pde_type = pde_type
        self.use_correction = use_correction
        self.use_learned_vx = use_learned_vx
        self.use_modulation = use_modulation

        scale = 1 / (in_channels * out_channels)

        # Main weights (modulated by the physics prior) — same as standard FNO
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

        # Low-rank correction: R_corr = U × V (optional)
        if use_correction:
            # corr_scale_param: ratio of the correction term initialization to the main weight scale
            #   None = 0.01 (the released 2D default, and the value used for the reported results)
            #   any other value = that multiple of the main weight scale
            corr_scale = scale * 0.01 if corr_scale_param is None else corr_scale_param * scale
            self.corr_U1 = nn.Parameter(corr_scale * torch.rand(in_channels, corr_rank, modes1, modes2, dtype=torch.cfloat))
            self.corr_V1 = nn.Parameter(corr_scale * torch.rand(corr_rank, out_channels, modes1, modes2, dtype=torch.cfloat))
            self.corr_U2 = nn.Parameter(corr_scale * torch.rand(in_channels, corr_rank, modes1, modes2, dtype=torch.cfloat))
            self.corr_V2 = nn.Parameter(corr_scale * torch.rand(corr_rank, out_channels, modes1, modes2, dtype=torch.cfloat))

        # Unified physics parameters (scalars)
        self.log_alpha = nn.Parameter(torch.zeros(1))       # diffusion coefficient α, initial = 1.0
        # β: convective loss rate — only needed for non-heat modes (β=0 is exact for the heat equation)
        if pde_type not in ('heat', 'wave'):
            self.log_beta = nn.Parameter(torch.full((1,), -5.0))  # initial ≈ 0.007

        # Advection-diffusion mode: learnable advection velocity parameters
        if pde_type == 'conv_diff':
            self.log_vx = nn.Parameter(torch.zeros(1))   # log(|vx|+eps)
            self.log_vy = nn.Parameter(torch.zeros(1))   # log(|vy|+eps)
            self.v_sign_x = nn.Parameter(torch.ones(1))  # tanh -> sign
            self.v_sign_y = nn.Parameter(torch.zeros(1))

        # Wave equation mode: learnable wave speed, damping, and viscoelastic parameters
        if pde_type == 'wave':
            self.log_c = nn.Parameter(torch.zeros(1))        # wave speed c, initial = 1.0
            self.log_gamma_w = nn.Parameter(torch.full((1,), -1.0))  # structural damping gamma, initial ≈ 0.37
            self.log_eta_w = nn.Parameter(torch.full((1,), -2.5))    # viscoelastic damping eta, initial ≈ 0.08

        # Precompute wavenumbers (k1, k2, |k|²)
        k1_pos = torch.arange(modes1).float()
        k2 = torch.arange(modes2).float()
        K1p, K2p = torch.meshgrid(k1_pos, k2, indexing='ij')
        self.register_buffer('k_sq_pos', (K1p ** 2 + K2p ** 2))
        self.register_buffer('k1_pos', K1p.clone())
        self.register_buffer('k2_pos', K2p.clone())

        # Negative frequency region: k1 takes negative values (FFT convention), k2 unchanged
        k1_neg = torch.arange(modes1, 0, -1).float()  # magnitude [modes1, ..., 1]
        K1n, K2n = torch.meshgrid(k1_neg, k2, indexing='ij')
        self.register_buffer('k_sq_neg', (K1n ** 2 + K2n ** 2))
        self.register_buffer('k1_neg', K1n.clone())   # magnitude (the advection term needs a sign, see _compute_decay)
        self.register_buffer('k2_neg', K2n.clone())

    def _compute_decay(self, k_sq, k1, k2, dt, neg_k1=False, sample_alpha=None, sample_vx=None, sample_beta=None, sample_c=None, sample_gamma_w=None):
        """Compute the PDE spectral decay factor (unified physics kernel)

        D(k,dt) = exp(-(α|k|² + β + iv·k)·dt)

        heat:           exp(-α|k|²·dt)                  real decay (pure diffusion, β=0)
        reaction_diff:  exp(-(α|k|² + β)·dt)            real decay (diffusion + surface cooling)
        conv_diff:      exp(-(α|k|² + β)·dt - iv·k·dt)  complex decay (diffusion + loss + advective phase shift)
        wave:           exp(-γ·dt) · exp(i·c·|k|·dt)    complex decay (damping + oscillation, hyperbolic)

        Returns: (m1, m2) or (batch, m1, m2) if per-sample parameters are present
        """
        # === Viscoelastic wave equation: frequency-dependent damping + oscillation ===
        if self.pde_type == 'wave':
            c_wave = self.log_c.exp()         # wave speed (scalar)
            gamma_w = self.log_gamma_w.exp()   # structural damping (scalar, frequency-independent)
            eta_w = self.log_eta_w.exp()       # viscoelastic damping (scalar, frequency dependence via |k|^2)

            # |k| = sqrt(k1^2 + k2^2); note this is the first power, not the second!
            k_abs = torch.sqrt(k_sq + 1e-8)  # (m1, m2), avoids sqrt(0)

            # Frequency-dependent damping: gamma + eta*|k|^2
            # - gamma: uniform decay (identical for all modes)
            # - eta*|k|^2: high-frequency modes decay quickly (diffusion-like smoothing effect)
            if sample_gamma_w is not None:
                # per-sample structural damping modulation: gamma_w * sample_gamma_w
                gamma_sample = sample_gamma_w.view(-1, 1, 1)
                damping = torch.exp(-(gamma_w * gamma_sample + eta_w * k_sq.unsqueeze(0)) * dt)  # (batch, m1, m2)
            else:
                damping = torch.exp(-(gamma_w + eta_w * k_sq) * dt)  # (m1, m2) mode-dependent!

            if sample_c is not None:
                # per-sample wave speed modulation
                c_sample = sample_c.view(-1, 1, 1)  # (batch, 1, 1)
                c_eff = c_wave * c_sample
                phase = c_eff * k_abs.unsqueeze(0) * dt  # (batch, m1, m2)
            else:
                phase = c_wave * k_abs * dt  # (m1, m2)

            # D_visco(k, dt) = exp(-(gamma + eta*|k|^2)*dt) * exp(i*c*|k|*dt)
            return damping * torch.exp(1j * phase)

        # === Parabolic PDE (heat equation family) ===
        alpha = self.log_alpha.exp()  # scalar
        beta = self.log_beta.exp() if hasattr(self, 'log_beta') else 0.0

        # Per-sample beta modulation (reaction_diff mode)
        if sample_beta is not None and hasattr(self, 'log_beta'):
            beta = beta * sample_beta.view(-1, 1, 1)  # (batch, 1, 1)

        if sample_alpha is not None:
            alpha_sample = sample_alpha.view(-1, 1, 1)  # (batch, 1, 1)
            alpha_eff = alpha * alpha_sample
            diffusion = -(alpha_eff * k_sq.unsqueeze(0) + beta) * dt  # (batch, m1, m2)
        else:
            diffusion = -(alpha * k_sq + beta) * dt  # (m1, m2) or (batch, m1, m2) if sample_beta

        if self.pde_type in ('heat', 'reaction_diff'):
            return torch.exp(diffusion)

        # conv_diff: advection term
        k1_signed = -k1 if neg_k1 else k1

        if sample_vx is not None:
            vx_base = sample_vx.view(-1, 1, 1)
            if self.use_learned_vx:
                vx_learned_corr = torch.tanh(self.v_sign_x) * self.log_vx.exp()
                vx = vx_base * vx_learned_corr
            else:
                # use_learned_vx=False: advection uses the data vx directly (multiplier = 1.0, most physically correct)
                vx = vx_base
            vy = torch.tanh(self.v_sign_y) * self.log_vy.exp()
            advection = -1j * (vx * k1_signed.unsqueeze(0) + vy * k2.unsqueeze(0)) * dt  # (batch, m1, m2)
            return torch.exp(diffusion + advection)
        else:
            vx = torch.tanh(self.v_sign_x) * self.log_vx.exp()
            vy = torch.tanh(self.v_sign_y) * self.log_vy.exp()
            advection = -1j * (vx * k1_signed + vy * k2) * dt  # (m1, m2)
            return torch.exp(diffusion + advection)

    def _low_rank_mul2d(self, x_ft, U, V):
        """Low-rank correction multiply (without expanding the full-rank matrix):
        out = x_ft × (U × V) = (x_ft × U) × V
        x_ft: (batch, in_ch, m1, m2)
        U: (in_ch, r, m1, m2), V: (r, out_ch, m1, m2)
        out: (batch, out_ch, m1, m2)
        """
        temp = torch.einsum("bixy,irxy->brxy", x_ft, U)  # (batch, r, m1, m2)
        return torch.einsum("brxy,roxy->boxy", temp, V)   # (batch, out_ch, m1, m2)

    def compl_mul2d(self, input, weights):
        """Complex matrix multiply: (batch, in, x, y) × (in, out, x, y) → (batch, out, x, y)"""
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x, dt=None, sample_alpha=None, sample_vx=None, sample_beta=None, sample_c=None, sample_gamma_w=None):
        """
        Args:
            x: (batch, in_channels, nx, ny)
            dt: time step scalar or tensor (if None, degenerates to a standard FNO)
            sample_alpha: per-sample normalized alpha (batch,) or None
            sample_vx: per-sample normalized vx (batch,) or None
            sample_beta: per-sample normalized beta (batch,) or None (reaction_diff)
            sample_c: per-sample normalized wave speed (batch,) or None (wave)
            sample_gamma_w: per-sample normalized structural damping (batch,) or None (wave)
        """
        batchsize = x.shape[0]
        nx, ny = x.shape[-2], x.shape[-1]

        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            batchsize, self.out_channels, nx, ny // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )

        # Slice each frequency band
        x_pos = x_ft[:, :, :self.modes1, :self.modes2]
        x_neg = x_ft[:, :, -self.modes1:, :self.modes2]

        if dt is not None and dt > 0 and self.use_modulation:
            # Physics-structured: decay ⊙ (W · x_ft) + (UV · x_ft)
            decay_pos = self._compute_decay(self.k_sq_pos, self.k1_pos, self.k2_pos, dt,
                                            neg_k1=False, sample_alpha=sample_alpha, sample_vx=sample_vx, sample_beta=sample_beta, sample_c=sample_c, sample_gamma_w=sample_gamma_w)
            decay_neg = self._compute_decay(self.k_sq_neg, self.k1_neg, self.k2_neg, dt,
                                            neg_k1=True, sample_alpha=sample_alpha, sample_vx=sample_vx, sample_beta=sample_beta, sample_c=sample_c, sample_gamma_w=sample_gamma_w)

            # decay shape: (m1, m2) or (batch, m1, m2) — the same decay for all channels
            if decay_pos.dim() == 2:
                # no per-sample parameters: (m1, m2) -> (1, 1, m1, m2) broadcast
                dp = decay_pos.unsqueeze(0).unsqueeze(0)
                dn = decay_neg.unsqueeze(0).unsqueeze(0)
            else:
                # per-sample parameters: (batch, m1, m2) -> (batch, 1, m1, m2)
                dp = decay_pos.unsqueeze(1)
                dn = decay_neg.unsqueeze(1)
            main_pos = self.compl_mul2d(x_pos, self.weights1) * dp
            main_neg = self.compl_mul2d(x_neg, self.weights2) * dn
        else:
            # use_modulation=False: no physics modulation, standard FNO main path
            main_pos = self.compl_mul2d(x_pos, self.weights1)
            main_neg = self.compl_mul2d(x_neg, self.weights2)

        # Low-rank correction: (U V) · x_ft (without expanding the full-rank matrix)
        if self.use_correction:
            corr_pos = self._low_rank_mul2d(x_pos, self.corr_U1, self.corr_V1)
            corr_neg = self._low_rank_mul2d(x_neg, self.corr_U2, self.corr_V2)
            out_ft[:, :, :self.modes1, :self.modes2] = main_pos + corr_pos
            out_ft[:, :, -self.modes1:, :self.modes2] = main_neg + corr_neg
        else:
            out_ft[:, :, :self.modes1, :self.modes2] = main_pos
            out_ft[:, :, -self.modes1:, :self.modes2] = main_neg

        return torch.fft.irfft2(out_ft, s=(nx, ny))

    def get_physics_ratio(self, dt):
        """Analyze the contribution ratio of the physics prior vs. the correction"""
        with torch.no_grad():
            if self.pde_type == 'wave':
                c_wave = self.log_c.exp().item()
                gamma_w = self.log_gamma_w.exp().item()
                k_abs = torch.sqrt(self.k_sq_pos + 1e-8)
                damping = np.exp(-gamma_w * dt)
                # amplitude of the oscillatory decay = damping (phase does not affect the amplitude)
                decay_pos = torch.full_like(self.k_sq_pos, damping)
                physics_norm = (self.weights1 * decay_pos.unsqueeze(0).unsqueeze(0)).abs().mean()
            else:
                alpha = self.log_alpha.exp().item()
                beta = self.log_beta.exp().item() if hasattr(self, 'log_beta') else 0.0
                # decay: (m1, m2)
                decay_pos = torch.exp(-(alpha * self.k_sq_pos + beta) * dt)
                physics_norm = (self.weights1 * decay_pos.unsqueeze(0).unsqueeze(0)).abs().mean()

            if self.use_correction:
                corr_full = torch.einsum("irxy,roxy->ioxy", self.corr_U1, self.corr_V1)
                corr_norm = corr_full.abs().mean()
            else:
                corr_norm = torch.tensor(0.0)

            total_norm = physics_norm + corr_norm + 1e-10
            result = {
                'physics_ratio': (physics_norm / total_norm).item(),
                'correction_ratio': (corr_norm / total_norm).item(),
            }
            if self.pde_type == 'wave':
                result['c'] = c_wave
                result['gamma_w'] = gamma_w
            else:
                result['alpha'] = self.log_alpha.exp().item()
                result['beta'] = self.log_beta.exp().item() if hasattr(self, 'log_beta') else 0.0
            if self.pde_type == 'conv_diff':
                result['vx'] = (torch.tanh(self.v_sign_x) * self.log_vx.exp()).item()
                result['vy'] = (torch.tanh(self.v_sign_y) * self.log_vy.exp()).item()
            return result


class PhysicsStructuredFNOBlock2d(nn.Module):
    """FNO block using the physics-structured spectral convolution"""

    def __init__(self, in_channels, out_channels, modes1, modes2, activation='gelu', corr_rank=8, pde_type='heat', use_correction=True, corr_init_scale=0.01, use_modulation=True, use_learned_vx=True, corr_scale_param=None):
        super().__init__()
        self.spectral_conv = PhysicsStructuredSpectralConv2d(
            in_channels, out_channels, modes1, modes2, corr_rank=corr_rank, pde_type=pde_type, use_correction=use_correction, corr_init_scale=corr_init_scale, use_modulation=use_modulation, use_learned_vx=use_learned_vx, corr_scale_param=corr_scale_param
        )
        self.linear = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        activations = {
            'relu': nn.ReLU(), 'gelu': nn.GELU(),
            'tanh': nn.Tanh(), 'silu': nn.SiLU()
        }
        self.activation = activations.get(activation, nn.GELU())

    def forward(self, x, dt=None, sample_alpha=None, sample_vx=None, sample_beta=None, sample_c=None, sample_gamma_w=None):
        """
        Args:
            x: (batch, channels, nx, ny)
            dt: time step (passed to the spectral convolution layer)
            sample_alpha: per-sample alpha (batch,) or None
            sample_vx: per-sample vx (batch,) or None
            sample_beta: per-sample beta (batch,) or None
            sample_c: per-sample wave speed (batch,) or None (wave)
            sample_gamma_w: per-sample structural damping (batch,) or None (wave)
        """
        x1 = self.spectral_conv(x, dt=dt, sample_alpha=sample_alpha, sample_vx=sample_vx, sample_beta=sample_beta, sample_c=sample_c, sample_gamma_w=sample_gamma_w)
        x2 = self.linear(x)
        return self.activation(x1 + x2)


class PhysicsStructuredPINO(nn.Module):
    """
    Physics-structured autoregressive neural operator

    Heat equation family: T(t+dt) = T(t) + scale · G_θ(T(t), dt, x)
    Wave equation: u(t+dt) = u(t) + scale · G_θ(u(t), u(t-dt), dt, x)

    where the spectral convolution kernels of G_θ are constrained by the PDE spectral semigroup prior
    """

    def __init__(self, modes=16, width=96, n_layers=5, device='cuda', corr_rank=8, pde_type='heat', use_correction=True, corr_init_scale=0.01, use_modulation=True, use_learned_vx=True, corr_scale_param=None):
        super().__init__()
        self.width = width
        self.n_layers = n_layers
        self.pde_type = pde_type

        # Input projection:
        #   heat equation family: T_current(1) + grid(2) + dt(1) = 4 channels
        #   wave equation: u_current(1) + u_prev(1) + grid(2) + dt(1) = 5 channels
        in_channels = 5 if pde_type == 'wave' else 4
        self.fc0 = nn.Linear(in_channels, width)

        # Physics-structured FNO backbone
        self.backbone = nn.ModuleList()
        for _ in range(n_layers):
            self.backbone.append(
                PhysicsStructuredFNOBlock2d(width, width, modes, modes, 'gelu', corr_rank=corr_rank, pde_type=pde_type, use_correction=use_correction, corr_init_scale=corr_init_scale, use_modulation=use_modulation, use_learned_vx=use_learned_vx, corr_scale_param=corr_scale_param)
            )

        # Output projection
        self.fc1 = nn.Linear(width, width // 2)
        self.fc2 = nn.Linear(width // 2, 1)

        # Residual scaling factor
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

        self.to(device)

    def forward(self, T_current, dt, grid, sample_alpha=None, sample_vx=None, sample_beta=None,
                u_prev=None, sample_c=None, sample_gamma_w=None):
        """
        Args:
            T_current: (batch, nx, ny) current field (temperature or displacement u(t))
            dt: time step (scalar or tensor)
            grid: (1, nx, ny, 2) spatial grid
            sample_alpha: per-sample normalized alpha (batch,) or None
            sample_vx: per-sample normalized vx (batch,) or None
            sample_beta: per-sample normalized beta (batch,) or None
            u_prev: (batch, nx, ny) displacement at the previous time step u(t-dt), used only in wave mode
            sample_c: per-sample normalized wave speed (batch,) or None
        Returns:
            T_next: (batch, nx, ny) field at the next time step
        """
        batch_size, nx, ny = T_current.shape

        if isinstance(dt, (int, float)):
            dt_val = dt
            dt = torch.tensor([dt], device=T_current.device).expand(batch_size)
        else:
            dt_val = dt.item() if dt.numel() == 1 else dt[0].item()

        dt_exp = dt.view(-1, 1, 1, 1).expand(batch_size, nx, ny, 1)

        # Assemble the input
        if self.pde_type == 'wave':
            # wave equation: [u_curr, u_prev, grid_x, grid_y, dt] = 5 channels
            assert u_prev is not None, "Wave mode requires u_prev input"
            x = torch.cat([
                T_current.unsqueeze(-1),
                u_prev.unsqueeze(-1),
                grid.expand(batch_size, -1, -1, -1),
                dt_exp
            ], dim=-1)
        else:
            # heat equation family: [T, grid_x, grid_y, dt] = 4 channels
            x = torch.cat([
                T_current.unsqueeze(-1),
                grid.expand(batch_size, -1, -1, -1),
                dt_exp
            ], dim=-1)

        # Project to the latent space
        x = self.fc0(x).permute(0, 3, 1, 2)  # (B, width, nx, ny)

        # Pass through the physics-structured FNO blocks (dt and per-sample parameters are passed in)
        for block in self.backbone:
            x = x + block(x, dt=dt_val, sample_alpha=sample_alpha, sample_vx=sample_vx, sample_beta=sample_beta, sample_c=sample_c, sample_gamma_w=sample_gamma_w)

        # Project to the output
        x = x.permute(0, 2, 3, 1)  # (B, nx, ny, width)
        x = F.gelu(self.fc1(x))
        delta_T = self.fc2(x).squeeze(-1)  # (B, nx, ny)

        # Residual connection
        return T_current + self.residual_scale * delta_T

    def analyze_physics_contribution(self, dt):
        """Analyze the physics prior contribution ratio of each layer"""
        results = []
        for i, block in enumerate(self.backbone):
            ratio = block.spectral_conv.get_physics_ratio(dt)
            ratio['layer'] = i
            results.append(ratio)
        return results


# ============================================================
# Standard FNO baseline (for comparison experiments)
# ============================================================

class StandardSpectralConv2d(nn.Module):
    """Standard FNO spectral convolution (same as the original fno.py, but with a unified interface)"""

    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x, dt=None):
        """The dt argument is ignored; the interface is kept consistent"""
        batchsize = x.shape[0]
        nx, ny = x.shape[-2], x.shape[-1]

        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            batchsize, self.out_channels, nx, ny // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )
        return torch.fft.irfft2(out_ft, s=(nx, ny))


class StandardFNOBlock2d(nn.Module):
    """Standard FNO block (for comparison)"""
    def __init__(self, in_channels, out_channels, modes1, modes2, activation='gelu'):
        super().__init__()
        self.spectral_conv = StandardSpectralConv2d(in_channels, out_channels, modes1, modes2)
        self.linear = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        activations = {'relu': nn.ReLU(), 'gelu': nn.GELU(), 'tanh': nn.Tanh(), 'silu': nn.SiLU()}
        self.activation = activations.get(activation, nn.GELU())

    def forward(self, x, dt=None):
        x1 = self.spectral_conv(x, dt=dt)
        x2 = self.linear(x)
        return self.activation(x1 + x2)


class StandardAutoregressivePINO(nn.Module):
    """Standard FNO autoregressive model (comparison baseline)

    Supports wave_mode=True for the wave equation (5-channel input)
    """
    def __init__(self, modes=16, width=96, n_layers=5, device='cuda', wave_mode=False):
        super().__init__()
        self.wave_mode = wave_mode
        in_channels = 5 if wave_mode else 4
        self.fc0 = nn.Linear(in_channels, width)
        self.backbone = nn.ModuleList()
        for _ in range(n_layers):
            self.backbone.append(StandardFNOBlock2d(width, width, modes, modes, 'gelu'))
        self.fc1 = nn.Linear(width, width // 2)
        self.fc2 = nn.Linear(width // 2, 1)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        self.to(device)

    def forward(self, T_current, dt, grid, **kwargs):
        # kwargs absorbs sample_alpha/sample_vx/u_prev etc.
        batch_size, nx, ny = T_current.shape
        if isinstance(dt, (int, float)):
            dt = torch.tensor([dt], device=T_current.device).expand(batch_size)
        dt_exp = dt.view(-1, 1, 1, 1).expand(batch_size, nx, ny, 1)

        if self.wave_mode:
            u_prev = kwargs.get('u_prev', None)
            assert u_prev is not None, "Wave mode requires u_prev input"
            x = torch.cat([
                T_current.unsqueeze(-1),
                u_prev.unsqueeze(-1),
                grid.expand(batch_size, -1, -1, -1),
                dt_exp
            ], dim=-1)
        else:
            x = torch.cat([T_current.unsqueeze(-1), grid.expand(batch_size, -1, -1, -1), dt_exp], dim=-1)

        x = self.fc0(x).permute(0, 3, 1, 2)
        for block in self.backbone:
            x = x + block(x)
        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        delta_T = self.fc2(x).squeeze(-1)
        return T_current + self.residual_scale * delta_T


# ============================================================
# Training strategies (comparison of three modes)
# ============================================================

def train_teacher_forcing(model, data, n_epochs=2000, device='cuda'):
    """Pure teacher forcing training (ground truth is used as input at every step)"""
    return _train_autoregressive(model, data, n_epochs, device, mode='teacher_forcing')


def train_scheduled_sampling(model, data, n_epochs=2000, device='cuda',
                              p_start=0.9, p_end=0.3, warmup=500, anneal_end=1500,
                              hard_topk=5, random_n=3, eval_data=None):
    """Scheduled sampling training. When eval_data is non-empty, the checkpoint is selected by val eval (default None = select by data)."""
    return _train_autoregressive(model, data, n_epochs, device, mode='scheduled_sampling',
                                  p_start=p_start, p_end=p_end,
                                  warmup=warmup, anneal_end=anneal_end,
                                  hard_topk=hard_topk, random_n=random_n,
                                  eval_data=eval_data)


def train_pushforward(model, data, n_epochs=2000, device='cuda',
                       noise_std=0.01, noise_ramp_epochs=500):
    """Pushforward trick training (noise is injected into the input to mimic the error distribution)"""
    return _train_autoregressive(model, data, n_epochs, device, mode='pushforward',
                                  noise_std=noise_std, noise_ramp_epochs=noise_ramp_epochs)


def _train_autoregressive(model, data, n_epochs, device, mode='teacher_forcing', **kwargs):
    """Unified autoregressive training framework"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=500, T_mult=2, eta_min=1e-6
    )

    mask, X, Y = data['mask'], data['X'], data['Y']
    T_all, t_array = data['T'], data['t']
    normalizer = data['normalizer']
    n_traj, nt = T_all.shape[0], len(t_array)
    grid = torch.stack([X, Y], dim=-1).unsqueeze(0)
    valid = mask > 0.5
    dt_array = t_array[1:] - t_array[:-1]

    # per-sample PDE parameters (for the sample-conditioned physics kernel)
    # only effective for PhysicsStructuredPINO
    # supports alpha-only (heat), alpha+beta (reaction_diff), alpha+vx (conv_diff)
    has_sample_alpha = ('alpha_array' in data and hasattr(model, 'pde_type'))
    has_sample_vx = 'vx_array' in data
    has_sample_beta = ('beta_array' in data and hasattr(model, 'pde_type'))
    if has_sample_alpha:
        alpha_arr = data['alpha_array']
        alpha_mean = alpha_arr.mean()
        alpha_norm = torch.tensor(alpha_arr / alpha_mean, dtype=torch.float32, device=device)
    if has_sample_vx:
        vx_arr = data['vx_array']
        vx_norm = torch.tensor(vx_arr, dtype=torch.float32, device=device)
    if has_sample_beta:
        beta_arr = data['beta_array']
        beta_mean = beta_arr.mean()
        beta_norm = torch.tensor(beta_arr / beta_mean, dtype=torch.float32, device=device)

    # Boundary weights
    weight = torch.ones_like(Y)
    hot_boundary = (Y < 0.15) & valid
    weight = torch.where(hot_boundary, weight * 3.0, weight)
    cold_boundary = (Y > 0.85) & valid
    weight = torch.where(cold_boundary, weight * 2.0, weight)
    boundary_weight = weight * mask

    sample_errors = np.ones(n_traj) * 10
    best_max, best_state = float('inf'), None

    # Default values (overridden by the mode-specific computation)
    p_teacher = kwargs.get('p_start', 0.9)
    noise_factor = kwargs.get('noise_std', 0.01)
    eval_data = kwargs.get('eval_data', None)  # non-empty = select the checkpoint by validation-set evaluation

    # Helper function that applies the BC
    def apply_bc(T, T_hot_norm):
        T_out = T.clone()
        hot = (Y < 0.1) & valid
        if hot.any():
            w = torch.clamp((0.1 - Y) / 0.08, 0, 1)
            T_out = torch.where(hot, T_out * (1 - w) + T_hot_norm * w, T_out)
        return T_out * mask

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        # Sampling strategy
        # hard_topk/random_n are tunable (default 5/3 = original protocol; increasing random_n improves trajectory coverage and reduces worst-seed dependence)
        hard_topk = kwargs.get('hard_topk', 5)
        random_n = kwargs.get('random_n', 3)
        if epoch < 200:
            batch_idx = np.random.choice(n_traj, min(8, n_traj), replace=False)
        else:
            hard_idx = np.argsort(sample_errors)[-hard_topk:]
            random_idx = np.random.choice(n_traj, random_n, replace=False)
            batch_idx = np.unique(np.concatenate([hard_idx, random_idx]))

        # Scheduled sampling probability computation
        if mode == 'scheduled_sampling':
            warmup = kwargs.get('warmup', 500)
            anneal_end = kwargs.get('anneal_end', 1500)
            p_start = kwargs.get('p_start', 0.9)
            p_end = kwargs.get('p_end', 0.3)
            if epoch < warmup:
                p_teacher = p_start
            elif epoch < anneal_end:
                p_teacher = p_start - (p_start - p_end) * (epoch - warmup) / (anneal_end - warmup)
            else:
                p_teacher = p_end

        # Pushforward noise magnitude
        if mode == 'pushforward':
            noise_ramp = kwargs.get('noise_ramp_epochs', 500)
            base_noise = kwargs.get('noise_std', 0.01)
            noise_factor = min(1.0, epoch / noise_ramp) * base_noise

        traj_losses = []
        batch_errors = {}

        for idx in batch_idx:
            T_traj = T_all[idx]
            traj_max_err = 0
            traj_loss = 0

            # Initial state
            T_input = T_traj[0]  # step 0 always uses the ground truth

            for t_idx in range(nt - 1):
                T_next_true = T_traj[t_idx + 1]
                dt = dt_array[t_idx]

                # Extract the BC value for the current time step at each step (it varies with time under a ramp BC)
                hot_mask = (Y < 0.1) & valid
                T_hot_norm = T_next_true[hot_mask].mean() if hot_mask.any() else T_traj[0][hot_mask].mean()

                # Select the input according to the training mode
                if mode == 'teacher_forcing':
                    T_input = T_traj[t_idx]

                elif mode == 'scheduled_sampling':
                    if t_idx == 0:
                        T_input = T_traj[0]
                    else:
                        use_teacher = torch.rand(1).item() < p_teacher
                        if use_teacher:
                            T_input = T_traj[t_idx]
                        # else: T_input is already the prediction from the previous step

                elif mode == 'pushforward':
                    T_input = T_traj[t_idx]
                    if epoch > 0:
                        noise = torch.randn_like(T_input) * noise_factor
                        T_input = T_input + noise * mask

                # Forward prediction (per-sample physics parameters are passed in)
                sa = alpha_norm[idx:idx+1] if has_sample_alpha else None
                sv = vx_norm[idx:idx+1] if has_sample_vx else None
                sb = beta_norm[idx:idx+1] if has_sample_beta else None
                T_next_pred = model(T_input.unsqueeze(0), dt.unsqueeze(0), grid,
                                    sample_alpha=sa, sample_vx=sv, sample_beta=sb).squeeze(0)
                T_next_pred = apply_bc(T_next_pred, T_hot_norm)

                # Keep the prediction for the next step (used by scheduled sampling)
                if mode == 'scheduled_sampling':
                    T_input = T_next_pred.detach()

                # Loss computation
                diff = (T_next_pred - T_next_true)[valid]
                w = boundary_weight[valid]
                loss_mse = ((diff ** 2) * w).mean()

                abs_diff = diff.abs()
                max_err = abs_diff.max()
                loss_max = max_err ** 2 * 0.8

                k = max(1, int(abs_diff.numel() * 0.01))
                top_k = torch.topk(abs_diff, k).values
                loss_topk = (top_k ** 2).mean() * 0.5

                # Uniform cooling bias penalty: suppresses the overall mean bias at the last step.
                # Physical meaning: for reaction_diff the k=0 mode (field mean) decays as exp(-βdt);
                # when the per-sample β is stuck at 0 the field is overall too hot;
                # when the per-sample cooling parameter can approximate that mode, the bias goes to 0.
                # This term anchors the "average cooling" as an explicit loss; default 0 = disabled.
                bias_w = kwargs.get('bias_weight', 0.0)
                if bias_w > 0:
                    bias_term = ((T_next_pred - T_next_true)[valid].mean() ** 2) * bias_w
                else:
                    bias_term = 0.0

                traj_loss += loss_mse + loss_max + loss_topk + bias_term
                traj_max_err = max(traj_max_err, (max_err * normalizer.scale).item())

            traj_losses.append(traj_loss / (nt - 1))
            batch_errors[idx] = traj_max_err

        for idx, err in batch_errors.items():
            sample_errors[idx] = 0.6 * sample_errors[idx] + 0.4 * err

        avg_loss = torch.stack(traj_losses).mean()
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 100 == 0 or epoch == 0:
            # when eval_data is non-empty, select the checkpoint by validation-set evaluation (protocol freeze requirement: test is only evaluated after freezing)
            eval_target = eval_data if eval_data is not None else data
            mae, max_err = _eval_model(model, eval_target, grid)
            lr = optimizer.param_groups[0]['lr']
            mode_str = mode.upper()
            extra = ''
            if mode == 'scheduled_sampling':
                extra = f' | p_teacher: {p_teacher:.2f}'
            elif mode == 'pushforward':
                extra = f' | noise: {noise_factor:.4f}'
            print(f'[{mode_str}] Epoch {epoch+1}/{n_epochs} | Loss: {avg_loss.item():.6f} '
                  f'| MAE: {mae:.4f}K | Max: {max_err:.4f}K | LR: {lr:.2e}{extra}', flush=True)

            if max_err < best_max:
                best_max = max_err
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                print(f'  -> New best! Max: {best_max:.4f}K', flush=True)

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print(f'[{mode.upper()}] Best Max Error: {best_max:.4f}K', flush=True)
    return model


def train_autoregressive_finetune(model, data, n_epochs=500, device='cuda',
                                   lr=1e-4, grad_accum_steps=4, eval_data=None,
                                   freeze_physics=False, weight_decay=1e-5):
    """
    Multi-step autoregressive finetuning: a pure autoregressive rollout over all time
    steps, with gradients flowing through the entire rollout.

    Key differences from SS training:
    1. p_teacher=0 (100% self-prediction)
    2. no detach; gradients flow from the last step back to the first step
    3. directly optimizes the true error accumulation at inference time
    4. when eval_data is non-empty, the checkpoint is selected by validation-set evaluation

    Args:
        model: a model already trained with SS
        n_epochs: number of finetuning epochs
        lr: learning rate (should be much smaller than the SS training lr)
        grad_accum_steps: number of gradient accumulation steps (reduces memory pressure)
        eval_data: validation set (optional, None = select by data)
        freeze_physics: freeze the physics kernel (α/vx/main weights weights1/2) and train
            only the low-rank correction terms corr_U/V.
            Motivation: pure AR rollout gradients perturb the physics kernel parameters,
            while the analytic form exp(-αk²dt-iv·k·dt) lacks numerical dissipation and
            ramp BC bias; once the perturbation direction is consistent it is amplified
            layer by layer. Locking the physics kernel at its post-SS state lets the
            data-driven correction term focus on compensating the rollout bias.
            Note: the weights must be frozen together with them — freezing only α/vx
            lets the AR gradient reshape weights to cancel the decay bias, degrading the
            physics kernel.
    """
    if freeze_physics:
        trainable = [p for n, p in model.named_parameters()
                     if ('corr_U' in n) or ('corr_V' in n)]
        for n, p in model.named_parameters():
            p.requires_grad = ('corr_U' in n) or ('corr_V' in n)
        n_frozen = sum(1 for n, _ in model.named_parameters() if not ('corr_U' in n or 'corr_V' in n))
        n_train = sum(p.numel() for p in trainable)
        print(f'[AR_FINETUNE] freeze_physics=True: trainable = {len(trainable)} correction terms'
              f' ({n_train/1e6:.1f}M), frozen physics kernel: {n_frozen} groups', flush=True)
        assert len(trainable) > 0, 'freeze_physics=True but no correction parameters (corr_U/V) are trainable'
        optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr * 0.01
    )

    mask, X, Y = data['mask'], data['X'], data['Y']
    T_all, t_array = data['T'], data['t']
    normalizer = data['normalizer']
    n_traj, nt = T_all.shape[0], len(t_array)
    grid = torch.stack([X, Y], dim=-1).unsqueeze(0)
    valid = mask > 0.5
    dt_array = t_array[1:] - t_array[:-1]

    # per-sample PDE parameters
    has_sample_alpha = 'alpha_array' in data and hasattr(model, 'pde_type')
    has_sample_vx = 'vx_array' in data
    has_sample_beta = 'beta_array' in data and hasattr(model, 'pde_type')
    if has_sample_alpha:
        alpha_arr = data['alpha_array']
        alpha_mean = alpha_arr.mean()
        alpha_norm = torch.tensor(alpha_arr / alpha_mean, dtype=torch.float32, device=device)
    if has_sample_vx:
        vx_arr = data['vx_array']
        vx_norm = torch.tensor(vx_arr, dtype=torch.float32, device=device)
    if has_sample_beta:
        beta_arr = data['beta_array']
        beta_mean = beta_arr.mean()
        beta_norm = torch.tensor(beta_arr / beta_mean, dtype=torch.float32, device=device)

    # Boundary weights
    weight = torch.ones_like(Y)
    hot_boundary = (Y < 0.15) & valid
    weight = torch.where(hot_boundary, weight * 3.0, weight)
    cold_boundary = (Y > 0.85) & valid
    weight = torch.where(cold_boundary, weight * 2.0, weight)
    boundary_weight = weight * mask

    sample_errors = np.ones(n_traj) * 10
    best_max, best_state = float('inf'), None

    def apply_bc(T, T_hot_norm):
        T_out = T.clone()
        hot = (Y < 0.1) & valid
        if hot.any():
            w = torch.clamp((0.1 - Y) / 0.08, 0, 1)
            T_out = torch.where(hot, T_out * (1 - w) + T_hot_norm * w, T_out)
        return T_out * mask

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        # Randomly select a few trajectories each epoch
        if epoch < 50:
            batch_idx = np.random.choice(n_traj, min(4, n_traj), replace=False)
        else:
            hard_idx = np.argsort(sample_errors)[-3:]
            random_idx = np.random.choice(n_traj, 2, replace=False)
            batch_idx = np.unique(np.concatenate([hard_idx, random_idx]))

        traj_losses = []
        batch_errors = {}

        for step_i, idx in enumerate(batch_idx):
            T_traj = T_all[idx]

            # Pure autoregressive rollout — no detach, gradients flow through all time steps
            T_current = T_traj[0]  # initial condition
            traj_loss = 0
            traj_max_err = 0

            for t_idx in range(nt - 1):
                T_next_true = T_traj[t_idx + 1]
                dt = dt_array[t_idx]

                # Extract the BC value for the current time step at each step (it varies with time under a ramp BC)
                hot_mask = (Y < 0.1) & valid
                T_hot_norm = T_next_true[hot_mask].mean() if hot_mask.any() else T_traj[0][hot_mask].mean()

                sa = alpha_norm[idx:idx+1] if has_sample_alpha else None
                sv = vx_norm[idx:idx+1] if has_sample_vx else None
                sb = beta_norm[idx:idx+1] if has_sample_beta else None
                T_next_pred = model(T_current.unsqueeze(0), dt.unsqueeze(0), grid,
                                    sample_alpha=sa, sample_vx=sv, sample_beta=sb).squeeze(0)
                T_next_pred = apply_bc(T_next_pred, T_hot_norm)

                # Loss: later time steps get larger weights (errors are larger late in the rollout)
                time_weight = 1.0 + 0.5 * t_idx / (nt - 2)

                diff = (T_next_pred - T_next_true)[valid]
                w = boundary_weight[valid]
                loss_mse = ((diff ** 2) * w).mean() * time_weight

                abs_diff = diff.abs()
                max_err = abs_diff.max()
                loss_max = max_err ** 2 * 0.8 * time_weight

                k = max(1, int(abs_diff.numel() * 0.01))
                top_k = torch.topk(abs_diff, k).values
                loss_topk = (top_k ** 2).mean() * 0.5 * time_weight

                traj_loss += loss_mse + loss_max + loss_topk
                traj_max_err = max(traj_max_err, (max_err * normalizer.scale).item())

                # Key: no detach, gradients flow through
                T_current = T_next_pred

            traj_losses.append(traj_loss / (nt - 1))
            batch_errors[idx] = traj_max_err

            # Gradient accumulation: update every grad_accum_steps trajectories or on the last one
            if (step_i + 1) % grad_accum_steps == 0 or step_i == len(batch_idx) - 1:
                avg_loss = torch.stack(traj_losses).mean()
                avg_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                optimizer.zero_grad()
                traj_losses = []

        for idx, err in batch_errors.items():
            sample_errors[idx] = 0.6 * sample_errors[idx] + 0.4 * err

        scheduler.step()

        if (epoch + 1) % 50 == 0 or epoch == 0:
            eval_target = eval_data if eval_data is not None else data
            mae, max_err = _eval_model(model, eval_target, grid)
            lr_now = optimizer.param_groups[0]['lr']
            loss_val = avg_loss.item() if isinstance(avg_loss, torch.Tensor) else 0.0
            print(f'[AR_FINETUNE] Epoch {epoch+1}/{n_epochs} | Loss: {loss_val:.6f} '
                  f'| MAE: {mae:.4f}K | Max: {max_err:.4f}K | LR: {lr_now:.2e}', flush=True)

            if max_err < best_max:
                best_max = max_err
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                print(f'  -> New best! Max: {best_max:.4f}K', flush=True)

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print(f'[AR_FINETUNE] Best Max Error: {best_max:.4f}K', flush=True)
    return model


@torch.no_grad()
def _eval_model(model, data, grid):
    """Evaluate the model - identical to eval_model in train_v10_clean.py"""
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
                hot = (Y < 0.1) & valid
                if hot.any():
                    w = torch.clamp((0.1 - Y) / 0.08, 0, 1)
                    T_next = torch.where(hot, T_next * (1 - w) + T_hot_norm * w, T_next)
                T_next = T_next * mask
                err = torch.abs(normalizer.denormalize(T_next) - normalizer.denormalize(T_true))[valid].max().item()
                traj_max = max(traj_max, err)
                T_current = T_next
            all_max.append(traj_max)

    model.train()
    return float(np.mean(all_max)), float(np.max(all_max))


# ============================================================
# Test
# ============================================================

def test_physics_structured():
    """Test the physics-structured FNO"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 60)
    print("Testing Physics-Structured FNO")
    print("=" * 60)

    # 1. Test the spectral convolution layer
    print("\n1. PhysicsStructuredSpectralConv2d")
    conv = PhysicsStructuredSpectralConv2d(32, 32, 16, 16).to(device)
    x = torch.randn(2, 32, 32, 32, device=device)
    y = conv(x, dt=0.1)
    print(f"   Input: {x.shape} -> Output: {y.shape}")

    ratio = conv.get_physics_ratio(0.1)
    print(f"   Physics ratio: {ratio['physics_ratio']:.4f}")
    print(f"   Correction ratio: {ratio['correction_ratio']:.4f}")
    print(f"   Learned alpha: {ratio['alpha']:.4f}")

    # 2. Test the full model
    print("\n2. PhysicsStructuredPINO")
    model = PhysicsStructuredPINO(modes=16, width=32, n_layers=4, device=device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {n_params:,}")

    T = torch.randn(2, 32, 32, device=device)
    dt = torch.tensor([0.1], device=device)
    grid = torch.stack(torch.meshgrid(
        torch.linspace(0, 1, 32, device=device),
        torch.linspace(0, 1, 32, device=device), indexing='ij'
    ), dim=-1).unsqueeze(0)
    T_next = model(T, dt, grid)
    print(f"   Input: {T.shape} -> Output: {T_next.shape}")

    # 3. Test the standard FNO baseline
    print("\n3. StandardAutoregressivePINO (baseline)")
    baseline = StandardAutoregressivePINO(modes=16, width=32, n_layers=4, device=device)
    n_params_base = sum(p.numel() for p in baseline.parameters())
    print(f"   Parameters: {n_params_base:,}")
    T_next_base = baseline(T, dt, grid)
    print(f"   Input: {T.shape} -> Output: {T_next_base.shape}")

    # 4. Parameter count comparison
    print(f"\nParameter count comparison:")
    print(f"   Standard FNO:           {n_params_base:,}")
    print(f"   Physics-Structured FNO: {n_params:,}")
    print(f"   Extra parameters: {n_params - n_params_base:,} ({(n_params-n_params_base)/n_params_base*100:.1f}%)")

    # 5. Physics contribution analysis
    print("\n5. Physics prior contribution analysis (dt=0.1):")
    analysis = model.analyze_physics_contribution(0.1)
    for item in analysis:
        print(f"   Layer {item['layer']}: physics={item['physics_ratio']:.3f}, "
              f"correction={item['correction_ratio']:.3f}, alpha={item['alpha']:.4f}")

    print("\nTests passed!")


if __name__ == '__main__':
    test_physics_structured()
