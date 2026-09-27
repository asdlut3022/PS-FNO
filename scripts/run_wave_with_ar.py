# Viscoelastic wave equation experiment: Standard vs Physics FNO (SS + AR finetuning)
# PDE: u_tt + gamma*u_t = c^2*laplacian(u) + eta*laplacian(u_t)
# Kelvin-Voigt viscoelastic model: frequency-dependent attenuation + oscillation
#
# Key improvement: the AR stage does not detach; BPTT spans every time step

import torch
import numpy as np
import sys
import time
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # release root

from irregular_domain import LShapedDomain
from physics_fno import PhysicsStructuredPINO, StandardAutoregressivePINO
from solve_wave import generate_wave_data, WaveNormalizer
from run_wave_experiment import train_wave_scheduled_sampling, eval_wave_full, eval_wave_model
from _release_cli import apply_cli, stamp_preset

# ============================================================
# Configuration — viscoelastic wave equation parameters (Kelvin-Voigt damping)
# ============================================================
SEEDS = [789]

WIDTH = 96
MODES = 16
N_LAYERS = 5
NX, NY = 32, 32
NT = 5              # 3 prediction steps (reduces AR accumulated phase error)
N_TRAIN = 250
N_TEST = 20
N_EPOCHS_SS = 2000
N_EPOCHS_AR = 800

# Viscoelastic wave equation physical parameters
T_TOTAL = 0.2           # prediction time horizon (shortened to reduce AR phase error accumulation)
C_RANGE = (0.9, 1.2)    # wave speed range
GAMMA_RANGE = (0.2, 0.5) # structural damping (can be lighter than for the pure wave equation since eta provides extra damping)
ETA = 0.08               # Kelvin-Voigt viscoelastic coefficient (moderately viscoelastic material: rubber/polymer/biological tissue)
AMP_RANGE = (0.5, 1.5)
SIGMA_RANGE = (0.08, 0.15)

DOMAIN = LShapedDomain(width=1.0, height=1.0, cut_size=0.5)

_DEFAULT_RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                                "results", "results_wave", "wave_ar_results.json")
_DEFAULT_CKPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
SEEDS, _RESULTS_FILE, _CKPT_DIR, _CFG = apply_cli(
    SEEDS, _DEFAULT_RESULTS, _DEFAULT_CKPT, device='cuda',
    description='Viscoelastic damped wave: Standard vs Physics FNO (SS + AR).')
device = _CFG['device']
NX = _CFG.get('nx', NX)
NY = _CFG.get('ny', NY)
NT = _CFG.get('nt', NT)
N_TRAIN = _CFG.get('n_train', N_TRAIN)
N_TEST = _CFG.get('n_test', N_TEST)
N_EPOCHS_SS = _CFG.get('ss_epochs', N_EPOCHS_SS)
N_EPOCHS_AR = _CFG.get('ar_epochs', N_EPOCHS_AR)

os.makedirs(_CKPT_DIR, exist_ok=True)
OUTPUT_DIR = os.path.dirname(_RESULTS_FILE)
# Anchored to the output root, not to the process working directory: a bare
# 'figures' would write into whatever directory the run was launched from, so a
# run aimed off the package with --output-dir would still drop figures inside it.
FIGURE_DIR = os.path.join(OUTPUT_DIR, 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)


# ============================================================
# Wave AR Finetuning (adapted from thermal version)
# ============================================================
def train_wave_ar_finetune(model, data, n_epochs=500, device='cuda',
                            lr=1e-4, grad_accum_steps=4, is_physics=False, feed_gamma=False):
    """Wave equation AR finetuning: pure autoregressive rollout, gradients span every time step.

    Key differences from SS training:
    1. p_teacher=0 (100% self-prediction)
    2. No detach: BPTT propagates from the last step back to the first
    3. Directly optimizes the accumulated error at inference time
    4. Time weighting: later time steps get a larger weight

    Wave-equation specifics:
    - Two-step state: (u_prev, u_curr)
    - The first two time steps are the initial condition; prediction starts at t_idx=1

    feed_gamma: additionally feed the per-sample gamma (False = default behavior unchanged)
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr * 0.01
    )

    mask = data['mask']
    U_all = data['U']         # (n_traj, nt, nx, ny)
    t_array = data['t']
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

    # Per-sample structural damping
    g_norm = None
    if feed_gamma:
        g_arr = data['gamma_array']
        g_mean = g_arr.mean()
        g_norm = torch.tensor(g_arr / g_mean, dtype=torch.float32, device=device)

    sample_errors = np.ones(n_traj) * 10.0
    best_max, best_state = float('inf'), None
    n_pred_steps = nt - 2  # NT=5 → 3 prediction steps

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        # Sampling: random for the first 50 epochs, then hard example mining
        if epoch < 50:
            batch_idx = np.random.choice(n_traj, min(4, n_traj), replace=False)
        else:
            hard_idx = np.argsort(sample_errors)[-3:]
            random_idx = np.random.choice(n_traj, 2, replace=False)
            batch_idx = np.unique(np.concatenate([hard_idx, random_idx]))

        traj_losses = []
        batch_errors = {}

        for step_i, idx in enumerate(batch_idx):
            U_traj = U_all[idx]

            # Wave equation: the first two steps are the initial condition
            u_prev = U_traj[0]    # u(t_0) — start of the computation graph
            u_curr = U_traj[1]    # u(t_1) — start of the computation graph
            traj_loss = 0
            traj_max_err = 0

            for t_idx in range(1, nt - 1):
                u_next_true = U_traj[t_idx + 1]
                dt = dt_array[t_idx]

                # Time weighting: later steps get a larger weight
                time_weight = 1.0 + 0.5 * (t_idx - 1) / max(1, n_pred_steps - 1)

                # Forward prediction
                sc = c_norm[idx:idx+1] if is_physics else None
                sgamma = g_norm[idx:idx+1] if (is_physics and feed_gamma) else None
                u_next_pred = model(
                    u_curr.unsqueeze(0), dt.unsqueeze(0), grid,
                    u_prev=u_prev.unsqueeze(0), sample_c=sc, sample_gamma_w=sgamma
                ).squeeze(0)
                u_next_pred = u_next_pred * mask

                # Loss computation
                diff = (u_next_pred - u_next_true)[valid]
                loss_mse = (diff ** 2).mean() * time_weight

                abs_diff = diff.abs()
                max_err = abs_diff.max()
                loss_max = max_err ** 2 * 0.8 * time_weight

                k = max(1, int(abs_diff.numel() * 0.01))
                top_k = torch.topk(abs_diff, k).values
                loss_topk = (top_k ** 2).mean() * 0.5 * time_weight

                traj_loss += loss_mse + loss_max + loss_topk
                traj_max_err = max(traj_max_err, (max_err * normalizer.scale).item())

                # Key point: no detach; BPTT spans every time step
                u_prev = u_curr
                u_curr = u_next_pred

            traj_losses.append(traj_loss / n_pred_steps)
            batch_errors[idx] = traj_max_err

            # Gradient accumulation
            if (step_i + 1) % grad_accum_steps == 0 or step_i == len(batch_idx) - 1:
                if traj_losses:
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
            avgmax, max_err_eval = eval_wave_model(model, data, grid, is_physics=is_physics, feed_gamma=feed_gamma)
            lr_now = optimizer.param_groups[0]['lr']
            print(f'[WAVE_AR] Epoch {epoch+1}/{n_epochs} | '
                  f'AvgMax: {avgmax:.6f} | Max: {max_err_eval:.6f} | LR: {lr_now:.2e}', flush=True)

            if max_err_eval < best_max:
                best_max = max_err_eval
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                print(f'  -> New best! Max: {best_max:.6f}', flush=True)

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print(f'[WAVE_AR] Best Max Error: {best_max:.6f}', flush=True)
    return model


# ============================================================
# Figure generation (same layout as before but with AR models)
# ============================================================
def generate_wave_figure(model_std, model_phys, test_data, output_dir):
    """Generate 5-row x 2-col comparison figure (mid + final time steps).

    The plotted trajectory is the **worst-case** test trajectory: the test
    trajectory with the largest per-trajectory maximum error across the two
    models (t >= 2, masked domain). This is the conservative convention used
    for the manuscript's field figure, and it is deliberately *not* selected
    for best appearance: the figure backs no reported metric and is not one of
    the archived result records.
    """
    mask = test_data['mask']
    U_all = test_data['U']
    t_array = test_data['t']
    normalizer = test_data['normalizer']
    valid_np = mask.cpu().numpy() > 0.5
    mask_np = mask.cpu().numpy()
    dt_array = t_array[1:] - t_array[:-1]
    nt = U_all.shape[1]
    X, Y = test_data['X'], test_data['Y']
    grid = torch.stack([X, Y], dim=-1).unsqueeze(0)

    c_arr = test_data['c_array']
    c_mean = c_arr.mean()
    c_norm = torch.tensor(c_arr / c_mean, dtype=torch.float32, device=test_data['X'].device)

    # Select the worst-case test trajectory (largest per-trajectory error
    # across the two models, t >= 2; not selected for best appearance)
    std_worst, phys_worst = [], []
    with torch.no_grad():
        model_std.eval()
        model_phys.eval()
        for traj_idx in range(U_all.shape[0]):
            U_traj = U_all[traj_idx]

            u_prev_s, u_curr_s = U_traj[0].clone(), U_traj[1].clone()
            u_prev_p, u_curr_p = U_traj[0].clone(), U_traj[1].clone()
            std_preds = [U_traj[0].cpu().numpy(), U_traj[1].cpu().numpy()]
            phys_preds = [U_traj[0].cpu().numpy(), U_traj[1].cpu().numpy()]

            for t_idx in range(1, nt - 1):
                dt = dt_array[t_idx]
                u_next_s = model_std(
                    u_curr_s.unsqueeze(0), dt.unsqueeze(0), grid,
                    u_prev=u_prev_s.unsqueeze(0), sample_c=None
                ).squeeze(0) * mask
                std_preds.append(u_next_s.cpu().numpy())
                u_prev_s, u_curr_s = u_curr_s, u_next_s

                sc = c_norm[traj_idx:traj_idx+1]
                u_next_p = model_phys(
                    u_curr_p.unsqueeze(0), dt.unsqueeze(0), grid,
                    u_prev=u_prev_p.unsqueeze(0), sample_c=sc
                ).squeeze(0) * mask
                phys_preds.append(u_next_p.cpu().numpy())
                u_prev_p, u_curr_p = u_curr_p, u_next_p

            std_preds_phys = np.array(std_preds) * normalizer.scale
            phys_preds_phys = np.array(phys_preds) * normalizer.scale
            trues_phys = U_traj.cpu().numpy() * normalizer.scale

            std_err = np.abs(std_preds_phys[2:] - trues_phys[2:])[:, valid_np].max()
            phys_err = np.abs(phys_preds_phys[2:] - trues_phys[2:])[:, valid_np].max()
            std_worst.append(std_err)
            phys_worst.append(phys_err)

    worst_idx = int(np.argmax(np.maximum(np.array(std_worst), np.array(phys_worst))))
    print(f'Selected worst-case trajectory {worst_idx}: '
          f'Std WorstMax={std_worst[worst_idx]:.4f}, '
          f'Phys WorstMax={phys_worst[worst_idx]:.4f}', flush=True)

    # Re-generate predictions for the selected worst-case trajectory
    U_traj = U_all[worst_idx]
    u_prev_s, u_curr_s = U_traj[0].clone(), U_traj[1].clone()
    u_prev_p, u_curr_p = U_traj[0].clone(), U_traj[1].clone()
    std_preds = [U_traj[0].cpu().numpy(), U_traj[1].cpu().numpy()]
    phys_preds = [U_traj[0].cpu().numpy(), U_traj[1].cpu().numpy()]

    with torch.no_grad():
        for t_idx in range(1, nt - 1):
            dt = dt_array[t_idx]
            u_next_s = model_std(
                u_curr_s.unsqueeze(0), dt.unsqueeze(0), grid,
                u_prev=u_prev_s.unsqueeze(0), sample_c=None
            ).squeeze(0) * mask
            std_preds.append(u_next_s.cpu().numpy())
            u_prev_s, u_curr_s = u_curr_s, u_next_s

            sc = c_norm[worst_idx:worst_idx+1]
            u_next_p = model_phys(
                u_curr_p.unsqueeze(0), dt.unsqueeze(0), grid,
                u_prev=u_prev_p.unsqueeze(0), sample_c=sc
            ).squeeze(0) * mask
            phys_preds.append(u_next_p.cpu().numpy())
            u_prev_p, u_curr_p = u_curr_p, u_next_p

    std_preds = np.array(std_preds) * normalizer.scale
    phys_preds = np.array(phys_preds) * normalizer.scale
    trues = U_traj.cpu().numpy() * normalizer.scale

    # Select 2 time steps: middle and final (skip early step where improvement is small)
    pred_start = 2  # first predicted frame
    pred_end = nt - 1  # last predicted frame
    if pred_end - pred_start >= 2:
        time_indices = [(pred_start + pred_end) // 2, pred_end]
    elif pred_end - pred_start >= 1:
        time_indices = [pred_end]
    else:
        time_indices = [pred_start]
    time_labels = [f't={t_array[ti].item():.3f}s' for ti in time_indices]
    n_cols = len(time_indices)

    X_np, Y_np = X.cpu().numpy(), Y.cpu().numpy()

    def apply_mask(field, m):
        result = field.copy()
        result[~(m > 0.5)] = np.nan
        return result

    # Figure: 5 rows x 2 columns (jet colormap style)
    fig, axes = plt.subplots(5, n_cols, figsize=(8, 15))
    plt.subplots_adjust(hspace=0.15, wspace=0.10, left=0.10, right=0.88, top=0.96, bottom=0.02)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    all_vals = [trues[ti][valid_np] for ti in time_indices]
    vmin_field = np.min(all_vals)
    vmax_field = np.max(all_vals)

    all_std_err = [np.abs(std_preds[ti] - trues[ti])[valid_np] for ti in time_indices]
    all_phys_err = [np.abs(phys_preds[ti] - trues[ti])[valid_np] for ti in time_indices]
    err_max = max(np.max(np.concatenate(all_std_err)), np.max(np.concatenate(all_phys_err)))
    err_max = max(err_max, 0.01)

    row_labels = ['Ground Truth', 'Standard FNO', '|Std Error|', 'Physics FNO', '|Phys Error|']

    for col_idx, (ti, tl) in enumerate(zip(time_indices, time_labels)):
        gt = apply_mask(trues[ti], mask_np)
        im_field = axes[0, col_idx].pcolormesh(X_np, Y_np, gt.T, cmap='jet',
                                                 vmin=vmin_field, vmax=vmax_field, shading='auto')
        axes[0, col_idx].set_title(tl, fontsize=13, fontweight='bold')

        std_f = apply_mask(std_preds[ti], mask_np)
        axes[1, col_idx].pcolormesh(X_np, Y_np, std_f.T, cmap='jet',
                                      vmin=vmin_field, vmax=vmax_field, shading='auto')

        std_err = apply_mask(np.abs(std_preds[ti] - trues[ti]), mask_np)
        im_err = axes[2, col_idx].pcolormesh(X_np, Y_np, std_err.T, cmap='jet',
                                               vmin=0, vmax=err_max, shading='auto')
        axes[2, col_idx].text(0.97, 0.95, f'max={np.nanmax(std_err):.3f}',
                               transform=axes[2, col_idx].transAxes, fontsize=8,
                               ha='right', va='top', fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))

        phys_f = apply_mask(phys_preds[ti], mask_np)
        axes[3, col_idx].pcolormesh(X_np, Y_np, phys_f.T, cmap='jet',
                                      vmin=vmin_field, vmax=vmax_field, shading='auto')

        phys_err = apply_mask(np.abs(phys_preds[ti] - trues[ti]), mask_np)
        axes[4, col_idx].pcolormesh(X_np, Y_np, phys_err.T, cmap='jet',
                                      vmin=0, vmax=err_max, shading='auto')
        axes[4, col_idx].text(0.97, 0.95, f'max={np.nanmax(phys_err):.3f}',
                               transform=axes[4, col_idx].transAxes, fontsize=8,
                               ha='right', va='top', fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))

        for row in range(5):
            axes[row, col_idx].set_aspect('equal')
            axes[row, col_idx].set_xticks([])
            axes[row, col_idx].set_yticks([])

    for row_idx, label in enumerate(row_labels):
        axes[row_idx, 0].set_ylabel(label, fontsize=11, fontweight='bold', labelpad=10)

    cbar_ax1 = fig.add_axes([0.90, 0.42, 0.018, 0.52])
    cb1 = fig.colorbar(im_field, cax=cbar_ax1)
    cb1.set_label('Displacement', fontsize=10)

    cbar_ax2 = fig.add_axes([0.90, 0.02, 0.018, 0.28])
    cb2 = fig.colorbar(im_err, cax=cbar_ax2)
    cb2.set_label('|Error|', fontsize=10)

    out_pdf = os.path.join(output_dir, 'fig_wave_field_std_vs_phys.pdf')
    out_png = os.path.join(output_dir, 'fig_wave_field_std_vs_phys.png')
    fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Figure saved to: {out_pdf}', flush=True)


# ============================================================
# Main experiment
# ============================================================
if __name__ == '__main__':
    print('=' * 70, flush=True)
    print('Wave Equation: SS + AR Finetuning Experiment', flush=True)
    print(f'SS: {N_EPOCHS_SS} epochs, AR: {N_EPOCHS_AR} epochs', flush=True)
    print(f'Seeds: {SEEDS}', flush=True)
    print('=' * 70, flush=True)

    all_results = {
        'seeds': SEEDS,
        'standard_ss': {},
        'standard_ss_ar': {},
        'physics_ss': {},
        'physics_ss_ar': {},
    }
    stamp_preset(all_results, _CFG)

    # Store seed 42 models for figure generation
    best_model_std = None
    best_model_phys = None
    best_test_data = None

    for seed_i, seed in enumerate(SEEDS):
        print(f'\n{"="*70}', flush=True)
        print(f'SEED {seed} ({seed_i+1}/{len(SEEDS)})', flush=True)
        print(f'{"="*70}', flush=True)

        torch.manual_seed(seed)
        np.random.seed(seed)
        if device.startswith('cuda'):
            torch.cuda.manual_seed_all(seed)

        # Generate data
        print('Generating data...', flush=True)
        train_data = generate_wave_data(DOMAIN, nx=NX, ny=NY, nt=NT, n_traj=N_TRAIN,
                                         device=device, seed=seed,
                                         t_total=T_TOTAL, c_range=C_RANGE,
                                         gamma_range=GAMMA_RANGE, amp_range=AMP_RANGE,
                                         sigma_range=SIGMA_RANGE,
                                         cut_frac_x=0.5, cut_frac_y=0.5,
                                         eta=ETA)
        test_data = generate_wave_data(DOMAIN, nx=NX, ny=NY, nt=NT, n_traj=N_TEST,
                                        device=device, seed=seed + 1000,
                                        t_total=T_TOTAL, c_range=C_RANGE,
                                        gamma_range=GAMMA_RANGE, amp_range=AMP_RANGE,
                                        sigma_range=SIGMA_RANGE,
                                        cut_frac_x=0.5, cut_frac_y=0.5,
                                        eta=ETA)

        # --- Standard FNO: SS ---
        print(f'\n--- Standard FNO SS (seed {seed}) ---', flush=True)
        torch.manual_seed(seed + 10000)
        if device.startswith('cuda'):
            torch.cuda.manual_seed_all(seed + 10000)
        model_std = StandardAutoregressivePINO(
            modes=MODES, width=WIDTH, n_layers=N_LAYERS, device=device, wave_mode=True
        )
        n_params_std = sum(p.numel() for p in model_std.parameters())

        t0 = time.time()
        model_std = train_wave_scheduled_sampling(
            model_std, train_data, n_epochs=N_EPOCHS_SS, device=device, is_physics=False
        )
        t_std_ss = time.time() - t0

        res_std_ss = eval_wave_full(model_std, test_data, f'Standard SS (seed {seed})', is_physics=False)
        res_std_ss['time'] = t_std_ss
        res_std_ss['params'] = n_params_std
        all_results['standard_ss'][seed] = res_std_ss

        # --- Standard FNO: AR finetune ---
        print(f'\n--- Standard FNO AR finetune (seed {seed}) ---', flush=True)
        t0 = time.time()
        model_std = train_wave_ar_finetune(
            model_std, train_data, n_epochs=N_EPOCHS_AR, device=device, is_physics=False
        )
        t_std_ar = time.time() - t0

        res_std_ar = eval_wave_full(model_std, test_data, f'Standard SS+AR (seed {seed})', is_physics=False)
        res_std_ar['time'] = t_std_ss + t_std_ar
        res_std_ar['params'] = n_params_std
        all_results['standard_ss_ar'][seed] = res_std_ar

        if seed == SEEDS[0]:
            best_model_std = model_std
            best_test_data = test_data

        del model_std
        if device.startswith('cuda'):
            torch.cuda.empty_cache()

        # --- Physics FNO: SS ---
        print(f'\n--- Physics FNO SS (seed {seed}) ---', flush=True)
        torch.manual_seed(seed + 10000)
        if device.startswith('cuda'):
            torch.cuda.manual_seed_all(seed + 10000)
        model_phys = PhysicsStructuredPINO(
            modes=MODES, width=WIDTH, n_layers=N_LAYERS,
            device=device, corr_rank=8, pde_type='wave'
        )
        n_params_phys = sum(p.numel() for p in model_phys.parameters())

        t0 = time.time()
        model_phys = train_wave_scheduled_sampling(
            model_phys, train_data, n_epochs=N_EPOCHS_SS, device=device, is_physics=True
        )
        t_phys_ss = time.time() - t0

        res_phys_ss = eval_wave_full(model_phys, test_data, f'Physics SS (seed {seed})', is_physics=True)
        res_phys_ss['time'] = t_phys_ss
        res_phys_ss['params'] = n_params_phys
        all_results['physics_ss'][seed] = res_phys_ss

        # --- Physics FNO: AR finetune ---
        print(f'\n--- Physics FNO AR finetune (seed {seed}) ---', flush=True)
        t0 = time.time()
        model_phys = train_wave_ar_finetune(
            model_phys, train_data, n_epochs=N_EPOCHS_AR, device=device, is_physics=True
        )
        t_phys_ar = time.time() - t0

        res_phys_ar = eval_wave_full(model_phys, test_data, f'Physics SS+AR (seed {seed})', is_physics=True)
        res_phys_ar['time'] = t_phys_ss + t_phys_ar
        res_phys_ar['params'] = n_params_phys

        # Physics ratio
        dt_avg = float((test_data['t'][2] - test_data['t'][1]).item())
        phys_analysis = model_phys.analyze_physics_contribution(dt_avg)
        avg_phys_ratio = np.mean([r['physics_ratio'] for r in phys_analysis])
        res_phys_ar['physics_ratio'] = float(avg_phys_ratio)
        res_phys_ss['physics_ratio'] = float(avg_phys_ratio)
        all_results['physics_ss'][seed]['physics_ratio'] = float(avg_phys_ratio)
        all_results['physics_ss_ar'][seed] = res_phys_ar

        if seed == SEEDS[0]:
            best_model_phys = model_phys

        del model_phys
        if device.startswith('cuda'):
            torch.cuda.empty_cache()

        # Per-seed summary
        print(f'\n--- Seed {seed} Summary ---', flush=True)
        for method_key, method_name in [('standard_ss', 'Std SS'), ('standard_ss_ar', 'Std SS+AR'),
                                         ('physics_ss', 'Phys SS'), ('physics_ss_ar', 'Phys SS+AR')]:
            r = all_results[method_key][seed]
            print(f'  {method_name}: MAE={r["mae"]:.4f}, AvgMax={r["avg_max"]:.4f}, WorstMax={r["worst_max"]:.4f}', flush=True)

        # Save intermediate
        partial_path = os.path.join(OUTPUT_DIR, 'wave_ar_results_partial.json')
        with open(partial_path, 'w') as f:
            json.dump(all_results, f, indent=2)

    # ============================================================
    # Summary statistics
    # ============================================================
    print('\n' + '=' * 70, flush=True)
    print('WAVE SS+AR RESULTS SUMMARY', flush=True)
    print('=' * 70, flush=True)

    summary = {}
    for method in ['standard_ss', 'standard_ss_ar', 'physics_ss', 'physics_ss_ar']:
        results = all_results[method]
        for k in ['mae', 'avg_max', 'worst_max']:
            vals = [results[seed][k] for seed in SEEDS]
            summary[f'{method}_{k}_mean'] = float(np.mean(vals))
            summary[f'{method}_{k}_std'] = float(np.std(vals))

    print('\nMethod comparison (three-seed average):', flush=True)
    for method, label in [('standard_ss', 'Std SS'), ('standard_ss_ar', 'Std SS+AR'),
                           ('physics_ss', 'Phys SS'), ('physics_ss_ar', 'Phys SS+AR')]:
        m = summary[f'{method}_mae_mean']
        a = summary[f'{method}_avg_max_mean']
        w = summary[f'{method}_worst_max_mean']
        print(f'  {label:12s}: MAE={m:.4f}, AvgMax={a:.4f}, WorstMax={w:.4f}', flush=True)

    # Key improvements
    for k_label, k in [('MAE', 'mae'), ('AvgMax', 'avg_max'), ('WorstMax', 'worst_max')]:
        s_ss = summary[f'standard_ss_{k}_mean']
        s_ar = summary[f'standard_ss_ar_{k}_mean']
        p_ss = summary[f'physics_ss_{k}_mean']
        p_ar = summary[f'physics_ss_ar_{k}_mean']
        # SS vs SS improvement
        imp_ss = (1 - p_ss / s_ss) * 100
        # SS+AR vs SS+AR improvement
        imp_ar = (1 - p_ar / s_ar) * 100
        # AR gain for Physics
        ar_gain = (1 - p_ar / p_ss) * 100
        print(f'\n  {k_label}:', flush=True)
        print(f'    Physics SS improvement over Std SS:     {imp_ss:+.1f}%', flush=True)
        print(f'    Physics SS+AR improvement over Std SS+AR: {imp_ar:+.1f}%', flush=True)
        print(f'    AR finetune gain for Physics:          {ar_gain:+.1f}%', flush=True)

    summary['improvement_ss_vs_ss'] = {
        k: float((1 - summary[f'physics_ss_{k}_mean'] / summary[f'standard_ss_{k}_mean']) * 100)
        for k in ['mae', 'avg_max', 'worst_max']
    }
    summary['improvement_ar_vs_ar'] = {
        k: float((1 - summary[f'physics_ss_ar_{k}_mean'] / summary[f'standard_ss_ar_{k}_mean']) * 100)
        for k in ['mae', 'avg_max', 'worst_max']
    }

    all_results['summary'] = summary

    # Save final results
    final_path = os.path.join(OUTPUT_DIR, 'wave_ar_results.json')
    with open(final_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nResults saved to: {final_path}', flush=True)

    # ============================================================
    # Generate figure (seed 42, SS+AR models)
    # ============================================================
    if best_model_std is not None and best_model_phys is not None:
        print('\n' + '=' * 70, flush=True)
        print('Generating comparison figure (seed 789, SS+AR)...', flush=True)
        generate_wave_figure(best_model_std, best_model_phys, best_test_data, FIGURE_DIR)

    print('\n' + '=' * 70, flush=True)
    print('DONE.', flush=True)
    print('=' * 70, flush=True)
