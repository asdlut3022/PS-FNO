# Convection-diffusion equation experiment: Standard FNO vs Physics FNO (conv_diff kernel)
# Goal: show that the physics kernel gives a larger improvement on a PDE with a more complex spectral structure

import torch
import os
import numpy as np
import sys
import time
import json

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))          # release root, for _release_cli

from irregular_domain import IrregularFinsDomain
from physics_fno import (
    PhysicsStructuredPINO,
    StandardAutoregressivePINO,
    train_scheduled_sampling,
    train_autoregressive_finetune,
)
from solve_conv_diff import generate_conv_diff_data
from _release_cli import apply_cli, stamp_preset

WIDTH = 96
MODES = 16
N_LAYERS = 5
NX, NY, NT = 32, 32, 10
N_TRAIN = 150
N_TEST = 20
N_EPOCHS_SS = 2000
N_EPOCHS_AR = 300
CKPT_DIR = os.path.join(_HERE, os.pardir, "checkpoints")
_DEFAULT_RESULTS = os.path.join(_HERE, os.pardir, "results", "conv_diff_results.json")

SEEDS, _RESULTS_FILE, CKPT_DIR, _CFG = apply_cli(
    [42], _DEFAULT_RESULTS, CKPT_DIR, device='cuda',
    description='Convection-diffusion: Standard FNO vs Physics FNO (SS + AR).')
SEED = SEEDS[0]
device = _CFG['device']
NX = _CFG.get('nx', NX)
NY = _CFG.get('ny', NY)
NT = _CFG.get('nt', NT)
N_TRAIN = _CFG.get('n_train', N_TRAIN)
N_TEST = _CFG.get('n_test', N_TEST)
N_EPOCHS_SS = _CFG.get('ss_epochs', N_EPOCHS_SS)
N_EPOCHS_AR = _CFG.get('ar_epochs', N_EPOCHS_AR)
torch.manual_seed(SEED)
np.random.seed(SEED)
if device.startswith('cuda'):
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.empty_cache()
os.makedirs(CKPT_DIR, exist_ok=True)

print('=' * 60, flush=True)
print('Experiment: Convection-Diffusion (Standard vs Physics FNO)', flush=True)
print(f'Config: width={WIDTH}, modes={MODES}, layers={N_LAYERS}', flush=True)
print(f'Data: {NX}x{NY}, {NT} steps, {N_TRAIN} train / {N_TEST} test', flush=True)
print(f'PDE: dT/dt = alpha*laplacian(T) - vx*dT/dx', flush=True)
print(f'vx ~ U(0.5, 2.0), Pe ~ 20-125', flush=True)
print('=' * 60, flush=True)

# Create domain and data
domain = IrregularFinsDomain(
    domain_size=(1.0, 1.0), fin_type='wavy', n_fins=3, base_height=0.1
)

print('\nGenerating conv-diff data...', flush=True)
train_data = generate_conv_diff_data(domain, nx=NX, ny=NY, nt=NT, n_traj=N_TRAIN,
                                      device=device, seed=SEED, ramp_fraction=0.5)
test_data = generate_conv_diff_data(domain, nx=NX, ny=NY, nt=NT, n_traj=N_TEST,
                                     device=device, seed=SEED + 1000, ramp_fraction=0.5)


def eval_full_model(model, data, name=''):
    """Full evaluation (autoregressive rollout over all time steps)"""
    model.eval()
    mask, X, Y = data['mask'], data['X'], data['Y']
    t_array, normalizer = data['t'], data['normalizer']
    valid_np = mask.cpu().numpy() > 0.5
    grid = torch.stack([X, Y], dim=-1).unsqueeze(0)
    nt = len(t_array)
    dt_array = t_array[1:] - t_array[:-1]

    # per-sample parameters (Physics model only)
    has_sample_params = ('vx_array' in data and 'alpha_array' in data
                         and hasattr(model, 'pde_type'))
    if has_sample_params:
        alpha_arr = data['alpha_array']
        vx_arr = data['vx_array']
        alpha_mean = alpha_arr.mean()

    all_mae, all_max = [], []

    for traj_idx in range(data['T'].shape[0]):
        T_traj = data['T'][traj_idx]
        T_current = T_traj[0].clone()

        # per-sample parameters
        if has_sample_params:
            sa = torch.tensor([alpha_arr[traj_idx] / alpha_mean], dtype=torch.float32, device=T_current.device)
            sv = torch.tensor([vx_arr[traj_idx]], dtype=torch.float32, device=T_current.device)
        else:
            sa, sv = None, None

        preds = [T_current.cpu().numpy()]
        with torch.no_grad():
            for t_idx in range(nt - 1):
                dt = dt_array[t_idx]
                T_next = model(T_current.unsqueeze(0), dt.unsqueeze(0), grid,
                               sample_alpha=sa, sample_vx=sv).squeeze(0)
                T_next = T_next * mask
                preds.append(T_next.cpu().numpy())
                T_current = T_next

        preds_K = normalizer.denormalize(np.array(preds))
        trues_K = normalizer.denormalize(T_traj.cpu().numpy())
        errors = np.abs(preds_K - trues_K)[:, valid_np]

        all_mae.append(errors.mean())
        all_max.append(errors.max())

    mean_mae = np.mean(all_mae)
    mean_max = np.mean(all_max)
    worst_max = np.max(all_max)
    p95_max = np.percentile(all_max, 95)
    print(f'\n{name}:', flush=True)
    print(f'  MAE:       {mean_mae:.6f} K', flush=True)
    print(f'  Avg Max:   {mean_max:.6f} K', flush=True)
    print(f'  Worst Max: {worst_max:.6f} K', flush=True)
    print(f'  P95 Max:   {p95_max:.6f} K', flush=True)
    return {'mae': float(mean_mae), 'avg_max': float(mean_max),
            'worst_max': float(worst_max), 'p95_max': float(p95_max)}


results = {}
stamp_preset(results, _CFG)
save_path = _RESULTS_FILE

def save_results():
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'  [Intermediate results saved to {save_path}]', flush=True)

# ===== Standard FNO + SS =====
print('\n' + '=' * 50, flush=True)
print('Standard FNO + SS on convection-diffusion data', flush=True)
print('=' * 50, flush=True)

torch.manual_seed(SEED)
model_std = StandardAutoregressivePINO(modes=MODES, width=WIDTH, n_layers=N_LAYERS, device=device)
n_params_std = sum(p.numel() for p in model_std.parameters())
print(f'Parameters: {n_params_std:,}', flush=True)

t0 = time.time()
model_std = train_scheduled_sampling(
    model_std, train_data,
    n_epochs=N_EPOCHS_SS, device=device,
    p_start=0.9, p_end=0.3, warmup=500, anneal_end=1500
)
t_std = time.time() - t0
print(f'Training time: {t_std:.1f}s', flush=True)

test_std = eval_full_model(model_std, test_data, 'Standard FNO + SS [Test]')
results['standard_ss'] = {
    'n_params': n_params_std, 'train_time': t_std, 'test': test_std
}
save_results()

# Free GPU memory
del model_std
if device.startswith('cuda'):
    torch.cuda.empty_cache()

# ===== Physics FNO + SS (convection-diffusion kernel) =====
print('\n' + '=' * 50, flush=True)
print('Physics FNO + SS (convection-diffusion kernel) on convection-diffusion data', flush=True)
print('=' * 50, flush=True)

torch.manual_seed(SEED)
model_phys = PhysicsStructuredPINO(
    modes=MODES, width=WIDTH, n_layers=N_LAYERS, device=device,
    corr_rank=8, pde_type='conv_diff'
)
n_params_phys = sum(p.numel() for p in model_phys.parameters())
print(f'Parameters: {n_params_phys:,}', flush=True)

t0 = time.time()
model_phys = train_scheduled_sampling(
    model_phys, train_data,
    n_epochs=N_EPOCHS_SS, device=device,
    p_start=0.9, p_end=0.3, warmup=500, anneal_end=1500
)
t_phys = time.time() - t0
print(f'Training time: {t_phys:.1f}s', flush=True)

test_phys = eval_full_model(model_phys, test_data, 'Physics FNO + SS [Test]')

# Physics contribution analysis
dt_avg = float((train_data['t'][1:] - train_data['t'][:-1]).mean())
physics_contrib = model_phys.analyze_physics_contribution(dt_avg)
print('\nPhysics contribution analysis:', flush=True)
for layer_info in physics_contrib:
    vx_str = f", vx={layer_info.get('vx', 'N/A'):.3f}" if 'vx' in layer_info else ""
    print(f"  Layer {layer_info['layer']}: phys={layer_info['physics_ratio']:.4f}, "
          f"corr={layer_info['correction_ratio']:.4f}, alpha={layer_info['alpha']:.4f}{vx_str}",
          flush=True)

results['physics_ss'] = {
    'n_params': n_params_phys, 'train_time': t_phys, 'test': test_phys,
    'physics_contribution': physics_contrib
}
save_results()

# Save the SS model for AR finetuning
torch.save(model_phys.state_dict(), os.path.join(CKPT_DIR, "conv_diff_phys_ss.pt"))

# ===== Physics FNO + SS + AR =====
print('\n' + '=' * 50, flush=True)
print('Physics FNO + SS + AR (convection-diffusion kernel)', flush=True)
print('=' * 50, flush=True)

t0 = time.time()
model_phys = train_autoregressive_finetune(
    model_phys, train_data,
    n_epochs=N_EPOCHS_AR, device=device,
    lr=1e-4, grad_accum_steps=4
)
t_ar = time.time() - t0
print(f'AR finetuning time: {t_ar:.1f}s', flush=True)

test_ar = eval_full_model(model_phys, test_data, 'Physics FNO + SS + AR [Test]')
results['physics_ss_ar'] = {
    'n_params': n_params_phys, 'train_time': t_phys + t_ar, 'test': test_ar
}

# Compute the improvement
std_wm = test_std['worst_max']
phys_wm = test_phys['worst_max']
ar_wm = test_ar['worst_max']
print('\n' + '=' * 60, flush=True)
print('SUMMARY: Conv-Diff Experiment', flush=True)
print(f'Standard FNO+SS WorstMax: {std_wm:.3f} K', flush=True)
print(f'Physics FNO+SS  WorstMax: {phys_wm:.3f} K  ({(1-phys_wm/std_wm)*100:+.1f}%)', flush=True)
print(f'Physics FNO+SS+AR WMax:   {ar_wm:.3f} K  ({(1-ar_wm/std_wm)*100:+.1f}%)', flush=True)
print('=' * 60, flush=True)

results['summary'] = {
    'physics_kernel_improvement_pct': float((1 - phys_wm / std_wm) * 100),
    'full_pipeline_improvement_pct': float((1 - ar_wm / std_wm) * 100),
}

# Save results
save_path = _RESULTS_FILE
with open(save_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f'\nResults saved to {save_path}', flush=True)
