"""
FIX for data leakage (Neural Networks major revision, structural risk P0-1):
Reaction-Diffusion AR finetuning previously selected the "best" checkpoint by
evaluating on the TEST set during training (see run_reaction_diff_ar.py L202-217).
This script fixes it by carving out an INDEPENDENT validation set
(seed+2000, 20 trajectories, disjoint from both train seed and test seed+1000)
and doing early stopping / checkpoint selection on validation worst-case error.
The test set is now touched exactly once, at the very end, for final reporting.

Reuses the existing SS-stage checkpoints (phys_reactdiff_ss_seed*.pth,
std_reactdiff_ss_seed*.pth) since SS-stage model selection was already done on
the training set (not the test set) -- SS stage has no leakage, only Stage-3 AR
finetuning did. This saves ~35 min/seed of SS retraining (2000 epochs). Those
checkpoints are not hosted in this repository (they are larger than the release
allows), so if a checkpoint is absent the SS stage is trained fresh here with
the standard 2D protocol via resolve_ss_checkpoint(); the resulting AR numbers
are then a fresh realization rather than the archived per-seed values.

Output: reaction_diff_ar_valfix_results.json
"""
import torch
import numpy as np
import sys
import json
import time
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))          # release root, for _release_cli

from irregular_domain import IrregularFinsDomain
from physics_fno import PhysicsStructuredPINO, StandardAutoregressivePINO
from solve_reaction_diff import generate_reaction_diff_data
from _release_cli import apply_cli, stamp_preset

SEEDS = [42, 123, 456]
NX, NY, NT = 32, 32, 10
N_TRAIN, N_VAL, N_TEST = 150, 20, 20
N_EPOCHS_SS = 2000
N_EPOCHS_AR = 300
MODES, WIDTH, N_LAYERS = 16, 96, 5
RESULTS_FILE = os.path.join(_HERE, os.pardir, "results", "reaction_diff_ar_valfix_results.json")
CKPT_DIR = os.path.join(_HERE, os.pardir, "checkpoints")

SEEDS, RESULTS_FILE, CKPT_DIR, _CFG = apply_cli(
    SEEDS, RESULTS_FILE, CKPT_DIR, device='cuda',
    description=(__doc__ or '').strip().splitlines()[0])
device = _CFG['device']
NX = _CFG.get('nx', NX)
NY = _CFG.get('ny', NY)
NT = _CFG.get('nt', NT)
N_TRAIN = _CFG.get('n_train', N_TRAIN)
N_VAL = _CFG.get('n_val', N_VAL)
N_TEST = _CFG.get('n_test', N_TEST)
N_EPOCHS_SS = _CFG.get('ss_epochs', N_EPOCHS_SS)
N_EPOCHS_AR = _CFG.get('ar_epochs', N_EPOCHS_AR)
os.makedirs(CKPT_DIR, exist_ok=True)

domain = IrregularFinsDomain(
    domain_size=(1.0, 1.0), fin_type='wavy', n_fins=3, base_height=0.1
)


def eval_model(model, data, name='', has_alpha=False):
    """Last-step evaluation (identical protocol to run_reaction_diff_ar.py)."""
    model.eval()
    mask, X, Y = data['mask'], data['X'], data['Y']
    t_array, normalizer = data['t'], data['normalizer']
    valid = mask > 0.5
    valid_np = mask.cpu().numpy() > 0.5
    grid = torch.stack([X, Y], dim=-1).unsqueeze(0)
    nt = len(t_array)
    dt_array = t_array[1:] - t_array[:-1]

    if has_alpha and 'alpha_array' in data:
        alpha_arr = data['alpha_array']
        alpha_mean = alpha_arr.mean()
        alpha_norm = torch.tensor(alpha_arr / alpha_mean, dtype=torch.float32, device=device)

    all_mae, all_max = [], []
    for traj_idx in range(data['T'].shape[0]):
        T_traj = data['T'][traj_idx]
        T_current = T_traj[0].clone()
        T_hot_norm = T_current[(Y < 0.1) & valid].mean()

        sa = alpha_norm[traj_idx:traj_idx+1] if has_alpha else None

        with torch.no_grad():
            for t_idx in range(nt - 1):
                dt = dt_array[t_idx]
                if sa is not None:
                    T_next = model(T_current.unsqueeze(0), dt.unsqueeze(0), grid,
                                   sample_alpha=sa).squeeze(0)
                else:
                    T_next = model(T_current.unsqueeze(0), dt.unsqueeze(0), grid).squeeze(0)
                hot = (Y < 0.1) & valid
                if hot.any():
                    w = torch.clamp((0.1 - Y) / 0.08, 0, 1)
                    T_next = torch.where(hot, T_next * (1 - w) + T_hot_norm * w, T_next)
                T_next = T_next * mask
                T_current = T_next

        preds_K = normalizer.denormalize(T_current.cpu().numpy())
        trues_K = normalizer.denormalize(T_traj[-1].cpu().numpy())
        errors = np.abs(preds_K - trues_K)[valid_np]
        all_mae.append(errors.mean())
        all_max.append(errors.max())

    r = {'mae': float(np.mean(all_mae)), 'avg_max': float(np.mean(all_max)),
         'worst_max': float(np.max(all_max))}
    if name:
        print(f'  {name}: MAE={r["mae"]:.4f}K, AvgMax={r["avg_max"]:.4f}K, '
              f'WorstMax={r["worst_max"]:.4f}K', flush=True)
    return r


def ar_finetune(model, train_data, val_data, test_data, has_alpha=False, n_epochs=300):
    """AR finetuning with early stopping on VALIDATION worst-case error (not test)."""
    mask = train_data['mask']
    X, Y = train_data['X'], train_data['Y']
    T_all, t_array = train_data['T'], train_data['t']
    n_traj, nt = T_all.shape[0], len(t_array)
    grid = torch.stack([X, Y], dim=-1).unsqueeze(0)
    valid = mask > 0.5
    dt_array = t_array[1:] - t_array[:-1]

    alpha_norm = None
    if has_alpha and 'alpha_array' in train_data:
        alpha_arr = train_data['alpha_array']
        alpha_mean = alpha_arr.mean()
        alpha_norm = torch.tensor(alpha_arr / alpha_mean, dtype=torch.float32, device=device)

    weight = torch.ones_like(Y)
    hot_boundary = (Y < 0.15) & valid
    weight = torch.where(hot_boundary, weight * 3.0, weight)
    cold_boundary = (Y > 0.85) & valid
    weight = torch.where(cold_boundary, weight * 2.0, weight)
    boundary_weight = weight * mask

    def apply_bc(T, T_hot_norm):
        T_out = T.clone()
        hot = (Y < 0.1) & valid
        if hot.any():
            w = torch.clamp((0.1 - Y) / 0.08, 0, 1)
            T_out = torch.where(hot, T_out * (1 - w) + T_hot_norm * w, T_out)
        return T_out * mask

    before_val = eval_model(model, val_data, has_alpha=has_alpha)
    before_test = eval_model(model, test_data, has_alpha=has_alpha)
    print(f'    Before AR: VAL WM={before_val["worst_max"]:.4f} | '
          f'TEST WM={before_test["worst_max"]:.4f} (test not used for selection)', flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)

    best_val_wm = before_val['worst_max']
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    patience_counter = 0
    max_patience = 80

    t0 = time.time()
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        batch_idx = np.random.choice(n_traj, min(8, n_traj), replace=False)
        traj_losses = []

        for idx in batch_idx:
            T_traj = T_all[idx]
            T_hot_norm = T_traj[0][(Y < 0.1) & valid].mean()
            T_current = T_traj[0]
            traj_loss = 0

            sa = alpha_norm[idx:idx+1] if alpha_norm is not None else None

            for t_idx in range(nt - 1):
                T_next_true = T_traj[t_idx + 1]
                dt = dt_array[t_idx]

                if sa is not None:
                    T_next_pred = model(T_current.unsqueeze(0), dt.unsqueeze(0), grid,
                                        sample_alpha=sa).squeeze(0)
                else:
                    T_next_pred = model(T_current.unsqueeze(0), dt.unsqueeze(0), grid).squeeze(0)

                T_next_pred = apply_bc(T_next_pred, T_hot_norm)

                diff = (T_next_pred - T_next_true)[valid]
                w = boundary_weight[valid]
                loss_mse = ((diff ** 2) * w).mean()

                abs_diff = diff.abs()
                max_err = abs_diff.max()
                loss_max = max_err ** 2 * 0.8

                k = max(1, int(abs_diff.numel() * 0.01))
                top_k = torch.topk(abs_diff, k).values
                loss_topk = (top_k ** 2).mean() * 0.5

                traj_loss += loss_mse + loss_max + loss_topk
                T_current = T_next_pred  # no detach - full BPTT

            traj_losses.append(traj_loss / (nt - 1))

        avg_loss = torch.stack(traj_losses).mean()
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 20 == 0 or epoch == 0:
            val_result = eval_model(model, val_data, has_alpha=has_alpha)
            print(f'    [AR] Epoch {epoch+1}/{n_epochs} | Loss: {avg_loss.item():.6f} | '
                  f'VAL WM: {val_result["worst_max"]:.4f}K', flush=True)

            if val_result['worst_max'] < best_val_wm:
                best_val_wm = val_result['worst_max']
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                print(f'      -> New best VAL WM: {best_val_wm:.4f}K', flush=True)
                patience_counter = 0
            else:
                patience_counter += 20

            if patience_counter >= max_patience:
                print(f'      Early stopping at epoch {epoch+1} (on VAL)', flush=True)
                break

    ar_time = time.time() - t0
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    after_val = eval_model(model, val_data, has_alpha=has_alpha)
    after_test = eval_model(model, test_data, has_alpha=has_alpha)
    print(f'    After AR: VAL WM={after_val["worst_max"]:.4f} | TEST WM={after_test["worst_max"]:.4f} '
          f'(time: {ar_time:.0f}s)', flush=True)
    return before_test, after_test, after_val, ar_time, best_state


def resolve_ss_checkpoint(arm, seed, model_factory, train_data):
    """Return the path to an SS-stage checkpoint, training one if it is absent.

    The SS-stage checkpoints behind the reported reaction-diffusion AR results
    are not hosted in this repository (see README, "Scope of the reproduction
    claim"). If the file is missing, the SS stage is retrained here with the
    standard 2D scheduled-sampling protocol (2000 epochs, p_teacher 0.9 -> 0.3,
    warmup 500, anneal 1500) used by run_conv_diff.py and by the fresh-SS
    fallback in run_heat_ar_gridsearch.py. A checkpoint produced this way gives
    a fresh AR realization, not a reproduction of the archived per-seed values.

    Lookup order: the canonical name first, then the legacy ``_gridnew_`` name.
    Earlier revisions saved the freshly trained SS checkpoint as
    ``{arm}_reactdiff_gridnew_ss_seed{seed}.pth`` while looking it up as
    ``{arm}_reactdiff_ss_seed{seed}.pth``, so a checkpoint written by one run
    was invisible to the next and the SS stage was retrained every time. Both
    names are now accepted on read; the canonical name is written.
    """
    canonical = os.path.join(CKPT_DIR, f'{arm}_reactdiff_ss_seed{seed}.pth')
    legacy = os.path.join(CKPT_DIR, f'{arm}_reactdiff_gridnew_ss_seed{seed}.pth')
    for ckpt in (canonical, legacy):
        if os.path.exists(ckpt):
            if ckpt == legacy:
                print(f'    Found existing SS checkpoint (legacy name): {ckpt}', flush=True)
            else:
                print(f'    Found existing SS checkpoint: {ckpt}', flush=True)
            return ckpt

    print(f'\n[SS] {arm} FNO, seed {seed}: no SS checkpoint present, '
          f'training fresh SS ({N_EPOCHS_SS} epochs)...', flush=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.startswith('cuda'):
        torch.cuda.manual_seed_all(seed)
    from physics_fno import train_scheduled_sampling
    model = model_factory()
    train_scheduled_sampling(model, train_data, n_epochs=N_EPOCHS_SS, device=device,
                             p_start=0.9, p_end=0.3, warmup=500, anneal_end=1500)
    ckpt = canonical
    torch.save({k: v.cpu().clone() for k, v in model.state_dict().items()}, ckpt)
    del model
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    print(f'    Trained fresh SS and saved: {ckpt}', flush=True)
    return ckpt


# ============================================================
# Main
# ============================================================
print('=' * 60, flush=True)
print('REACTION-DIFFUSION: AR VALIDATION-SPLIT FIX (3 seeds)', flush=True)
print('Independent val set (seed+2000, 20 traj) for early stopping.', flush=True)
print('Test set (seed+1000) touched only for final reporting.', flush=True)
print(f'AR: {N_EPOCHS_AR} epochs | SS checkpoints reused if present, else trained fresh', flush=True)
print('=' * 60, flush=True)

results = {'standard_ar': {}, 'physics_ar': {}, 'standard_ar_val': {}, 'physics_ar_val': {}}
stamp_preset(results, _CFG)

for seed in SEEDS:
    print(f'\n{"=" * 60}', flush=True)
    print(f'SEED {seed}', flush=True)
    print(f'{"=" * 60}', flush=True)

    print('  Generating data (train/val/test, ramp_fraction=0.7)...', flush=True)
    train_data = generate_reaction_diff_data(domain, nx=NX, ny=NY, nt=NT, n_traj=N_TRAIN,
                                              device=device, seed=seed, ramp_fraction=0.7)
    val_data = generate_reaction_diff_data(domain, nx=NX, ny=NY, nt=NT, n_traj=N_VAL,
                                            device=device, seed=seed + 2000, ramp_fraction=0.7)
    test_data = generate_reaction_diff_data(domain, nx=NX, ny=NY, nt=NT, n_traj=N_TEST,
                                             device=device, seed=seed + 1000, ramp_fraction=0.7)

    # ---- Standard FNO: resolve SS checkpoint, redo AR with val-based stopping ----
    print(f'\n  [Standard AR-valfix] Resolving SS checkpoint...', flush=True)
    ss_ckpt_std = resolve_ss_checkpoint(
        'std', seed,
        lambda: StandardAutoregressivePINO(modes=MODES, width=WIDTH, n_layers=N_LAYERS, device=device),
        train_data)
    model_std = StandardAutoregressivePINO(modes=MODES, width=WIDTH, n_layers=N_LAYERS, device=device)
    model_std.load_state_dict(torch.load(ss_ckpt_std, map_location=device))
    model_std = model_std.to(device)
    print(f'    Loaded: {ss_ckpt_std}', flush=True)

    torch.manual_seed(seed + 5000)
    np.random.seed(seed + 5000)
    if device.startswith('cuda'):
        torch.cuda.manual_seed_all(seed + 5000)

    before_std, after_std_test, after_std_val, ar_time_std, ar_state_std = ar_finetune(
        model_std, train_data, val_data, test_data, has_alpha=False, n_epochs=N_EPOCHS_AR)

    torch.save(ar_state_std, os.path.join(CKPT_DIR, f'std_reactdiff_ar_valfix_seed{seed}.pth'))

    results['standard_ar'][str(seed)] = {**after_std_test, 'time': ar_time_std}
    results['standard_ar_val'][str(seed)] = after_std_val
    del model_std
    if device.startswith('cuda'):
        torch.cuda.empty_cache()

    # ---- Physics FNO: resolve SS checkpoint, redo AR with val-based stopping ----
    print(f'\n  [Physics AR-valfix] Resolving SS checkpoint...', flush=True)
    ss_ckpt_phys = resolve_ss_checkpoint(
        'phys', seed,
        lambda: PhysicsStructuredPINO(modes=MODES, width=WIDTH, n_layers=N_LAYERS,
                                      device=device, corr_rank=8, pde_type='heat'),
        train_data)
    model_phys = PhysicsStructuredPINO(modes=MODES, width=WIDTH, n_layers=N_LAYERS,
                                        device=device, corr_rank=8, pde_type='heat')
    model_phys.load_state_dict(torch.load(ss_ckpt_phys, map_location=device))
    model_phys = model_phys.to(device)
    print(f'    Loaded: {ss_ckpt_phys}', flush=True)

    torch.manual_seed(seed + 5000)
    np.random.seed(seed + 5000)
    if device.startswith('cuda'):
        torch.cuda.manual_seed_all(seed + 5000)

    before_phys, after_phys_test, after_phys_val, ar_time_phys, ar_state_phys = ar_finetune(
        model_phys, train_data, val_data, test_data, has_alpha=True, n_epochs=N_EPOCHS_AR)

    torch.save(ar_state_phys, os.path.join(CKPT_DIR, f'phys_reactdiff_ar_valfix_seed{seed}.pth'))

    results['physics_ar'][str(seed)] = {**after_phys_test, 'time': ar_time_phys}
    results['physics_ar_val'][str(seed)] = after_phys_val
    del model_phys
    if device.startswith('cuda'):
        torch.cuda.empty_cache()

    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)

# ============================================================
# Summary
# ============================================================
print(f'\n{"=" * 60}', flush=True)
print('SUMMARY: Reaction-Diffusion AR (validation-split fix, 3 seeds)', flush=True)
print(f'{"=" * 60}', flush=True)

for method in ['standard_ar', 'physics_ar']:
    vals = results[method]
    if not vals:
        continue
    mae_list = [vals[str(s)]['mae'] for s in SEEDS if str(s) in vals]
    am_list = [vals[str(s)]['avg_max'] for s in SEEDS if str(s) in vals]
    wm_list = [vals[str(s)]['worst_max'] for s in SEEDS if str(s) in vals]
    print(f'\n{method} (TEST, reported only after selection on VAL):')
    print(f'  MAE:     {np.mean(mae_list):.3f} +/- {np.std(mae_list):.3f} K')
    print(f'  AvgMax:  {np.mean(am_list):.3f} +/- {np.std(am_list):.3f} K')
    print(f'  WorstMax:{np.mean(wm_list):.3f} +/- {np.std(wm_list):.3f} K')
    print(f'  Per-seed WM: {[f"{w:.3f}" for w in wm_list]}')

with open(RESULTS_FILE, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nSaved: {RESULTS_FILE}')
print('DONE!')
