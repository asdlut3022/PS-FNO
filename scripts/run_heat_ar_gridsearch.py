"""
Symmetric AR hyperparameter grid search for heat (2D wavy-fin diffusion),
Physics FNO vs Standard FNO -- fair tuning per AE requirement ("Authors need
to tune all methods fairly. Tuning steps and tuned parameter values must be
included.")

Design (agreed protocol, both models treated identically):
  - Reuses existing SS checkpoints: seeds 42/123 use
    phys/std_heat_valfix_ss_seed{42,123}.pth; seed 456 uses the non-valfix
    names phys/std_heat_ss_seed456.pth. If the resolved checkpoint is absent,
    the script trains a fresh SS stage (2000 epochs) before the AR grid.
  - AR hyperparameter grid: lr in {1e-5, 3e-5, 1e-4}, patience in {40, 80}
    (6 combinations), IDENTICAL grid and IDENTICAL search protocol for both
    Physics FNO and Standard FNO.
  - Model selection at every stage is on the INDEPENDENT VALIDATION set
    (seed+2000, 20 trajectories) -- never on test.
  - Seeds fixed to {42,123,456} for the main means and the cross-benchmark tests.
  - For each (model, seed): run all 6 AR configs, pick the config with best
    VALIDATION worst-case error, report that config's TEST worst-case error.
    This is the standard "tune on val, report on test" protocol -- symmetric
    for both models, no cherry-picking of favorable results.
  - Test set (seed+1000) is touched exactly once per (model, seed), after
    the winning AR config has already been chosen on validation.

Output: heat_ar_gridsearch_results.json
"""
import torch
import numpy as np
import sys
import json
import os
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))          # release root, for _release_cli
import train_v10_clean as old_train

from irregular_domain import IrregularFinsDomain
from physics_fno import PhysicsStructuredPINO, StandardAutoregressivePINO
from _release_cli import apply_cli, stamp_preset

SEEDS = [42, 123, 456]  # three seeds, symmetric AR grid for both models
NX, NY, NT = 32, 32, 10
N_TRAIN, N_VAL, N_TEST = 150, 20, 20
N_EPOCHS_SS = 2000
N_EPOCHS_AR = 300
MODES, WIDTH, N_LAYERS = 16, 96, 5
LR_GRID = [1e-5, 3e-5, 1e-4]
PATIENCE_GRID = [40, 80]
RESULTS_FILE = os.path.join(_HERE, os.pardir, "results", "heat_ar_gridsearch_results.json")
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


def atomic_save(results):
    """Write results atomically: temp file then rename (crash-safe)."""
    tmp = RESULTS_FILE + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    os.replace(tmp, RESULTS_FILE)

domain = IrregularFinsDomain(
    domain_size=(1.0, 1.0), fin_type='wavy', n_fins=3, base_height=0.1
)


def eval_model(model, data):
    model.eval()
    mask, X, Y = data['mask'], data['X'], data['Y']
    t_array, normalizer = data['t'], data['normalizer']
    valid = mask > 0.5
    valid_np = mask.cpu().numpy() > 0.5
    grid = torch.stack([X, Y], dim=-1).unsqueeze(0)
    nt = len(t_array)
    dt_array = t_array[1:] - t_array[:-1]

    all_mae, all_max = [], []
    for traj_idx in range(data['T'].shape[0]):
        T_traj = data['T'][traj_idx]
        T_current = T_traj[0].clone()
        T_hot_norm = T_current[(Y < 0.1) & valid].mean()

        preds_list = [T_current.cpu().numpy()]
        with torch.no_grad():
            for t_idx in range(nt - 1):
                dt = dt_array[t_idx]
                T_next = model(T_current.unsqueeze(0), dt.unsqueeze(0), grid).squeeze(0)
                hot = (Y < 0.1) & valid
                if hot.any():
                    w = torch.clamp((0.1 - Y) / 0.08, 0, 1)
                    T_next = torch.where(hot, T_next * (1 - w) + T_hot_norm * w, T_next)
                T_next = T_next * mask
                preds_list.append(T_next.cpu().numpy())
                T_current = T_next

        preds_K = normalizer.denormalize(np.array(preds_list))
        trues_K = normalizer.denormalize(T_traj.cpu().numpy())
        errors = np.abs(preds_K - trues_K)[:, valid_np]
        all_mae.append(errors.mean())
        all_max.append(errors.max())

    return {
        'mae': float(np.mean(all_mae)),
        'avg_max': float(np.mean(all_max)),
        'worst_max': float(np.max(all_max)),
    }


def ar_finetune_config(model_state, train_data, val_data, lr, patience, n_epochs=300):
    """Run one AR config from a fixed SS starting point; select on VAL only."""
    model = model_state['ctor']()
    model.load_state_dict(model_state['ss_state'])
    model = model.to(device)

    mask, X, Y = train_data['mask'], train_data['X'], train_data['Y']
    T_all, t_array = train_data['T'], train_data['t']
    n_traj, nt = T_all.shape[0], len(t_array)
    grid = torch.stack([X, Y], dim=-1).unsqueeze(0)
    valid = mask > 0.5
    dt_array = t_array[1:] - t_array[:-1]

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

    before_val = eval_model(model, val_data)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)

    best_val_wm = before_val['worst_max']
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    patience_counter = 0

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

            for t_idx in range(nt - 1):
                T_next_true = T_traj[t_idx + 1]
                dt = dt_array[t_idx]
                T_next_pred = model(T_current.unsqueeze(0), dt.unsqueeze(0), grid).squeeze(0)
                T_next_pred = apply_bc(T_next_pred, T_hot_norm)

                diff = (T_next_pred - T_next_true)[valid]
                w = boundary_weight[valid]
                loss_mse = ((diff ** 2) * w).mean()
                abs_diff = diff.abs()
                loss_max = abs_diff.max() ** 2 * 0.8
                k = max(1, int(abs_diff.numel() * 0.01))
                loss_topk = (torch.topk(abs_diff, k).values ** 2).mean() * 0.5

                traj_loss += loss_mse + loss_max + loss_topk
                T_current = T_next_pred

            traj_losses.append(traj_loss / (nt - 1))

        avg_loss = torch.stack(traj_losses).mean()
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 20 == 0 or epoch == 0:
            val_result = eval_model(model, val_data)
            if val_result['worst_max'] < best_val_wm:
                best_val_wm = val_result['worst_max']
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 20
            if patience_counter >= patience:
                break

    ar_time = time.time() - t0
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    after_val = eval_model(model, val_data)
    del model
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    return best_val_wm, after_val, best_state, ar_time


def run_seed(model_name, ctor, ss_ckpt_path, seed):
    print(f'\n{"="*60}', flush=True)
    print(f'{model_name} | SEED {seed} | grid search over {len(LR_GRID)}x{len(PATIENCE_GRID)} configs', flush=True)
    print(f'{"="*60}', flush=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    train_data = old_train.generate_data(domain, nx=NX, ny=NY, nt=NT, n_traj=N_TRAIN,
                                          device=device, seed=seed)
    val_data = old_train.generate_data(domain, nx=NX, ny=NY, nt=NT, n_traj=N_VAL,
                                        device=device, seed=seed + 2000)
    test_data = old_train.generate_data(domain, nx=NX, ny=NY, nt=NT, n_traj=N_TEST,
                                         device=device, seed=seed + 1000)

    ss_state = torch.load(ss_ckpt_path, map_location=device)
    model_state = {'ctor': ctor, 'ss_state': ss_state}

    ss_model = ctor()
    ss_model.load_state_dict(ss_state)
    ss_model = ss_model.to(device)
    ss_val = eval_model(ss_model, val_data)
    ss_test = eval_model(ss_model, test_data)
    del ss_model
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    print(f'  SS checkpoint: VAL WM={ss_val["worst_max"]:.4f} TEST WM={ss_test["worst_max"]:.4f}', flush=True)

    configs_result = []
    for lr in LR_GRID:
        for patience in PATIENCE_GRID:
            torch.manual_seed(seed + 5000)
            np.random.seed(seed + 5000)
            torch.cuda.manual_seed_all(seed + 5000)
            val_wm, after_val, best_state, ar_time = ar_finetune_config(
                model_state, train_data, val_data, lr, patience, n_epochs=N_EPOCHS_AR)
            print(f'    lr={lr:.0e} patience={patience}: VAL WM={val_wm:.4f} ({ar_time:.0f}s)', flush=True)
            configs_result.append({
                'lr': lr, 'patience': patience, 'val_wm': val_wm,
                'ar_time': ar_time, 'state': best_state,
            })

    # Pick config with best VAL worst-case error (never look at test for selection)
    best_config = min(configs_result, key=lambda c: c['val_wm'])
    print(f'  BEST config: lr={best_config["lr"]:.0e} patience={best_config["patience"]} '
          f'(VAL WM={best_config["val_wm"]:.4f})', flush=True)

    winner_model = ctor()
    winner_model.load_state_dict(best_config['state'])
    winner_model = winner_model.to(device)
    final_test = eval_model(winner_model, test_data)
    final_val = eval_model(winner_model, val_data)
    del winner_model
    if device.startswith('cuda'):
        torch.cuda.empty_cache()

    print(f'  FINAL (winning config, reported once): VAL WM={final_val["worst_max"]:.4f} '
          f'TEST WM={final_test["worst_max"]:.4f}', flush=True)

    return {
        'ss_val': ss_val,
        'ss_test': ss_test,
        'grid': [{'lr': c['lr'], 'patience': c['patience'], 'val_wm': c['val_wm'],
                  'ar_time': c['ar_time']} for c in configs_result],
        'best_lr': best_config['lr'],
        'best_patience': best_config['patience'],
        'final_val': final_val,
        'final_test': final_test,
    }


# ============================================================
# Main
# ============================================================
print('=' * 60, flush=True)
print('HEAT: SYMMETRIC AR HYPERPARAMETER GRID SEARCH', flush=True)
print(f'Seeds (fixed in advance): {SEEDS}', flush=True)
print(f'Grid: lr in {LR_GRID}, patience in {PATIENCE_GRID} ({len(LR_GRID)*len(PATIENCE_GRID)} configs)', flush=True)
print('IDENTICAL protocol for Physics FNO and Standard FNO.', flush=True)
print('Selection on independent VAL set; TEST reported once per seed after selection.', flush=True)
print('=' * 60, flush=True)

results = {'physics': {}, 'standard': {}}
if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
        results = json.load(f)
    print(f'[resume] loaded existing results: physics={sorted(results["physics"])}, '
          f'standard={sorted(results["standard"])}', flush=True)
stamp_preset(results, _CFG)

for seed in SEEDS:
    if str(seed) in results['physics']:
        print(f'[skip] physics/{seed} already in results', flush=True)
    else:
        valfix = os.path.join(CKPT_DIR, f'phys_heat_valfix_ss_seed{seed}.pth')
        plain = os.path.join(CKPT_DIR, f'phys_heat_ss_seed{seed}.pth')
        legacy = os.path.join(CKPT_DIR, f'phys_heat_gridnew_ss_seed{seed}.pth')
        # Lookup order: valfix (seeds 42/123 only) -> plain name -> legacy _gridnew_ name.
        # A freshly trained checkpoint is written under the "plain" name, i.e. the name the lookup
        # will hit; previously it was written as _gridnew_ while the lookup used the plain name, so every run retrained SS. The training strategy was not changed.
        cands = ([valfix] if seed in [42, 123] else []) + [plain, legacy]
        ss_ckpt = next((p for p in cands if os.path.exists(p)), plain)
        if os.path.exists(ss_ckpt):
            print(f'    Found existing SS checkpoint: {ss_ckpt}', flush=True)
        if not os.path.exists(ss_ckpt):
            # Missing SS checkpoint: need fresh SS training first
            print(f'\n[SS] Physics FNO, seed {seed}: no existing checkpoint, training fresh SS (2000 epochs)...', flush=True)
            torch.manual_seed(seed)
            np.random.seed(seed)
            torch.cuda.manual_seed_all(seed)
            train_data_ss = old_train.generate_data(domain, nx=NX, ny=NY, nt=NT, n_traj=N_TRAIN,
                                                      device=device, seed=seed)
            from physics_fno import train_scheduled_sampling
            model = PhysicsStructuredPINO(modes=MODES, width=WIDTH, n_layers=N_LAYERS, device=device)
            train_scheduled_sampling(model, train_data_ss, n_epochs=N_EPOCHS_SS, device=device,
                                     p_start=0.9, p_end=0.3)
            ss_ckpt = plain
            torch.save({k: v.cpu().clone() for k, v in model.state_dict().items()}, ss_ckpt)
            del model
            if device.startswith('cuda'):
                torch.cuda.empty_cache()

        results['physics'][str(seed)] = run_seed(
            'Physics', lambda: PhysicsStructuredPINO(modes=MODES, width=WIDTH, n_layers=N_LAYERS, device=device),
            ss_ckpt, seed)
        atomic_save(results)

    if str(seed) in results['standard']:
        print(f'[skip] standard/{seed} already in results', flush=True)
    else:
        valfix_std = os.path.join(CKPT_DIR, f'std_heat_valfix_ss_seed{seed}.pth')
        plain_std = os.path.join(CKPT_DIR, f'std_heat_ss_seed{seed}.pth')
        legacy_std = os.path.join(CKPT_DIR, f'std_heat_gridnew_ss_seed{seed}.pth')
        cands_std = ([valfix_std] if seed in [42, 123] else []) + [plain_std, legacy_std]
        ss_ckpt_std = next((p for p in cands_std if os.path.exists(p)), plain_std)
        if os.path.exists(ss_ckpt_std):
            print(f'    Found existing SS checkpoint: {ss_ckpt_std}', flush=True)
        if not os.path.exists(ss_ckpt_std):
            # Missing SS checkpoint: need fresh SS training first
            print(f'\n[SS] Standard FNO, seed {seed}: no existing checkpoint, training fresh SS (2000 epochs)...', flush=True)
            torch.manual_seed(seed)
            np.random.seed(seed)
            torch.cuda.manual_seed_all(seed)
            train_data_ss = old_train.generate_data(domain, nx=NX, ny=NY, nt=NT, n_traj=N_TRAIN,
                                                      device=device, seed=seed)
            from physics_fno import train_scheduled_sampling
            model = StandardAutoregressivePINO(modes=MODES, width=WIDTH, n_layers=N_LAYERS, device=device)
            train_scheduled_sampling(model, train_data_ss, n_epochs=N_EPOCHS_SS, device=device,
                                     p_start=0.9, p_end=0.3)
            ss_ckpt_std = plain_std
            torch.save({k: v.cpu().clone() for k, v in model.state_dict().items()}, ss_ckpt_std)
            del model
            if device.startswith('cuda'):
                torch.cuda.empty_cache()

        results['standard'][str(seed)] = run_seed(
            'Standard', lambda: StandardAutoregressivePINO(modes=MODES, width=WIDTH, n_layers=N_LAYERS, device=device),
            ss_ckpt_std, seed)
        atomic_save(results)

# ============================================================
# Summary
# ============================================================
print(f'\n{"=" * 60}', flush=True)
print('SUMMARY: Heat AR grid search (3 seeds, symmetric tuning)', flush=True)
print(f'{"=" * 60}', flush=True)

for method in ['physics', 'standard']:
    wm_list = [results[method][str(s)]['final_test']['worst_max'] for s in SEEDS]
    mae_list = [results[method][str(s)]['final_test']['mae'] for s in SEEDS]
    print(f'\n{method} FNO SS+AR (best-of-grid, TEST, n={len(SEEDS)}):')
    print(f'  MAE:     {np.mean(mae_list):.3f} +/- {np.std(mae_list):.3f} K')
    print(f'  WorstMax:{np.mean(wm_list):.3f} +/- {np.std(wm_list):.3f} K')
    print(f'  Per-seed WM: {[f"{w:.3f}" for w in wm_list]}')
    configs_used = [(results[method][str(s)]["best_lr"], results[method][str(s)]["best_patience"]) for s in SEEDS]
    print(f'  Winning configs per seed: {configs_used}')

atomic_save(results)
print(f'\nSaved: {RESULTS_FILE}')
print('DONE!')
