"""Train and evaluate GRUEncoderGRUDecoder v2 for activity-suffix, time, and
concurrency prediction.

Usage
-----
    python approach_suffix_v2/models/run_gru_suffix_v2.py <log_name> [results_dir]

Loads pre-built datasets from approach_suffix_v2/results_per_log/<log_name>/:
    train_seqdataset.pt, val_seqdataset.pt, test_seqdataset.pt
    test_seq_concurrent_mask.pt
    <log_name>_seq_stats.pkl

Run create_sequential_data.py first to generate these files.
"""
import argparse
import csv
import os
import pickle
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from create_sequential_data import build_dataloaders
from model_gru_suffix_v2 import GRUEncoderGRUDecoder

# ─── Hyperparameters ──────────────────────────────────────────────────────────
D_MODEL            = 64
DROPOUT            = 0.4
N_LAYERS           = 1
LR                 = 0.002
MAX_EPOCHS         = 200
PATIENCE           = 24
LR_PATIENCE        = 10
MAX_NORM           = 2.0
BATCH_SIZE         = 128
SEED               = 24
CONSISTENCY_WEIGHT = 0.1   # weight for concurrency-time consistency penalty

# ─── Scheduled sampling ───────────────────────────────────────────────────────
USE_SCHEDULED_SAMPLING = True
SS_P_TEACHER_START     = 1.0
SS_P_TEACHER_END       = 0.0
SS_ANNEAL_EPOCHS       = MAX_EPOCHS

METHOD_NAME = 'gru_suffix_v2'


# ─── Device helper ────────────────────────────────────────────────────────────

def _to_device(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


# ─── Loss ─────────────────────────────────────────────────────────────────────

def _loss(act_logits, time_pred, conc_logits, consistency, data, num_classes):
    ce = F.cross_entropy(
        act_logits.view(-1, num_classes),
        data['suffix_act'].view(-1),
        ignore_index=0,
    )
    mask = data['suffix_mask'].view(-1).float()
    mse  = (((time_pred.view(-1) - data['suffix_dt'].view(-1)) ** 2) * mask).sum() \
           / mask.sum().clamp(min=1)
    bce_raw = F.binary_cross_entropy_with_logits(
        conc_logits.view(-1), data['suffix_conc'].view(-1).float(), reduction='none')
    bce  = (bce_raw * mask).sum() / mask.sum().clamp(min=1)
    cons = (consistency.view(-1) * mask).sum() / mask.sum().clamp(min=1)
    return ce + mse + bce + CONSISTENCY_WEIGHT * cons


# ─── Metrics ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def _compute_metrics(pred_acts, pred_dt, pred_conc,
                     suf_acts, suf_dt, suf_conc, suf_mask,
                     num_classes, dt_mean, dt_std):
    device  = pred_acts.device
    N, W    = pred_acts.shape
    eos_idx = num_classes - 1

    actual_length = (suf_acts == eos_idx).to(torch.int64).argmax(dim=-1)
    has_eos       = (pred_acts == eos_idx).any(dim=-1)
    first_eos     = (pred_acts == eos_idx).to(torch.int64).argmax(dim=-1)
    pred_length   = torch.where(has_eos, first_eos, torch.full_like(first_eos, W - 1))

    counting    = torch.arange(W, device=device).unsqueeze(0)
    batch_range = torch.arange(N, device=device)

    # ── Damerau-Levenshtein similarity ────────────────────────────────────────
    len_pred   = pred_length   + 1
    len_actual = actual_length + 1
    max_len    = torch.maximum(len_pred, len_actual).float()

    d  = torch.zeros(N, W + 1, W + 1, dtype=torch.int64, device=device)
    ar = torch.arange(W + 1, device=device).unsqueeze(0)
    d[:, 0, :] = ar
    d[:, :, 0] = ar
    for i in range(1, W + 1):
        for j in range(1, W + 1):
            cost         = torch.where(pred_acts[:, i-1] == suf_acts[:, j-1], 0, 1)
            deletion     = d[:, i-1, j]   + 1
            insertion    = d[:, i, j-1]   + 1
            substitution = d[:, i-1, j-1] + cost
            d[:, i, j]  = torch.minimum(torch.minimum(deletion, insertion), substitution)
            if i > 1 and j > 1:
                tpos_true   = (
                    (pred_acts[:, i-1] == suf_acts[:, j-2]) &
                    (pred_acts[:, i-2] == suf_acts[:, j-1])
                )
                min_og_tpos = torch.minimum(d[:, i, j], d[:, i-2, j-2] + cost)
                d[:, i, j]  = torch.where(tpos_true, min_og_tpos, d[:, i, j])
    dl_sim = (1.0 - d[batch_range, len_pred, len_actual].float() / max_len).mean().item()

    # ── dt MAE (unnormalized, minutes) ────────────────────────────────────────
    def _unnorm(x):
        return (torch.exp(x.float() * dt_std + dt_mean) - 1.0).clamp(min=0.0)

    pred_dt_sec = _unnorm(pred_dt)
    suf_dt_sec  = _unnorm(suf_dt)
    pred_dt_sec[counting > pred_length.unsqueeze(-1)] = 0.0

    before_eos = counting < actual_length.unsqueeze(-1)
    dt_mae_sec = torch.abs(pred_dt_sec - suf_dt_sec)[before_eos].mean().item() \
                 if before_eos.any() else 0.0

    # ── RRT MAE (minutes) ─────────────────────────────────────────────────────
    rrt_pred = pred_dt_sec.clone()
    rrt_pred[counting >= pred_length.unsqueeze(-1)] = 0.0
    rrt_pred = rrt_pred.sum(dim=-1)

    rrt_gt = suf_dt_sec.clone()
    rrt_gt[counting >= actual_length.unsqueeze(-1)] = 0.0
    rrt_gt = rrt_gt.sum(dim=-1)

    rrt_mae_sec = torch.abs(rrt_pred - rrt_gt).mean().item()

    # ── Concurrency accuracy ──────────────────────────────────────────────────
    valid    = suf_mask & (suf_acts != eos_idx)
    conc_acc = (pred_conc[valid] == suf_conc[valid]).float().mean().item() \
               if valid.any() else 0.0

    mean_pred_len   = (pred_length   + 1).float().mean().item()
    mean_actual_len = (actual_length + 1).float().mean().item()
    return dl_sim, dt_mae_sec / 60.0, rrt_mae_sec / 60.0, conc_acc, mean_pred_len, mean_actual_len


# ─── Collect predictions ──────────────────────────────────────────────────────

@torch.no_grad()
def _collect_preds(model, loader, device):
    model.eval()
    all_acts, all_dt, all_conc = [], [], []
    all_suf_acts, all_suf_dt, all_suf_conc, all_suf_mask = [], [], [], []
    for batch in loader:
        batch = _to_device(batch, device)
        acts, dt, conc = model(batch)
        all_acts.append(acts.cpu())
        all_dt.append(dt.cpu())
        all_conc.append(conc.cpu())
        all_suf_acts.append(batch['suffix_act'].cpu())
        all_suf_dt.append(batch['suffix_dt'].cpu())
        all_suf_conc.append(batch['suffix_conc'].cpu())
        all_suf_mask.append(batch['suffix_mask'].cpu())
    return (
        torch.cat(all_acts), torch.cat(all_dt), torch.cat(all_conc),
        torch.cat(all_suf_acts), torch.cat(all_suf_dt),
        torch.cat(all_suf_conc), torch.cat(all_suf_mask),
    )


# ─── Validation pass ──────────────────────────────────────────────────────────

@torch.no_grad()
def _evaluate(model, loader, device, num_classes, dt_mean, dt_std):
    pa, pd, pc, sa, sd, sc, sm = _collect_preds(model, loader, device)
    return _compute_metrics(pa, pd, pc, sa, sd, sc, sm,
                            num_classes, dt_mean, dt_std)


# ─── Next-activity metrics ────────────────────────────────────────────────────

def _next_act_metrics(pred_col0, gt_col0):
    pred = pred_col0.numpy()
    gt   = gt_col0.numpy()
    acc  = float(accuracy_score(gt, pred))
    f1   = float(f1_score(gt, pred, average='weighted', zero_division=0))
    return acc, f1


# ─── Main ─────────────────────────────────────────────────────────────────────

def run(log_name: str, results_dir: str = None):
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    results_dir = results_dir or f'results_{METHOD_NAME}'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}\nLog : {log_name}\nDevice : {device}\n{'='*60}")

    # ── Load pre-built datasets ───────────────────────────────────────────────
    data_dir   = os.path.join('approach_suffix_v2', 'results_per_log', log_name)
    train_data = torch.load(os.path.join(data_dir, 'train_seqdataset.pt'), weights_only=False)
    val_data   = torch.load(os.path.join(data_dir, 'val_seqdataset.pt'),   weights_only=False)
    test_data  = torch.load(os.path.join(data_dir, 'test_seqdataset.pt'),  weights_only=False)

    with open(os.path.join(data_dir, f'{log_name}_seq_stats.pkl'), 'rb') as f:
        stats = pickle.load(f)
    num_classes = stats['num_classes']
    eos_idx     = stats['eos_idx']
    window_size = stats['window_size']
    dt_mean     = stats['dt_mean']
    dt_std      = stats['dt_std']
    print(f"num_classes={num_classes}  eos_idx={eos_idx}  window_size={window_size}  "
          f"dt_mean={dt_mean:.4f}  dt_std={dt_std:.4f}")

    train_loader, val_loader, test_loader = build_dataloaders(
        train_data, val_data, test_data, BATCH_SIZE)

    # ── Build model ───────────────────────────────────────────────────────────
    model = GRUEncoderGRUDecoder(
        num_classes=num_classes,
        d_model=D_MODEL,
        dropout=DROPOUT,
        n_layers=N_LAYERS,
        use_scheduled_sampling=USE_SCHEDULED_SAMPLING,
    ).to(device)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {num_trainable_params:,}")

    optimizer    = torch.optim.NAdam(model.parameters(), lr=LR)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=LR_PATIENCE, threshold=1e-4, min_lr=0)

    os.makedirs(results_dir, exist_ok=True)
    best_model_path = os.path.join(results_dir, f'{log_name}_{METHOD_NAME}.pt')

    best_dl_sim    = -1.0
    best_dt_mae    =  1e9
    best_rrt_mae   =  1e9
    patience_count = 0
    train_start    = time.time()

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(MAX_EPOCHS):
        torch.manual_seed(epoch)
        model.train()
        total_loss, n_batches = 0.0, 0

        if USE_SCHEDULED_SAMPLING:
            progress  = epoch / max(SS_ANNEAL_EPOCHS - 1, 1)
            p_teacher = max(SS_P_TEACHER_END,
                            SS_P_TEACHER_START - (SS_P_TEACHER_START - SS_P_TEACHER_END) * progress)
        else:
            p_teacher = 1.0

        for batch in train_loader:
            batch = _to_device(batch, device)
            optimizer.zero_grad()
            act_logits, time_pred, conc_logits, consistency = model(batch, p_teacher=p_teacher)
            loss = _loss(act_logits, time_pred, conc_logits, consistency, batch, num_classes)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        train_loss = total_loss / max(n_batches, 1)

        dl_sim, dt_mae_min, rrt_mae_min, conc_acc, mean_pred_len, mean_actual_len = \
            _evaluate(model, val_loader, device, num_classes, dt_mean, dt_std)

        lr_scheduler.step(1.0 - dl_sim)
        lr    = optimizer.param_groups[0]['lr']
        ss_info = f"  p_teacher={p_teacher:.3f}" if USE_SCHEDULED_SAMPLING else ""
        print(f"[{log_name}] Epoch {epoch+1:4d}  loss={train_loss:.4f}  "
              f"DL={dl_sim:.4f}  dt_MAE={dt_mae_min:.2f}min  "
              f"RRT={rrt_mae_min:.2f}min  conc_acc={conc_acc:.4f}  "
              f"lr={lr:.2e}  len_pred={mean_pred_len:.1f}  len_gt={mean_actual_len:.1f}{ss_info}")

        if dl_sim > best_dl_sim:
            torch.save(model.state_dict(), best_model_path)

        better = (dl_sim > best_dl_sim or
                  dt_mae_min  < best_dt_mae or
                  rrt_mae_min < best_rrt_mae)
        if better:
            best_dl_sim  = max(best_dl_sim,  dl_sim)
            best_dt_mae  = min(best_dt_mae,  dt_mae_min)
            best_rrt_mae = min(best_rrt_mae, rrt_mae_min)
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}.")
                break

    training_time = time.time() - train_start

    # ── Test ──────────────────────────────────────────────────────────────────
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    model.to(device)
    test_start = time.time()

    pa, pd, pc, sa, sd, sc, sm = _collect_preds(model, test_loader, device)

    dl_sim, dt_mae_min, rrt_mae_min, conc_acc, _, _ = _compute_metrics(
        pa, pd, pc, sa, sd, sc, sm, num_classes, dt_mean, dt_std)

    next_acc, next_f1 = _next_act_metrics(pa[:, 0], sa[:, 0])

    # ── Concurrent-ending-prefix subset ──────────────────────────────────────
    conc_mask = torch.load(
        os.path.join(data_dir, 'test_seq_concurrent_mask.pt'), weights_only=True)
    n_conc = conc_mask.sum().item()
    if n_conc > 0:
        conc_dl, conc_dt, conc_rrt, sub_conc_acc, _, _ = _compute_metrics(
            pa[conc_mask], pd[conc_mask], pc[conc_mask],
            sa[conc_mask], sd[conc_mask], sc[conc_mask], sm[conc_mask],
            num_classes, dt_mean, dt_std)
        conc_next_acc, conc_next_f1 = _next_act_metrics(
            pa[conc_mask, 0], sa[conc_mask, 0])
    else:
        conc_dl = conc_dt = conc_rrt = sub_conc_acc = conc_next_acc = conc_next_f1 = None

    testing_time = time.time() - test_start

    print(f"\n{'─'*60}")
    print(f"DL similarity         : {dl_sim:.4f}")
    print(f"dt MAE                : {dt_mae_min:.2f} min")
    print(f"RRT MAE               : {rrt_mae_min:.2f} min")
    print(f"Concurrency accuracy  : {conc_acc:.4f}")
    print(f"Next-act accuracy     : {next_acc:.4f}")
    print(f"Next-act F1 (weighted): {next_f1:.4f}")
    print(f"Concurrent subset     : {n_conc} / {len(test_data)} samples")
    if n_conc > 0:
        print(f"  DL similarity     : {conc_dl:.4f}")
        print(f"  dt MAE            : {conc_dt:.2f} min")
        print(f"  RRT MAE           : {conc_rrt:.2f} min")
        print(f"  Conc accuracy     : {sub_conc_acc:.4f}")
        print(f"  Next-act accuracy : {conc_next_acc:.4f}")
        print(f"  Next-act F1       : {conc_next_f1:.4f}")
    print(f"Training time         : {training_time:.1f}s")
    print(f"Testing time          : {testing_time:.1f}s")

    # ── Save results ──────────────────────────────────────────────────────────
    csv_path   = os.path.join(results_dir, 'results_gru_suffix_v2.csv')
    fieldnames = ['log', 'model', 'method',
                  'dl_similarity', 'dt_mae_minutes', 'rrt_mae_minutes',
                  'conc_accuracy', 'next_act_accuracy', 'next_act_f1_weighted',
                  'conc_n_samples',
                  'conc_dl_similarity', 'conc_dt_mae_minutes', 'conc_rrt_mae_minutes',
                  'conc_conc_accuracy', 'conc_next_act_accuracy', 'conc_next_act_f1_weighted',
                  'training_time_seconds', 'testing_time_seconds',
                  'num_trainable_params']
    new_row = {
        'log':                       log_name,
        'model':                     'gru_encoder_gru_decoder_v2',
        'method':                    METHOD_NAME,
        'dl_similarity':             round(dl_sim,      6),
        'dt_mae_minutes':            round(dt_mae_min,  6),
        'rrt_mae_minutes':           round(rrt_mae_min, 6),
        'conc_accuracy':             round(conc_acc,    6),
        'next_act_accuracy':         round(next_acc,    6),
        'next_act_f1_weighted':      round(next_f1,     6),
        'conc_n_samples':            n_conc,
        'conc_dl_similarity':     (round(conc_dl,       6) if conc_dl       is not None else ''),
        'conc_dt_mae_minutes':    (round(conc_dt,       6) if conc_dt       is not None else ''),
        'conc_rrt_mae_minutes':   (round(conc_rrt,      6) if conc_rrt      is not None else ''),
        'conc_conc_accuracy':     (round(sub_conc_acc,  6) if sub_conc_acc  is not None else ''),
        'conc_next_act_accuracy': (round(conc_next_acc, 6) if conc_next_acc is not None else ''),
        'conc_next_act_f1_weighted': (round(conc_next_f1, 6) if conc_next_f1 is not None else ''),
        'training_time_seconds':  round(training_time, 2),
        'testing_time_seconds':   round(testing_time,  2),
        'num_trainable_params':   num_trainable_params,
    }

    lock_path = csv_path + '.lock'
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            time.sleep(0.05)
    try:
        rows = []
        if os.path.isfile(csv_path):
            with open(csv_path, newline='') as f:
                rows = list(csv.DictReader(f))
        updated = False
        for row in rows:
            if row['log'] == log_name and row['method'] == METHOD_NAME:
                row.update(new_row)
                updated = True
                break
        if not updated:
            rows.append(new_row)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    finally:
        os.remove(lock_path)

    print(f"Results saved → {csv_path}")
    return {
        'dl_similarity':             dl_sim,
        'dt_mae_minutes':            dt_mae_min,
        'rrt_mae_minutes':           rrt_mae_min,
        'conc_accuracy':             conc_acc,
        'next_act_accuracy':         next_acc,
        'next_act_f1_weighted':      next_f1,
        'conc_n_samples':            n_conc,
        'conc_dl_similarity':        conc_dl,
        'conc_dt_mae_minutes':       conc_dt,
        'conc_rrt_mae_minutes':      conc_rrt,
        'conc_conc_accuracy':        sub_conc_acc,
        'conc_next_act_accuracy':    conc_next_acc,
        'conc_next_act_f1_weighted': conc_next_f1,
    }


# ─── Entry point ──────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description='Train GRU encoder-decoder v2; evaluate with DL similarity, '
                    'dt MAE, RRT MAE, and concurrency accuracy.')
    p.add_argument('log_name',    help='Log name (must match results_per_log/<log_name>/)')
    p.add_argument('results_dir', nargs='?', default=None)
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    run(args.log_name, args.results_dir)
