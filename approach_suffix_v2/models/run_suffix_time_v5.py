"""Train and evaluate GATv2EncoderGRUDecoderMultiLabel for activity suffix + TTNE
prediction with explicit partial-order decoding.

Differences from run_suffix_time_v2.py
---------------------------------------
- Activity loss   : BCEWithLogitsLoss (multi-label) instead of CrossEntropy
- Decoder steps   : one per distinct timestamp group (concurrent events merged)
- Ground-truth DL : act_label_seq is canonically sorted (concurrent activities
                    ordered by class index) before Damerau-Levenshtein so that
                    the metric is independent of the original linearisation.

Usage
-----
    python run_suffix_time_v5.py <log_name> [results_dir]
"""
import argparse
import csv
import os
import pickle
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from model_suffix_time_v5 import GATv2EncoderGRUDecoderMultiLabel

# ─── Hyperparameters ──────────────────────────────────────────────────────────
D_MODEL     = 64
DROPOUT     = 0.4
N_LAYERS    = 1
LR          = 0.002
MAX_EPOCHS  = 200
PATIENCE    = 24
LR_PATIENCE = 10
MAX_NORM    = 2.0
BATCH_SIZE  = 128
SEED        = 24

USE_SCHEDULED_SAMPLING = True
SS_P_TEACHER_START     = 1.0
SS_P_TEACHER_END       = 0.0
SS_ANNEAL_EPOCHS       = MAX_EPOCHS

METHOD_NAME = 'gatv2_gru_multilabel'


# ─── Loss ─────────────────────────────────────────────────────────────────────

def _bce_act(act_logits, group_targets, group_mask):
    """BCEWithLogitsLoss over valid groups, excluding the padding class (idx 0)."""
    logits  = act_logits[group_mask][:, 1:]    # (N_valid, C-1)
    targets = group_targets[group_mask][:, 1:] # (N_valid, C-1)
    return F.binary_cross_entropy_with_logits(logits, targets, reduction='mean')


def _mae_ttne(ttne_out, group_ttne, group_mask):
    pred = ttne_out.squeeze(-1)[group_mask]    # (N_valid,)
    true = group_ttne[group_mask]              # (N_valid,)
    mask = (true != -100).float()
    return (torch.abs(pred - true) * mask).sum() / mask.sum().clamp(min=1)


def _stop_bce(stop_logits, group_targets, group_mask, num_activities, avg_suffix_len):
    """BCE stop head: target=1 for groups that contain the END token."""
    end_tok = num_activities - 1
    logits  = stop_logits[group_mask]              # (N_valid,)
    is_end  = group_targets[group_mask][:, end_tok] # (N_valid,) float
    pos_weight = torch.tensor([avg_suffix_len], device=logits.device)
    return F.binary_cross_entropy_with_logits(
        logits, is_end, pos_weight=pos_weight, reduction='mean')


def _loss(act_logits, ttne_out, stop_logits,
          group_targets, group_mask, group_ttne,
          num_activities, avg_suffix_len):
    return (_bce_act(act_logits, group_targets, group_mask)
            + _mae_ttne(ttne_out, group_ttne, group_mask)
            + _stop_bce(stop_logits, group_targets, group_mask,
                        num_activities, avg_suffix_len))


# ─── Canonical sort for ground-truth labels ───────────────────────────────────

def _canonical_sort(act_labels, tss, window_size):
    """
    Sort activities within concurrent groups (equal tss) by class index.

    act_labels : (B, W) long
    tss        : (B, W) float  — normalised ts_start for each suffix position
    Returns    : (B, W) long   — same events, concurrent groups sorted
    """
    B = act_labels.shape[0]
    out = act_labels.clone()
    for b in range(B):
        pos = 0
        while pos < window_size:
            if act_labels[b, pos] == 0:   # padding → done
                break
            group_tss = tss[b, pos]
            end_pos = pos + 1
            while (end_pos < window_size
                   and tss[b, end_pos] == group_tss
                   and act_labels[b, end_pos] != 0):
                end_pos += 1
            out[b, pos:end_pos] = act_labels[b, pos:end_pos].sort().values
            pos = end_pos
    return out


# ─── Metrics ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def _compute_metrics(suffix_acts, suffix_ttne, act_labels, ttne_labels, rrt_labels,
                     num_activities, mean_std_ttne, mean_std_rrt):
    device  = suffix_acts.device
    N, W    = suffix_acts.shape
    end_tok = num_activities - 1
    ttne_mean, ttne_std = mean_std_ttne
    rrt_mean,  rrt_std  = mean_std_rrt

    actual_length = (act_labels == end_tok).to(torch.int64).argmax(dim=-1)

    has_end     = (suffix_acts == end_tok).any(dim=-1)
    first_end   = (suffix_acts == end_tok).to(torch.int64).argmax(dim=-1)
    pred_length = torch.where(has_end, first_end, torch.full_like(first_end, W - 1))

    counting    = torch.arange(W, device=device).unsqueeze(0)
    batch_range = torch.arange(N, device=device)

    # ── Damerau-Levenshtein similarity ────────────────────────────────────────
    len_pred   = pred_length   + 1
    len_actual = actual_length + 1
    max_len    = torch.maximum(len_pred, len_actual).float()

    d  = torch.full((N, W + 1, W + 1), fill_value=0, dtype=torch.int64, device=device)
    ar = torch.arange(W + 1, device=device).unsqueeze(0)
    d[:, 0, :] = ar
    d[:, :, 0] = ar
    for i in range(1, W + 1):
        for j in range(1, W + 1):
            cost         = torch.where(suffix_acts[:, i-1] == act_labels[:, j-1], 0, 1)
            deletion     = d[:, i-1, j]   + 1
            insertion    = d[:, i, j-1]   + 1
            substitution = d[:, i-1, j-1] + cost
            d[:, i, j]   = torch.minimum(torch.minimum(deletion, insertion), substitution)
            if i > 1 and j > 1:
                tpos_true   = (
                    (suffix_acts[:, i-1] == act_labels[:, j-2]) &
                    (suffix_acts[:, i-2] == act_labels[:, j-1])
                )
                min_og_tpos = torch.minimum(d[:, i, j], d[:, i-2, j-2] + cost)
                d[:, i, j]  = torch.where(tpos_true, min_og_tpos, d[:, i, j])
    dl_sim = (1.0 - d[batch_range, len_pred, len_actual].float() / max_len).mean().item()

    # ── TTNE MAE ──────────────────────────────────────────────────────────────
    ttne_preds_sec  = (suffix_ttne.clone() * ttne_std + ttne_mean).clamp(min=0)
    ttne_labels_sec = (ttne_labels.clone() * ttne_std + ttne_mean).clamp(min=0)
    ttne_preds_sec[counting > pred_length.unsqueeze(-1)] = 0.0

    before_end   = counting <= actual_length.unsqueeze(-1)
    ttne_mae_sec = torch.abs(ttne_preds_sec - ttne_labels_sec)[before_end].mean().item()

    # ── RRT MAE ───────────────────────────────────────────────────────────────
    rrt_preds_sec = ttne_preds_sec.clone()
    rrt_preds_sec[batch_range, pred_length] = 0.0
    rrt_preds_sec = rrt_preds_sec.sum(dim=-1)

    rrt_label_sec = (rrt_labels[:, 0] * rrt_std + rrt_mean).clamp(min=0)
    rrt_mae_sec   = torch.abs(rrt_preds_sec - rrt_label_sec).mean().item()

    mean_pred_len   = (pred_length   + 1).float().mean().item()
    mean_actual_len = (actual_length + 1).float().mean().item()
    return dl_sim, ttne_mae_sec / 60.0, rrt_mae_sec / 60.0, mean_pred_len, mean_actual_len


# ─── Evaluation pass ──────────────────────────────────────────────────────────

@torch.no_grad()
def _evaluate(model, loader, device, num_activities, window_size,
              mean_std_ttne, mean_std_tss, mean_std_tsp, mean_std_rrt):
    model.eval()
    all_acts, all_ttne, all_lbl_acts, all_lbl_ttne, all_lbl_rrt = [], [], [], [], []

    for data in loader:
        data = data.to(device)
        B    = data.num_graphs
        acts, ttne = model(
            data,
            window_size=window_size,
            mean_std_ttne=mean_std_ttne,
            mean_std_tss=mean_std_tss,
            mean_std_tsp=mean_std_tsp,
        )

        # Canonical sort of ground-truth labels before DL computation
        lbl_acts = data.act_label_seq.view(B, window_size).cpu()
        tss_cpu  = data.suffix_num.reshape(B, window_size, 2)[:, :, 0].cpu()
        lbl_acts_canonical = _canonical_sort(lbl_acts, tss_cpu, window_size)

        all_acts.append(acts.cpu())
        all_ttne.append(ttne.cpu())
        all_lbl_acts.append(lbl_acts_canonical)
        all_lbl_ttne.append(data.ttnext_label.reshape(B, window_size).cpu())
        all_lbl_rrt.append(data.rtime_label.squeeze(-1).view(B, window_size).cpu())

    sa = torch.cat(all_acts,     dim=0)
    st = torch.cat(all_ttne,     dim=0)
    la = torch.cat(all_lbl_acts, dim=0)
    lt = torch.cat(all_lbl_ttne, dim=0)
    lr = torch.cat(all_lbl_rrt,  dim=0)

    return _compute_metrics(sa, st, la, lt, lr, num_activities, mean_std_ttne, mean_std_rrt)


# ─── Main ─────────────────────────────────────────────────────────────────────

def run(log_name: str, results_dir: str = None):
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    results_dir = results_dir or f'results_time_{METHOD_NAME}'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}\nLog : {log_name}\nDevice : {device}\n{'='*60}")

    # ── Load pre-built datasets ───────────────────────────────────────────────
    data_dir   = os.path.join("approach_suffix_v2", 'results_per_log', log_name)
    train_data = torch.load(os.path.join(data_dir, 'train_graphdataset.pt'), weights_only=False)
    val_data   = torch.load(os.path.join(data_dir, 'val_graphdataset.pt'),   weights_only=False)
    test_data  = torch.load(os.path.join(data_dir, 'test_graphdataset.pt'),  weights_only=False)

    # ── Load normalization stats ──────────────────────────────────────────────
    with open(os.path.join(data_dir, f'{log_name}_train_means_dict.pkl'), 'rb') as f:
        means = pickle.load(f)
    with open(os.path.join(data_dir, f'{log_name}_train_std_dict.pkl'), 'rb') as f:
        stds  = pickle.load(f)

    mean_std_ttne = [means['timeLabel_df'][0], stds['timeLabel_df'][0]]
    mean_std_rrt  = [means['timeLabel_df'][1], stds['timeLabel_df'][1]]
    mean_std_tss  = [means['suffix_df'][0],    stds['suffix_df'][0]]
    mean_std_tsp  = [means['suffix_df'][1],    stds['suffix_df'][1]]

    with open(os.path.join(data_dir, f'{log_name}_cardin_list_prefix.pkl'), 'rb') as f:
        pref_cat_cars = pickle.load(f)
    window_size    = train_data[0].suffix_act.shape[0]
    num_activities = pref_cat_cars[-1] + 2

    avg_suffix_len = sum(
        (s.act_label_seq != 0).sum().item() for s in train_data
    ) / len(train_data)
    print(f"num_activities={num_activities}  window_size={window_size}  "
          f"avg_suffix_len={avg_suffix_len:.1f}")

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False)

    # ── Build model ───────────────────────────────────────────────────────────
    model = GATv2EncoderGRUDecoderMultiLabel(
        num_activities=num_activities,
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

    best_dl_sim   = -1.0
    best_ttne_mae =  1e9
    best_rrt_mae  =  1e9
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

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            act_logits, ttne_preds, stop_logits, group_targets, group_mask, group_ttne = model(
                data,
                p_teacher=p_teacher,
                mean_std_ttne=mean_std_ttne if USE_SCHEDULED_SAMPLING else None,
                mean_std_tss=mean_std_tss   if USE_SCHEDULED_SAMPLING else None,
                mean_std_tsp=mean_std_tsp   if USE_SCHEDULED_SAMPLING else None,
            )
            loss = _loss(act_logits, ttne_preds, stop_logits,
                         group_targets, group_mask, group_ttne,
                         num_activities, avg_suffix_len)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        train_loss = total_loss / max(n_batches, 1)

        dl_sim, ttne_mae_min, rrt_mae_min, mean_pred_len, mean_actual_len = _evaluate(
            model, val_loader, device, num_activities, window_size,
            mean_std_ttne, mean_std_tss, mean_std_tsp, mean_std_rrt)

        lr_scheduler.step(1.0 - dl_sim)
        lr = optimizer.param_groups[0]['lr']
        ss_info = f"  p_teacher={p_teacher:.3f}" if USE_SCHEDULED_SAMPLING else ""
        print(f"[{log_name}] Epoch {epoch+1:4d}  loss={train_loss:.4f}  "
              f"DL={dl_sim:.4f}  TTNE={ttne_mae_min:.2f}min  "
              f"RRT={rrt_mae_min:.2f}min  lr={lr:.2e}  "
              f"len_pred={mean_pred_len:.1f}  len_gt={mean_actual_len:.1f}{ss_info}")

        if dl_sim > best_dl_sim:
            torch.save(model.state_dict(), best_model_path)

        better = (dl_sim > best_dl_sim or
                  ttne_mae_min < best_ttne_mae or
                  rrt_mae_min  < best_rrt_mae)
        if better:
            best_dl_sim   = max(best_dl_sim,   dl_sim)
            best_ttne_mae = min(best_ttne_mae, ttne_mae_min)
            best_rrt_mae  = min(best_rrt_mae,  rrt_mae_min)
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

    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    test_start  = time.time()
    dl_sim, ttne_mae_min, rrt_mae_min, _, _ = _evaluate(
        model, test_loader, device, num_activities, window_size,
        mean_std_ttne, mean_std_tss, mean_std_tsp, mean_std_rrt)
    testing_time = time.time() - test_start

    print(f"\n{'─'*60}")
    print(f"DL similarity : {dl_sim:.4f}")
    print(f"TTNE MAE      : {ttne_mae_min:.2f} min")
    print(f"RRT MAE       : {rrt_mae_min:.2f} min")
    print(f"Training time : {training_time:.1f}s")
    print(f"Testing time  : {testing_time:.1f}s")

    # ── Save results ──────────────────────────────────────────────────────────
    csv_path   = os.path.join(results_dir, 'results_suffix_time_gnn.csv')
    fieldnames = ['log', 'model', 'method',
                  'dl_similarity', 'ttne_mae_minutes', 'rrt_mae_minutes',
                  'training_time_seconds', 'testing_time_seconds',
                  'num_trainable_params']
    new_row = {
        'log':                   log_name,
        'model':                 'suffix_time_multilabel',
        'method':                METHOD_NAME,
        'dl_similarity':         round(dl_sim,        6),
        'ttne_mae_minutes':      round(ttne_mae_min,  6),
        'rrt_mae_minutes':       round(rrt_mae_min,   6),
        'training_time_seconds': round(training_time, 2),
        'testing_time_seconds':  round(testing_time,  2),
        'num_trainable_params':  num_trainable_params,
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
    return {'dl_similarity': dl_sim, 'ttne_mae_minutes': ttne_mae_min,
            'rrt_mae_minutes': rrt_mae_min}


# ─── Entry point ──────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description='Train multi-label GRU model and evaluate for suffix + RRT')
    p.add_argument('log_name',    help='Log name (must match results_per_log/<log_name>/)')
    p.add_argument('results_dir', nargs='?', default=None)
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    run(args.log_name, args.results_dir)
