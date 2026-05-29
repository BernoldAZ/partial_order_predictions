"""Train and evaluate GATv2GRUMultilabel for multi-label next-activity +
time-to-next-block prediction.

Usage
-----
    python run_multilabel_v1.py <log_name> [results_dir]

Loads pre-built multilabel datasets from results_per_log/<log_name>/:
    train_multilabel_dataset.pt, val_multilabel_dataset.pt,
    test_multilabel_dataset.pt, <log_name>_T_max.pkl,
    <log_name>_train_means_dict.pkl, <log_name>_train_std_dict.pkl

Run create_multilabel_data.py first to generate these files.

Evaluation
----------
Predicted and ground-truth multi-hot sequences are converted to canonical
event sequences (active activities within each block sorted by index, blocks
concatenated, stop at END) before computing Damerau-Levenshtein similarity.
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

from model_multilabel_v1 import GATv2GRUMultilabel

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

def _multilabel_bce(logits, label_mh):
    """BCE averaged over all non-padding positions and all C classes.

    A position is padding when its label multi-hot is all-zero.
    logits, label_mh : (B, T, C)
    """
    active = label_mh.sum(dim=-1) > 0          # (B, T)
    bce = F.binary_cross_entropy_with_logits(
        logits, label_mh.float(), reduction='none')   # (B, T, C)
    return bce[active].mean()


def _masked_tsp_mae(tsp_pred, label_tsp):
    """MAE over positions where label_tsp != -100.

    tsp_pred : (B, T, 1)
    label_tsp : (B, T)
    """
    pred = tsp_pred.squeeze(-1)                # (B, T)
    mask = label_tsp != -100
    if not mask.any():
        return tsp_pred.sum() * 0.0
    return torch.abs(pred[mask] - label_tsp[mask]).mean()


def _loss(logits, tsp_pred, data):
    B  = data.num_graphs
    C  = logits.shape[-1]
    T  = logits.shape[1]
    label_mh  = data.suffix_multihot.view(B, T, C)
    label_tsp = data.label_tsp.view(B, T)
    bce = _multilabel_bce(logits, label_mh)
    mae = _masked_tsp_mae(tsp_pred, label_tsp)
    return bce + mae


# ─── Canonical sequence conversion ───────────────────────────────────────────

def _multihot_to_canonical(pred_blocks, C):
    """Convert (T, C) multi-hot block sequence to a canonical integer sequence.

    Stops at the first block where END (index C-1) is predicted.
    Within each block, active regular activities are sorted by index.
    Activity integers returned are 1-indexed (add 1 to 0-indexed multi-hot index).
    """
    seq = []
    for t in range(pred_blocks.shape[0]):
        active = pred_blocks[t].nonzero(as_tuple=False).squeeze(-1).tolist()
        if C - 1 in active:       # END token predicted
            break
        if not active:
            continue              # skip empty (padding) block
        for idx in sorted(active):
            seq.append(idx + 1)   # 0-indexed → 1-indexed activity integer
    return seq


def _gt_multihot_to_canonical(gt_blocks, C):
    """Same as _multihot_to_canonical but for ground-truth blocks.

    gt_blocks : (T, C) float multi-hot
    """
    return _multihot_to_canonical(gt_blocks, C)


# ─── DL similarity (loop-based, handles variable-length sequences) ────────────

def _dl_distance(s1, s2):
    """Damerau-Levenshtein distance between two integer lists."""
    n, m = len(s1), len(s2)
    if n == 0 and m == 0:
        return 0
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): d[i][0] = i
    for j in range(m + 1): d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + cost)
            if i > 1 and j > 1 and s1[i-1] == s2[j-2] and s1[i-2] == s2[j-1]:
                d[i][j] = min(d[i][j], d[i-2][j-2] + cost)
    return d[n][m]


def _dl_similarity(s1, s2):
    dist = _dl_distance(s1, s2)
    denom = max(len(s1), len(s2), 1)
    return 1.0 - dist / denom


# ─── Metrics ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def _compute_metrics(pred_blocks_all, pred_tsp_all, gt_mh_all,
                     label_tsp_all, label_rrt_all, C, mean_std_tsp, mean_std_rrt):
    """Compute DL similarity, TSP MAE, and RRT MAE.

    pred_blocks_all : (N, T, C) float multi-hot predictions
    pred_tsp_all    : (N, T) normalised TSP predictions
    gt_mh_all       : (N, T, C) float ground-truth multi-hot
    label_tsp_all   : (N, T) normalised TSP labels (-100 = masked)
    label_rrt_all   : (N,) normalised RRT labels
    """
    tsp_mean, tsp_std = mean_std_tsp
    rrt_mean, rrt_std = mean_std_rrt
    N, T = pred_tsp_all.shape

    dl_sims = []
    for b in range(N):
        pred_seq = _multihot_to_canonical(pred_blocks_all[b], C)
        gt_seq   = _gt_multihot_to_canonical(gt_mh_all[b], C)
        dl_sims.append(_dl_similarity(pred_seq, gt_seq))

    dl_sim = sum(dl_sims) / max(len(dl_sims), 1)

    # TSP MAE in seconds
    pred_secs = (pred_tsp_all * tsp_std + tsp_mean).clamp(min=0)
    mask      = label_tsp_all != -100
    if mask.any():
        gt_secs  = (label_tsp_all[mask] * tsp_std + tsp_mean).clamp(min=0)
        tsp_mae  = torch.abs(pred_secs[mask] - gt_secs).mean().item() / 60.0
    else:
        tsp_mae = 0.0

    # RRT MAE in minutes
    # pred_tsp[t] predicts suffix_tsp[t+1] (time from block t to block t+1).
    # RRT prediction = sum of pred_tsp up to (but not including) predicted END block.
    # RRT label = pre-stored rtime at first suffix position, de-normalised with rrt stats.
    end_predicted = pred_blocks_all[:, :, C - 1] > 0.5        # (N, T) bool
    has_end       = end_predicted.any(dim=-1)
    pred_end      = torch.where(
        has_end,
        end_predicted.to(torch.int64).argmax(dim=-1),
        torch.full((N,), T - 1, dtype=torch.int64))

    counting      = torch.arange(T).unsqueeze(0)               # (1, T)
    rrt_pred_sec  = pred_secs.clone()
    rrt_pred_sec[counting >= pred_end.unsqueeze(-1)] = 0.0
    rrt_pred_sec  = rrt_pred_sec.sum(dim=-1)                   # (N,)

    rrt_label_sec = (label_rrt_all * rrt_std + rrt_mean).clamp(min=0)  # (N,)
    rrt_mae       = torch.abs(rrt_pred_sec - rrt_label_sec).mean().item() / 60.0

    return dl_sim, tsp_mae, rrt_mae


# ─── Evaluation pass ──────────────────────────────────────────────────────────

@torch.no_grad()
def _evaluate(model, loader, device, C, T_max, mean_std_tsp, mean_std_tss, mean_std_rrt):
    model.eval()
    all_pred, all_tsp, all_gt_mh, all_lbl_tsp, all_lbl_rrt = [], [], [], [], []

    for data in loader:
        data = data.to(device)
        B    = data.num_graphs
        pred_blocks, pred_tsp = model(
            data,
            T_max=T_max,
            mean_std_tsp=mean_std_tsp,
            mean_std_tss=mean_std_tss,
        )
        all_pred.append(pred_blocks.cpu())
        all_tsp.append(pred_tsp.cpu())
        all_gt_mh.append(data.suffix_multihot.view(B, T_max, C).cpu())
        all_lbl_tsp.append(data.label_tsp.view(B, T_max).cpu())
        all_lbl_rrt.append(data.label_rrt.view(B).cpu())

    pred_all  = torch.cat(all_pred,    dim=0)
    tsp_all   = torch.cat(all_tsp,     dim=0)
    gt_all    = torch.cat(all_gt_mh,   dim=0)
    ltsp_all  = torch.cat(all_lbl_tsp, dim=0)
    lrrt_all  = torch.cat(all_lbl_rrt, dim=0)

    return _compute_metrics(pred_all, tsp_all, gt_all, ltsp_all, lrrt_all, C, mean_std_tsp, mean_std_rrt)


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

    data_dir = os.path.join("approach_suffix_v2", 'results_per_log', log_name)

    train_data = torch.load(
        os.path.join(data_dir, 'train_multilabel_dataset.pt'), weights_only=False)
    val_data   = torch.load(
        os.path.join(data_dir, 'val_multilabel_dataset.pt'),   weights_only=False)
    test_data  = torch.load(
        os.path.join(data_dir, 'test_multilabel_dataset.pt'),  weights_only=False)

    with open(os.path.join(data_dir, f'{log_name}_train_means_dict.pkl'), 'rb') as f:
        means = pickle.load(f)
    with open(os.path.join(data_dir, f'{log_name}_train_std_dict.pkl'), 'rb') as f:
        stds  = pickle.load(f)
    with open(os.path.join(data_dir, f'{log_name}_T_max.pkl'), 'rb') as f:
        T_max = pickle.load(f)
    with open(os.path.join(data_dir, f'{log_name}_cardin_list_prefix.pkl'), 'rb') as f:
        pref_cat_cars = pickle.load(f)

    # suffix_df normalisation stats: index 0 = ts_start (tss), index 1 = ts_prev (tsp)
    mean_std_tss = [means['suffix_df'][0], stds['suffix_df'][0]]
    mean_std_tsp = [means['suffix_df'][1], stds['suffix_df'][1]]
    # timeLabel_df normalisation stats: index 0 = tt_next, index 1 = rtime (rrt)
    mean_std_rrt = [means['timeLabel_df'][1], stds['timeLabel_df'][1]]

    num_activities = pref_cat_cars[-1] + 2
    C = num_activities - 1

    print(f"num_activities={num_activities}  C={C}  T_max={T_max}")

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False)

    model = GATv2GRUMultilabel(
        num_activities=num_activities,
        d_model=D_MODEL,
        dropout=DROPOUT,
        n_layers=N_LAYERS,
        use_scheduled_sampling=USE_SCHEDULED_SAMPLING,
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {num_params:,}")

    optimizer    = torch.optim.NAdam(model.parameters(), lr=LR)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=LR_PATIENCE,
        threshold=1e-4, min_lr=0)

    os.makedirs(results_dir, exist_ok=True)
    best_model_path = os.path.join(results_dir, f'{log_name}_{METHOD_NAME}.pt')

    best_dl_sim   = -1.0
    best_tsp_mae  =  1e9
    best_rrt_mae  =  1e9
    patience_count = 0
    train_start    = time.time()

    for epoch in range(MAX_EPOCHS):
        torch.manual_seed(epoch)
        model.train()
        total_loss, n_batches = 0.0, 0

        if USE_SCHEDULED_SAMPLING:
            progress  = epoch / max(SS_ANNEAL_EPOCHS - 1, 1)
            p_teacher = max(SS_P_TEACHER_END,
                            SS_P_TEACHER_START -
                            (SS_P_TEACHER_START - SS_P_TEACHER_END) * progress)
        else:
            p_teacher = 1.0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            logits, tsp_pred = model(
                data,
                p_teacher=p_teacher,
                mean_std_tsp=mean_std_tsp if USE_SCHEDULED_SAMPLING else None,
                mean_std_tss=mean_std_tss if USE_SCHEDULED_SAMPLING else None,
            )
            loss = _loss(logits, tsp_pred, data)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        train_loss = total_loss / max(n_batches, 1)

        dl_sim, tsp_mae, rrt_mae = _evaluate(
            model, val_loader, device, C, T_max, mean_std_tsp, mean_std_tss, mean_std_rrt)

        lr_scheduler.step(1.0 - dl_sim)
        lr = optimizer.param_groups[0]['lr']
        ss_info = f"  p_teacher={p_teacher:.3f}" if USE_SCHEDULED_SAMPLING else ""
        print(f"[{log_name}] Epoch {epoch+1:4d}  loss={train_loss:.4f}  "
              f"DL={dl_sim:.4f}  TSP_MAE={tsp_mae:.2f}min  RRT_MAE={rrt_mae:.2f}min  "
              f"lr={lr:.2e}{ss_info}")

        if dl_sim > best_dl_sim:
            torch.save(model.state_dict(), best_model_path)

        better = dl_sim > best_dl_sim or tsp_mae < best_tsp_mae or rrt_mae < best_rrt_mae
        if better:
            best_dl_sim  = max(best_dl_sim,  dl_sim)
            best_tsp_mae = min(best_tsp_mae, tsp_mae)
            best_rrt_mae = min(best_rrt_mae, rrt_mae)
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}.")
                break

    training_time = time.time() - train_start

    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    model.to(device)

    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    test_start  = time.time()
    dl_sim, tsp_mae, rrt_mae = _evaluate(
        model, test_loader, device, C, T_max, mean_std_tsp, mean_std_tss, mean_std_rrt)
    testing_time = time.time() - test_start

    print(f"\n{'─'*60}")
    print(f"DL similarity : {dl_sim:.4f}")
    print(f"TSP MAE       : {tsp_mae:.2f} min")
    print(f"RRT MAE       : {rrt_mae:.2f} min")
    print(f"Training time : {training_time:.1f}s")
    print(f"Testing time  : {testing_time:.1f}s")

    csv_path   = os.path.join(results_dir, 'results_multilabel.csv')
    fieldnames = ['log', 'model', 'method',
                  'dl_similarity', 'tsp_mae_minutes', 'rrt_mae_minutes',
                  'training_time_seconds', 'testing_time_seconds',
                  'num_trainable_params']
    new_row = {
        'log':                   log_name,
        'model':                 'multilabel_v1',
        'method':                METHOD_NAME,
        'dl_similarity':         round(dl_sim,       6),
        'tsp_mae_minutes':       round(tsp_mae,      6),
        'rrt_mae_minutes':       round(rrt_mae,      6),
        'training_time_seconds': round(training_time, 2),
        'testing_time_seconds':  round(testing_time,  2),
        'num_trainable_params':  num_params,
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
    return {'dl_similarity': dl_sim, 'tsp_mae_minutes': tsp_mae, 'rrt_mae_minutes': rrt_mae}


def _parse_args():
    p = argparse.ArgumentParser(
        description='Train multilabel GRU model and evaluate with canonical DL')
    p.add_argument('log_name',    help='Log name (must match results_per_log/<log_name>/)')
    p.add_argument('results_dir', nargs='?', default=None)
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    run(args.log_name, args.results_dir)
