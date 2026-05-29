"""Train and evaluate GATv2EncoderGRUDecoderStop for activity suffix + TTNE
prediction. Extends run_suffix_time_v2.py with:
  - Next-activity accuracy and weighted F1 on the full test set.
  - The same metrics (plus DL / TTNE / RRT) on the concurrent-ending-prefix
    subset: samples where the last events of the input prefix are concurrent
    (i.e. the last node in the prefix graph has at least one intra-block
    outgoing edge).

Usage
-----
    python run_suffix_time_v2_1.py <log_name> [results_dir]

Loads pre-built datasets from results_per_log/<log_name>/:
    train_graphdataset.pt, val_graphdataset.pt, test_graphdataset.pt
    <log_name>_train_means_dict.pkl, <log_name>_train_std_dict.pkl

Run create_general_data.py first to generate these files.
"""
import argparse
import csv
import os
import pickle
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.loader import DataLoader

from model_suffix_time_v2 import GATv2EncoderGRUDecoderStop

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

# ─── Scheduled sampling ───────────────────────────────────────────────────────
USE_SCHEDULED_SAMPLING = True
SS_P_TEACHER_START     = 1.0
SS_P_TEACHER_END       = 0.0
SS_ANNEAL_EPOCHS       = MAX_EPOCHS

METHOD_NAME = 'gatv2_gru_stop'


# ─── Loss ─────────────────────────────────────────────────────────────────────

def _masked_ce(act_logits, act_targets, num_activities):
    return F.cross_entropy(
        act_logits.view(-1, num_activities),
        act_targets.view(-1),
        ignore_index=0,
    )


def _masked_mae(ttne_preds, ttne_targets):
    p    = ttne_preds.view(-1)
    t    = ttne_targets.view(-1)
    mask = (t != -100).float()
    return (torch.abs(p - t) * mask).sum() / mask.sum().clamp(min=1)


def _stop_bce(stop_logits, act_targets, num_activities, avg_suffix_len):
    """BCE over all non-padding positions; target=1 only at the END token."""
    logits  = stop_logits.view(-1)
    targets = act_targets.view(-1)
    is_end  = (targets == num_activities - 1).float()
    is_pad  = (targets == 0)
    pos_weight = torch.tensor([avg_suffix_len], device=logits.device)
    raw = F.binary_cross_entropy_with_logits(
        logits, is_end, pos_weight=pos_weight, reduction='none')
    return raw[~is_pad].mean()


def _loss(act_logits, ttne_preds, stop_logits, data, num_activities, avg_suffix_len):
    ce   = _masked_ce(act_logits, data.act_label_seq, num_activities)
    mae  = _masked_mae(ttne_preds, data.ttnext_label)
    stop = _stop_bce(stop_logits, data.act_label_seq, num_activities, avg_suffix_len)
    return ce + mae + stop


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
            d[:, i, j]  = torch.minimum(torch.minimum(deletion, insertion), substitution)
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


# ─── Evaluation pass (used during training / validation) ──────────────────────

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
        all_acts.append(acts.cpu())
        all_ttne.append(ttne.cpu())
        all_lbl_acts.append(data.act_label_seq.view(B, window_size).cpu())
        all_lbl_ttne.append(data.ttnext_label.squeeze(-1).view(B, window_size).cpu())
        all_lbl_rrt.append(data.rtime_label.squeeze(-1).view(B, window_size).cpu())

    sa = torch.cat(all_acts,     dim=0)
    st = torch.cat(all_ttne,     dim=0)
    la = torch.cat(all_lbl_acts, dim=0)
    lt = torch.cat(all_lbl_ttne, dim=0)
    lr = torch.cat(all_lbl_rrt,  dim=0)

    return _compute_metrics(sa, st, la, lt, lr, num_activities, mean_std_ttne, mean_std_rrt)


# ─── Test-time helpers ────────────────────────────────────────────────────────

@torch.no_grad()
def _collect_preds(model, loader, device, window_size,
                   mean_std_ttne, mean_std_tss, mean_std_tsp):
    """Run inference and return raw prediction / label tensors (N, W)."""
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
        all_acts.append(acts.cpu())
        all_ttne.append(ttne.cpu())
        all_lbl_acts.append(data.act_label_seq.view(B, window_size).cpu())
        all_lbl_ttne.append(data.ttnext_label.squeeze(-1).view(B, window_size).cpu())
        all_lbl_rrt.append(data.rtime_label.squeeze(-1).view(B, window_size).cpu())
    return (torch.cat(all_acts), torch.cat(all_ttne),
            torch.cat(all_lbl_acts), torch.cat(all_lbl_ttne), torch.cat(all_lbl_rrt))


def _next_act_metrics(pred_col0, gt_col0):
    """Accuracy and weighted F1 for the first suffix position (next activity)."""
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
    model = GATv2EncoderGRUDecoderStop(
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
            act_logits, ttne_preds, stop_logits = model(
                data,
                p_teacher=p_teacher,
                mean_std_ttne=mean_std_ttne if USE_SCHEDULED_SAMPLING else None,
                mean_std_tss=mean_std_tss   if USE_SCHEDULED_SAMPLING else None,
                mean_std_tsp=mean_std_tsp   if USE_SCHEDULED_SAMPLING else None,
            )
            loss = _loss(act_logits, ttne_preds, stop_logits,
                         data, num_activities, avg_suffix_len)
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

    sa, st, la, lt, lr = _collect_preds(
        model, test_loader, device, window_size,
        mean_std_ttne, mean_std_tss, mean_std_tsp)

    dl_sim, ttne_mae_min, rrt_mae_min, _, _ = _compute_metrics(
        sa, st, la, lt, lr, num_activities, mean_std_ttne, mean_std_rrt)

    next_acc, next_f1 = _next_act_metrics(sa[:, 0], la[:, 0])

    # ── Concurrent-ending-prefix subset ──────────────────────────────────────
    conc_mask = torch.load(os.path.join(data_dir, 'test_concurrent_mask.pt'), weights_only=True)
    n_conc    = conc_mask.sum().item()
    if n_conc > 0:
        conc_dl, conc_ttne, conc_rrt, _, _ = _compute_metrics(
            sa[conc_mask], st[conc_mask], la[conc_mask],
            lt[conc_mask], lr[conc_mask],
            num_activities, mean_std_ttne, mean_std_rrt)
        conc_acc, conc_f1 = _next_act_metrics(sa[conc_mask, 0], la[conc_mask, 0])
    else:
        conc_dl = conc_ttne = conc_rrt = conc_acc = conc_f1 = None

    testing_time = time.time() - test_start

    print(f"\n{'─'*60}")
    print(f"DL similarity         : {dl_sim:.4f}")
    print(f"TTNE MAE              : {ttne_mae_min:.2f} min")
    print(f"RRT MAE               : {rrt_mae_min:.2f} min")
    print(f"Next-act accuracy     : {next_acc:.4f}")
    print(f"Next-act F1 (weighted): {next_f1:.4f}")
    print(f"Concurrent subset     : {n_conc} / {len(test_data)} samples")
    if n_conc > 0:
        print(f"  DL similarity     : {conc_dl:.4f}")
        print(f"  TTNE MAE          : {conc_ttne:.2f} min")
        print(f"  RRT MAE           : {conc_rrt:.2f} min")
        print(f"  Next-act accuracy : {conc_acc:.4f}")
        print(f"  Next-act F1       : {conc_f1:.4f}")
    print(f"Training time         : {training_time:.1f}s")
    print(f"Testing time          : {testing_time:.1f}s")

    # ── Save results ──────────────────────────────────────────────────────────
    csv_path   = os.path.join(results_dir, 'results_suffix_time_gnn_v2_1.csv')
    fieldnames = ['log', 'model', 'method',
                  'dl_similarity', 'ttne_mae_minutes', 'rrt_mae_minutes',
                  'next_act_accuracy', 'next_act_f1_weighted',
                  'conc_n_samples',
                  'conc_dl_similarity', 'conc_ttne_mae_minutes', 'conc_rrt_mae_minutes',
                  'conc_next_act_accuracy', 'conc_next_act_f1_weighted',
                  'training_time_seconds', 'testing_time_seconds',
                  'num_trainable_params']
    new_row = {
        'log':                    log_name,
        'model':                  'suffix_time_stop',
        'method':                 METHOD_NAME,
        'dl_similarity':          round(dl_sim,       6),
        'ttne_mae_minutes':       round(ttne_mae_min, 6),
        'rrt_mae_minutes':        round(rrt_mae_min,  6),
        'next_act_accuracy':      round(next_acc,     6),
        'next_act_f1_weighted':      round(next_f1,      6),
        'conc_n_samples':         n_conc,
        'conc_dl_similarity':     (round(conc_dl,   6) if conc_dl   is not None else ''),
        'conc_ttne_mae_minutes':  (round(conc_ttne, 6) if conc_ttne is not None else ''),
        'conc_rrt_mae_minutes':   (round(conc_rrt,  6) if conc_rrt  is not None else ''),
        'conc_next_act_accuracy': (round(conc_acc,  6) if conc_acc  is not None else ''),
        'conc_next_act_f1_weighted': (round(conc_f1,   6) if conc_f1   is not None else ''),
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
        'dl_similarity':          dl_sim,
        'ttne_mae_minutes':       ttne_mae_min,
        'rrt_mae_minutes':        rrt_mae_min,
        'next_act_accuracy':      next_acc,
        'next_act_f1_weighted':      next_f1,
        'conc_n_samples':         n_conc,
        'conc_dl_similarity':     conc_dl,
        'conc_ttne_mae_minutes':  conc_ttne,
        'conc_rrt_mae_minutes':   conc_rrt,
        'conc_next_act_accuracy': conc_acc,
        'conc_next_act_f1_weighted': conc_f1,
    }


# ─── Entry point ──────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description='Train GRU+stop-head model; evaluate with next-act metrics and concurrent subset')
    p.add_argument('log_name',    help='Log name (must match results_per_log/<log_name>/)')
    p.add_argument('results_dir', nargs='?', default=None)
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    run(args.log_name, args.results_dir)
