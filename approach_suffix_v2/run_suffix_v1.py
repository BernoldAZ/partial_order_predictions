"""Train and evaluate GATv2EncoderGRUDecoderSuffix for activity suffix prediction only
(no time/TTNE prediction).

Usage
-----
    python run_suffix_v1.py <log_name> [results_dir]

Loads pre-built datasets from results_per_log/<log_name>/:
    train_graphdataset.pt, val_graphdataset.pt, test_graphdataset.pt

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
from torch_geometric.loader import DataLoader

from model_suffix_v1 import GATv2EncoderGRUDecoderSuffix

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


# ─── Loss ─────────────────────────────────────────────────────────────────────

def _loss(act_logits, data, num_activities):
    return F.cross_entropy(
        act_logits.view(-1, num_activities),
        data.act_label_seq.view(-1),
        ignore_index=0,
    )


# ─── Metrics ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def _compute_metrics(suffix_acts, act_labels, num_activities):
    device  = suffix_acts.device
    N, W    = suffix_acts.shape
    end_tok = num_activities - 1

    actual_length = (act_labels == end_tok).to(torch.int64).argmax(dim=-1)

    has_end     = (suffix_acts == end_tok).any(dim=-1)
    first_end   = (suffix_acts == end_tok).to(torch.int64).argmax(dim=-1)
    pred_length = torch.where(has_end, first_end, torch.full_like(first_end, W - 1))

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

    mean_pred_len   = (pred_length   + 1).float().mean().item()
    mean_actual_len = (actual_length + 1).float().mean().item()
    return dl_sim, mean_pred_len, mean_actual_len


# ─── Evaluation pass ──────────────────────────────────────────────────────────

@torch.no_grad()
def _evaluate(model, loader, device, num_activities, window_size):
    model.eval()
    all_acts, all_lbl_acts = [], []

    for data in loader:
        data = data.to(device)
        B    = data.num_graphs
        acts = model(data, window_size=window_size)
        all_acts.append(acts.cpu())
        all_lbl_acts.append(data.act_label_seq.view(B, window_size).cpu())

    sa = torch.cat(all_acts,     dim=0)
    la = torch.cat(all_lbl_acts, dim=0)

    return _compute_metrics(sa, la, num_activities)


# ─── Main ─────────────────────────────────────────────────────────────────────

def run(log_name: str, results_dir: str = None, model_type: str = 'gatv2_gru'):
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    results_dir = results_dir or f'results_suffix_{model_type}'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}\nLog : {log_name}\nDevice : {device}\n{'='*60}")

    # ── Load pre-built datasets ───────────────────────────────────────────────
    data_dir   = os.path.join("approach_suffix_v2", 'results_per_log', log_name)
    train_data = torch.load(os.path.join(data_dir, 'train_graphdataset.pt'), weights_only=False)
    val_data   = torch.load(os.path.join(data_dir, 'val_graphdataset.pt'),   weights_only=False)
    test_data  = torch.load(os.path.join(data_dir, 'test_graphdataset.pt'),  weights_only=False)

    with open(os.path.join(data_dir, f'{log_name}_cardin_list_prefix.pkl'), 'rb') as f:
        pref_cat_cars = pickle.load(f)
    window_size    = train_data[0].suffix_act.shape[0]
    num_activities = pref_cat_cars[-1] + 2

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False)

    # ── Build model ───────────────────────────────────────────────────────────
    model = GATv2EncoderGRUDecoderSuffix(
        num_activities=num_activities,
        d_model=D_MODEL,
        dropout=DROPOUT,
        n_layers=N_LAYERS,
        use_scheduled_sampling=USE_SCHEDULED_SAMPLING,
    ).to(device)
    print(f"num_activities={num_activities}  window_size={window_size}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer    = torch.optim.NAdam(model.parameters(), lr=LR)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=LR_PATIENCE, threshold=1e-4, min_lr=0)

    os.makedirs(results_dir, exist_ok=True)
    best_model_path = os.path.join(results_dir, f'{log_name}_{model_type}.pt')

    best_dl_sim    = -1.0
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
            act_logits = model(data, p_teacher=p_teacher)
            loss = _loss(act_logits, data, num_activities)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        train_loss = total_loss / max(n_batches, 1)

        dl_sim, mean_pred_len, mean_actual_len = _evaluate(
            model, val_loader, device, num_activities, window_size)

        lr_scheduler.step(1.0 - dl_sim)
        lr = optimizer.param_groups[0]['lr']
        ss_info = f"  p_teacher={p_teacher:.3f}" if USE_SCHEDULED_SAMPLING else ""
        print(f"[{log_name}] Epoch {epoch+1:4d}  loss={train_loss:.4f}  "
              f"DL={dl_sim:.4f}  lr={lr:.2e}  "
              f"len_pred={mean_pred_len:.1f}  len_gt={mean_actual_len:.1f}{ss_info}")

        if dl_sim > best_dl_sim:
            torch.save(model.state_dict(), best_model_path)
            best_dl_sim   = dl_sim
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
    dl_sim, _, _ = _evaluate(model, test_loader, device, num_activities, window_size)
    testing_time = time.time() - test_start

    print(f"\n{'─'*60}")
    print(f"DL similarity : {dl_sim:.4f}")
    print(f"Training time : {training_time:.1f}s")
    print(f"Testing time  : {testing_time:.1f}s")

    # ── Save results ──────────────────────────────────────────────────────────
    csv_path   = os.path.join(results_dir, 'results_suffix_gnn.csv')
    fieldnames = ['log', 'model', 'method',
                  'dl_similarity',
                  'training_time_seconds', 'testing_time_seconds']
    new_row = {
        'log':                   log_name,
        'model':                 'suffix',
        'method':                model_type,
        'dl_similarity':         round(dl_sim,        6),
        'training_time_seconds': round(training_time, 2),
        'testing_time_seconds':  round(testing_time,  2),
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
            if row['log'] == log_name and row['method'] == model_type:
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
    return {'dl_similarity': dl_sim}


# ─── Entry point ──────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description='Train and evaluate GNN suffix-only prediction model')
    p.add_argument('log_name',    help='Log name (must match results_per_log/<log_name>/)')
    p.add_argument('results_dir', nargs='?', default=None)
    p.add_argument('--model', choices=['gatv2_gru'], default='gatv2_gru')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    run(args.log_name, args.results_dir, model_type=args.model)
