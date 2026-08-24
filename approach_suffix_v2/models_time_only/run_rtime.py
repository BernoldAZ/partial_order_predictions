"""Train and evaluate GATv2RemainingTime for remaining time prediction.

Usage
-----
    python run_rtime.py <log_name> [results_dir]

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
from torch_geometric.loader import DataLoader

from model_rtime import GATv2RemainingTime

# ─── Hyperparameters ──────────────────────────────────────────────────────────
D_MODEL     = 64
DROPOUT     = 0.4
LR          = 0.002
MAX_EPOCHS  = 200
PATIENCE    = 24
LR_PATIENCE = 10
MAX_NORM    = 2.0
BATCH_SIZE  = 128

METHOD_NAME = 'gatv2_rtime'


# ─── Loss ─────────────────────────────────────────────────────────────────────

def _loss(rtime_pred, rtime_target):
    return F.l1_loss(rtime_pred.view(-1), rtime_target.view(-1))


# ─── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def _evaluate(model, loader, device, window_size, mean_std_rrt):
    model.eval()
    rrt_mean, rrt_std = mean_std_rrt
    all_preds, all_labels = [], []

    for data in loader:
        data  = data.to(device)
        B     = data.num_graphs
        pred  = model(data)                                             # (B, 1)
        label = data.rtime_label.view(B, window_size)[:, 0].unsqueeze(-1)  # (B, 1)
        all_preds.append(pred.cpu())
        all_labels.append(label.cpu())

    preds  = torch.cat(all_preds,  dim=0)
    labels = torch.cat(all_labels, dim=0)

    pred_sec  = (preds  * rrt_std + rrt_mean).clamp(min=0)
    label_sec = (labels * rrt_std + rrt_mean).clamp(min=0)
    return torch.abs(pred_sec - label_sec).mean().item() / 60.0     # minutes


# ─── Main ─────────────────────────────────────────────────────────────────────

def run(log_name: str, results_dir: str = None, run_id: int = 0):
    torch.manual_seed(run_id)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_id)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    results_dir = results_dir or f'results_{METHOD_NAME}'
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

    mean_std_rrt = [means['timeLabel_df'][1], stds['timeLabel_df'][1]]

    with open(os.path.join(data_dir, f'{log_name}_cardin_list_prefix.pkl'), 'rb') as f:
        pref_cat_cars = pickle.load(f)
    window_size    = train_data[0].suffix_act.shape[0]
    num_activities = pref_cat_cars[-1] + 2

    print(f"num_activities={num_activities}  window_size={window_size}")

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False)

    # ── Build model ───────────────────────────────────────────────────────────
    model = GATv2RemainingTime(
        num_activities=num_activities,
        d_model=D_MODEL,
        dropout=DROPOUT,
    ).to(device)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {num_trainable_params:,}")

    optimizer    = torch.optim.NAdam(model.parameters(), lr=LR)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=LR_PATIENCE, threshold=1e-4, min_lr=0)

    os.makedirs(results_dir, exist_ok=True)
    best_model_path = os.path.join(results_dir, f'{log_name}_{METHOD_NAME}.pt')

    best_rrt_mae   = 1e9
    patience_count = 0
    train_start    = time.time()

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(MAX_EPOCHS):
        torch.manual_seed(epoch)
        model.train()
        total_loss, n_batches = 0.0, 0

        for data in train_loader:
            data  = data.to(device)
            B     = data.num_graphs
            pred  = model(data)                                             # (B, 1)
            label = data.rtime_label.view(B, window_size)[:, 0].unsqueeze(-1)  # (B, 1)

            loss = _loss(pred, label)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        train_loss = total_loss / max(n_batches, 1)
        rrt_mae    = _evaluate(model, val_loader, device, window_size, mean_std_rrt)

        lr_scheduler.step(rrt_mae)
        lr = optimizer.param_groups[0]['lr']
        print(f"[{log_name}] Epoch {epoch+1:4d}  loss={train_loss:.4f}  "
              f"RRT={rrt_mae:.2f}min  lr={lr:.2e}")

        if rrt_mae < best_rrt_mae:
            best_rrt_mae   = rrt_mae
            patience_count = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}.")
                break

    training_time = time.time() - train_start

    # ── Test ──────────────────────────────────────────────────────────────────
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    model.to(device)

    test_loader  = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    test_start   = time.time()
    rrt_mae_test = _evaluate(model, test_loader, device, window_size, mean_std_rrt)
    testing_time = time.time() - test_start

    print(f"\n{'─'*60}")
    print(f"RRT MAE        : {rrt_mae_test:.2f} min")
    print(f"Training time  : {training_time:.1f}s")
    print(f"Testing time   : {testing_time:.1f}s")

    # ── Save results ──────────────────────────────────────────────────────────
    csv_path   = os.path.join(results_dir, 'results_rtime.csv')
    fieldnames = ['log', 'model', 'method', 'rrt_mae_minutes',
                  'training_time_seconds', 'testing_time_seconds', 'num_trainable_params']
    new_row = {
        'log':                   log_name,
        'model':                 'gatv2_rtime',
        'method':                METHOD_NAME,
        'rrt_mae_minutes':       round(rrt_mae_test,  6),
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
    return {'rrt_mae_minutes': rrt_mae_test}


# ─── Entry point ──────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description='Train GATv2 remaining-time model')
    p.add_argument('log_name',    help='Log name (must match results_per_log/<log_name>/)')
    p.add_argument('results_dir', nargs='?', default=None)
    p.add_argument('--run_id',    type=int, default=0)
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    run(args.log_name, args.results_dir, args.run_id)
