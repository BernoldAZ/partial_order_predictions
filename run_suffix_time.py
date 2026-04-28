"""
Train and evaluate GNN encoder + LSTM decoder for Activity Suffix + Time Prediction.

Reports the same activity metrics as run_suffix.py plus:
  MAE TTNE (min) — mean absolute error on time till next event
  MAE RRT  (min) — mean absolute error on remaining runtime

Usage
-----
    python run_suffix_time.py <log_path> [log_name] [results_dir]
"""

import argparse
import csv
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

from data_pipeline_suffix import build_suffix_dataloaders
from model_suffix_time import GNNSuffixTimeModel, train_epoch, evaluate

# ─────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────

SEED            = 42
HIDDEN_CHANNELS = 128
EMB_DIM         = 64
LSTM_HIDDEN     = 128
DROPOUT         = 0.3
LR              = 1e-3
WEIGHT_DECAY    = 1e-4
MAX_EPOCHS      = 100
PATIENCE        = 10
BATCH_SIZE      = 32
TRUNCATION      = 'none'
LAMBDA_TTNE     = 1.0   # weight of TTNE L1 loss relative to CE
LAMBDA_RRT      = 1.0   # weight of RRT  L1 loss relative to CE


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _compute_time_norms(dataset):
    """Compute mean TTNE and mean RRT (seconds) from a graph dataset."""
    ttne_sum = rrt_sum = 0.0
    n = len(dataset)
    for g in dataset:
        ttne_sum += g.ttne.item()
        rrt_sum  += g.rrt.item()
    # Guard against degenerate logs where all values are 0
    mean_ttne = max(ttne_sum / n, 1.0)
    mean_rrt  = max(rrt_sum  / n, 1.0)
    return mean_ttne, mean_rrt


# ─────────────────────────────────────────────
# Main routine
# ─────────────────────────────────────────────

def run(log_path: str, log_name: str, results_dir: str):
    _set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Log    : {log_name}")
    print(f"Device : {device}")
    print(f"{'='*60}")

    # ── Data ─────────────────────────────────────────────────────────────
    (train_loader, val_loader, test_loader,
     activity_to_idx, end_token_idx, max_suffix_len) = \
        build_suffix_dataloaders(log_path,
                                 truncation_level=TRUNCATION,
                                 batch_size=BATCH_SIZE)

    # Normalisation constants from training set (computed once, before training)
    mean_ttne, mean_rrt = _compute_time_norms(train_loader.dataset)
    print(f"Time norms  — mean TTNE: {mean_ttne/60:.1f} min  "
          f"mean RRT: {mean_rrt/60:.1f} min")

    num_activities = len(activity_to_idx)

    # ── Model ─────────────────────────────────────────────────────────────
    model = GNNSuffixTimeModel(
        in_channels     = num_activities,
        hidden_channels = HIDDEN_CHANNELS,
        num_activities  = num_activities,
        emb_dim         = EMB_DIM,
        lstm_hidden     = LSTM_HIDDEN,
        dropout         = DROPOUT,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    print(f"Parameters : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Vocab size : {num_activities + 2}  |  max_suffix_len: {max_suffix_len}")

    # ── Training ──────────────────────────────────────────────────────────
    best_val_dls   = -1.0
    patience_count = 0
    best_state     = None
    train_start    = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device,
                                 mean_ttne, mean_rrt, LAMBDA_TTNE, LAMBDA_RRT)
        val_loss, val_m = evaluate(model, val_loader, criterion, device,
                                   end_token_idx, max_suffix_len,
                                   mean_ttne, mean_rrt)

        print(f"Epoch {epoch:3d}  "
              f"train={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"acc={val_m['activity_accuracy']:.4f}  "
              f"dls={val_m['mean_dls']:.4f}  "
              f"ttne={val_m['mae_ttne_minutes']:.2f}m  "
              f"rrt={val_m['mae_rrt_minutes']:.2f}m")

        if val_m['mean_dls'] > best_val_dls:
            best_val_dls   = val_m['mean_dls']
            patience_count = 0
            best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"Early stopping at epoch {epoch}.")
                break

    training_time = time.time() - train_start

    # ── Test evaluation ───────────────────────────────────────────────────
    model.load_state_dict(best_state)
    model.to(device)

    test_start = time.time()
    _, test_m  = evaluate(model, test_loader, criterion, device,
                          end_token_idx, max_suffix_len,
                          mean_ttne, mean_rrt)
    testing_time = time.time() - test_start

    print(f"\n{'─'*60}")
    print(f"Test activity accuracy : {test_m['activity_accuracy']:.4f}")
    print(f"Test mean DLS          : {test_m['mean_dls']:.4f}")
    print(f"Test MAE TTNE          : {test_m['mae_ttne_minutes']:.2f} min")
    print(f"Test MAE RRT           : {test_m['mae_rrt_minutes']:.2f} min")
    print(f"Training time          : {training_time:.1f}s")
    print(f"Testing time           : {testing_time:.1f}s")

    # ── Save results ──────────────────────────────────────────────────────
    os.makedirs(results_dir, exist_ok=True)
    csv_path   = os.path.join(results_dir, 'results_suffix_gnn_time.csv')
    fieldnames = ['log', 'model', 'activity_accuracy', 'mean_dls',
                  'mae_ttne_minutes', 'mae_rrt_minutes',
                  'training_time_seconds', 'testing_time_seconds']

    new_row = {
        'log':                    log_name,
        'model':                  'GNN_Suffix_Time',
        'activity_accuracy':      round(test_m['activity_accuracy'],  6),
        'mean_dls':               round(test_m['mean_dls'],           6),
        'mae_ttne_minutes':       round(test_m['mae_ttne_minutes'],   4),
        'mae_rrt_minutes':        round(test_m['mae_rrt_minutes'],    4),
        'training_time_seconds':  round(training_time, 2),
        'testing_time_seconds':   round(testing_time,  2),
    }

    rows = []
    if os.path.isfile(csv_path):
        with open(csv_path, newline='') as f:
            rows = list(csv.DictReader(f))

    updated = False
    for row in rows:
        if row['log'] == log_name and row['model'] == 'GNN_Suffix_Time':
            row.update(new_row)
            updated = True
            break
    if not updated:
        rows.append(new_row)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Results saved → {csv_path}")
    return test_m


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description='GNN+LSTM suffix prediction with time targets (TTNE & RRT)')
    parser.add_argument('log_path',    help='Path to event log (.xes or .csv)')
    parser.add_argument('log_name',    nargs='?', default=None)
    parser.add_argument('results_dir', nargs='?', default='results')
    return parser.parse_args()


if __name__ == '__main__':
    args     = _parse_args()
    log_name = args.log_name or os.path.splitext(os.path.basename(args.log_path))[0]
    run(args.log_path, log_name, args.results_dir)
