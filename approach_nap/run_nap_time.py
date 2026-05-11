"""
Train and evaluate the GNN-based Next Activity + Timestamp Prediction model.

Usage
-----
    python run_nap_time.py <log_path> [log_name] [results_dir]
"""

import argparse
import csv
import os
import time

import torch
import torch.nn as nn

from approach_nap.data_pipeline_nap import build_nap_dataloaders
from approach_nap.model_nap_time import GNNNextActivityTime, train_epoch, evaluate

# ─────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────

HIDDEN_CHANNELS = 128
DROPOUT         = 0.3
LR              = 1e-3
WEIGHT_DECAY    = 1e-4
MAX_EPOCHS      = 100
PATIENCE        = 10
BATCH_SIZE      = 32
TRUNCATION      = 'none'
LAMBDA_TIME     = 1.0     # weight of time loss relative to activity loss

# ─────────────────────────────────────────────
# Main routine
# ─────────────────────────────────────────────

def run(log_path: str, log_name: str, results_dir: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Log    : {log_name}")
    print(f"Device : {device}")
    print(f"{'='*60}")

    # ── Data ─────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, activity_to_idx, *_ = \
        build_nap_dataloaders(log_path,
                              truncation_level=TRUNCATION,
                              batch_size=BATCH_SIZE)

    num_activities = len(activity_to_idx)

    # ── Model ─────────────────────────────────────────────────────────────
    emb_dim = min(128, max(8, 4 * int(num_activities ** 0.5)))
    print(f"Activities: {num_activities}  →  emb_dim: {emb_dim}")
    model = GNNNextActivityTime(
        num_activities  = num_activities,
        emb_dim         = emb_dim,
        hidden_channels = HIDDEN_CHANNELS,
        out_channels    = num_activities,
        dropout         = DROPOUT,
    ).to(device)

    optimizer      = torch.optim.Adam(model.parameters(), lr=LR,
                                      weight_decay=WEIGHT_DECAY)
    scheduler      = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6)
    criterion_act  = nn.CrossEntropyLoss()
    criterion_time = nn.L1Loss()

    print(f"\nModel parameters: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ── Training with early stopping ──────────────────────────────────────
    best_val_acc   = 0.0
    patience_count = 0
    best_state     = None
    train_start    = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer,
                                 criterion_act, criterion_time, device, LAMBDA_TIME)
        val_loss, val_metrics = evaluate(model, val_loader,
                                         criterion_act, criterion_time, device, LAMBDA_TIME)

        scheduler.step(val_metrics['accuracy'])
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch:3d}  "
              f"train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"val_acc={val_metrics['accuracy']:.4f}  "
              f"val_f1={val_metrics['f1']:.4f}  "
              f"val_time_mae={val_metrics['time_mae']:.4f}  "
              f"lr={current_lr:.2e}")

        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc   = val_metrics['accuracy']
            patience_count = 0
            best_state     = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
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
    _, test_metrics = evaluate(model, test_loader,
                               criterion_act, criterion_time, device, LAMBDA_TIME)
    testing_time = time.time() - test_start

    print(f"\n{'─'*60}")
    print(f"Test accuracy  : {test_metrics['accuracy']:.4f}")
    print(f"Test F1        : {test_metrics['f1']:.4f}")
    print(f"Test time MAE  : {test_metrics['time_mae']:.4f}")
    print(f"Training time  : {training_time:.1f}s")
    print(f"Testing time   : {testing_time:.1f}s")

    # ── Save results ──────────────────────────────────────────────────────
    os.makedirs(results_dir, exist_ok=True)
    csv_path   = os.path.join(results_dir, 'results_nap_gnn_time.csv')
    fieldnames = ['log', 'model', 'method', 'accuracy', 'f1', 'time_mae',
                  'training_time_seconds', 'testing_time_seconds']

    new_row = {
        'log':                    log_name,
        'model':                  'GNN_time',
        'method':                 'nap_time',
        'accuracy':               round(test_metrics['accuracy'], 6),
        'f1':                     round(test_metrics['f1'], 6),
        'time_mae':               round(test_metrics['time_mae'], 6),
        'training_time_seconds':  round(training_time, 2),
        'testing_time_seconds':   round(testing_time, 2),
    }

    rows = []
    if os.path.isfile(csv_path):
        with open(csv_path, newline='') as f:
            rows = list(csv.DictReader(f))

    updated = False
    for row in rows:
        if row['log'] == log_name and row['model'] == 'GNN_time':
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
    return test_metrics


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description='Train and evaluate GNN for NAP with timestamp prediction')
    parser.add_argument('log_path',    help='Path to event log (.xes or .csv)')
    parser.add_argument('log_name',    nargs='?', default=None,
                        help='Log name for CSV output (default: filename stem)')
    parser.add_argument('results_dir', nargs='?', default='results',
                        help='Output directory for CSV (default: results/)')
    return parser.parse_args()


if __name__ == '__main__':
    args     = _parse_args()
    log_name = args.log_name or os.path.splitext(
        os.path.basename(args.log_path))[0]
    run(args.log_path, log_name, args.results_dir)
