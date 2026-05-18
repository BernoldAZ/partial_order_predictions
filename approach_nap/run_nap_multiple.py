"""
Train and evaluate the GNN-based Multiple-Activity Prediction model.

Predicts the FULL SET of activities that occur at the next concurrent
timestamp block (multi-label), rather than a single next activity.

Results CSV columns:
  log, model, method, exact_match, f1_set,
  training_time_seconds, testing_time_seconds

Usage
-----
    python run_nap_multiple.py <log_path> [log_name] [results_dir]

Examples
--------
    python run_nap_multiple.py data/BPIC17.xes BPIC17
    python run_nap_multiple.py data/BPIC17.xes BPIC17 my_results/
"""

import argparse
import csv
import os
import time

import torch
import torch.nn as nn

from approach_nap.data_pipeline_nap_multiple import build_multiple_dataloaders
from approach_nap.model_nap_multiple import GNNMultipleActivity, train_epoch, evaluate

# ─────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────

HIDDEN_CHANNELS = 128
DROPOUT         = 0.3
LR              = 1e-3
WEIGHT_DECAY    = 1e-4
MAX_EPOCHS      = 200
PATIENCE        = 24          # early-stopping patience
LR_PATIENCE     = 10           # ReduceLROnPlateau patience
BATCH_SIZE      = 32
TRUNCATION      = 'none'

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
        build_multiple_dataloaders(log_path,
                                   truncation_level=TRUNCATION,
                                   batch_size=BATCH_SIZE)

    num_activities = len(activity_to_idx)

    # ── Model ─────────────────────────────────────────────────────────────
    emb_dim = min(128, max(8, 4 * int(num_activities ** 0.5)))
    print(f"Activities: {num_activities}  →  emb_dim: {emb_dim}")
    model = GNNMultipleActivity(
        num_activities  = num_activities,
        emb_dim         = emb_dim,
        hidden_channels = HIDDEN_CHANNELS,
        out_channels    = num_activities,
        dropout         = DROPOUT,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR,
                                 weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=LR_PATIENCE, min_lr=1e-6)
    criterion = nn.BCEWithLogitsLoss()

    print(f"\nModel parameters: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ── Training with early stopping ──────────────────────────────────────
    os.makedirs(results_dir, exist_ok=True)
    best_model_path = os.path.join(results_dir, f'{log_name}_model.pt')

    best_val_exact = 0.0
    patience_count = 0
    train_start    = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss            = train_epoch(model, train_loader, optimizer,
                                            criterion, device)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_metrics['exact_match'])
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch:3d}  "
              f"train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"val_exact={val_metrics['exact_match']:.4f}  "
              f"val_f1_set={val_metrics['f1_set']:.4f}  "
              f"lr={current_lr:.2e}")

        if val_metrics['exact_match'] > best_val_exact:
            best_val_exact = val_metrics['exact_match']
            patience_count = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"Early stopping at epoch {epoch}.")
                break

    training_time = time.time() - train_start

    # ── Test evaluation ───────────────────────────────────────────────────
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    model.to(device)

    test_start                = time.time()
    _, test_metrics           = evaluate(model, test_loader, criterion, device)
    testing_time              = time.time() - test_start

    n_single = test_metrics['n_single']
    n_multi  = test_metrics['n_multi']

    print(f"\n{'─'*60}")
    print(f"Test exact match (all,    n={n_single+n_multi}) : "
          f"{test_metrics['exact_match']:.4f}")
    print(f"Test F1 set      (all           ) : "
          f"{test_metrics['f1_set']:.4f}")
    print(f"Test exact match (single, n={n_single:>5}) : "
          f"{test_metrics['exact_match_single']:.4f}")
    print(f"Test F1 set      (single        ) : "
          f"{test_metrics['f1_set_single']:.4f}")
    print(f"Test exact match (multi,  n={n_multi:>5}) : "
          f"{test_metrics['exact_match_multi']:.4f}")
    print(f"Test F1 set      (multi         ) : "
          f"{test_metrics['f1_set_multi']:.4f}")
    print(f"Training time    : {training_time:.1f}s")
    print(f"Testing time     : {testing_time:.1f}s")

    # ── Save results ──────────────────────────────────────────────────────
    csv_path   = os.path.join(results_dir, 'results_nap_multiple_gnn.csv')
    fieldnames = [
        'log', 'model', 'method',
        'exact_match', 'f1_set',
        'exact_match_single', 'f1_set_single', 'n_single',
        'exact_match_multi',  'f1_set_multi',  'n_multi',
        'training_time_seconds', 'testing_time_seconds',
    ]

    new_row = {
        'log':                   log_name,
        'model':                 'GNN',
        'method':                'nap_multiple',
        'exact_match':           round(test_metrics['exact_match'], 6),
        'f1_set':                round(test_metrics['f1_set'], 6),
        'exact_match_single':    round(test_metrics['exact_match_single'], 6),
        'f1_set_single':         round(test_metrics['f1_set_single'], 6),
        'n_single':              n_single,
        'exact_match_multi':     round(test_metrics['exact_match_multi'], 6),
        'f1_set_multi':          round(test_metrics['f1_set_multi'], 6),
        'n_multi':               n_multi,
        'training_time_seconds': round(training_time, 2),
        'testing_time_seconds':  round(testing_time, 2),
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
            if row['log'] == log_name and row['model'] == 'GNN':
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
    return test_metrics


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description='Train and evaluate GNN for multiple-activity prediction')
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
