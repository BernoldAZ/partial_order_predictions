"""Pre-compute scaffold graph datasets from existing graph datasets.

For each split (train / val / test), builds W growing scaffold graphs
per sample:
  scaffold[t] = prefix graph + first t ground-truth suffix events

Output files (written to results_per_log/<log_name>/):
  train_scaffold_graphdataset.pt
  val_scaffold_graphdataset.pt
  test_scaffold_graphdataset.pt

Each file is a list of N items; item[i] is a list of W
Data(x, cat_x, edge_index, edge_attr) objects.

Usage
-----
    python create_scaffold_data.py <log_name> [results_dir]

Requires create_graph_data.py to have been run first so that
train/val/test_graphdataset.pt and the pkl stats files exist.
"""
import argparse
import csv
import os
import pickle
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed

from Preprocessing.scaffold_graph_builder import build_scaffold_dataset


def build_scaffold_data(log_name, results_dir=None):
    data_dir = results_dir or os.path.join('results_per_log', log_name)

    with open(os.path.join(data_dir, f'{log_name}_train_means_dict.pkl'), 'rb') as f:
        means = pickle.load(f)
    with open(os.path.join(data_dir, f'{log_name}_train_std_dict.pkl'), 'rb') as f:
        stds  = pickle.load(f)

    tss_mean = means['suffix_df'][0];  tss_std = stds['suffix_df'][0]
    tsp_mean = means['suffix_df'][1];  tsp_std = stds['suffix_df'][1]

    with open(os.path.join(data_dir, f'{log_name}_cardin_list_prefix.pkl'), 'rb') as f:
        pref_cat_cars = pickle.load(f)
    end_tok = pref_cat_cars[-1] + 2 - 1   # num_activities - 1

    for split in ('train', 'val', 'test'):
        src = os.path.join(data_dir, f'{split}_graphdataset.pt')
        dst = os.path.join(data_dir, f'{split}_scaffold_graphdataset.pt')
        dataset = torch.load(src, weights_only=False)
        print(f'Building {split} scaffold dataset ({len(dataset)} samples) ...')
        scaffold = build_scaffold_dataset(
            dataset, end_tok, tss_mean, tss_std, tsp_mean, tsp_std)
        torch.save(scaffold, dst)
        print(f'  Saved -> {dst}')


# ─────────────────────────────────────────────
# Batch runner (all logs in a folder)
# ─────────────────────────────────────────────

def _run_one_log(args):
    """Top-level worker for ProcessPoolExecutor: build scaffold for one log."""
    log_name, results_base = args
    try:
        build_scaffold_data(log_name, os.path.join(results_base, log_name))
        return {'log': log_name, 'error': ''}
    except Exception as exc:
        return {'log': log_name, 'error': str(exc)}


def run_all_logs(results_base='results_per_log', output_file='scaffold_summary.csv',
                 n_workers=None):
    """Run build_scaffold_data for every log directory under *results_base*.

    Parameters
    ----------
    results_base : str
        Directory containing per-log subdirectories (e.g. results_per_log/).
    output_file : str
        Path for the output CSV (created or overwritten).
    n_workers : int or None
        Number of parallel worker processes. None = os.cpu_count().
    """
    log_names = sorted(
        d for d in os.listdir(results_base)
        if os.path.isdir(os.path.join(results_base, d))
    )
    if not log_names:
        print(f"No log directories found in '{results_base}'.")
        return

    workers = min(n_workers or os.cpu_count(), len(log_names))
    args_list = [(name, results_base) for name in log_names]

    print(f"Processing {len(log_names)} logs with {workers} workers ...")
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['log', 'error'])
        writer.writeheader()

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_run_one_log, args): args[0] for args in args_list}
            for future in as_completed(futures):
                r = future.result()
                if r['error']:
                    print(f"  ERROR  {r['log']}: {r['error']}")
                else:
                    print(f"  {r['log']}  done")
                writer.writerow(r)
                f.flush()

    print(f"\nSummary written to '{output_file}'.")


def _parse_args():
    p = argparse.ArgumentParser(description='Pre-compute scaffold graph datasets')
    p.add_argument('log_name')
    p.add_argument('results_dir', nargs='?', default=None)
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    build_scaffold_data(args.log_name, args.results_dir)
