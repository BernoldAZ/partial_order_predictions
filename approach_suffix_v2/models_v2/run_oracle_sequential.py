"""Oracle-Sequential reference point: predicts every activity label of the
test suffix correctly, in log order, but places every event in its own block
(a total order / chain graph). Upper bound for any approach predicting a
totally ordered suffix without concurrency. No training needed.

Usage (from the project root)
-----
    python approach_suffix_v2/models_v2/run_oracle_sequential.py

Reads results_per_log/<log_name>/test_graphdataset.pt (same test set as
run_suffix_v2.py) and writes one row per log to
results_oracle_sequential/run_1/results_suffix_time_gnn.csv, plus the
per-length dicts <log_name>_{prefix,suffix}_length_results_dict.pkl.

Docker
------
    docker run -it --rm -v $(pwd):/workspace ml-jupyter-gpu python approach_suffix_v2/models_v2/run_oracle_sequential.py
"""
import csv
import os
import pickle
from concurrent.futures import ProcessPoolExecutor

import torch

from partial_order_metrics import ges_exact_compute_sample, bucket_order_similarity

EVENT_LOGS = [
    "Sepsis",
    "BPI_Challenge_2012_A",
    "BPI_Challenge_2012_O",
    "BPIC15_1",
    "BPIC15_2",
    "BPIC15_3",
    "BPIC15_4",
    "BPIC15_5",
]

METHOD_NAME = 'oracle_sequential'
RESULTS_DIR = os.path.join("approach_suffix_v2", 'results_oracle_sequential', 'run_1')
GES_WORKERS = 4


def _score_sample(args):
    return (ges_exact_compute_sample(args), *bucket_order_similarity(*args))


def run(log_name, pool):
    data_dir  = os.path.join("approach_suffix_v2", 'results_per_log', log_name)
    test_data = torch.load(os.path.join(data_dir, 'test_graphdataset.pt'), weights_only=False)
    with open(os.path.join(data_dir, f'{log_name}_cardin_list_prefix.pkl'), 'rb') as f:
        pref_cat_cars = pickle.load(f)
    end_tok = pref_cat_cars[-1] + 1   # num_activities - 1

    args_list, pref_lens, suf_lens = [], [], []
    for s in test_data:
        L    = int((s.act_label_seq == end_tok).to(torch.int64).argmax())
        acts = s.act_label_seq[:L].tolist()
        args_list.append((acts, [True] * L, acts, (s.new_block_label[:L] > 0.5).tolist()))
        pref_lens.append(int(s.num_nodes))
        suf_lens.append(L)

    ges, bo_order, bo_content, bo_comb = zip(*pool.map(_score_sample, args_list, chunksize=4))

    # Per-length dicts, same layout and file names as run_suffix_v2.py:
    # [dl, rrt, count, ges, bo_order, bo_content, bo_combined]
    for lens, tag in ((pref_lens, 'prefix'), (suf_lens, 'suffix')):
        groups = {}
        for length, *scores in zip(lens, ges, bo_order, bo_content, bo_comb):
            groups.setdefault(length, []).append(scores)
        len_dict = {k: [1.0, float('nan'), len(v),
                        *(sum(col) / len(v) for col in zip(*v))]
                    for k, v in groups.items()}
        with open(os.path.join(RESULTS_DIR, f'{log_name}_{tag}_length_results_dict.pkl'), 'wb') as f:
            pickle.dump(len_dict, f)
    row = {
        'log':           log_name,
        'method':        METHOD_NAME,
        'dl_similarity': 1.0,
        'ges_exact':     round(sum(ges)        / len(ges),        6),
        'bo_order_sim':  round(sum(bo_order)   / len(bo_order),   6),
        'bo_content_f1': round(sum(bo_content) / len(bo_content), 6),
        'bo_combined':   round(sum(bo_comb)    / len(bo_comb),    6),
    }
    print(row, flush=True)
    return row


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with ProcessPoolExecutor(max_workers=GES_WORKERS) as pool:
        rows = [run(log_name, pool) for log_name in EVENT_LOGS]
    csv_path = os.path.join(RESULTS_DIR, 'results_suffix_time_gnn.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results saved → {csv_path}")
