"""GES / bucket-order evaluation of the TTNE-predicting baselines with
concurrent blocks derived from the predicted TTNE.

Suffix event j >= 1 joins the block of event j-1 when its predicted TTNE (in
seconds) is <= X, otherwise it opens a new block. X is rate-matched per log,
model and run: q = fraction of concurrent events in the train suffixes, and
X = q-quantile of the model's predicted test TTNEs (no test labels used).

Run from the baselines/ folder (relative results_per_log/ paths).

python3 eval_ttne_partial_order.py --workers 10 --chunksize 1

docker run -it --rm -v $(pwd):/app ppm-sutran-best python3 eval_ttne_partial_order.py --workers 20 --chunksize 1
"""

import argparse
import glob
import os
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from partial_order_metrics import ges_exact_compute_sample, bucket_order_similarity
from run_all_suffix_baselines import EVENT_LOGS

MODELS = {
    "SEP-LSTM":     "SEP_LSTM_results",
    "ED-LSTM":      "ED_LSTM_results",
    "SuTraN (DA)":  "SUTRAN_DA_results",
    "SuTraN (NDA)": "SUTRAN_NDA_results",
}
SUMMARY_CSV = "po_ttne_summary.csv"


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def end_positions(acts, end_tok):
    """Index of the first END token per row (row width if absent)."""
    is_end = acts == end_tok
    return torch.where(is_end.any(dim=1), is_end.int().argmax(dim=1),
                       torch.full((acts.shape[0],), acts.shape[1])).tolist()


def train_concurrency_rate(log_dir, end_tok):
    act_lbl = torch.load(os.path.join(log_dir, 'train_tensordataset.pt'), weights_only=True)[-1]
    nb = torch.load(os.path.join(log_dir, 'train_new_block_labels.pt'), weights_only=True)
    n_conc, n_tot = 0, 0
    for i, L in enumerate(end_positions(act_lbl, end_tok)):
        n_conc += int((nb[i, 1:L] < 0.5).sum())
        n_tot += max(L - 1, 0)
    return n_conc / n_tot if n_tot else 0.0


def _score_sample(args):
    """Worker: exact GES and bucket-order (order, content, combined) of one sample."""
    return (ges_exact_compute_sample(args), *bucket_order_similarity(*args))


def evaluate_run(pool, chunksize, desc, res_path, mean_std_ttne, end_tok, q, nb_labels):
    acts = torch.load(os.path.join(res_path, 'suffix_acts_decoded.pt'), weights_only=True)
    ttne = torch.load(os.path.join(res_path, 'suffix_ttne_preds.pt'), weights_only=True)
    act_lbl = torch.load(os.path.join(res_path, 'labels.pt'), weights_only=True)[-1]
    sec = torch.clamp(ttne * mean_std_ttne[1] + mean_std_ttne[0], min=0).numpy()

    pred_lens = end_positions(acts, end_tok)
    lbl_lens = end_positions(act_lbl, end_tok)

    pooled = np.concatenate([sec[i, 1:pl] for i, pl in enumerate(pred_lens)])
    X = float(np.quantile(pooled, q)) if q > 0 and len(pooled) else -np.inf  # -inf: no merging

    po_args, n_conc = [], 0
    for i, (pl, al) in enumerate(zip(pred_lens, lbl_lens)):
        nb_pred = [True] + (sec[i, 1:pl] > X).tolist()
        n_conc += pl - sum(nb_pred[:pl])
        po_args.append((acts[i, :pl].tolist(), nb_pred[:pl],
                        act_lbl[i, :al].tolist(),
                        (nb_labels[i, :al] > 0.5).tolist()))

    ges, bo_order, bo_content, bo_comb = (
        list(v) for v in zip(*tqdm(pool.map(_score_sample, po_args, chunksize=chunksize),
                                   total=len(po_args), desc=desc)))

    return {
        'X': X,
        'q': q,
        'frac_concurrent_pred': n_conc / len(pooled) if len(pooled) else 0.0,
        'ges_exact':     float(np.mean(ges)),
        'bo_order_sim':  float(np.mean(bo_order)),
        'bo_content_f1': float(np.mean(bo_content)),
        'bo_combined':   float(np.mean(bo_comb)),
        'per_sample': {
            'ges':         torch.tensor(ges,        dtype=torch.float32),
            'bo_order':    torch.tensor(bo_order,   dtype=torch.float32),
            'bo_content':  torch.tensor(bo_content, dtype=torch.float32),
            'bo_combined': torch.tensor(bo_comb,    dtype=torch.float32),
        },
    }


def main(workers, chunksize):
    rows = []
    pool = ProcessPoolExecutor(max_workers=workers)

    # Per-log context, loaded once (q, TTNE de-standardisation, END token, test labels)
    log_ctx = {}
    for log_name in EVENT_LOGS:
        log_dir = os.path.join('results_per_log', log_name)
        if not os.path.isfile(os.path.join(log_dir, 'test_new_block_labels.pt')):
            continue
        means = load_pkl(os.path.join(log_dir, log_name + '_train_means_dict.pkl'))
        stds = load_pkl(os.path.join(log_dir, log_name + '_train_std_dict.pkl'))
        end_tok = load_pkl(os.path.join(log_dir, log_name + '_cardin_dict.pkl'))['concept:name'] + 1
        log_ctx[log_name] = dict(
            log_dir=log_dir,
            mean_std_ttne=[means['timeLabel_df'][0], stds['timeLabel_df'][0]],
            end_tok=end_tok,
            q=train_concurrency_rate(log_dir, end_tok),
            nb_labels=torch.load(os.path.join(log_dir, 'test_new_block_labels.pt'), weights_only=True),
        )

    run_ids = sorted({int(p.rsplit('_run', 1)[-1])
                      for c in log_ctx.values() for dir_base in MODELS.values()
                      for p in glob.glob(os.path.join(c['log_dir'], f"{dir_base}_run*"))})

    # Order: run -> model -> log
    for run_id in run_ids:
        for model_name, dir_base in MODELS.items():
            for log_name, c in log_ctx.items():
                res_path = os.path.join(c['log_dir'], f"{dir_base}_run{run_id}", 'TEST_SET_RESULTS')
                if not os.path.isfile(os.path.join(res_path, 'suffix_ttne_preds.pt')):
                    continue
                q = c['q']
                po_path = os.path.join(res_path, 'po_ttne_results.pkl')
                if os.path.isfile(po_path):
                    res = load_pkl(po_path)  # already computed: reuse for the summary
                    print(f"{log_name} | {model_name} | run {run_id} | already done, skipped")
                else:
                    res = evaluate_run(pool, chunksize, f"{log_name} | {model_name} | run {run_id}",
                                       res_path, c['mean_std_ttne'], c['end_tok'], q, c['nb_labels'])
                    with open(po_path, 'wb') as f:
                        pickle.dump(res, f)
                print(f"{log_name} | {model_name} | run {run_id} | X={res['X']:.1f}s "
                      f"q={q:.3f} GES={res['ges_exact']:.4f} BOS={res['bo_combined']:.4f}")
                rows.append({'log': log_name, 'model': model_name, 'run': run_id,
                             **{k: v for k, v in res.items() if k != 'per_sample'}})

    pool.shutdown()
    pd.DataFrame(rows).to_csv(SUMMARY_CSV, index=False)
    print(f"Summary written to {SUMMARY_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                        help="processes scoring samples in parallel (default: all cores)")
    parser.add_argument("--chunksize", type=int, default=4,
                        help="samples sent to a worker per task (default: 4)")
    args = parser.parse_args()
    main(args.workers, args.chunksize)


