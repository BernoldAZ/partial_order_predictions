"""GES / bucket-order evaluation of the TTNE-predicting baselines with a fixed
threshold X = 0.0, compared against the rate-matched threshold (theta) of
eval_ttne_partial_order.py.

Suffix event j >= 1 joins the block of event j-1 when its predicted TTNE (in
seconds, clamped at 0) is <= 0.0, otherwise it opens a new block.

Run from the baselines/ folder (relative results_per_log/ paths), after
eval_ttne_partial_order.py so the theta results (po_ttne_results.pkl) exist.

python3 eval_ttne_partial_order_zero.py --workers 10 --chunksize 1

docker run -it --rm -v $(pwd):/app ppm-sutran-best python3 eval_ttne_partial_order_zero.py --workers 20 --chunksize 1
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

from eval_ttne_partial_order import MODELS, load_pkl, end_positions, _score_sample
from run_all_suffix_baselines import EVENT_LOGS

X_ZERO = 0.0
COMPARE_CSV = "po_ttne_zero_vs_theta.csv"


def evaluate_run_zero(pool, chunksize, desc, res_path, mean_std_ttne, end_tok, nb_labels):
    acts = torch.load(os.path.join(res_path, 'suffix_acts_decoded.pt'), weights_only=True)
    ttne = torch.load(os.path.join(res_path, 'suffix_ttne_preds.pt'), weights_only=True)
    act_lbl = torch.load(os.path.join(res_path, 'labels.pt'), weights_only=True)[-1]
    sec = torch.clamp(ttne * mean_std_ttne[1] + mean_std_ttne[0], min=0).numpy()

    pred_lens = end_positions(acts, end_tok)
    lbl_lens = end_positions(act_lbl, end_tok)
    n_pooled = sum(max(pl - 1, 0) for pl in pred_lens)

    po_args, n_conc = [], 0
    for i, (pl, al) in enumerate(zip(pred_lens, lbl_lens)):
        nb_pred = [True] + (sec[i, 1:pl] > X_ZERO).tolist()
        n_conc += pl - sum(nb_pred[:pl])
        po_args.append((acts[i, :pl].tolist(), nb_pred[:pl],
                        act_lbl[i, :al].tolist(),
                        (nb_labels[i, :al] > 0.5).tolist()))

    ges, bo_order, bo_content, bo_comb = (
        list(v) for v in zip(*tqdm(pool.map(_score_sample, po_args, chunksize=chunksize),
                                   total=len(po_args), desc=desc)))

    return {
        'X': X_ZERO,
        'frac_concurrent_pred': n_conc / n_pooled if n_pooled else 0.0,
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


def win_tie_loss(zero, theta):
    """Fraction of samples where zero beats / ties / loses to theta."""
    d = zero - theta
    return float((d > 1e-9).float().mean()), float((d.abs() <= 1e-9).float().mean()), \
        float((d < -1e-9).float().mean())


def main(workers, chunksize):
    rows = []
    pool = ProcessPoolExecutor(max_workers=workers)

    # Per-log context, loaded once (TTNE de-standardisation, END token, test labels)
    log_ctx = {}
    for log_name in EVENT_LOGS:
        log_dir = os.path.join('results_per_log', log_name)
        if not os.path.isfile(os.path.join(log_dir, 'test_new_block_labels.pt')):
            continue
        means = load_pkl(os.path.join(log_dir, log_name + '_train_means_dict.pkl'))
        stds = load_pkl(os.path.join(log_dir, log_name + '_train_std_dict.pkl'))
        log_ctx[log_name] = dict(
            log_dir=log_dir,
            mean_std_ttne=[means['timeLabel_df'][0], stds['timeLabel_df'][0]],
            end_tok=load_pkl(os.path.join(log_dir, log_name + '_cardin_dict.pkl'))['concept:name'] + 1,
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
                desc = f"{log_name} | {model_name} | run {run_id}"
                zero_path = os.path.join(res_path, 'po_ttne_zero_results.pkl')
                if os.path.isfile(zero_path):
                    zero = load_pkl(zero_path)  # already computed: reuse for the comparison
                    print(f"{desc} | already done, skipped")
                else:
                    zero = evaluate_run_zero(pool, chunksize, desc, res_path,
                                             c['mean_std_ttne'], c['end_tok'], c['nb_labels'])
                    with open(zero_path, 'wb') as f:
                        pickle.dump(zero, f)

                row = {'log': log_name, 'model': model_name, 'run': run_id,
                       'frac_conc_zero': zero['frac_concurrent_pred'],
                       'ges_zero': zero['ges_exact'], 'bos_zero': zero['bo_combined']}

                theta_path = os.path.join(res_path, 'po_ttne_results.pkl')
                if os.path.isfile(theta_path):
                    theta = load_pkl(theta_path)
                    row.update({
                        'X_theta': theta['X'],
                        'frac_conc_theta': theta['frac_concurrent_pred'],
                        'ges_theta': theta['ges_exact'], 'bos_theta': theta['bo_combined'],
                        'ges_diff': zero['ges_exact'] - theta['ges_exact'],
                        'bos_diff': zero['bo_combined'] - theta['bo_combined'],
                    })
                    for key, ps in (('ges', 'ges'), ('bos', 'bo_combined')):
                        w, t, l = win_tie_loss(zero['per_sample'][ps], theta['per_sample'][ps])
                        row.update({f'{key}_win': w, f'{key}_tie': t, f'{key}_loss': l})
                    print(f"{desc} | GES zero={row['ges_zero']:.4f} theta={row['ges_theta']:.4f} "
                          f"| BOS zero={row['bos_zero']:.4f} theta={row['bos_theta']:.4f}")
                else:
                    print(f"{desc} | no theta results (po_ttne_results.pkl) to compare against")
                rows.append(row)

    pool.shutdown()
    df = pd.DataFrame(rows)
    df.to_csv(COMPARE_CSV, index=False)
    print(f"Comparison written to {COMPARE_CSV}")

    if 'ges_theta' in df:
        cols = ['frac_conc_theta', 'frac_conc_zero', 'ges_theta', 'ges_zero', 'ges_diff',
                'bos_theta', 'bos_zero', 'bos_diff']
        cmp = df.dropna(subset=['ges_theta'])
        with pd.option_context('display.width', 200, 'display.max_rows', None,
                               'display.float_format', '{:.4f}'.format):
            print("\nMean over runs per (log, model)  [diff = zero - theta]")
            print(cmp.groupby(['log', 'model'])[cols].mean())
            print("\nMean over logs and runs per model")
            print(cmp.groupby('model')[cols].mean())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                        help="processes scoring samples in parallel (default: all cores)")
    parser.add_argument("--chunksize", type=int, default=4,
                        help="samples sent to a worker per task (default: 4)")
    args = parser.parse_args()
    main(args.workers, args.chunksize)
