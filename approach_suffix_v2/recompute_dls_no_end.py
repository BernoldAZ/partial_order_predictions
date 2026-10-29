"""Recompute the DL similarity of suffix_v2 / suffix_v2_seq WITHOUT the END
token and overwrite the stored DLS in place:
  results_suffix_time_gnn[_<tag>].csv        dl_similarity
  <log>_per_sample_metrics[_<tag>].pt        'dl_similarity'
  <log>_{prefix,suffix}_length_results_dict.pkl   index 0 (normal test set only)

Predictions are not stored, so the saved model of each log / run is loaded and
greedy inference is rerun on the normal test set and on every perturbed test
set (prefixflip, prefixrandom, flip, random) that has results. A result is only
overwritten when the recomputed DLS WITH the END token matches the stored one;
otherwise it is skipped (e.g. already converted).

Usage (from the project root)
-----
    python approach_suffix_v2/recompute_dls_no_end.py

Docker
------
    docker run -it --rm -v $(pwd):/workspace --gpus all ml-jupyter-gpu python approach_suffix_v2/recompute_dls_no_end.py
"""
import csv
import glob
import os
import pickle
import sys

import torch
from torch_geometric.loader import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, 'models_v2'))

import run_suffix_v2
import run_suffix_v2_seq
from run_all_suffix import MODEL_CONFIGS

# model -> (run module, test dataset stem)
MODELS = {
    'suffix_v2':     (run_suffix_v2,     'graphdataset'),
    'suffix_v2_seq': (run_suffix_v2_seq, 'seqgraphdataset'),
}
TAGS   = ['', 'prefixflip', 'prefixrandom', 'flip', 'random']   # '' = normal test set
TOL    = 1e-3   # max |recomputed with-END DLS - stored DLS| to accept a result
CHUNK  = 4096   # samples per DP batch (bounds the (N, W+1, W+1) matrix)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _dl_chunk(pred, lbl, end_tok):
    """(with END, without END) normalised DL similarity per row, same DP as
    run_suffix_v2._compute_metrics."""
    N, W = pred.shape
    has_end   = (pred == end_tok).any(dim=-1)
    first_end = (pred == end_tok).to(torch.int64).argmax(dim=-1)
    pred_len  = torch.where(has_end, first_end, torch.full_like(first_end, W - 1))
    act_len   = (lbl == end_tok).to(torch.int64).argmax(dim=-1)

    d  = torch.zeros((N, W + 1, W + 1), dtype=torch.int64, device=pred.device)
    ar = torch.arange(W + 1, device=pred.device).unsqueeze(0)
    d[:, 0, :] = ar
    d[:, :, 0] = ar
    for i in range(1, W + 1):
        for j in range(1, W + 1):
            cost       = torch.where(pred[:, i-1] == lbl[:, j-1], 0, 1)
            d[:, i, j] = torch.minimum(torch.minimum(d[:, i-1, j] + 1, d[:, i, j-1] + 1),
                                       d[:, i-1, j-1] + cost)
            if i > 1 and j > 1:
                tpos       = (pred[:, i-1] == lbl[:, j-2]) & (pred[:, i-2] == lbl[:, j-1])
                d[:, i, j] = torch.where(tpos, torch.minimum(d[:, i, j], d[:, i-2, j-2] + cost),
                                         d[:, i, j])

    rows = torch.arange(N, device=pred.device)

    def sim(lp, la):
        return 1.0 - d[rows, lp, la].float() / torch.maximum(lp, la).clamp(min=1).float()

    # Without END: no END predicted -> all W predicted activities are kept
    pred_len_no_end = torch.where(has_end, first_end, torch.full_like(first_end, W))
    return sim(pred_len + 1, act_len + 1).cpu(), sim(pred_len_no_end, act_len).cpu(), act_len.cpu()


def dl_similarity(pred, lbl, end_tok):
    parts = [_dl_chunk(pred[s:s + CHUNK].to(DEVICE), lbl[s:s + CHUNK].to(DEVICE), end_tok)
             for s in range(0, pred.shape[0], CHUNK)]
    return tuple(torch.cat([p[k] for p in parts]) for k in range(3))


@torch.no_grad()
def _predict(model, test_data, window_size, batch_size):
    model.eval()
    acts, lbls = [], []
    for data in DataLoader(test_data, batch_size=batch_size, shuffle=False):
        data = data.to(DEVICE)
        a, _ = model(data, window_size=window_size)
        acts.append(a.cpu())
        lbls.append(data.act_label_seq.view(data.num_graphs, window_size).cpu())
    return torch.cat(acts), torch.cat(lbls)


def _read_csv(path):
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def _write_csv(path, fieldnames, rows):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def recompute_log(mod, ds_stem, run_dir, log_name):
    data_dir = os.path.join("approach_suffix_v2", 'results_per_log', log_name)
    with open(os.path.join(data_dir, f'{log_name}_cardin_list_prefix.pkl'), 'rb') as f:
        num_activities = pickle.load(f)[-1] + 2
    end_tok = num_activities - 1

    model = mod.GATv2EncoderGRUDecoderNewBlockV2NoTime(
        num_activities=num_activities,
        d_model=mod.D_MODEL,
        dropout=mod.DROPOUT,
        n_gru_layers=mod.N_GRU_LAYERS,
        n_gnn_layers=mod.N_GNN_LAYERS,
    ).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(run_dir, f'{log_name}_{mod.METHOD_NAME}.pt'),
                                     map_location=DEVICE, weights_only=True))

    for tag in TAGS:
        sfx       = f'_{tag}' if tag else ''
        test_path = os.path.join(data_dir, f'test_{ds_stem}{sfx}.pt')
        csv_path  = os.path.join(run_dir, f'results_suffix_time_gnn{sfx}.csv')
        if not (os.path.isfile(test_path) and os.path.isfile(csv_path)):
            continue
        fieldnames, rows = _read_csv(csv_path)
        row = next((r for r in rows if r['log'] == log_name
                    and r['method'] == mod.METHOD_NAME), None)
        if row is None:
            continue

        test_data   = torch.load(test_path, weights_only=False)
        window_size = test_data[0].suffix_act.shape[0]
        pred, lbl   = _predict(model, test_data, window_size, mod.BATCH_SIZE)
        dl_end, dl_no_end, suf_lens = dl_similarity(pred, lbl, end_tok)

        label = f"{log_name} | {mod.METHOD_NAME} | {os.path.basename(run_dir)} | {tag or 'test'}"
        old = float(row['dl_similarity'])
        if abs(dl_end.mean().item() - old) > TOL:
            print(f"{label} | SKIPPED: stored DLS {old:.4f} != recomputed with-END DLS "
                  f"{dl_end.mean().item():.4f}")
            continue

        row['dl_similarity'] = round(dl_no_end.mean().item(), 6)
        _write_csv(csv_path, fieldnames, rows)

        ps_path = os.path.join(run_dir, f'{log_name}_per_sample_metrics{sfx}.pt')
        if os.path.isfile(ps_path):
            per_sample = torch.load(ps_path, weights_only=True)
            per_sample['dl_similarity'] = dl_no_end.to(torch.float32)
            torch.save(per_sample, ps_path)

        # Per-length dicts [dl, rrt, count, ...]: same grouping as run_suffix_v2.py
        if not tag:
            pref_lens = torch.tensor([s.num_nodes for s in test_data])
            for fname, lens in (('prefix', pref_lens), ('suffix', suf_lens)):
                pkl_path = os.path.join(run_dir, f'{log_name}_{fname}_length_results_dict.pkl')
                if not os.path.isfile(pkl_path):
                    continue
                with open(pkl_path, 'rb') as f:
                    len_dict = pickle.load(f)
                for k, vals in len_dict.items():
                    vals[0] = dl_no_end[lens == k].mean().item()
                with open(pkl_path, 'wb') as f:
                    pickle.dump(len_dict, f)

        print(f"{label} | DLS {old:.4f} -> {row['dl_similarity']:.4f} (no END)")


def main():
    for model_key, (mod, ds_stem) in MODELS.items():
        results_sub = os.path.join(_HERE, MODEL_CONFIGS[model_key]['results_subdir'])
        for run_dir in sorted(glob.glob(os.path.join(results_sub, 'run_*'))):
            csv_path = os.path.join(run_dir, 'results_suffix_time_gnn.csv')
            if not os.path.isfile(csv_path):
                continue
            logs = [r['log'] for r in _read_csv(csv_path)[1] if r['method'] == mod.METHOD_NAME]
            for log_name in logs:
                if not os.path.isfile(os.path.join(run_dir, f'{log_name}_{mod.METHOD_NAME}.pt')):
                    print(f"{log_name} | {mod.METHOD_NAME} | {os.path.basename(run_dir)} | "
                          f"SKIPPED: no saved model")
                    continue
                recompute_log(mod, ds_stem, run_dir, log_name)


if __name__ == '__main__':
    main()
