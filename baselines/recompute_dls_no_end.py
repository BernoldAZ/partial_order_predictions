"""Recompute the DL similarity of the suffix baselines WITHOUT the END token
and overwrite the stored DLS in place (averaged_results.pkl 'DL sim',
dam_lev_similarity.pt, index 0 of the prefix/suffix-length dicts).

Uses the saved predictions (suffix_acts_decoded.pt, labels.pt), so no
inference is needed. A run is only overwritten when the recomputed DLS WITH
the END token matches the stored one; otherwise it is skipped (e.g. already
converted, or predictions that do not match the stored results).

Run from the baselines/ folder (relative results_per_log/ paths).

python3 recompute_dls_no_end.py

docker run -it --rm -v $(pwd):/app ppm-sutran-best python3 recompute_dls_no_end.py
"""

import glob
import os
import pickle

import torch

from run_all_suffix_baselines import EVENT_LOGS, _RESULT_DIRS

TOL = 1e-3          # max |recomputed with-END DLS - stored DLS| to accept a run
CHUNK = 4096        # samples per DP batch (bounds the (N, W+1, W+1) matrix)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def dump_pkl(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def _dl_chunk(pred, lbl, end_tok):
    """(with END, without END) normalised DL similarity per row."""
    N, W = pred.shape
    # No END predicted: the last position is treated as the END step
    has_end = (pred == end_tok).any(dim=-1)
    pred = pred.clone()
    pred[~has_end, W - 1] = end_tok
    pred_len = (pred == end_tok).to(torch.int64).argmax(dim=-1)
    act_len = (lbl == end_tok).to(torch.int64).argmax(dim=-1)

    d = torch.zeros((N, W + 1, W + 1), dtype=torch.int64, device=pred.device)
    ar = torch.arange(W + 1, device=pred.device).unsqueeze(0)
    d[:, 0, :] = ar
    d[:, :, 0] = ar
    for i in range(1, W + 1):
        for j in range(1, W + 1):
            cost = torch.where(pred[:, i-1] == lbl[:, j-1], 0, 1)
            d[:, i, j] = torch.minimum(torch.minimum(d[:, i-1, j] + 1, d[:, i, j-1] + 1),
                                       d[:, i-1, j-1] + cost)
            if i > 1 and j > 1:
                tpos = (pred[:, i-1] == lbl[:, j-2]) & (pred[:, i-2] == lbl[:, j-1])
                d[:, i, j] = torch.where(tpos, torch.minimum(d[:, i, j], d[:, i-2, j-2] + cost),
                                         d[:, i, j])

    rows = torch.arange(N, device=pred.device)

    def sim(lp, la):
        return 1.0 - d[rows, lp, la].float() / torch.maximum(lp, la).clamp(min=1).float()

    return sim(pred_len + 1, act_len + 1).cpu(), sim(pred_len, act_len).cpu()


def dl_similarity(pred, lbl, end_tok):
    parts = [_dl_chunk(pred[s:s + CHUNK].to(DEVICE), lbl[s:s + CHUNK].to(DEVICE), end_tok)
             for s in range(0, pred.shape[0], CHUNK)]
    return torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])


def recompute_run(res_path, end_tok):
    avg_path = os.path.join(res_path, 'averaged_results.pkl')
    avg = load_pkl(avg_path)
    pred = torch.load(os.path.join(res_path, 'suffix_acts_decoded.pt'), weights_only=True)
    lbl = torch.load(os.path.join(res_path, 'labels.pt'), weights_only=True)[-1]

    dl_end, dl_no_end = dl_similarity(pred, lbl, end_tok)
    old = float(avg['DL sim'])
    if abs(dl_end.mean().item() - old) > TOL:
        return old, dl_end.mean().item(), False

    avg['DL sim'] = dl_no_end.mean().item()
    dump_pkl(avg, avg_path)
    torch.save(dl_no_end, os.path.join(res_path, 'dam_lev_similarity.pt'))

    # Per-length dicts: [dl, rrt, count, ...] -> overwrite dl, same grouping as before
    for fname, lens_file in (('prefix_length_results_dict.pkl', 'pref_len.pt'),
                             ('suffix_length_results_dict.pkl', 'suf_len.pt')):
        dict_path = os.path.join(res_path, fname)
        lens = torch.load(os.path.join(res_path, lens_file), weights_only=True).cpu()
        len_dict = load_pkl(dict_path)
        for k, vals in len_dict.items():
            vals[0] = dl_no_end[lens == k].mean().item()
        dump_pkl(len_dict, dict_path)

    return old, avg['DL sim'], True


def main():
    for log_name in EVENT_LOGS:
        log_dir = os.path.join('results_per_log', log_name)
        cardin_path = os.path.join(log_dir, log_name + '_cardin_dict.pkl')
        if not os.path.isfile(cardin_path):
            continue
        end_tok = load_pkl(cardin_path)['concept:name'] + 1

        for model, dir_base in _RESULT_DIRS.items():
            for run_dir in sorted(glob.glob(os.path.join(log_dir, f"{dir_base}_run*"))):
                res_path = os.path.join(run_dir, 'TEST_SET_RESULTS')
                if not all(os.path.isfile(os.path.join(res_path, f))
                           for f in ('suffix_acts_decoded.pt', 'averaged_results.pkl')):
                    continue
                run_id = run_dir.rsplit('_run', 1)[-1]
                old, new, ok = recompute_run(res_path, end_tok)
                if ok:
                    print(f"{log_name} | {model} | run {run_id} | DLS {old:.4f} -> {new:.4f} (no END)")
                else:
                    print(f"{log_name} | {model} | run {run_id} | SKIPPED: stored DLS {old:.4f} "
                          f"!= recomputed with-END DLS {new:.4f}")


if __name__ == "__main__":
    main()
