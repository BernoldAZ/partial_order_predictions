#!/usr/bin/env python
"""Preprocessing endpoint for the ASTON baseline.

Builds ASTON inputs for exactly the same prefix-suffix instances (same
order, same labels) as the SuTraN baselines, so the results are directly
comparable. The SuTraN tensors must therefore exist first
(``baselines/create_general_data.py``).

Per event log this script:
  1. Re-runs the SuTraN dataframe pipeline (same defaults as
     ``create_general_data.construct_datasets``) to recover the raw
     timestamps, which the SuTraN tensors only keep as standardised deltas.
     Every instance is checked against the saved SuTraN tensors.
  2. Computes ASTON's 6 time features per event (time since midnight, month,
     weekday, hour, time since last event, time since case start), with the
     same transformations as ASTON's ``data_loader.py``.
  3. Mines a Petri net on the train + validation cases with the pm4py
     Inductive Miner (the original repo used Split Miner, which needs Java).
  4. Replays every trace on the net with ASTON's ``GraphVectorizer`` to get
     the per-event place features of the GRNN.

Outputs in ``baselines/results_per_log/<log_name>/`` (next to the SuTraN tensors):
  aston_data.pkl   – per-split model inputs, replay features, model metadata
  petri_net.pnml   – the mined process model

Usage:
    python create_aston_data.py                     # every log in run_all_suffix_baselines.EVENT_LOGS
    python create_aston_data.py --logs Sepsis BPIC15_1
    python create_aston_data.py --force             # rebuild even if aston_data.pkl exists

Run this in the ml-jupyter-gpu image: the Petri-net replay calls pm4py's
apply_hidden_trans with the signature of pm4py 2.7.20 (ppm-sutran-best has
2.7.8). Training (run_all_suffix_baselines.py) can run in either image.

Every log file in a folder, in parallel, with a summary CSV
(like ``create_general_data.run_all_logs``):
    from create_aston_data import run_all_logs
    run_all_logs("Logs/", "aston_split_summary.csv")
"""

import argparse
import csv
import os
import pickle
import sys
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "ASTON"))

import pm4py
from data_loader.data_loader_grnn import GraphVectorizer
from create_general_data import load_log, preprocess_log, infer_feature_columns
from Preprocessing.dataframes_pipeline import main_dataframe_pipeline
from run_all_suffix_baselines import EVENT_LOGS, _RESULTS_BASE as SUTRAN_RESULTS, _find_log_file, data_exists

RESULTS_BASE = SUTRAN_RESULTS
SPLITS = ("train", "val", "test")

CASE_ID = 'case:concept:name'
ACT_LABEL = 'concept:name'
TIMESTAMP = 'time:timestamp'
RESOURCE = 'org:resource'

# Same as create_general_data.construct_datasets defaults
TEST_LEN_SHARE = 0.20
VAL_LEN_SHARE = 0.20
MODE = 'preferred'

# ASTON's data_loader.py: log(0) guard for the log-transformed time features
EPSILON = 0.001


def _load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def _sutran_prefix_dfs(log_path, log_name, window_size):
    """Re-run the SuTraN dataframe pipeline and return {split: prefix_df}.

    Mirrors steps 1-5 of ``create_general_data.construct_datasets``. The
    pipeline writes its metadata pickles relative to the cwd, so it is run
    in a throwaway temp directory to leave the SuTraN results untouched.
    """
    log = preprocess_log(load_log(log_path), timestamp_col=TIMESTAMP)
    cat_casefts, num_casefts, cat_eventfts, num_eventfts = infer_feature_columns(
        log, CASE_ID, ACT_LABEL, TIMESTAMP)

    ts = pd.to_datetime(log[TIMESTAMP], utc=True)
    tmp = log.copy()
    tmp['_ts'] = ts
    durations = tmp.groupby(CASE_ID)['_ts'].agg(lambda x: (x.max() - x.min()).total_seconds())
    max_days = float(durations.max() / (24 * 3600))

    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            train_ps, val_ps, test_ps, *_ = main_dataframe_pipeline(
                log, log_name, None, None, None, max_days, TEST_LEN_SHARE, VAL_LEN_SHARE,
                window_size, False, MODE, CASE_ID, ACT_LABEL, TIMESTAMP,
                cat_casefts, num_casefts, cat_eventfts, num_eventfts, None)
        finally:
            os.chdir(cwd)
    return {'train': train_ps[0], 'val': val_ps[0], 'test': test_ps[0]}


def _time_features(prefix_df):
    """Per-row ASTON time features, in ASTON's column order:
    timesincemidnight, month, weekday, hour, timesincelastevent, timesincecasestart.
    The last two are standardised later with training statistics."""
    ts = prefix_df[TIMESTAMP]
    grp = prefix_df.groupby(CASE_ID, sort=False)[TIMESTAMP]
    since_last = (ts - grp.shift(1)).dt.total_seconds().fillna(0.0)
    since_start = np.log((ts - grp.transform('first')).dt.total_seconds() + EPSILON)
    since_midnight = np.log((ts - ts.dt.normalize()).dt.total_seconds() + EPSILON)
    return np.stack([since_midnight, ts.dt.month, ts.dt.weekday, ts.dt.hour,
                     since_last, since_start], axis=1).astype(np.float64)


def construct_aston_datasets(log_path, log_name):
    sutran_dir = os.path.join(SUTRAN_RESULTS, log_name)
    out_dir = os.path.join(RESULTS_BASE, log_name)
    os.makedirs(out_dir, exist_ok=True)

    cat_cols = _load_pickle(os.path.join(sutran_dir, log_name + '_cat_cols_dict.pkl'))['prefix_df']
    cardin_dict = _load_pickle(os.path.join(sutran_dir, log_name + '_cardin_dict.pkl'))
    act_idx = len(cat_cols) - 1                  # activity is the last categorical
    res_idx = cat_cols.index(RESOURCE) if RESOURCE in cat_cols else None
    num_activities = cardin_dict[ACT_LABEL] + 2  # padding 0, activities, EOS
    eos = num_activities - 1
    num_resources = cardin_dict[RESOURCE] + 1 if res_idx is not None else 0

    tensors = {s: torch.load(os.path.join(sutran_dir, f'{s}_tensordataset.pt'),
                                 weights_only=False) for s in SPLITS}
    W = tensors['train'][act_idx].shape[1]

    # 1. Raw timestamps from the SuTraN dataframe pipeline
    print(f"[{log_name}] Re-running SuTraN dataframe pipeline ...", flush=True)
    prefix_dfs = _sutran_prefix_dfs(log_path, log_name, W)

    # 2. Per-row features + instance/position index of every row
    rows = {}
    for s in SPLITS:
        df = prefix_dfs[s]
        data = tensors[s]
        inst = pd.factorize(df[CASE_ID])[0]   # first-appearance order == SuTraN instance order
        pos = df.groupby(CASE_ID, sort=False).cumcount().to_numpy()
        acts = data[act_idx]
        if inst.max() + 1 != acts.shape[0] or not torch.equal(
                acts[inst, pos], torch.from_numpy(df[ACT_LABEL].to_numpy().astype(np.int64) + 1)):
            raise RuntimeError(f"[{log_name}] {s}: re-run pipeline does not match the saved SuTraN "
                               f"tensors. Was the log preprocessed with non-default settings?")
        rows[s] = dict(inst=inst, pos=pos, feats=_time_features(df),
                       is_event=(df['prefix_nr'] == df['case_length']).to_numpy(),
                       df=df)

    # Standardise the two time-delta features on the training events (each
    # event once: the full-length prefix of every training case).
    for col in (4, 5):
        scaler = StandardScaler().fit(rows['train']['feats'][rows['train']['is_event'], col:col + 1])
        for s in SPLITS:
            rows[s]['feats'][:, col] = scaler.transform(rows[s]['feats'][:, col:col + 1])[:, 0]

    # 3. Petri net on the train + val cases
    print(f"[{log_name}] Mining Petri net (Inductive Miner) ...", flush=True)
    ev = pd.concat([rows[s]['df'][rows[s]['is_event']] for s in ('train', 'val')],
                   ignore_index=True)
    mining_log = pd.DataFrame({
        CASE_ID: ev['orig_case_id'].astype(str),
        ACT_LABEL: (ev[ACT_LABEL].astype(np.int64) + 1).astype(str),
        TIMESTAMP: ev[TIMESTAMP],
    })
    net, im, fm = pm4py.discover_petri_net_inductive(
        mining_log, activity_key=ACT_LABEL, case_id_key=CASE_ID, timestamp_key=TIMESTAMP)
    pm4py.write_pnml(net, im, fm, os.path.join(out_dir, 'petri_net.pnml'))
    vectorizer = GraphVectorizer(net, im, fm)
    adjacency = GraphVectorizer.normalized_adjacency(vectorizer.adjacency_matrix, symmetric=False)
    print(f"[{log_name}] Petri net: {len(net.places)} places, {len(net.transitions)} transitions",
          flush=True)

    # 4. Model inputs (left-padded, as in ASTON) + replay features.
    # The replay of a prefix equals the first k rows of the replay of its full
    # trace, so each distinct full trace is replayed once.
    trace_ids, traces = {}, []
    out = {}
    for s in SPLITS:
        data = tensors[s]
        acts = data[act_idx]
        labels = data[-1]
        N = acts.shape[0]
        pref_len = (~data[act_idx + 2]).sum(dim=1).numpy()   # padding mask: True = pad
        suf_len = (labels == eos).int().argmax(dim=1).numpy()

        n_feats = 1 + (res_idx is not None) + 6
        X = np.zeros((N, W, n_feats), dtype=np.float32)
        r = rows[s]
        left = W - pref_len[r['inst']] + r['pos']
        X[r['inst'], left, 0] = acts[r['inst'], r['pos']].numpy()
        if res_idx is not None:
            X[r['inst'], left, 1] = data[res_idx][r['inst'], r['pos']].numpy()
        X[r['inst'], left, n_feats - 6:] = r['feats']

        Y = labels.clone()
        Y[Y == 0] = eos                                     # ASTON pads targets with EOS

        trace_idx = np.empty(N, dtype=np.int64)
        for i in tqdm(range(N), desc=f"[{log_name}] {s} replay"):
            full = tuple(acts[i, :pref_len[i]].tolist() + labels[i, :suf_len[i]].tolist())
            if full not in trace_ids:
                trace_ids[full] = len(traces)
                traces.append(vectorizer.vectorize([{'task': str(a)} for a in full]).astype(np.uint8))
            trace_idx[i] = trace_ids[full]

        out[s] = {'X': X, 'X_valid_len': pref_len.astype(np.int64), 'Y': Y.numpy(),
                  'Y_valid_len': (suf_len + 1).astype(np.int64), 'trace_idx': trace_idx}

    out['traces'] = traces
    out['meta'] = {'num_activities': num_activities, 'num_resources': num_resources,
                   'N_places': vectorizer.N, 'F': vectorizer.F, 'adjacency': adjacency,
                   'window_size': W}
    with open(os.path.join(out_dir, 'aston_data.pkl'), 'wb') as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    counts = ', '.join(f"{s}={len(out[s]['X'])}" for s in SPLITS)
    print(f"[{log_name}] Saved to '{out_dir}' ({len(traces)} distinct traces; instances {counts})",
          flush=True)
    return {**{f'{s}_pairs': len(out[s]['X']) for s in SPLITS},
            'n_traces': len(traces), 'n_places': len(net.places),
            'n_transitions': len(net.transitions)}


def aston_data_exists(log_name):
    return os.path.isfile(os.path.join(RESULTS_BASE, log_name, 'aston_data.pkl'))


# ─────────────────────────────────────────────
# Batch runner (all logs in a folder)
# ─────────────────────────────────────────────

_SUMMARY_FIELDS = ['log', 'train_pairs', 'val_pairs', 'test_pairs', 'n_traces',
                   'n_places', 'n_transitions', 'error']


def _run_one_log(args):
    """Top-level worker for ProcessPoolExecutor: run one log, return counts."""
    log_path, log_name = args
    try:
        if not data_exists(log_name):
            raise FileNotFoundError("no SuTraN tensors — run create_general_data.py first")
        counts = construct_aston_datasets(log_path, log_name)
        return {'log': log_name, **counts, 'error': ''}
    except Exception as exc:
        return {**{k: '' for k in _SUMMARY_FIELDS}, 'log': log_name, 'error': str(exc)}


def run_all_logs(folder, output_file, n_workers=None):
    """Run construct_aston_datasets for every log in *folder* and write a summary CSV.

    Mirrors ``create_general_data.run_all_logs``. Each log needs its SuTraN
    tensors (same log name = file name without extension) to exist already.

    Parameters
    ----------
    folder : str
        Directory containing .xes, .xes.gz, or .csv event-log files.
    output_file : str
        Path for the output CSV (created or overwritten).
    n_workers : int or None
        Number of parallel worker processes. None = os.cpu_count().
    """
    _SUPPORTED_EXT = {'.xes', '.gz', '.csv'}

    def _stem(fname):
        for ext in ('.xes.gz', '.xes', '.csv'):
            if fname.endswith(ext):
                return fname[:-len(ext)]
        return os.path.splitext(fname)[0]

    files = sorted(
        (os.path.join(folder, f), _stem(f))
        for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
        and os.path.splitext(f)[1].lower() in _SUPPORTED_EXT
    )
    if not files:
        print(f"No log files found in '{folder}'.")
        return

    workers = min(n_workers or os.cpu_count(), len(files))
    print(f"Processing {len(files)} logs with {workers} workers ...")
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDS)
        writer.writeheader()

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_run_one_log, args): args[1] for args in files}
            for future in as_completed(futures):
                r = future.result()
                if r['error']:
                    print(f"  ERROR  {r['log']}: {r['error']}")
                else:
                    print(f"  {r['log']}  "
                          f"train={r['train_pairs']}  val={r['val_pairs']}  "
                          f"test={r['test_pairs']} pairs  traces={r['n_traces']}  "
                          f"net={r['n_places']}p/{r['n_transitions']}t")
                writer.writerow(r)
                f.flush()

    print(f"\nSummary written to '{output_file}'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build ASTON inputs from the SuTraN splits.")
    parser.add_argument("--logs", nargs="+", default=EVENT_LOGS,
                        help="Event logs to process (default: run_all_suffix_baselines.EVENT_LOGS)")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even if aston_data.pkl already exists")
    args = parser.parse_args()

    for log_name in args.logs:
        if not args.force and aston_data_exists(log_name):
            print(f"[DATA-SKIP] {log_name}", flush=True)
            continue
        if not data_exists(log_name):
            print(f"[NO-SUTRAN-DATA] {log_name} — run create_general_data.py first",
                  flush=True)
            continue
        log_file = _find_log_file(log_name)
        if log_file is None:
            print(f"[DATA-MISSING] {log_name} — not found in Logs/", flush=True)
            continue
        print(f"[DATA-RUNNING] {log_name}", flush=True)
        try:
            construct_aston_datasets(log_file, log_name)
            print(f"[DATA-DONE] {log_name}", flush=True)
        except Exception:
            print(f"[DATA-ERROR] {log_name}\n{traceback.format_exc()}", flush=True)
