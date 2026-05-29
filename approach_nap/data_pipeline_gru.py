"""
Sequential data pipeline for GRU-based Next Event Prediction.

Reuses split/vocab logic from data_pipeline_nap.py.
One sample per trace (seq2seq): events[0..N-2] → events[1..N-1].

Concurrency: conc[i]=1 if timestamp[i]==timestamp[i-1], else 0; conc[0]=0.
Time: dt[i]=(ts[i]-ts[i-1]).total_seconds(), dt[0]=0.
      dt_log=log(1+dt), normalized with mean/std from training data only.

Splits: 64% train / 16% val / 20% test (Weytjens preferred mode).
"""

import csv
import hashlib
import math
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from approach_nap.data_pipeline_nap import (
    build_activity_vocab,
    build_splits,
    df_to_traces,
    load_log,
    sort_log_by_start,
)


# ─────────────────────────────────────────────
# 1. Trace → raw feature lists
# ─────────────────────────────────────────────

def _trace_to_raw(trace, activity_to_idx, timestamp='time:timestamp'):
    """
    Extract (acts, dt_logs, concs) for a trace.

    Returns None if the trace has fewer than 2 events.
    dt_logs are log(1+dt) in seconds, unnormalized.
    """
    events = sorted(trace['events'], key=lambda e: e.get(timestamp))
    if len(events) < 2:
        return None

    unk_idx = len(activity_to_idx)
    acts, dt_logs, concs = [], [], []

    for i, ev in enumerate(events):
        act = ev.get('concept:name')
        acts.append(activity_to_idx.get(act, unk_idx))

        ts = ev.get(timestamp)
        if i == 0:
            dt_logs.append(0.0)
            concs.append(0)
        else:
            prev_ts = events[i - 1].get(timestamp)
            dt_sec = (ts - prev_ts).total_seconds()
            dt_logs.append(math.log(1.0 + dt_sec))
            concs.append(1 if ts == prev_ts else 0)

    return acts, dt_logs, concs


def _make_sample(acts, dt_logs_norm, concs):
    """Build input/target tensor dict from normalized parallel lists."""
    return {
        'act':         torch.tensor(acts[:-1],          dtype=torch.long),
        'dt':          torch.tensor(dt_logs_norm[:-1],  dtype=torch.float),
        'conc':        torch.tensor(concs[:-1],          dtype=torch.long),
        'act_target':  torch.tensor(acts[1:],           dtype=torch.long),
        'dt_target':   torch.tensor(dt_logs_norm[1:],   dtype=torch.float),
        'conc_target': torch.tensor(concs[1:],           dtype=torch.long),
        'length':      len(acts) - 1,
    }


# ─────────────────────────────────────────────
# 2. Dataset + collate
# ─────────────────────────────────────────────

class GRUEventDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    lengths = [s['length'] for s in batch]
    max_len = max(lengths)
    B = len(batch)

    act         = torch.zeros(B, max_len, dtype=torch.long)
    dt          = torch.zeros(B, max_len, dtype=torch.float)
    conc        = torch.zeros(B, max_len, dtype=torch.long)
    act_target  = torch.zeros(B, max_len, dtype=torch.long)
    dt_target   = torch.zeros(B, max_len, dtype=torch.float)
    conc_target = torch.zeros(B, max_len, dtype=torch.long)
    mask        = torch.zeros(B, max_len, dtype=torch.bool)

    for i, s in enumerate(batch):
        L = s['length']
        act[i, :L]         = s['act']
        dt[i, :L]          = s['dt']
        conc[i, :L]        = s['conc']
        act_target[i, :L]  = s['act_target']
        dt_target[i, :L]   = s['dt_target']
        conc_target[i, :L] = s['conc_target']
        mask[i, :L]        = True

    return {
        'act': act, 'dt': dt, 'conc': conc,
        'act_target': act_target, 'dt_target': dt_target,
        'conc_target': conc_target,
        'mask': mask,
    }


# ─────────────────────────────────────────────
# 3. Full pipeline
# ─────────────────────────────────────────────

def _cache_path(log_path, test_len, val_len_share, case_id, act_label, timestamp):
    mtime = os.path.getmtime(log_path)
    key = (f"{os.path.abspath(log_path)}|{mtime}|"
           f"{test_len}|{val_len_share}|{case_id}|{act_label}|{timestamp}|gru_v1")
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(log_path)), "gru_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{h}.pt")


def build_gru_dataloaders(log_path,
                          batch_size=32,
                          test_len=0.20,
                          val_len_share=0.20,
                          case_id='case:concept:name',
                          act_label='concept:name',
                          timestamp='time:timestamp'):
    """
    End-to-end GRU data pipeline.

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
    activity_to_idx : dict
    dt_mean, dt_std : float  (log(1+dt) space; for inference unnormalization)
    counts : dict
    """
    cache_file = _cache_path(log_path, test_len, val_len_share,
                             case_id, act_label, timestamp)

    if os.path.exists(cache_file):
        print(f"Loading cached GRU splits from {cache_file} …")
        cache = torch.load(cache_file, weights_only=False)
        train_samples   = cache['train_samples']
        val_samples     = cache['val_samples']
        test_samples    = cache['test_samples']
        activity_to_idx = cache['activity_to_idx']
        dt_mean         = cache['dt_mean']
        dt_std          = cache['dt_std']
        n_train = cache['n_train']
        n_val   = cache['n_val']
        n_test  = cache['n_test']
    else:
        print("Loading event log …")
        df = load_log(log_path, case_id, timestamp)
        df.drop_duplicates(inplace=True, ignore_index=True)
        df = sort_log_by_start(df, case_id, timestamp)

        df_train, df_val, df_test, _ = build_splits(
            df, test_len, val_len_share, case_id, timestamp)

        n_train = df_train[case_id].nunique()
        n_val   = df_val[case_id].nunique()
        n_test  = df_test[case_id].nunique()

        activity_to_idx = build_activity_vocab(
            pd.concat([df_train, df_val], ignore_index=True), act_label)
        print(f"Vocabulary size (train+val): {len(activity_to_idx)}")

        traces_train = df_to_traces(df_train, case_id, timestamp)
        traces_val   = df_to_traces(df_val,   case_id, timestamp)
        traces_test  = df_to_traces(df_test,  case_id, timestamp)

        # ── Extract raw features ──────────────────────────────────────────
        def _get_raw(traces, desc):
            raw = []
            for trace in tqdm(traces, desc=desc):
                result = _trace_to_raw(trace, activity_to_idx, timestamp)
                if result is not None:
                    raw.append(result)
            return raw

        raw_train = _get_raw(traces_train, "Raw train")
        raw_val   = _get_raw(traces_val,   "Raw val  ")
        raw_test  = _get_raw(traces_test,  "Raw test ")

        # ── Compute dt stats from training data only ──────────────────────
        all_dt_logs = [v for _, dt_logs, _ in raw_train for v in dt_logs]
        dt_mean = float(np.mean(all_dt_logs))
        dt_std  = max(float(np.std(all_dt_logs)), 1e-8)
        print(f"dt_log  mean={dt_mean:.4f}  std={dt_std:.4f}")

        def _normalize(dt_logs):
            return [(v - dt_mean) / dt_std for v in dt_logs]

        def _to_samples(raw):
            return [_make_sample(acts, _normalize(dt_logs), concs)
                    for acts, dt_logs, concs in raw]

        train_samples = _to_samples(raw_train)
        val_samples   = _to_samples(raw_val)
        test_samples  = _to_samples(raw_test)

        torch.save({
            'train_samples':   train_samples,
            'val_samples':     val_samples,
            'test_samples':    test_samples,
            'activity_to_idx': activity_to_idx,
            'dt_mean':         dt_mean,
            'dt_std':          dt_std,
            'n_train':         n_train,
            'n_val':           n_val,
            'n_test':          n_test,
        }, cache_file)
        print(f"Cached GRU splits saved to {cache_file}")

    train_pairs = sum(s['length'] for s in train_samples)
    val_pairs   = sum(s['length'] for s in val_samples)
    test_pairs  = sum(s['length'] for s in test_samples)
    print(f"train={n_train} ({len(train_samples)} traces, {train_pairs} pairs)  "
          f"val={n_val} ({len(val_samples)} traces, {val_pairs} pairs)  "
          f"test={n_test} ({len(test_samples)} traces, {test_pairs} pairs)")

    train_loader = DataLoader(GRUEventDataset(train_samples),
                              batch_size=batch_size, shuffle=True,
                              drop_last=True, collate_fn=collate_fn)
    val_loader   = DataLoader(GRUEventDataset(val_samples),
                              batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn)
    test_loader  = DataLoader(GRUEventDataset(test_samples),
                              batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn)

    counts = {
        'n_train': n_train, 'train_traces': len(train_samples), 'train_pairs': train_pairs,
        'n_val':   n_val,   'val_traces':   len(val_samples),   'val_pairs':   val_pairs,
        'n_test':  n_test,  'test_traces':  len(test_samples),  'test_pairs':  test_pairs,
    }
    return train_loader, val_loader, test_loader, activity_to_idx, dt_mean, dt_std, counts


# ─────────────────────────────────────────────
# 4. Batch runner (all logs in a folder)
# ─────────────────────────────────────────────

def _run_one_log(args):
    """Top-level worker for ProcessPoolExecutor: run one log, return counts."""
    log_path, log_name, kw = args
    try:
        _, _, _, _, _, _, counts = build_gru_dataloaders(log_path, **kw)
        return {'log': log_name, **counts, 'error': ''}
    except Exception as exc:
        return {'log': log_name, 'n_train': '', 'train_traces': '', 'train_pairs': '',
                'n_val': '', 'val_traces': '', 'val_pairs': '',
                'n_test': '', 'test_traces': '', 'test_pairs': '',
                'error': str(exc)}


def run_all_logs(folder, output_file,
                 batch_size=32,
                 test_len=0.20,
                 val_len_share=0.20,
                 case_id='case:concept:name',
                 act_label='concept:name',
                 timestamp='time:timestamp',
                 n_workers=None):
    """Build GRU dataloaders for every log in *folder* and write a summary CSV.

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

    kw = dict(batch_size=batch_size, test_len=test_len,
              val_len_share=val_len_share, case_id=case_id,
              act_label=act_label, timestamp=timestamp)

    workers = min(n_workers or os.cpu_count(), len(files))
    args_list = [(path, name, kw) for path, name in files]

    print(f"Processing {len(files)} logs with {workers} workers ...")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(_run_one_log, args_list))

    for r in results:
        if r['error']:
            print(f"  ERROR  {r['log']}: {r['error']}")
        else:
            print(f"  {r['log']}  "
                  f"train={r['n_train']} ({r['train_traces']} traces, {r['train_pairs']} pairs)  "
                  f"val={r['n_val']} ({r['val_traces']} traces, {r['val_pairs']} pairs)  "
                  f"test={r['n_test']} ({r['test_traces']} traces, {r['test_pairs']} pairs)")

    fieldnames = ['log', 'n_train', 'train_traces', 'train_pairs',
                  'n_val', 'val_traces', 'val_pairs',
                  'n_test', 'test_traces', 'test_pairs', 'error']
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSummary written to '{output_file}'.")
