"""
Data pipeline for Multiple-Activity Prediction using PyTorch Geometric.

Like data_pipeline_nap.py but the prediction target is the FULL SET of
activities that occur at the next timestamp block (concurrently), encoded
as a multi-hot vector.

One graph sample is generated per BLOCK TRANSITION (not per individual event):
  prefix graph = partial-order DAG of all events in blocks 0 .. t-1
  target y     = multi-hot of all activities in block t

Time features (edge_attr, node_pos, y_time) are kept for a future
time-prediction variant but are not used by the current model.
"""

import hashlib
import os
from collections import defaultdict

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import pandas as pd

from approach_nap.data_pipeline_nap import (
    load_log,
    sort_log_by_start,
    build_splits,
    build_activity_vocab,
    df_to_traces,
    _build_prefix_graph,
)


# ─────────────────────────────────────────────
# Prefix-graph generation (multiple activities)
# ─────────────────────────────────────────────

def trace_to_multiple_graphs(trace, activity_to_idx,
                              timestamp='time:timestamp',
                              truncation_level='none'):
    """
    Generate one prefix graph per block transition.

    For a trace with timestamp blocks [B0, B1, ..., Bn]:
      sample t  →  prefix = events in B0..B(t-1),  target = multi-hot of Bt

    Parameters
    ----------
    trace : dict
    activity_to_idx : dict
    timestamp : str
    truncation_level : str

    Returns
    -------
    list of torch_geometric.data.Data
    """
    if truncation_level != 'none':
        from utilities import truncate_trace_timestamps
        trace = truncate_trace_timestamps(trace, truncation_level)

    events = sorted(trace['events'], key=lambda e: e.get(timestamp))
    if len(events) < 2:
        return []

    time_groups = defaultdict(list)
    for ev in events:
        ts = ev.get(timestamp)
        time_groups[ts].append(ev)
    sorted_times = sorted(time_groups.keys())

    if len(sorted_times) < 2:
        return []

    n_acts  = len(activity_to_idx)
    dataset = []

    for t in range(1, len(sorted_times)):
        # Prefix: all events in blocks 0 .. t-1
        prefix_events = []
        for prev_ts in sorted_times[:t]:
            prefix_events.extend(time_groups[prev_ts])

        # Multi-hot target: all activities in block t
        target_ts = sorted_times[t]
        y = torch.zeros(n_acts)
        valid = False
        for ev in time_groups[target_ts]:
            act = ev.get('concept:name')
            if act in activity_to_idx:
                y[activity_to_idx[act]] = 1.0
                valid = True
        if not valid:
            continue

        x, edge_index, edge_attr, _ = _build_prefix_graph(
            prefix_events, activity_to_idx, timestamp)

        last_prefix_ts = sorted_times[t - 1]
        y_time = torch.tensor(
            [(target_ts - last_prefix_ts).total_seconds()], dtype=torch.float)

        dataset.append(Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y.unsqueeze(0),   # (1, n_acts) → (B, n_acts) after batching
            y_time=y_time,
        ))

    return dataset


# ─────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────

def _cache_path(log_path, truncation_level, test_len, val_len_share, mode,
                case_id, act_label, timestamp):
    mtime = os.path.getmtime(log_path)
    key = (f"{os.path.abspath(log_path)}|{mtime}|{truncation_level}|"
           f"{test_len}|{val_len_share}|{mode}|{case_id}|{act_label}|{timestamp}")
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    cache_dir = os.path.join(
        os.path.dirname(os.path.abspath(log_path)), "nap_multiple_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{h}.pt")


def build_multiple_dataloaders(log_path,
                                truncation_level='none',
                                batch_size=32,
                                test_len=0.20,
                                val_len_share=0.20,
                                mode='preferred',
                                case_id='case:concept:name',
                                act_label='concept:name',
                                timestamp='time:timestamp'):
    """
    End-to-end pipeline for multiple-activity prediction.

    Parameters and split logic are identical to build_nap_dataloaders.
    The only difference is the graph-generation step: one sample per block
    transition with a multi-hot target instead of one sample per event.

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
    activity_to_idx : dict
    """
    if mode != 'preferred':
        raise NotImplementedError("Only 'preferred' mode is currently supported.")

    # ── Cache check ───────────────────────────────────────────────────────
    cache_file = _cache_path(log_path, truncation_level, test_len,
                             val_len_share, mode, case_id, act_label, timestamp)
    if os.path.exists(cache_file):
        print(f"Loading cached data splits from {cache_file} …")
        cache = torch.load(cache_file, weights_only=False)
        train_graphs    = cache['train_graphs']
        val_graphs      = cache['val_graphs']
        test_graphs     = cache['test_graphs']
        activity_to_idx = cache['activity_to_idx']
        n_train         = cache['n_train']
        n_val           = cache['n_val']
        n_test          = cache['n_test']
        print(f"train={n_train} ({len(train_graphs)} pairs)  "
              f"val={n_val} ({len(val_graphs)} pairs)  "
              f"test={n_test} ({len(test_graphs)} pairs)")
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

        def _process_split(traces, desc):
            graphs = []
            for trace in tqdm(traces, desc=desc):
                graphs.extend(
                    trace_to_multiple_graphs(trace, activity_to_idx, timestamp,
                                             truncation_level)
                )
            return graphs

        train_graphs = _process_split(traces_train, "Train graphs")
        val_graphs   = _process_split(traces_val,   "Val graphs  ")
        test_graphs  = _process_split(traces_test,  "Test graphs ")

        print(f"train={n_train} ({len(train_graphs)} pairs)  "
              f"val={n_val} ({len(val_graphs)} pairs)  "
              f"test={n_test} ({len(test_graphs)} pairs)")

        torch.save({
            'train_graphs'   : train_graphs,
            'val_graphs'     : val_graphs,
            'test_graphs'    : test_graphs,
            'activity_to_idx': activity_to_idx,
            'n_train'        : n_train,
            'n_val'          : n_val,
            'n_test'         : n_test,
        }, cache_file)
        print(f"Cached data splits saved to {cache_file}")

    train_loader = DataLoader(train_graphs, batch_size=batch_size,
                              shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_graphs,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_graphs,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, activity_to_idx
