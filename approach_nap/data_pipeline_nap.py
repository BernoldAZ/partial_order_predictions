"""
Data extraction pipeline for Next Activity Prediction (NAP) using PyTorch Geometric.

Each trace is converted into a DAG based on timestamp ordering (partial order):
  - Each event gets its own node with a one-hot activity feature
  - Events with identical timestamps form a "block" layer
  - Edges run from every node in the previous layer to every node in the current layer
  - Edge features are the time delta (in seconds) between the two endpoint nodes

Train/val/test split matches the baseline script generate_new_event_log_splits.py:
  - Cases are sorted chronologically by their start timestamp
  - Temporal out-of-time split: 64% train / 16% val / 20% test
  - Train+val vs test: Weytjens 'preferred' mode (overlapping cases filtered)
  - Train vs val: simple chronological case assignment (no overlap handling)
  - Activity vocabulary is built from the train+val union (same as baselines)
"""

import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm


# ─────────────────────────────────────────────
# 1. Data loading
# ─────────────────────────────────────────────

def load_log(log_path,
             case_id='case:concept:name',
             timestamp='time:timestamp'):
    """Load an XES (.xes / .gz) or CSV event log into a DataFrame."""
    ext = os.path.splitext(log_path)[1].lower()
    if ext in ('.xes', '.gz'):
        from pm4py.objects.log.importer.xes import importer as xes_importer
        from pm4py.objects.conversion.log import converter
        event_log = xes_importer.apply(log_path)
        df = converter.apply(event_log, variant=converter.Variants.TO_DATA_FRAME)
    elif ext == '.csv':
        df = pd.read_csv(log_path)
    else:
        raise ValueError(f"Unsupported log format: '{ext}'")

    df[timestamp] = pd.to_datetime(df[timestamp], utc=True)
    return df


def sort_log_by_start(df,
                      case_id='case:concept:name',
                      timestamp='time:timestamp'):
    """Sort cases by start time, then events within each case by timestamp."""
    case_starts = df.groupby(case_id)[timestamp].min()
    case_order = {c: i for i, c in enumerate(case_starts.sort_values().index)}
    df = df.copy()
    df['_order'] = df[case_id].map(case_order)
    df = df.sort_values(['_order', timestamp], kind='mergesort')
    return df.drop(columns='_order').reset_index(drop=True)


# ─────────────────────────────────────────────
# 2. Temporal split (Weytjens 'preferred' mode)
# ─────────────────────────────────────────────

def _temporal_split_preferred(df, split_fraction,
                               case_id='case:concept:name',
                               timestamp='time:timestamp'):
    """
    Chronological split keeping `split_fraction` of cases in the 'train' portion.

    Overlapping cases (start before split, end at/after split) are dropped from
    both sides to keep the splits clean.

    Returns
    -------
    df_train : DataFrame  (cases ending strictly before split time)
    df_test  : DataFrame  (cases starting at or after split time)
    split_time : Timestamp
    """
    case_starts = df.groupby(case_id)[timestamp].min()
    case_ends   = df.groupby(case_id)[timestamp].max()

    sorted_start_times = case_starts.sort_values()
    n = len(sorted_start_times)
    first_test_idx = int(n * split_fraction)
    split_time = sorted_start_times.iloc[first_test_idx]

    # Pure train: cases ending strictly before split time
    train_case_ids = case_ends[case_ends < split_time].index.tolist()
    df_train = df[df[case_id].isin(train_case_ids)].copy().reset_index(drop=True)

    # Pure test: cases starting at or after split time (timestamp comparison,
    # matches generate_new_event_log_splits.py; overlapping cases excluded)
    test_case_ids = case_starts[case_starts >= split_time].index.tolist()
    df_test = df[df[case_id].isin(test_case_ids)].copy().reset_index(drop=True)

    return df_train, df_test, split_time


def _val_case_split(df_trainval, val_len_share,
                    case_id='case:concept:name',
                    timestamp='time:timestamp'):
    """
    Simple chronological split of train+val into train vs val.

    Mirrors _val_case_split() in generate_new_event_log_splits.py:
    cases are ordered by start time; the last val_len_share fraction becomes val.
    No overlap handling — each case goes entirely to one split.

    Returns
    -------
    df_train, df_val : DataFrames
    """
    case_starts = df_trainval.groupby(case_id)[timestamp].min().sort_values()
    n = len(case_starts)
    first_val_idx = int(n * (1.0 - val_len_share))
    val_case_ids   = set(case_starts.iloc[first_val_idx:].index)
    train_case_ids = set(case_starts.iloc[:first_val_idx].index)
    df_val   = df_trainval[df_trainval[case_id].isin(val_case_ids)].copy().reset_index(drop=True)
    df_train = df_trainval[df_trainval[case_id].isin(train_case_ids)].copy().reset_index(drop=True)
    return df_train, df_val


def build_splits(df,
                 test_len=0.20,
                 val_len_share=0.20,
                 case_id='case:concept:name',
                 timestamp='time:timestamp'):
    """
    Two-stage split matching baseline script → 64% train / 16% val / 20% test.

    Stage 1: Weytjens 'preferred' split at (1 - test_len) → train+val vs test.
    Stage 2: simple chronological split of train+val at val_len_share.

    Returns
    -------
    df_train, df_val, df_test : DataFrames
    """
    train_val_fraction = 1.0 - test_len
    df_trainval, df_test, _ = _temporal_split_preferred(
        df, train_val_fraction, case_id, timestamp)

    df_train, df_val = _val_case_split(df_trainval, val_len_share, case_id, timestamp)

    return df_train, df_val, df_test


# ─────────────────────────────────────────────
# 3. Vocabulary
# ─────────────────────────────────────────────

def build_activity_vocab(df_train, act_label='concept:name'):
    """Build {activity: index} mapping using training data only."""
    activities = sorted(df_train[act_label].dropna().unique().tolist())
    return {act: i for i, act in enumerate(activities)}


# ─────────────────────────────────────────────
# 4. DataFrame → trace dicts
# ─────────────────────────────────────────────

def df_to_traces(df,
                 case_id='case:concept:name',
                 timestamp='time:timestamp'):
    """
    Convert a DataFrame to a list of trace dicts compatible with utilities.py.
    Timestamps are converted to timezone-aware Python datetime objects.
    """
    traces = []
    for cid, case_df in df.groupby(case_id, sort=False):
        case_df = case_df.sort_values(timestamp, kind='mergesort')
        events = []
        for _, row in case_df.iterrows():
            ev = row.to_dict()
            # Ensure timestamp is a Python datetime (not pd.Timestamp)
            ts = ev.get(timestamp)
            if ts is not None and hasattr(ts, 'to_pydatetime'):
                ev[timestamp] = ts.to_pydatetime()
            events.append(ev)
        traces.append({
            'trace_attributes': {case_id: cid},
            'events': events
        })
    return traces


# ─────────────────────────────────────────────
# 5. Prefix DAG builder (shared helper)
# ─────────────────────────────────────────────

def _build_prefix_graph(prefix_events, activity_to_idx, timestamp):
    """
    Build a PyG Data (x, edge_index, edge_attr) from an ordered list of
    prefix events.  Events with the same timestamp form a concurrent layer;
    edges run from every node in layer t to every node in layer t+1.
    """
    n_acts = len(activity_to_idx)

    time_groups = defaultdict(list)
    for ev in prefix_events:
        ts = ev.get(timestamp)
        if ts is not None:
            time_groups[ts].append(ev)

    sorted_times = sorted(time_groups.keys())

    node_activities = []
    node_timestamps = []
    edge_list       = []
    edge_attr_list  = []
    previous_indices = []

    for ts in sorted_times:
        current_indices = []
        for ev in time_groups[ts]:
            idx = len(node_activities)
            act = ev.get('concept:name')
            node_activities.append(act if act in activity_to_idx else None)
            node_timestamps.append(ts)
            current_indices.append(idx)

        for prev in previous_indices:
            for curr in current_indices:
                delta = (node_timestamps[curr] - node_timestamps[prev]).total_seconds()
                edge_list.append((prev, curr))
                edge_attr_list.append(delta)

        previous_indices = current_indices

    x_rows = []
    for act in node_activities:
        row = torch.zeros(n_acts)
        if act is not None:
            row[activity_to_idx[act]] = 1.0
        x_rows.append(row)
    x = torch.stack(x_rows)

    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr  = torch.tensor(edge_attr_list, dtype=torch.float).unsqueeze(1)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr  = torch.empty((0, 1), dtype=torch.float)

    return x, edge_index, edge_attr


# ─────────────────────────────────────────────
# 6. Prefix-graph generation (NAP)
# ─────────────────────────────────────────────

def trace_to_nap_graphs(trace, activity_to_idx,
                        timestamp='time:timestamp',
                        truncation_level='none'):
    """
    Generate one prefix graph per event (same granularity as the baselines).

    For event at sequential position i (1-indexed from the second event),
    the graph is the partial-order DAG of events 0..i-1 and the target is
    the single activity at position i.

    When multiple events share a timestamp, the trace order determines which
    comes first; earlier events are added to the prefix before later ones of
    the same timestamp, so each concurrent event gets its own prefix-output
    tuple with a progressively richer prefix.

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

    n_acts  = len(activity_to_idx)
    dataset = []

    for i in range(1, len(events)):
        target_act = events[i].get('concept:name')
        if target_act not in activity_to_idx:
            continue

        x, edge_index, edge_attr = _build_prefix_graph(
            events[:i], activity_to_idx, timestamp)

        y = torch.zeros(n_acts)
        y[activity_to_idx[target_act]] = 1.0

        dataset.append(Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y.unsqueeze(0),
        ))

    return dataset


# ─────────────────────────────────────────────
# 7. Full pipeline
# ─────────────────────────────────────────────

def build_nap_dataloaders(log_path,
                          truncation_level='none',
                          batch_size=32,
                          test_len=0.20,
                          val_len_share=0.20,
                          mode='preferred',
                          case_id='case:concept:name',
                          act_label='concept:name',
                          timestamp='time:timestamp'):
    """
    End-to-end NAP data pipeline.

    Loads an event log, applies the Weytjens temporal split, builds activity
    vocabulary from training data only, and returns PyG DataLoaders.

    Parameters
    ----------
    log_path : str
        Path to XES (.xes / .gz) or CSV event log.
    truncation_level : str
        Timestamp truncation to introduce concurrency ('none', 'day', 'hour',
        'minute', 'second').  'none' keeps original precision.
    batch_size : int
    test_len : float
        Fraction of cases (by start time) assigned to the test split.
    val_len_share : float
        Fraction of train+val cases assigned to the validation split (default 0.20
        → ~16% of all cases, matching the baseline 64/16/20 ratio).
    mode : str
        Currently only 'preferred' is supported (Weytjens default).
    case_id, act_label, timestamp : str
        Column names for case ID, activity label, and timestamp.

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
    activity_to_idx : dict
    """
    if mode != 'preferred':
        raise NotImplementedError("Only 'preferred' mode is currently supported.")

    # ── Load & sort ───────────────────────────────────────────────────────
    print("Loading event log …")
    df = load_log(log_path, case_id, timestamp)
    df.drop_duplicates(inplace=True, ignore_index=True)
    df = sort_log_by_start(df, case_id, timestamp)

    # ── Temporal split ────────────────────────────────────────────────────
    df_train, df_val, df_test = build_splits(
        df, test_len, val_len_share, case_id, timestamp)

    n_train = df_train[case_id].nunique()
    n_val   = df_val[case_id].nunique()
    n_test  = df_test[case_id].nunique()
    print(f"Cases  – train: {n_train}  val: {n_val}  test: {n_test}")

    # ── Vocabulary (train + val union, matching baselines) ────────────────
    activity_to_idx = build_activity_vocab(
        pd.concat([df_train, df_val], ignore_index=True), act_label)
    print(f"Vocabulary size (train+val): {len(activity_to_idx)}")

    # ── Convert splits to trace dicts ─────────────────────────────────────
    traces_train = df_to_traces(df_train, case_id, timestamp)
    traces_val   = df_to_traces(df_val,   case_id, timestamp)
    traces_test  = df_to_traces(df_test,  case_id, timestamp)

    # ── Generate prefix graphs ────────────────────────────────────────────
    def _process_split(traces, desc):
        graphs = []
        for trace in tqdm(traces, desc=desc):
            graphs.extend(
                trace_to_nap_graphs(trace, activity_to_idx, timestamp,
                                    truncation_level)
            )
        return graphs

    train_graphs = _process_split(traces_train, "Train graphs")
    val_graphs   = _process_split(traces_val,   "Val graphs  ")
    test_graphs  = _process_split(traces_test,  "Test graphs ")

    print(f"Graphs – train: {len(train_graphs)}  "
          f"val: {len(val_graphs)}  test: {len(test_graphs)}")

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_graphs,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_graphs,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, activity_to_idx


# ─────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python data_pipeline_nap.py <path_to_log>")
        sys.exit(1)

    train_loader, val_loader, test_loader, act_vocab = build_nap_dataloaders(
        sys.argv[1], truncation_level='none')

    batch = next(iter(train_loader))
    print(f"\nSample batch – nodes: {batch.x.shape}, "
          f"edges: {batch.edge_index.shape[1]}, "
          f"labels: {batch.y.shape}")
    print(f"Activity vocab: {list(act_vocab.items())[:5]} …")
