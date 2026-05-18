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

import csv
import hashlib
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

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
    Chronological split matching generate_new_event_log_splits.py preferred mode.

    train+val : cases whose last event is strictly before split_time
    test      : cases whose last event is at or after split_time
                (includes overlapping cases that started before split_time)

    Cases with fewer than 2 events are dropped from both sides.

    Returns
    -------
    df_trainval : DataFrame
    df_test     : DataFrame
    split_time  : Timestamp
    """
    case_starts = df.groupby(case_id)[timestamp].min()
    case_ends   = df.groupby(case_id)[timestamp].max()

    sorted_start_times = case_starts.sort_values()
    n = len(sorted_start_times)
    first_test_idx = int(n * split_fraction)
    split_time = sorted_start_times.iloc[first_test_idx]

    trainval_ids = case_ends[case_ends <  split_time].index
    test_ids     = case_ends[case_ends >= split_time].index

    df_trainval = df[df[case_id].isin(trainval_ids)].copy().reset_index(drop=True)
    df_test     = df[df[case_id].isin(test_ids)].copy().reset_index(drop=True)

    def _drop_short(frame):
        counts = frame.groupby(case_id)[timestamp].transform('count')
        return frame[counts >= 2].reset_index(drop=True)

    return _drop_short(df_trainval), _drop_short(df_test), split_time


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

    Stage 1: preferred split at (1 - test_len) → train+val vs test.
    Stage 2: window_size filter (98.5th percentile of case lengths) applied to
             both sides — mirrors the suffix pipeline order in the baseline.
    Stage 3: simple chronological split of train+val at val_len_share.

    Returns
    -------
    df_train, df_val, df_test : DataFrames
    window_size : int
    """
    train_val_fraction = 1.0 - test_len
    df_trainval, df_test, _ = _temporal_split_preferred(
        df, train_val_fraction, case_id, timestamp)

    # Auto-derive window_size from the full dataset (before split)
    window_size = int(np.percentile(df.groupby(case_id).size(), 98.5))
    print(f"Auto-derived window_size (98.5th percentile): {window_size}")

    def _filter_window(frame):
        lengths = frame.groupby(case_id)[case_id].transform('count')
        return frame[lengths <= window_size].reset_index(drop=True)

    df_trainval = _filter_window(df_trainval)
    df_test     = _filter_window(df_test)

    df_train, df_val = _val_case_split(df_trainval, val_len_share, case_id, timestamp)

    return df_train, df_val, df_test, window_size


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
    Build a PyG Data (x, edge_index, edge_attr) from an ordered list of prefix
    events.  Events with the same timestamp form a concurrent layer; edges run
    from every node in layer t to every node in layer t+1.

    edge_attr holds one value per edge: the time delta in seconds between the
    source block timestamp and the destination block timestamp (0.0 for
    intra-block edges between concurrent events).
    """
    time_groups = defaultdict(list)
    for ev in prefix_events:
        ts = ev.get(timestamp)
        if ts is not None:
            time_groups[ts].append(ev)

    sorted_times = sorted(time_groups.keys())

    node_activities  = []
    edge_list        = []
    edge_attr_list   = []
    previous_indices = []

    for block_idx, ts in enumerate(sorted_times):
        prev_ts   = sorted_times[block_idx - 1] if block_idx > 0 else ts
        dt_seconds = (ts - prev_ts).total_seconds()

        current_indices = []
        for ev in time_groups[ts]:
            idx = len(node_activities)
            act = ev.get('concept:name')
            node_activities.append(act if act in activity_to_idx else None)
            current_indices.append(idx)

        for prev in previous_indices:
            for curr in current_indices:
                edge_list.append((prev, curr))
                edge_attr_list.append([dt_seconds])

        for k in range(len(current_indices)):
            for l in range(k + 1, len(current_indices)):
                edge_list.append((current_indices[k], current_indices[l]))
                edge_attr_list.append([0.0])
                edge_list.append((current_indices[l], current_indices[k]))
                edge_attr_list.append([0.0])

        previous_indices = current_indices

    unk_idx = len(activity_to_idx)
    x_ids = [activity_to_idx[act] if act is not None else unk_idx
             for act in node_activities]
    x = torch.tensor(x_ids, dtype=torch.long)

    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr  = torch.tensor(edge_attr_list, dtype=torch.float)  # (E, 1)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr  = torch.empty((0, 1), dtype=torch.float)

    n = len(node_activities)
    last_layer_mask = torch.zeros(n, dtype=torch.bool)
    for idx in previous_indices:
        last_layer_mask[idx] = True

    return x, edge_index, edge_attr, last_layer_mask


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

        x, edge_index, edge_attr, last_layer_mask = _build_prefix_graph(
            events[:i], activity_to_idx, timestamp)

        y = torch.zeros(n_acts)
        if target_act in activity_to_idx:
            y[activity_to_idx[target_act]] = 1.0
        # else: zero label — unseen activity, matches baseline behaviour

        last_ts = events[i - 1].get(timestamp)
        next_ts = events[i].get(timestamp)
        y_time  = torch.tensor(
            [(next_ts - last_ts).total_seconds()], dtype=torch.float)
        y_layer = torch.tensor(
            [0 if next_ts == last_ts else 1], dtype=torch.long)

        dataset.append(Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y.unsqueeze(0),
            y_time=y_time,
            y_layer=y_layer,
            last_layer_mask=last_layer_mask,
        ))

    return dataset


# ─────────────────────────────────────────────
# 7. Full pipeline
def _cache_path(log_path, truncation_level, test_len, val_len_share, mode,
                case_id, act_label, timestamp):
    mtime = os.path.getmtime(log_path)
    key = (f"{os.path.abspath(log_path)}|{mtime}|{truncation_level}|"
           f"{test_len}|{val_len_share}|{mode}|{case_id}|{act_label}|{timestamp}|v2")
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(log_path)), "nap_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{h}.pt")


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
        # ── Load & sort ───────────────────────────────────────────────────
        print("Loading event log …")
        df = load_log(log_path, case_id, timestamp)
        df.drop_duplicates(inplace=True, ignore_index=True)
        df = sort_log_by_start(df, case_id, timestamp)

        # ── Temporal split ────────────────────────────────────────────────
        df_train, df_val, df_test, _ = build_splits(
            df, test_len, val_len_share, case_id, timestamp)

        n_train = df_train[case_id].nunique()
        n_val   = df_val[case_id].nunique()
        n_test  = df_test[case_id].nunique()

        # ── Vocabulary (train + val union, matching baselines) ────────────
        activity_to_idx = build_activity_vocab(
            pd.concat([df_train, df_val], ignore_index=True), act_label)
        print(f"Vocabulary size (train+val): {len(activity_to_idx)}")

        # ── Convert splits to trace dicts ─────────────────────────────────
        traces_train = df_to_traces(df_train, case_id, timestamp)
        traces_val   = df_to_traces(df_val,   case_id, timestamp)
        traces_test  = df_to_traces(df_test,  case_id, timestamp)

        # ── Generate prefix graphs ────────────────────────────────────────
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

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_graphs,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_graphs,  batch_size=batch_size, shuffle=False)

    counts = {
        'n_train':     n_train,     'train_pairs': len(train_graphs),
        'n_val':       n_val,       'val_pairs':   len(val_graphs),
        'n_test':      n_test,      'test_pairs':  len(test_graphs),
    }
    return train_loader, val_loader, test_loader, activity_to_idx, counts


# ─────────────────────────────────────────────
# 8. Batch runner (all logs in a folder)
# ─────────────────────────────────────────────

def _run_one_log(args):
    """Top-level worker for ProcessPoolExecutor: run one log, return counts."""
    log_path, log_name, kw = args
    try:
        _, _, _, _, counts = build_nap_dataloaders(log_path, **kw)
        return {'log': log_name, **counts, 'error': ''}
    except Exception as exc:
        return {'log': log_name, 'n_train': '', 'train_pairs': '',
                'n_val': '', 'val_pairs': '', 'n_test': '', 'test_pairs': '',
                'error': str(exc)}


def run_all_logs(folder, output_file,
                 truncation_level='none',
                 batch_size=32,
                 test_len=0.20,
                 val_len_share=0.20,
                 mode='preferred',
                 case_id='case:concept:name',
                 act_label='concept:name',
                 timestamp='time:timestamp',
                 n_workers=None):
    """Build NAP dataloaders for every log in *folder* and write a summary CSV.

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

    kw = dict(truncation_level=truncation_level, batch_size=batch_size,
              test_len=test_len, val_len_share=val_len_share, mode=mode,
              case_id=case_id, act_label=act_label, timestamp=timestamp)

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
                  f"train={r['n_train']} ({r['train_pairs']} pairs)  "
                  f"val={r['n_val']} ({r['val_pairs']} pairs)  "
                  f"test={r['n_test']} ({r['test_pairs']} pairs)")

    fieldnames = ['log', 'n_train', 'train_pairs', 'n_val', 'val_pairs',
                  'n_test', 'test_pairs', 'error']
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSummary written to '{output_file}'.")
