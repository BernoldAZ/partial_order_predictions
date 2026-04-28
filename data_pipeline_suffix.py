"""
Data extraction pipeline for Suffix Prediction using PyTorch Geometric.

Each trace is converted into a DAG based on timestamp ordering (partial order):
  - Each event gets its own node with a one-hot activity feature
  - Events with identical timestamps form a "block" layer
  - Edges run from every node in the previous layer to every node in the current layer
  - Edge features are the time delta (in seconds) between the two endpoint nodes

For each prefix (graph), the prediction target is the full activity suffix –
the remaining events from the next position to the end of the trace –
terminated with an END_TOKEN and zero-padded to a fixed length `max_suffix_len`.

One prefix-suffix pair is generated per event (same granularity as baselines).
Within concurrent events (same timestamp), earlier events in trace order are
added to the prefix before later ones, so each gets its own tuple.

Activity indices in the suffix target are 1-based so that 0 can serve as a
padding token.  END_TOKEN is assigned index `len(activity_to_idx) + 1`.

Train/val/test split matches the SuTraN baseline (Preprocessing/from_log_to_tensors.py):
  - Temporal out-of-time split: 60% train / 15% val / 25% test
  - Train+val vs test: Weytjens 'preferred' mode (test_len_share=0.25)
  - Train vs val: simple chronological case assignment (val_len_share=0.20 of train+val)
  - Cases longer than the 98.5th percentile of training case lengths are discarded
  - Activity vocabulary is built from train+val union
  - max_suffix_len = window_size (98.5th pct of training case lengths)
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

    Returns
    -------
    df_train : DataFrame  (cases ending before split time)
    df_test  : DataFrame  (cases ending at or after split time)
    first_prefix_dict : dict
        Maps overlapping case IDs → 0-based index of their first event that
        occurs strictly after the split time.
    split_time : Timestamp
    """
    case_starts = df.groupby(case_id)[timestamp].min()
    case_ends   = df.groupby(case_id)[timestamp].max()

    sorted_start_times = case_starts.sort_values()
    n = len(sorted_start_times)
    first_test_idx = int(n * split_fraction)
    split_time = sorted_start_times.iloc[first_test_idx]

    # Cases entirely before split → train
    train_case_ids = case_ends[case_ends < split_time].index.tolist()
    df_train = df[df[case_id].isin(train_case_ids)].copy().reset_index(drop=True)

    # Cases ending at or after split → test (includes overlapping cases)
    test_case_ids_all = case_ends[case_ends >= split_time].index.tolist()
    df_test = df[df[case_id].isin(test_case_ids_all)].copy().reset_index(drop=True)

    # Overlapping cases: start before split but end at or after
    test_case_ids_start_after = sorted_start_times.iloc[first_test_idx:].index.tolist()
    overlap_ids = set(test_case_ids_all) - set(test_case_ids_start_after)

    # For each overlapping case, record the event index of its first post-split event
    first_prefix_dict = {}
    if overlap_ids:
        df_overlap = df_test[df_test[case_id].isin(overlap_ids)].copy()
        df_overlap['_evt_idx'] = df_overlap.groupby(case_id).cumcount()
        df_post = df_overlap[df_overlap[timestamp] > split_time]
        first_post = df_post.groupby(case_id, as_index=False)['_evt_idx'].first()
        first_prefix_dict = dict(zip(first_post[case_id], first_post['_evt_idx']))

    return df_train, df_test, first_prefix_dict, split_time


def _val_case_split(df_trainval, val_len_share,
                    case_id='case:concept:name',
                    timestamp='time:timestamp'):
    """
    Simple chronological split of train+val into train vs val.
    Mirrors split_train_val() in SuTraN Preprocessing/dataframes_pipeline.py.
    """
    case_starts = df_trainval.groupby(case_id)[timestamp].min().sort_values()
    n = len(case_starts)
    first_val_idx  = int(n * (1.0 - val_len_share))
    val_case_ids   = set(case_starts.iloc[first_val_idx:].index)
    train_case_ids = set(case_starts.iloc[:first_val_idx].index)
    df_val   = df_trainval[df_trainval[case_id].isin(val_case_ids)].copy().reset_index(drop=True)
    df_train = df_trainval[df_trainval[case_id].isin(train_case_ids)].copy().reset_index(drop=True)
    return df_train, df_val


def build_splits(df,
                 test_len=0.25,
                 val_len_share=0.20,
                 case_id='case:concept:name',
                 timestamp='time:timestamp'):
    """
    Two-stage split matching SuTraN baseline → 60% train / 15% val / 25% test.

    Stage 1: Weytjens 'preferred' split at (1 - test_len) → train+val vs test.
    Stage 2: simple chronological split of train+val at val_len_share.

    Returns
    -------
    df_train, df_val, df_test : DataFrames
    fpd_test : first_prefix_dict for test overlapping cases
    """
    train_val_fraction = 1.0 - test_len
    df_trainval, df_test, fpd_test, _ = _temporal_split_preferred(
        df, train_val_fraction, case_id, timestamp)

    df_train, df_val = _val_case_split(df_trainval, val_len_share, case_id, timestamp)

    return df_train, df_val, df_test, fpd_test


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
    """Build a PyG (x, edge_index, edge_attr) from an ordered prefix event list."""
    n_acts = len(activity_to_idx)

    time_groups = defaultdict(list)
    for ev in prefix_events:
        ts = ev.get(timestamp)
        if ts is not None:
            time_groups[ts].append(ev)

    sorted_times     = sorted(time_groups.keys())
    node_activities  = []
    node_timestamps  = []
    edge_list        = []
    edge_attr_list   = []
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
# 6. Window size and case filtering
# ─────────────────────────────────────────────

def compute_window_size(df_train, case_id='case:concept:name', percentile=98.5):
    """
    Compute window_size as the percentile-th percentile of training case lengths.
    Matches the SuTraN baseline: cases longer than window_size are discarded.
    max_suffix_len = window_size (longest suffix = window_size-1 events + END_TOKEN).
    """
    case_lengths = df_train.groupby(case_id).size().values
    return int(np.percentile(case_lengths, percentile))


def filter_long_cases(df, window_size, case_id='case:concept:name'):
    """Discard cases with more events than window_size (matching SuTraN baseline)."""
    case_lengths = df.groupby(case_id).size()
    valid_cases  = case_lengths[case_lengths <= window_size].index
    n_dropped    = df[case_id].nunique() - len(valid_cases)
    if n_dropped:
        print(f"  Dropping {n_dropped} cases with length > {window_size}")
    return df[df[case_id].isin(valid_cases)].copy().reset_index(drop=True)


# ─────────────────────────────────────────────
# 7. Prefix-graph + suffix target generation
# ─────────────────────────────────────────────

def trace_to_suffix_graphs(trace, activity_to_idx, end_token_idx, max_suffix_len,
                            timestamp='time:timestamp',
                            min_event_idx=0,
                            truncation_level='none'):
    """
    Generate one suffix-prediction instance per event (same granularity as baselines).

    For event at sequential position i (1 to L-1), the graph is the partial-order
    DAG of events 0..i-1 and the target is the full remaining suffix events[i..L-1]
    encoded as 1-based activity indices + END_TOKEN, padded to max_suffix_len.

    Encoding convention:
      0                       → PAD
      activity_to_idx[a] + 1  → activity a  (1-based)
      end_token_idx           → END_TOKEN

    Parameters
    ----------
    trace : dict
    activity_to_idx : dict
    end_token_idx : int
    max_suffix_len : int
    timestamp : str
    min_event_idx : int
        Only generate tuples where i >= this value.
        Used for Weytjens overlapping test cases.
    truncation_level : str

    Returns
    -------
    list of torch_geometric.data.Data  (y shape: (1, max_suffix_len))
    """
    if truncation_level != 'none':
        from utilities import truncate_trace_timestamps
        trace = truncate_trace_timestamps(trace, truncation_level)

    events = sorted(trace['events'], key=lambda e: e.get(timestamp))
    if len(events) < 2:
        return []

    dataset = []

    for i in range(1, len(events) + 1):
        if i < min_event_idx:
            continue

        x, edge_index, edge_attr = _build_prefix_graph(
            events[:i], activity_to_idx, timestamp)

        # Suffix: all remaining events after position i (1-based indices)
        suffix_acts = []
        for ev in events[i:]:
            act = ev.get('concept:name')
            if act in activity_to_idx:
                suffix_acts.append(activity_to_idx[act] + 1)  # 1-based

        suffix_acts.append(end_token_idx)

        if len(suffix_acts) >= max_suffix_len:
            suffix_acts = suffix_acts[:max_suffix_len]
        else:
            suffix_acts = suffix_acts + [0] * (max_suffix_len - len(suffix_acts))

        y = torch.tensor(suffix_acts, dtype=torch.long).unsqueeze(0)

        # Time targets (seconds) for the prefix ending at events[i-1]
        t_prev = events[i - 1].get(timestamp)
        t_next = events[i].get(timestamp) if i < len(events) else None
        t_end  = events[-1].get(timestamp)
        ttne_s = (t_next - t_prev).total_seconds() if t_next is not None else 0.0
        rrt_s  = (t_end  - t_prev).total_seconds()

        dataset.append(Data(
            x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
            ttne=torch.tensor([ttne_s], dtype=torch.float),
            rrt=torch.tensor([rrt_s],  dtype=torch.float),
        ))

    return dataset


# ─────────────────────────────────────────────
# 8. Full pipeline
# ─────────────────────────────────────────────

def build_suffix_dataloaders(log_path,
                              truncation_level='none',
                              batch_size=32,
                              test_len=0.25,
                              val_len_share=0.20,
                              window_size=None,
                              window_size_percentile=98.5,
                              mode='preferred',
                              case_id='case:concept:name',
                              act_label='concept:name',
                              timestamp='time:timestamp'):
    """
    End-to-end suffix-prediction data pipeline matching the SuTraN baseline.

    Parameters
    ----------
    log_path : str
    truncation_level : str
    batch_size : int
    test_len : float
        Fraction of cases assigned to test (default 0.25, matching SuTraN).
    val_len_share : float
        Fraction of train+val cases assigned to val (default 0.20).
    window_size : int or None
        Max case length.  Cases longer than this are discarded; also used as
        max_suffix_len.  If None, derived as the 98.5th percentile of training
        case lengths (matching SuTraN baseline).
    window_size_percentile : float
    mode : str  ('preferred' only)
    case_id, act_label, timestamp : str

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
    activity_to_idx : dict
    end_token_idx : int
    max_suffix_len : int
    """
    if mode != 'preferred':
        raise NotImplementedError("Only 'preferred' mode is currently supported.")

    # ── Load & sort ───────────────────────────────────────────────────────
    print("Loading event log …")
    df = load_log(log_path, case_id, timestamp)
    df = sort_log_by_start(df, case_id, timestamp)

    # ── Window size from full log before any split (matching SuTraN) ──────
    if window_size is None:
        case_lengths = df.groupby(case_id).size().values
        window_size = int(np.percentile(case_lengths, window_size_percentile))
        print(f"window_size (p{window_size_percentile:.0f} of full log): {window_size}")

    # ── Stage 1: preferred split → train+val vs test ──────────────────────
    df_trainval, df_test, _, _ = _temporal_split_preferred(
        df, 1.0 - test_len, case_id, timestamp)

    # ── Filter long cases before train/val split (matching SuTraN order) ──
    print("Filtering long cases …")
    df_trainval = filter_long_cases(df_trainval, window_size, case_id)
    df_test     = filter_long_cases(df_test,     window_size, case_id)

    # ── Stage 2: split train+val → train vs val ───────────────────────────
    df_train, df_val = _val_case_split(df_trainval, val_len_share, case_id, timestamp)

    # ── Vocabulary (train + val union, matching SuTraN baseline) ─────────
    activity_to_idx = build_activity_vocab(
        pd.concat([df_train, df_val], ignore_index=True), act_label)
    end_token_idx   = len(activity_to_idx) + 1
    max_suffix_len  = window_size          # max suffix = window_size-1 events + END
    print(f"Vocabulary size (train+val): {len(activity_to_idx)}")
    print(f"END_TOKEN index: {end_token_idx}  |  max_suffix_len: {max_suffix_len}")

    # ── Convert splits to trace dicts ─────────────────────────────────────
    traces_train = df_to_traces(df_train, case_id, timestamp)
    traces_val   = df_to_traces(df_val,   case_id, timestamp)
    traces_test  = df_to_traces(df_test,  case_id, timestamp)

    # ── Generate prefix graphs + suffix targets ───────────────────────────
    def _process_split(traces, fpd, desc):
        graphs = []
        for trace in tqdm(traces, desc=desc):
            cid = trace['trace_attributes'].get(case_id)
            min_event_idx = fpd.get(cid, 0)
            graphs.extend(
                trace_to_suffix_graphs(
                    trace, activity_to_idx, end_token_idx, max_suffix_len,
                    timestamp, min_event_idx, truncation_level)
            )
        return graphs

    train_graphs = _process_split(traces_train, {}, "Train graphs")
    val_graphs   = _process_split(traces_val,   {}, "Val graphs  ")
    test_graphs  = _process_split(traces_test,  {}, "Test graphs ")

    n_train = df_train[case_id].nunique()
    n_val   = df_val[case_id].nunique()
    n_test  = df_test[case_id].nunique()
    print(f"Cases (after filter) – train: {n_train}  val: {n_val}  test: {n_test}")

    print(f"Graphs – train: {len(train_graphs)}  "
          f"val: {len(val_graphs)}  test: {len(test_graphs)}")

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_graphs,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_graphs,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, activity_to_idx, end_token_idx, max_suffix_len


# ─────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python data_pipeline_suffix.py <path_to_log>")
        sys.exit(1)

    (train_loader, val_loader, test_loader,
     act_vocab, end_tok, max_suf) = build_suffix_dataloaders(sys.argv[1])

    batch = next(iter(train_loader))
    print(f"\nSample batch – nodes: {batch.x.shape}, "
          f"edges: {batch.edge_index.shape[1]}, "
          f"suffix targets: {batch.y.shape}")
    print(f"END_TOKEN index: {end_tok}  |  max_suffix_len: {max_suf}")
    print(f"Activity vocab: {list(act_vocab.items())[:5]} …")
