"""Sequential preprocessing pipeline for suffix prediction.

Drop-in companion to from_log_to_tensors.py and from_log_to_tensors_graph.py.
Identical outer structure (sort → train/test split → window filter →
train/val split → time columns → pair building → normalization → tensors),
but prefixes are represented as lightweight (act, log_dt, conc) sequences
rather than padded SuTraN tensors or PyG graphs.

Token convention
----------------
  0        = PAD  (prefix padding, suffix padding)
  1..N     = activity classes (1-indexed)
  N+1      = OOV  (unseen test activities)
  N+2      = EOS  (end-of-sequence marker appended to every suffix)
  num_classes = N+3

Pair generation
---------------
For a case of length T, prefix lengths 1..T are generated (matching the
graph pipeline). The full-case prefix (length T) produces a suffix
containing only the EOS token.

Returns
-------
train_data, val_data, test_data : list of dict
stats : dict
conc_mask : BoolTensor
counts : dict
"""

import math

import numpy as np
import pandas as pd
import torch

from Preprocessing.create_benchmarks import remainTimeOrClassifBenchmark
from Preprocessing.dataframes_pipeline import (
    create_numeric_timeCols,
    sort_log,
    split_train_val,
)


# ---------------------------------------------------------------------------
# Raw pair builder
# ---------------------------------------------------------------------------

def _df_to_raw(df, split_name, case_id, act_label, timestamp,
               act_to_idx, oov_idx, eos_idx, prefix_dict, mode):
    """Build raw prefix-suffix pair dicts for one DataFrame split.

    split_name : 'train_val' | 'test'
    """
    raw = []
    for cid, grp in df.groupby(case_id, sort=False):
        grp = grp.sort_values(timestamp).reset_index(drop=True)
        T   = len(grp)

        acts      = [act_to_idx.get(a, oov_idx) for a in grp[act_label]]
        prevs     = grp['ts_prev'].values.astype(float)
        ts_starts = grp['ts_start'].values.astype(float)
        concs     = [1 if prevs[i] == 0.0 and i > 0 else 0 for i in range(T)]
        dt_logs   = [math.log(1.0 + max(prevs[i], 0.0)) for i in range(T)]

        if split_name == 'test' and mode == 'preferred' and cid in prefix_dict:
            # prefix_dict[cid] = 0-based idx of first post-sep event
            # only prefixes that include that event are valid
            min_plen = prefix_dict[cid] + 1
            max_plen = T
        elif split_name == 'train_val' and mode == 'workaround' and cid in prefix_dict:
            # prefix_dict[cid] = 0-based idx of last pre-sep event
            # prefix must not cross sep time
            min_plen = 1
            max_plen = min(prefix_dict[cid] + 1, T)
        else:
            min_plen = 1
            max_plen = T

        for plen in range(min_plen, max_plen + 1):
            suf_acts = acts[plen:]    + [eos_idx]
            suf_dt   = dt_logs[plen:] + [0.0]
            suf_conc = concs[plen:]   + [0]
            raw.append({
                'prefix_act':       acts[:plen],
                'prefix_dt':        dt_logs[:plen],
                'prefix_conc':      concs[:plen],
                'suffix_act':       suf_acts,
                'suffix_dt':        suf_dt,
                'suffix_conc':      suf_conc,
                'slen':             len(suf_acts),
                'last_prefix_act':      acts[plen - 1],
                'last_prefix_dt':       dt_logs[plen - 1],
                'last_prefix_conc':     concs[plen - 1],
                'last_prefix_ts_start': ts_starts[plen - 1],
                'prev_prefix_ts_start': ts_starts[plen - 2] if plen >= 2 else 0.0,
            })
    return raw


# ---------------------------------------------------------------------------
# Tensor converter
# ---------------------------------------------------------------------------

def _to_tensors(raw, dt_mean, dt_std, W):
    """Convert raw pair dicts to fixed-size padded tensor dicts."""
    dataset = []
    for s in raw:
        plen = len(s['prefix_act'])
        slen = min(s['slen'], W)

        p_act  = torch.tensor(s['prefix_act'],  dtype=torch.long)
        p_dt   = torch.tensor(
            [(v - dt_mean) / dt_std for v in s['prefix_dt']], dtype=torch.float)
        p_conc = torch.tensor(s['prefix_conc'], dtype=torch.long)

        suf_act  = torch.zeros(W, dtype=torch.long)
        suf_dt   = torch.zeros(W, dtype=torch.float)
        suf_conc = torch.zeros(W, dtype=torch.long)
        suf_mask = torch.zeros(W, dtype=torch.bool)

        suf_act[:slen]  = torch.tensor(s['suffix_act'][:slen],  dtype=torch.long)
        suf_dt[:slen]   = torch.tensor(
            [(v - dt_mean) / dt_std for v in s['suffix_dt'][:slen]], dtype=torch.float)
        suf_conc[:slen] = torch.tensor(s['suffix_conc'][:slen], dtype=torch.long)
        suf_mask[:slen] = True

        dataset.append({
            'prefix_act':       p_act,
            'prefix_dt':        p_dt,
            'prefix_conc':      p_conc,
            'suffix_act':       suf_act,
            'suffix_dt':        suf_dt,
            'suffix_conc':      suf_conc,
            'suffix_mask':      suf_mask,
            'prefix_len':       plen,
            'last_prefix_act':  s['last_prefix_act'],
            'last_prefix_dt':   float((s['last_prefix_dt'] - dt_mean) / dt_std),
            'last_prefix_conc': s['last_prefix_conc'],
        })
    return dataset


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def log_to_sequential_tensors(
    log, log_name, start_date, start_before_date, end_date, max_days,
    test_len_share, val_len_share, window_size, mode,
    case_id='case:concept:name',
    act_label='concept:name',
    timestamp='time:timestamp',
):
    """Build (act, log_dt, conc) prefix-suffix tensor pairs.

    Same parameters and outer structure as log_to_tensors() and
    log_to_graphs(). Only activity + timestamp are used; no graph
    construction or feature encoding.

    Parameters
    ----------
    log : pd.DataFrame
    log_name : str
    start_date, start_before_date, end_date : str or None
    max_days : float
    test_len_share, val_len_share : float
    window_size : int
    mode : {'preferred', 'workaround'}
    case_id, act_label, timestamp : str

    Returns
    -------
    train_data, val_data, test_data : list of dict
    stats : dict
    conc_mask : BoolTensor
    counts : dict
    """
    # ── 1. Sort + chronological train/test split ──────────────────────────
    log = sort_log(log, case_id=case_id, timestamp=timestamp, act_label=act_label)
    train_df, test_df, prefix_dict = remainTimeOrClassifBenchmark(
        dataset=log, file_name=log_name,
        start_date=start_date, start_before_date=start_before_date,
        end_date=end_date, max_days=max_days,
        test_len_share=test_len_share,
        case_id=case_id, timestamp=timestamp, mode=mode,
    )

    # ── 2. Filter traces longer than window_size ──────────────────────────
    def _filter_len(df):
        lens = df.groupby(case_id, sort=False)[act_label].transform('count')
        return df[lens <= window_size].reset_index(drop=True)

    train_df = _filter_len(train_df)
    test_df  = _filter_len(test_df)

    # ── 3. Further split train → train + val ─────────────────────────────
    train_df, val_df = split_train_val(train_df, val_len_share, case_id, timestamp)

    # ── 4. Add ts_prev / ts_start / etc. time columns ────────────────────
    train_df = create_numeric_timeCols(train_df, case_id, timestamp, act_label)
    val_df   = create_numeric_timeCols(val_df,   case_id, timestamp, act_label)
    test_df  = create_numeric_timeCols(test_df,  case_id, timestamp, act_label)

    # ── 5. Activity vocabulary from train + val (1-indexed; 0 = PAD) ─────
    train_val_acts = sorted(
        pd.concat([train_df, val_df], ignore_index=True)[act_label].unique()
    )
    act_to_idx  = {a: i + 1 for i, a in enumerate(train_val_acts)}
    num_acts    = len(act_to_idx)
    oov_idx     = num_acts + 1
    eos_idx     = num_acts + 2
    num_classes = num_acts + 3

    # ── 6. Build raw prefix-suffix pairs per split ────────────────────────
    raw_train = _df_to_raw(train_df, 'train_val', case_id, act_label, timestamp,
                           act_to_idx, oov_idx, eos_idx, prefix_dict, mode)
    raw_val   = _df_to_raw(val_df,   'train_val', case_id, act_label, timestamp,
                           act_to_idx, oov_idx, eos_idx, prefix_dict, mode)
    raw_test  = _df_to_raw(test_df,  'test',      case_id, act_label, timestamp,
                           act_to_idx, oov_idx, eos_idx, prefix_dict, mode)

    # ── 7. dt normalization stats (training data only; exclude EOS zeros) ─
    all_dt  = [v for s in raw_train
               for v in s['prefix_dt'] + s['suffix_dt'][:-1]]
    dt_mean = float(np.mean(all_dt)) if all_dt else 0.0
    dt_std  = max(float(np.std(all_dt)), 1e-8)

    # ts_start normalization stats (training data only; matches baseline pipeline)
    ts_start_mean = float(train_df['ts_start'].mean())
    ts_start_std  = max(float(train_df['ts_start'].std()), 1e-8)

    # ── 8. Convert raw → padded fixed-size tensor dicts ──────────────────
    W = window_size

    train_data = _to_tensors(raw_train, dt_mean, dt_std, W)
    val_data   = _to_tensors(raw_val,   dt_mean, dt_std, W)
    test_data  = _to_tensors(raw_test,  dt_mean, dt_std, W)

    # ── 9. Concurrent-ending-prefix mask for test set ─────────────────────
    # Uses standardized float32 ts_start comparison to match the baseline pipeline.
    def _conc_check(s):
        if len(s['prefix_act']) < 2:
            return False
        last = np.float32((s['last_prefix_ts_start'] - ts_start_mean) / ts_start_std)
        prev = np.float32((s['prev_prefix_ts_start'] - ts_start_mean) / ts_start_std)
        return bool(last == prev)

    conc_mask = torch.tensor([_conc_check(s) for s in raw_test], dtype=torch.bool)

    stats = {
        'dt_mean':     dt_mean,
        'dt_std':      dt_std,
        'num_classes': num_classes,
        'num_acts':    num_acts,
        'oov_idx':     oov_idx,
        'eos_idx':     eos_idx,
        'window_size': W,
        'act_to_idx':  act_to_idx,
    }
    counts = {
        'n_train':     train_df[case_id].nunique(),
        'train_pairs': int(train_df.groupby(case_id, sort=False)[act_label].count().sum()),
        'n_val':       val_df[case_id].nunique(),
        'val_pairs':   int(val_df.groupby(case_id, sort=False)[act_label].count().sum()),
        'n_test':      test_df[case_id].nunique(),
        'test_pairs':  int(test_df.groupby(case_id, sort=False)[act_label].count().sum()),
    }
    return train_data, val_data, test_data, stats, conc_mask, counts
