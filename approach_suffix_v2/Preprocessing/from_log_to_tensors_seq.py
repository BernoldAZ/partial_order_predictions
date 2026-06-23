"""Sequential preprocessing pipeline.

Produces padded prefix sequence tensors instead of PyG graphs.
The prefix is flattened into a temporal sequence of events; each event carries:
  - activity label  (int64, shifted +1, 0 = padding)
  - ts_start        (float32, normalised)
  - ts_prev         (float32, normalised)
  - new_block_flag  (float32, 1.0 = first event of a new timestamp block, 0.0 = concurrent)

All prefixes are padded to window_size (matching the suffix padding, same as the baselines).
Prefixes longer than window_size are truncated to the last window_size events.
Suffix tensors are identical to the graph pipeline.

Public entry point: log_to_sequences()

Returned dicts (one per split) contain:
  prefix_act       (N, P)    int64
  prefix_num       (N, P, 2) float32  [ts_start, ts_prev]
  prefix_nb        (N, P)    float32
  last_prefix_num  (N, 2)    float32  [ts_start, ts_prev] of last prefix event
  prefix_len       (N,)      int64    actual (unpadded) prefix length
  suffix_act       (N, W)    int64    shifted +1, 0 = pad
  suffix_num       (N, W, 2) float32  -1 = pad
  ttnext_label     (N, W)    float32  -100 = pad
  rtime_label      (N, W)    float32  -100 = pad
  act_label_seq    (N, W)    int64    incl. END token, 0 = pad
  new_block_label  (N, W)    float32  0 = pad
"""

import os
import pickle

import numpy as np
import torch

from Preprocessing.dataframes_pipeline import main_dataframe_pipeline


def _make_act_label_mapping(cardinality_dict, act_label):
    car  = cardinality_dict[act_label] + 1
    keys = [str(i) for i in range(car - 1)] + ['END_TOKEN']
    return dict(zip(keys, range(1, car + 1)))


def _collect_samples(pref_suff, case_id, act_label, timestamp,
                     suffix_num_cols, window_size, outcome, act_mapping):
    """Convert one pref_suff tuple into a list of sample dicts (prefix unpadded).

    Returns (samples, max_actual_prefix_len_in_split).
    """
    prefix_df    = pref_suff[0]
    suffix_df    = pref_suff[1]
    timeLabel_df = pref_suff[2]
    actLabel_df  = pref_suff[3]
    if outcome:
        outcomeLabel_df = pref_suff[4]

    prefix_ids = list(prefix_df.drop_duplicates(subset=case_id)[case_id])
    pref_grp = dict(list(prefix_df.groupby(case_id, sort=False)))
    suff_grp = dict(list(suffix_df.groupby(case_id, sort=False)))
    tlab_grp = dict(list(timeLabel_df.groupby(case_id, sort=False)))
    alab_grp = dict(list(actLabel_df.groupby(case_id, sort=False)))
    if outcome:
        out_grp = dict(list(outcomeLabel_df.groupby(case_id, sort=False)))

    W        = window_size
    samples  = []
    max_plen = 0

    for pid in prefix_ids:
        pref_rows = pref_grp[pid]
        suff_rows = suff_grp[pid]
        time_rows = tlab_grp[pid]
        act_rows  = alab_grp[pid]

        k = len(pref_rows)

        # Prefix activity (int64, shifted +1)
        p_act = pref_rows[act_label].to_numpy().astype(np.int64) + 1   # (k,)

        # Prefix [ts_start, ts_prev]
        p_num = pref_rows[['ts_start', 'ts_prev']].to_numpy().astype(np.float32)  # (k, 2)

        # Prefix new_block_flag: 1.0 if event starts a new timestamp block
        pref_ts = pref_rows[timestamp].to_numpy()
        p_nb    = np.ones(k, dtype=np.float32)                         # first event always new block
        if k > 1:
            p_nb[1:] = (pref_ts[1:] != pref_ts[:-1]).astype(np.float32)

        # Truncate prefix to window_size (keep last W events, matching baseline behaviour)
        if k > window_size:
            p_act = p_act[-window_size:]
            p_num = p_num[-window_size:]
            p_nb  = p_nb[-window_size:]
            k     = window_size

        if k > max_plen:
            max_plen = k

        # Last prefix event [ts_start, ts_prev] — seed for autoregressive time updates
        last_pnum = p_num[-1].copy()

        # Suffix activity (padded 0)
        s_act_raw = suff_rows[act_label].to_numpy().astype(np.int64) + 1
        s_act     = np.zeros(W, dtype=np.int64)
        s_act[:len(s_act_raw)] = s_act_raw

        # Suffix [ts_start, ts_prev] (padded -1)
        s_num_raw = suff_rows[suffix_num_cols].to_numpy().astype(np.float32)
        s_num     = np.full((W, 2), -1.0, dtype=np.float32)
        s_num[:len(s_num_raw)] = s_num_raw

        # Time labels (padded -100)
        ttnext  = time_rows['tt_next'].to_numpy().astype(np.float32)
        rtime   = time_rows['rtime'].to_numpy().astype(np.float32)
        ttn_buf = np.full(W, -100.0, dtype=np.float32)
        rt_buf  = np.full(W, -100.0, dtype=np.float32)
        ttn_buf[:len(ttnext)] = ttnext
        rt_buf[:len(rtime)]   = rtime

        # Ground-truth activity label sequence (incl. END token, padded 0)
        act_seq     = (act_rows[act_label].astype(str).map(act_mapping)
                       .to_numpy().astype(np.int64))
        act_seq_buf = np.zeros(W, dtype=np.int64)
        act_seq_buf[:len(act_seq)] = act_seq

        # New-block label for suffix
        suff_ts = suff_rows[timestamp].to_numpy()
        prev_ts = np.concatenate([[pref_ts[-1]], suff_ts[:-1]])
        nb_raw  = (suff_ts != prev_ts).astype(np.float32)
        nb_buf  = np.zeros(W, dtype=np.float32)
        nb_buf[:len(nb_raw)] = nb_raw

        samples.append({
            'p_act':           p_act,
            'p_num':           p_num,
            'p_nb':            p_nb,
            'last_pnum':       last_pnum,
            'k':               k,
            's_act':           s_act,
            's_num':           s_num,
            'ttn':             ttn_buf,
            'rtime':           rt_buf,
            'act_label_seq':   act_seq_buf,
            'new_block_label': nb_buf,
        })

    return samples, max_plen


def _pad_and_stack(samples, max_prefix_len):
    """Pad all prefix tensors to max_prefix_len and stack into a dict of tensors."""
    N = len(samples)
    W = samples[0]['s_act'].shape[0]

    prefix_act      = np.zeros((N, max_prefix_len), dtype=np.int64)
    prefix_num      = np.zeros((N, max_prefix_len, 2), dtype=np.float32)
    prefix_nb       = np.zeros((N, max_prefix_len), dtype=np.float32)
    last_prefix_num = np.zeros((N, 2), dtype=np.float32)
    prefix_len      = np.zeros(N, dtype=np.int64)
    suffix_act      = np.zeros((N, W), dtype=np.int64)
    suffix_num      = np.full((N, W, 2), -1.0, dtype=np.float32)
    ttnext_label    = np.full((N, W), -100.0, dtype=np.float32)
    rtime_label     = np.full((N, W), -100.0, dtype=np.float32)
    act_label_seq   = np.zeros((N, W), dtype=np.int64)
    new_block_label = np.zeros((N, W), dtype=np.float32)

    for i, s in enumerate(samples):
        k = s['k']
        prefix_act[i, :k]  = s['p_act']
        prefix_num[i, :k]  = s['p_num']
        prefix_nb[i, :k]   = s['p_nb']
        last_prefix_num[i] = s['last_pnum']
        prefix_len[i]      = k
        suffix_act[i]      = s['s_act']
        suffix_num[i]      = s['s_num']
        ttnext_label[i]    = s['ttn']
        rtime_label[i]     = s['rtime']
        act_label_seq[i]   = s['act_label_seq']
        new_block_label[i] = s['new_block_label']

    return {
        'prefix_act':      torch.from_numpy(prefix_act),
        'prefix_num':      torch.from_numpy(prefix_num),
        'prefix_nb':       torch.from_numpy(prefix_nb),
        'last_prefix_num': torch.from_numpy(last_prefix_num),
        'prefix_len':      torch.from_numpy(prefix_len),
        'suffix_act':      torch.from_numpy(suffix_act),
        'suffix_num':      torch.from_numpy(suffix_num),
        'ttnext_label':    torch.from_numpy(ttnext_label),
        'rtime_label':     torch.from_numpy(rtime_label),
        'act_label_seq':   torch.from_numpy(act_label_seq),
        'new_block_label': torch.from_numpy(new_block_label),
    }


def log_to_sequences(log,
                     log_name,
                     start_date,
                     start_before_date,
                     end_date,
                     max_days,
                     test_len_share,
                     val_len_share,
                     window_size,
                     mode,
                     case_id='case:concept:name',
                     act_label='concept:name',
                     timestamp='time:timestamp',
                     cat_casefts=[],
                     num_casefts=[],
                     cat_eventfts=[],
                     num_eventfts=[],
                     outcome=None):
    """Return train/val/test as dicts of stacked tensors.

    Calls the same dataframe pipeline as log_to_graphs().  Prefix events are
    flattened into a padded sequence rather than a graph.

    Parameters
    ----------
    (Same as log_to_graphs.)

    Returns
    -------
    train_dict, val_dict, test_dict : dict of torch.Tensor
    counts : dict
        n_train, train_pairs, n_val, val_pairs, n_test, test_pairs
    num_activities : int
    max_prefix_len : int
    """
    log_transformed = False
    print("Generating Dataframes...")
    (train_pref_suff, val_pref_suff, test_pref_suff,
     cardinality_dict, num_cols_dict, cat_cols_dict,
     train_means_dict, train_std_dict) = main_dataframe_pipeline(
        log, log_name, start_date, start_before_date, end_date, max_days,
        test_len_share, val_len_share, window_size, log_transformed, mode,
        case_id, act_label, timestamp,
        cat_casefts, num_casefts, cat_eventfts, num_eventfts, outcome)

    n_train = train_pref_suff[0]['orig_case_id'].nunique()
    n_val   = val_pref_suff[0]['orig_case_id'].nunique()
    n_test  = test_pref_suff[0]['orig_case_id'].nunique()
    p_train = int(train_pref_suff[0].drop_duplicates('orig_case_id')['case_length'].sum())
    p_val   = int(val_pref_suff[0].drop_duplicates('orig_case_id')['case_length'].sum())
    p_test  = int(test_pref_suff[0].drop_duplicates('orig_case_id')['case_length'].sum())
    print(f"Cases – train: {n_train}  val: {n_val}  test: {n_test}")
    print(f"Pairs – train: {p_train}  val: {p_val}  test: {p_test}")

    suffix_num_cols = num_cols_dict['suffix_df']
    act_mapping     = _make_act_label_mapping(cardinality_dict, act_label)
    num_activities  = cardinality_dict[act_label] + 2

    print("Collecting train samples...")
    train_raw, train_maxp = _collect_samples(
        train_pref_suff, case_id, act_label, timestamp,
        suffix_num_cols, window_size, outcome, act_mapping)

    print("Collecting val samples...")
    val_raw, val_maxp = _collect_samples(
        val_pref_suff, case_id, act_label, timestamp,
        suffix_num_cols, window_size, outcome, act_mapping)

    print("Collecting test samples...")
    test_raw, test_maxp = _collect_samples(
        test_pref_suff, case_id, act_label, timestamp,
        suffix_num_cols, window_size, outcome, act_mapping)

    max_prefix_len = window_size
    print(f"max_prefix_len = {max_prefix_len} (= window_size)  num_activities = {num_activities}")

    print("Padding and stacking tensors...")
    train_dict = _pad_and_stack(train_raw, max_prefix_len)
    val_dict   = _pad_and_stack(val_raw,   max_prefix_len)
    test_dict  = _pad_and_stack(test_raw,  max_prefix_len)

    # Save cardinality lists (same values as graph pipeline)
    pref_cat_cars = [cardinality_dict[c] for c in cat_cols_dict['prefix_df']]
    suff_cat_cars = [cardinality_dict[c] for c in cat_cols_dict['suffix_df']]
    output_dir = os.path.join('results_per_log', log_name)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, log_name + '_cardin_list_prefix.pkl'), 'wb') as f:
        pickle.dump(pref_cat_cars, f)
    with open(os.path.join(output_dir, log_name + '_cardin_list_suffix.pkl'), 'wb') as f:
        pickle.dump(suff_cat_cars, f)

    counts = {
        'n_train': n_train, 'train_pairs': p_train,
        'n_val':   n_val,   'val_pairs':   p_val,
        'n_test':  n_test,  'test_pairs':  p_test,
    }
    return train_dict, val_dict, test_dict, counts, num_activities, max_prefix_len
