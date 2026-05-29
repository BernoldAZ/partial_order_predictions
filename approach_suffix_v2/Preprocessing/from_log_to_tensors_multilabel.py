"""Multilabel graph-based preprocessing pipeline.

Groups suffix events that share the same timestamp into 'blocks' and stores
them as multi-hot vectors. Designed for multi-label next-activity prediction
in partial-order process logs.

Each Data object has the same prefix graph as the standard graph pipeline,
plus block-level suffix attributes:

  suffix_multihot : float32 [T_max, C]
      Multi-hot of each block. Positions 0..T_actual-1 are real blocks,
      position T_actual is the END block (index C-1 set to 1), and the
      rest are zero-padded.
      C = num_activities - 1  (regular activities 0..cardinality-1 at
      indices 0..cardinality-1, END_TOKEN at index cardinality = C-1).

  suffix_tss : float32 [T_max]
      Normalised time-since-case-start for each block. -1 for padding.

  suffix_tsp : float32 [T_max]
      Normalised time-since-previous-block for each block. -1 for padding.

  label_tsp : float32 [T_max]
      Target "time to next block" for each decoder step.
      label_tsp[t] = suffix_tsp[t+1]  for t in 0..T_actual-2
      label_tsp[T_actual-1] = -100  (no next block before END)
      label_tsp[T_actual..]  = -100  (padding)
"""

import os
import pickle

import numpy as np
import torch
from torch_geometric.data import Data

from Preprocessing.dataframes_pipeline import main_dataframe_pipeline
from Preprocessing.from_log_to_tensors_graph import _build_prefix_graph


# ---------------------------------------------------------------------------
# Block grouping
# ---------------------------------------------------------------------------

def _group_suffix_blocks(suff_rows, act_col, raw_ts_col, tss_col, tsp_col, cardinality):
    """Group sorted suffix rows into timestamp blocks.

    Returns list of (multihot, tss, tsp) tuples, one per block.
    multihot has shape (C,) where C = cardinality + 1:
      - indices 0..cardinality-1: regular activities (0-indexed as in suffix_df)
      - index cardinality (= C-1): END_TOKEN slot (never set here; added separately)
    tss, tsp are normalised float scalars.
    """
    C = cardinality + 1
    blocks = []
    if len(suff_rows) == 0:
        return blocks

    ts_vals  = suff_rows[raw_ts_col].values
    act_vals = suff_rows[act_col].values.astype(np.int64)
    tss_vals = suff_rows[tss_col].values.astype(np.float32)
    tsp_vals = suff_rows[tsp_col].values.astype(np.float32)

    cur_ts  = ts_vals[0]
    cur_mh  = np.zeros(C, dtype=np.float32)
    cur_tss = float(tss_vals[0])
    cur_tsp = float(tsp_vals[0])

    for i in range(len(ts_vals)):
        if ts_vals[i] != cur_ts:
            blocks.append((cur_mh.copy(), cur_tss, cur_tsp))
            cur_ts  = ts_vals[i]
            cur_mh  = np.zeros(C, dtype=np.float32)
            cur_tss = float(tss_vals[i])
            cur_tsp = float(tsp_vals[i])
        a = int(act_vals[i])
        if 0 <= a < cardinality:
            cur_mh[a] = 1.0

    blocks.append((cur_mh.copy(), cur_tss, cur_tsp))
    return blocks


# ---------------------------------------------------------------------------
# Per-split dataset builder
# ---------------------------------------------------------------------------

def _generate_multilabel_dataset(pref_suff, case_id, act_label, timestamp,
                                  cat_cols, num_node_cols, suffix_num_cols,
                                  cardinality_dict, window_size, outcome,
                                  T_max):
    """Convert one pref_suff tuple into a list of PyG Data objects with
    block-level multilabel attributes."""
    C          = cardinality_dict[act_label] + 1   # multi-hot dimension
    cardinality = cardinality_dict[act_label]

    prefix_df    = pref_suff[0]
    suffix_df    = pref_suff[1]
    timeLabel_df = pref_suff[2]   # unused here but kept for future use
    actLabel_df  = pref_suff[3]
    if outcome:
        outcomeLabel_df = pref_suff[4]

    prefix_ids = list(prefix_df.drop_duplicates(subset=case_id)[case_id])
    pref_grp = dict(list(prefix_df.groupby(case_id, sort=False)))
    suff_grp = dict(list(suffix_df.groupby(case_id, sort=False)))
    tlab_grp = dict(list(timeLabel_df.groupby(case_id, sort=False)))
    if outcome:
        out_grp = dict(list(outcomeLabel_df.groupby(case_id, sort=False)))

    tss_col, tsp_col = suffix_num_cols[0], suffix_num_cols[1]  # ts_start, ts_prev

    data_list = []

    for pid in prefix_ids:
        pref_rows = pref_grp[pid]
        suff_rows = suff_grp[pid]
        time_rows = tlab_grp[pid]

        # ── Prefix graph ─────────────────────────────────────────────────
        data = _build_prefix_graph(pref_rows, cat_cols, num_node_cols, timestamp)

        # ── Group suffix into blocks ─────────────────────────────────────
        blocks = _group_suffix_blocks(
            suff_rows, act_label, timestamp, tss_col, tsp_col, cardinality)
        T_actual = len(blocks)

        # ── Build suffix_multihot [T_max, C] ────────────────────────────
        mh_arr  = np.zeros((T_max, C), dtype=np.float32)
        tss_arr = np.full(T_max, -1.0, dtype=np.float32)
        tsp_arr = np.full(T_max, -1.0, dtype=np.float32)
        ltsp    = np.full(T_max, -100.0, dtype=np.float32)

        # Clip to T_max-1 real blocks (reserve last slot for END if possible)
        T_write = min(T_actual, T_max - 1)
        for t in range(T_write):
            mh, tss, tsp = blocks[t]
            mh_arr[t]  = mh
            tss_arr[t] = tss
            tsp_arr[t] = tsp

        # END block: at T_write if T_actual <= T_max-1, else overwrite last slot
        end_pos = T_write if T_actual < T_max else T_max - 1
        mh_arr[end_pos, C - 1] = 1.0   # END_TOKEN at index C-1

        # label_tsp[t] = tsp of block t+1  (TTNE equivalent for blocks)
        for t in range(T_write - 1):
            ltsp[t] = blocks[t + 1][2]
        # ltsp[T_write-1] stays -100 (last written block → END has no TTNE)

        data.suffix_multihot = torch.from_numpy(mh_arr)
        data.suffix_tss      = torch.from_numpy(tss_arr)
        data.suffix_tsp      = torch.from_numpy(tsp_arr)
        data.label_tsp       = torch.from_numpy(ltsp)
        data.label_rrt       = torch.tensor([float(time_rows['rtime'].iloc[0])], dtype=torch.float32)

        if outcome:
            out_val = float(out_grp[pid][outcome].iloc[0])
            data.outcome_label = torch.tensor([[out_val]], dtype=torch.float32)

        data_list.append(data)

    return data_list


# ---------------------------------------------------------------------------
# T_max computation
# ---------------------------------------------------------------------------

def _compute_T_max(pref_suff, case_id, act_label, timestamp,
                   cardinality_dict, suffix_num_cols):
    """Return the maximum number of timestamp blocks in any suffix + 1 for END."""
    cardinality = cardinality_dict[act_label]
    suffix_df   = pref_suff[1]
    tss_col, tsp_col = suffix_num_cols[0], suffix_num_cols[1]
    max_blocks  = 0
    for _, suff_rows in suffix_df.groupby(case_id, sort=False):
        blocks = _group_suffix_blocks(
            suff_rows, act_label, timestamp, tss_col, tsp_col, cardinality)
        max_blocks = max(max_blocks, len(blocks))
    return max_blocks + 1  # +1 for END block


# ---------------------------------------------------------------------------
# Train / val / test orchestration
# ---------------------------------------------------------------------------

def _multilabel_data_train_test(train_pref_suff, val_pref_suff, test_pref_suff,
                                 case_id, act_label, timestamp,
                                 cardinality_dict, num_cols_dict, cat_cols_dict,
                                 window_size, outcome, log_name):
    cat_cols        = cat_cols_dict['prefix_df']
    suffix_num_cols = num_cols_dict['suffix_df']   # ['ts_start', 'ts_prev']
    num_node_cols   = [c for c in num_cols_dict['prefix_df'] if c != 'ts_prev']

    print("Computing T_max from training set ...")
    T_max = _compute_T_max(train_pref_suff, case_id, act_label, timestamp,
                           cardinality_dict, suffix_num_cols)
    print(f"T_max = {T_max}")

    print("Computing train multilabel dataset ...")
    train_data = _generate_multilabel_dataset(
        train_pref_suff, case_id, act_label, timestamp,
        cat_cols, num_node_cols, suffix_num_cols,
        cardinality_dict, window_size, outcome, T_max)

    print("Computing validation multilabel dataset ...")
    val_data = _generate_multilabel_dataset(
        val_pref_suff, case_id, act_label, timestamp,
        cat_cols, num_node_cols, suffix_num_cols,
        cardinality_dict, window_size, outcome, T_max)

    print("Computing test multilabel dataset ...")
    test_data = _generate_multilabel_dataset(
        test_pref_suff, case_id, act_label, timestamp,
        cat_cols, num_node_cols, suffix_num_cols,
        cardinality_dict, window_size, outcome, T_max)

    pref_cat_cars  = [cardinality_dict[c] for c in cat_cols]
    num_activities = cardinality_dict[act_label] + 2

    output_dir = os.path.join('results_per_log', log_name)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, log_name + '_cardin_list_prefix.pkl'), 'wb') as f:
        pickle.dump(pref_cat_cars, f)
    with open(os.path.join(output_dir, log_name + '_T_max.pkl'), 'wb') as f:
        pickle.dump(T_max, f)

    return train_data, val_data, test_data, pref_cat_cars, T_max, num_activities


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def log_to_graphs_multilabel(log, log_name,
                              start_date, start_before_date, end_date, max_days,
                              test_len_share, val_len_share, window_size, mode,
                              case_id='case:concept:name', act_label='concept:name',
                              timestamp='time:timestamp',
                              cat_casefts=[], num_casefts=[],
                              cat_eventfts=[], num_eventfts=[],
                              outcome=None):
    """Return train/val/test as lists of PyG Data objects with multilabel attributes.

    Preprocessing is identical to log_to_graphs(). Only the suffix representation
    differs: events with the same timestamp are merged into a single multi-hot block.

    Returns
    -------
    train_data, val_data, test_data : list of torch_geometric.data.Data
    counts : dict
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

    print("Generating Multilabel Graph Datasets...")
    train_data, val_data, test_data, *_ = _multilabel_data_train_test(
        train_pref_suff, val_pref_suff, test_pref_suff,
        case_id, act_label, timestamp,
        cardinality_dict, num_cols_dict, cat_cols_dict,
        window_size, outcome, log_name)

    counts = {
        'n_train': n_train, 'train_pairs': p_train,
        'n_val':   n_val,   'val_pairs':   p_val,
        'n_test':  n_test,  'test_pairs':  p_test,
    }
    return train_data, val_data, test_data, counts
