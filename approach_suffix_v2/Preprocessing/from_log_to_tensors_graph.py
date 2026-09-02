"""Graph-based preprocessing pipeline.

Drop-in companion to from_log_to_tensors.py. Identical preprocessing steps,
but prefixes are returned as lists of PyG Data objects instead of padded tensors.

Graph construction (per prefix):
  - One node per prefix event.
  - Node features:
      cat_x  (int64,   [k, num_cat])  all categoricals (act_label + cat_casefts +
                                       cat_eventfts), integer-encoded, shifted +1
                                       so that 0 serves as the padding token.
      x      (float32, [k, num_num])  all numeric prefix features EXCEPT ts_prev
                                       (ts_start + num_casefts + num_eventfts +
                                       missing-value indicators).
  - Blocks: consecutive events sharing the same timestamp form a block.
  - Edges:
      Intra-block  both directions, pairwise fully connected.
                   edge_attr = standardised ts_prev of any non-first node in the
                   block, which equals (0 - mean_ts_prev) / std_ts_prev because
                   events in the same block have 0 raw elapsed time between them.
      Inter-block  every node in block_i -> every node in block_{i+1}.
                   edge_attr = standardised ts_prev of the first node in block_{i+1},
                   which holds the actual inter-block elapsed time.
  - last_block_mask : bool [k]  True for nodes in the last concurrent block of the prefix.
  - Suffix (padded to window_size, attached directly to the Data object):
      suffix_act      (int64,   [W])    activity labels of suffix events (shifted +1, 0=pad)
      suffix_num      (float32, [W, 2]) [ts_start, ts_prev] of suffix events (-1=pad)
      ttnext_label    (float32, [W, 1]) time-till-next-event targets (-100=pad)
      rtime_label     (float32, [W, 1]) remaining-runtime targets (-100=pad)
      act_label_seq   (int64,   [W])    ground-truth activity labels incl. END token (0=pad)
      new_block_label (float32, [W])    1.0 if suffix event starts a new block (timestamp
                                        differs from previous event), 0.0 if concurrent
                                        with previous event (0=pad)
      outcome_label   (float32, [1, 1]) binary outcome (only present if outcome is not None)
"""

import os
import pickle

import numpy as np
import torch
from torch_geometric.data import Data

from Preprocessing.dataframes_pipeline import main_dataframe_pipeline


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def _build_prefix_graph(rows, cat_cols, num_node_cols, timestamp_col):
    """Build a PyG Data object for one prefix from its (ordered) event rows."""
    k = len(rows)

    cat_arr     = rows[cat_cols].to_numpy().astype(np.int64) + 1    # (k, num_cat)
    num_arr     = rows[num_node_cols].to_numpy().astype(np.float32)  # (k, num_num)
    ts_prev_arr = rows['ts_prev'].to_numpy().astype(np.float32)      # (k,)
    timestamps  = rows[timestamp_col].to_numpy()

    # Group consecutive events with the same timestamp into blocks
    blocks, cur = [], [0]
    for i in range(1, k):
        if timestamps[i] == timestamps[i - 1]:
            cur.append(i)
        else:
            blocks.append(cur)
            cur = [i]
    blocks.append(cur)

    src_list, dst_list, attr_list = [], [], []

    # Intra-block: pairwise fully connected (both directions).
    # Events in the same block share the same timestamp, so the raw elapsed
    # time between them is 0 seconds.  Their standardised ts_prev is therefore
    # (0 - mean) / std.  In the DataFrame this value is already stored for
    # every non-first event in the block (their ts_prev = standardised 0).
    # We read it directly from block[1] when the block has >=2 events.
    for block in blocks:
        if len(block) < 2:
            continue
        # ts_prev of block[1] = standardised(0) = (0 - mean_ts_prev) / std_ts_prev
        intra_val = float(ts_prev_arr[block[1]])
        for i in block:
            for j in block:
                if i != j:
                    src_list.append(i)
                    dst_list.append(j)
                    attr_list.append(intra_val)

    # Inter-block: every node in block_i -> every node in block_{i+1}.
    # edge_attr = standardised ts_prev of the first node in block_{i+1},
    # which holds the actual elapsed time from block_i to block_{i+1}.
    for bi in range(len(blocks) - 1):
        edge_val = float(ts_prev_arr[blocks[bi + 1][0]])
        for u in blocks[bi]:
            for v in blocks[bi + 1]:
                src_list.append(u)
                dst_list.append(v)
                attr_list.append(edge_val)

    if src_list:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr  = torch.tensor(attr_list, dtype=torch.float32).unsqueeze(1)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, 1), dtype=torch.float32)

    last_block_mask = torch.zeros(k, dtype=torch.bool)
    for idx in blocks[-1]:
        last_block_mask[idx] = True

    return Data(
        x=torch.from_numpy(num_arr),
        cat_x=torch.from_numpy(cat_arr),
        edge_index=edge_index,
        edge_attr=edge_attr,
        last_block_mask=last_block_mask,
    )


# ---------------------------------------------------------------------------
# Activity-label integer mapping (includes END_TOKEN)
# ---------------------------------------------------------------------------

def _make_act_label_mapping(cardinality_dict, act_label):
    """Return str->int mapping for ground-truth activity labels."""
    car  = cardinality_dict[act_label] + 1
    keys = [str(i) for i in range(car - 1)] + ['END_TOKEN']
    return dict(zip(keys, range(1, car + 1)))


# ---------------------------------------------------------------------------
# Per-split dataset builder
# ---------------------------------------------------------------------------

def _generate_graph_dataset(pref_suff, case_id, act_label, timestamp,
                             cat_cols, num_node_cols, suffix_num_cols,
                             cardinality_dict, window_size, outcome,
                             act_mapping):
    """Convert one pref_suff tuple into a list of PyG Data objects."""
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

    W       = window_size
    num_suf = len(suffix_num_cols)
    data_list = []

    for pid in prefix_ids:
        pref_rows = pref_grp[pid]
        suff_rows = suff_grp[pid]
        time_rows = tlab_grp[pid]
        act_rows  = alab_grp[pid]

        # --- Prefix graph ---
        data = _build_prefix_graph(pref_rows, cat_cols, num_node_cols, timestamp)
        last_num = pref_rows[['ts_start', 'ts_prev']].iloc[-1].to_numpy().astype(np.float32)
        data.last_prefix_num = torch.from_numpy(last_num).unsqueeze(0)  # (1, 2) → batched: (B, 2)
        pref_ts = pref_rows[timestamp].to_numpy()
        last_nb_val = float(len(pref_ts) == 1 or pref_ts[-1] != pref_ts[-2])
        data.last_prefix_nb = torch.tensor([last_nb_val], dtype=torch.float32)

        # --- Suffix activity (int64, shifted +1, padded 0) ---
        s_act = suff_rows[act_label].to_numpy().astype(np.int64) + 1
        buf = np.zeros(W, dtype=np.int64)
        buf[:len(s_act)] = s_act
        data.suffix_act = torch.from_numpy(buf)

        # --- Suffix numeric [ts_start, ts_prev] (float32, padded -1) ---
        s_num = suff_rows[suffix_num_cols].to_numpy().astype(np.float32)
        buf2 = np.full((W, num_suf), -1.0, dtype=np.float32)
        buf2[:len(s_num)] = s_num
        data.suffix_num = torch.from_numpy(buf2)

        # --- Time labels (float32, padded -100) ---
        ttnext = time_rows['tt_next'].to_numpy().astype(np.float32)
        rtime  = time_rows['rtime'].to_numpy().astype(np.float32)
        ttn_buf = np.full(W, -100.0, dtype=np.float32)
        rt_buf  = np.full(W, -100.0, dtype=np.float32)
        ttn_buf[:len(ttnext)] = ttnext
        rt_buf[:len(rtime)]   = rtime
        data.ttnext_label = torch.from_numpy(ttn_buf).unsqueeze(1)
        data.rtime_label  = torch.from_numpy(rt_buf).unsqueeze(1)

        # --- Ground-truth activity label sequence (int64, padded 0) ---
        act_seq = (act_rows[act_label]
                   .astype(str)
                   .map(act_mapping)
                   .to_numpy().astype(np.int64))
        act_buf = np.zeros(W, dtype=np.int64)
        act_buf[:len(act_seq)] = act_seq
        data.act_label_seq = torch.from_numpy(act_buf)

        # --- New-block label (float32, padded 0.0) ---
        # 1.0 = suffix event starts a new block (timestamp differs from previous event)
        # 0.0 = concurrent with previous event (same block), or padding
        suff_ts     = suff_rows[timestamp].to_numpy()
        prev_ts     = np.concatenate([[pref_rows[timestamp].iloc[-1]], suff_ts[:-1]])
        nb_arr      = (suff_ts != prev_ts).astype(np.float32)
        nb_buf      = np.zeros(W, dtype=np.float32)
        nb_buf[:len(nb_arr)] = nb_arr
        data.new_block_label = torch.from_numpy(nb_buf)

        # --- Outcome (optional) ---
        if outcome:
            out_val = float(out_grp[pid][outcome].iloc[0])
            data.outcome_label = torch.tensor([[out_val]], dtype=torch.float32)

        data_list.append(data)

    return data_list


# ---------------------------------------------------------------------------
# Train / val / test orchestration
# ---------------------------------------------------------------------------

def _graph_data_train_test(train_pref_suff, val_pref_suff, test_pref_suff,
                            case_id, act_label, timestamp,
                            cardinality_dict, num_cols_dict, cat_cols_dict,
                            window_size, outcome, log_name):
    cat_cols        = cat_cols_dict['prefix_df']
    suffix_num_cols = num_cols_dict['suffix_df']
    # Node numeric features = all prefix numeric cols except ts_prev (moves to edges)
    num_node_cols   = [c for c in num_cols_dict['prefix_df'] if c != 'ts_prev']
    act_mapping     = _make_act_label_mapping(cardinality_dict, act_label)

    print("Computing train graph dataset ...")
    train_data = _generate_graph_dataset(
        train_pref_suff, case_id, act_label, timestamp,
        cat_cols, num_node_cols, suffix_num_cols,
        cardinality_dict, window_size, outcome, act_mapping)

    print("Computing validation graph dataset ...")
    val_data = _generate_graph_dataset(
        val_pref_suff, case_id, act_label, timestamp,
        cat_cols, num_node_cols, suffix_num_cols,
        cardinality_dict, window_size, outcome, act_mapping)

    print("Computing test graph dataset ...")
    test_data = _generate_graph_dataset(
        test_pref_suff, case_id, act_label, timestamp,
        cat_cols, num_node_cols, suffix_num_cols,
        cardinality_dict, window_size, outcome, act_mapping)

    pref_cat_cars  = [cardinality_dict[c] for c in cat_cols]
    suff_cat_cars  = [cardinality_dict[c] for c in cat_cols_dict['suffix_df']]
    num_activities = cardinality_dict[act_label] + 2

    output_dir = os.path.join('results_per_log', log_name)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, log_name + '_cardin_list_prefix.pkl'), 'wb') as f:
        pickle.dump(pref_cat_cars, f)
    with open(os.path.join(output_dir, log_name + '_cardin_list_suffix.pkl'), 'wb') as f:
        pickle.dump(suff_cat_cars, f)

    return (train_data, val_data, test_data,
            len(cat_cols), len(cat_cols_dict['suffix_df']),
            pref_cat_cars, suff_cat_cars, num_activities)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def log_to_graphs(log,
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
    """Return train/val/test as lists of PyG Data objects.

    Same parameters and preprocessing as log_to_tensors(). The only
    difference is how prefixes are represented: padded (N, W) tensors are
    replaced by per-instance PyG Data graphs. Suffix tensors, time labels,
    and activity labels are stored as extra attributes on each Data object.

    Node features per graph
    -----------------------
    data.cat_x  : int64   [k, num_cat]   categorical features (shifted +1)
    data.x      : float32 [k, num_num]   numeric features, ts_prev excluded

    Edge features per graph
    -----------------------
    data.edge_index : int64   [2, E]
    data.edge_attr  : float32 [E, 1]
        Intra-block: standardised(0) = (0 - mean_ts_prev) / std_ts_prev
        Inter-block: standardised ts_prev of the destination block's first event

    Suffix attributes (padded to window_size W)
    -------------------------------------------
    data.suffix_act    : int64   [W]
    data.suffix_num    : float32 [W, 2]
    data.ttnext_label  : float32 [W, 1]
    data.rtime_label   : float32 [W, 1]
    data.act_label_seq : int64   [W]
    data.outcome_label : float32 [1, 1]  (only if outcome is not None)

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
    print(f"Cases – train: {n_train}  val: {n_val}  test: {n_test}")

    print("Generating Graph Datasets...")
    train_data, val_data, test_data, *_ = _graph_data_train_test(
        train_pref_suff, val_pref_suff, test_pref_suff,
        case_id, act_label, timestamp,
        cardinality_dict, num_cols_dict, cat_cols_dict,
        window_size, outcome, log_name)

    # Prefix-suffix pairs actually retained (after out-of-time split debiasing).
    p_train, p_val, p_test = len(train_data), len(val_data), len(test_data)
    print(f"Pairs – train: {p_train}  val: {p_val}  test: {p_test}")

    counts = {
        'n_train': n_train, 'train_pairs': p_train,
        'n_val':   n_val,   'val_pairs':   p_val,
        'n_test':  n_test,  'test_pairs':  p_test,
    }
    return train_data, val_data, test_data, counts
