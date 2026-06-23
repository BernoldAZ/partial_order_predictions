"""SAGE graph preprocessing pipeline.

Identical to from_log_to_tensors_graph.py except:
  - ts_prev is kept in node features (not moved to edge_attr).
  - Edge features (edge_attr) are dropped; edges carry topology only.

Node features per graph
-----------------------
data.cat_x  : int64   [k, num_cat]
data.x      : float32 [k, num_num]   all numeric prefix features INCLUDING ts_prev
                                      column order: num_casefts + num_eventfts +
                                      [ts_start, ts_prev] + indicator_cols
                                      → ts_start at x[:, -2], ts_prev at x[:, -1]

Edge features per graph
-----------------------
data.edge_index : int64 [2, E]   topology only, no edge_attr

Suffix attributes (padded to window_size W)
-------------------------------------------
data.suffix_act    : int64   [W]
data.suffix_num    : float32 [W, 2]
data.ttnext_label  : float32 [W, 1]
data.rtime_label   : float32 [W, 1]
data.act_label_seq : int64   [W]
data.new_block_label : float32 [W]
data.outcome_label : float32 [1, 1]  (only if outcome is not None)
"""

import os
import pickle

import numpy as np
import torch
from torch_geometric.data import Data

from Preprocessing.dataframes_pipeline import main_dataframe_pipeline


def _build_prefix_graph(rows, cat_cols, num_node_cols, timestamp_col):
    """Build a PyG Data object for one prefix. ts_prev is in num_node_cols."""
    k = len(rows)

    cat_arr    = rows[cat_cols].to_numpy().astype(np.int64) + 1     # (k, num_cat)
    num_arr    = rows[num_node_cols].to_numpy().astype(np.float32)   # (k, num_num)
    timestamps = rows[timestamp_col].to_numpy()

    blocks, cur = [], [0]
    for i in range(1, k):
        if timestamps[i] == timestamps[i - 1]:
            cur.append(i)
        else:
            blocks.append(cur)
            cur = [i]
    blocks.append(cur)

    src_list, dst_list = [], []

    for block in blocks:
        if len(block) < 2:
            continue
        for i in block:
            for j in block:
                if i != j:
                    src_list.append(i)
                    dst_list.append(j)

    for bi in range(len(blocks) - 1):
        for u in blocks[bi]:
            for v in blocks[bi + 1]:
                src_list.append(u)
                dst_list.append(v)

    if src_list:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    last_block_mask = torch.zeros(k, dtype=torch.bool)
    for idx in blocks[-1]:
        last_block_mask[idx] = True

    return Data(
        x=torch.from_numpy(num_arr),
        cat_x=torch.from_numpy(cat_arr),
        edge_index=edge_index,
        last_block_mask=last_block_mask,
    )


def _make_act_label_mapping(cardinality_dict, act_label):
    car  = cardinality_dict[act_label] + 1
    keys = [str(i) for i in range(car - 1)] + ['END_TOKEN']
    return dict(zip(keys, range(1, car + 1)))


def _generate_sage_dataset(pref_suff, case_id, act_label, timestamp,
                            cat_cols, num_node_cols, suffix_num_cols,
                            cardinality_dict, window_size, outcome,
                            act_mapping):
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

        data = _build_prefix_graph(pref_rows, cat_cols, num_node_cols, timestamp)
        last_num = pref_rows[['ts_start', 'ts_prev']].iloc[-1].to_numpy().astype(np.float32)
        data.last_prefix_num = torch.from_numpy(last_num).unsqueeze(0)
        pref_ts = pref_rows[timestamp].to_numpy()
        last_nb_val = float(len(pref_ts) == 1 or pref_ts[-1] != pref_ts[-2])
        data.last_prefix_nb = torch.tensor([last_nb_val], dtype=torch.float32)

        s_act = suff_rows[act_label].to_numpy().astype(np.int64) + 1
        buf = np.zeros(W, dtype=np.int64)
        buf[:len(s_act)] = s_act
        data.suffix_act = torch.from_numpy(buf)

        s_num = suff_rows[suffix_num_cols].to_numpy().astype(np.float32)
        buf2 = np.full((W, num_suf), -1.0, dtype=np.float32)
        buf2[:len(s_num)] = s_num
        data.suffix_num = torch.from_numpy(buf2)

        ttnext = time_rows['tt_next'].to_numpy().astype(np.float32)
        rtime  = time_rows['rtime'].to_numpy().astype(np.float32)
        ttn_buf = np.full(W, -100.0, dtype=np.float32)
        rt_buf  = np.full(W, -100.0, dtype=np.float32)
        ttn_buf[:len(ttnext)] = ttnext
        rt_buf[:len(rtime)]   = rtime
        data.ttnext_label = torch.from_numpy(ttn_buf).unsqueeze(1)
        data.rtime_label  = torch.from_numpy(rt_buf).unsqueeze(1)

        act_seq = (act_rows[act_label]
                   .astype(str)
                   .map(act_mapping)
                   .to_numpy().astype(np.int64))
        act_buf = np.zeros(W, dtype=np.int64)
        act_buf[:len(act_seq)] = act_seq
        data.act_label_seq = torch.from_numpy(act_buf)

        suff_ts = suff_rows[timestamp].to_numpy()
        prev_ts = np.concatenate([[pref_rows[timestamp].iloc[-1]], suff_ts[:-1]])
        nb_arr  = (suff_ts != prev_ts).astype(np.float32)
        nb_buf  = np.zeros(W, dtype=np.float32)
        nb_buf[:len(nb_arr)] = nb_arr
        data.new_block_label = torch.from_numpy(nb_buf)

        if outcome:
            out_val = float(out_grp[pid][outcome].iloc[0])
            data.outcome_label = torch.tensor([[out_val]], dtype=torch.float32)

        data_list.append(data)

    return data_list


def _sage_data_train_test(train_pref_suff, val_pref_suff, test_pref_suff,
                           case_id, act_label, timestamp,
                           cardinality_dict, num_cols_dict, cat_cols_dict,
                           window_size, outcome, log_name):
    cat_cols        = cat_cols_dict['prefix_df']
    suffix_num_cols = num_cols_dict['suffix_df']
    # Include ts_prev in node features (unlike graph version which excludes it)
    num_node_cols   = num_cols_dict['prefix_df']
    act_mapping     = _make_act_label_mapping(cardinality_dict, act_label)

    print("Computing train SAGE dataset ...")
    train_data = _generate_sage_dataset(
        train_pref_suff, case_id, act_label, timestamp,
        cat_cols, num_node_cols, suffix_num_cols,
        cardinality_dict, window_size, outcome, act_mapping)

    print("Computing validation SAGE dataset ...")
    val_data = _generate_sage_dataset(
        val_pref_suff, case_id, act_label, timestamp,
        cat_cols, num_node_cols, suffix_num_cols,
        cardinality_dict, window_size, outcome, act_mapping)

    print("Computing test SAGE dataset ...")
    test_data = _generate_sage_dataset(
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


def log_to_sage_graphs(log,
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
    """Return train/val/test as lists of PyG Data objects with ts_prev on nodes.

    Same interface as log_to_graphs() but ts_prev is included in data.x
    instead of edge_attr. Edges carry topology only (no edge_attr).
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

    print("Generating SAGE Datasets...")
    train_data, val_data, test_data, *_ = _sage_data_train_test(
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
