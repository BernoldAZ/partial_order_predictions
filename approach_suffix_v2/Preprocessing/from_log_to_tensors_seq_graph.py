"""Sequence-graph preprocessing pipeline.

Same as from_log_to_tensors_graph.py, except the prefix graph is a linear
chain instead of a partial-order (timestamp-block) graph.

Graph construction (per prefix):
  - One node per prefix event.
  - Node features: same as from_log_to_tensors_graph.py.
  - Edges: event_i -> event_{i+1} for i in 0..k-2  (directed chain).
    edge_attr = standardised ts_prev of the destination node, i.e. the
    elapsed time from event i to event i+1.
  - last_block_mask : bool [k]  True only for the last node.

Suffix attributes are identical to from_log_to_tensors_graph.py.
Saves datasets as train/val/test_seqgraphdataset.pt.
"""

import os
import pickle

import numpy as np
import torch
from torch_geometric.data import Data

from Preprocessing.dataframes_pipeline import main_dataframe_pipeline


# ---------------------------------------------------------------------------
# Graph builder — chain topology
# ---------------------------------------------------------------------------

def _build_prefix_graph(rows, cat_cols, num_node_cols, timestamp_col):
    """Build a PyG Data object for one prefix as a directed chain graph."""
    k = len(rows)

    cat_arr     = rows[cat_cols].to_numpy().astype(np.int64) + 1
    num_arr     = rows[num_node_cols].to_numpy().astype(np.float32)
    ts_prev_arr = rows['ts_prev'].to_numpy().astype(np.float32)

    src_list, dst_list, attr_list = [], [], []
    for i in range(k - 1):
        src_list.append(i)
        dst_list.append(i + 1)
        attr_list.append(float(ts_prev_arr[i + 1]))

    if src_list:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr  = torch.tensor(attr_list, dtype=torch.float32).unsqueeze(1)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, 1), dtype=torch.float32)

    last_block_mask = torch.zeros(k, dtype=torch.bool)
    if k > 0:
        last_block_mask[-1] = True

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

        suff_ts     = suff_rows[timestamp].to_numpy()
        prev_ts     = np.concatenate([[pref_rows[timestamp].iloc[-1]], suff_ts[:-1]])
        nb_arr      = (suff_ts != prev_ts).astype(np.float32)
        nb_buf      = np.zeros(W, dtype=np.float32)
        nb_buf[:len(nb_arr)] = nb_arr
        data.new_block_label = torch.from_numpy(nb_buf)

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
    num_node_cols   = [c for c in num_cols_dict['prefix_df'] if c != 'ts_prev']
    act_mapping     = _make_act_label_mapping(cardinality_dict, act_label)

    print("Computing train seq-graph dataset ...")
    train_data = _generate_graph_dataset(
        train_pref_suff, case_id, act_label, timestamp,
        cat_cols, num_node_cols, suffix_num_cols,
        cardinality_dict, window_size, outcome, act_mapping)

    print("Computing validation seq-graph dataset ...")
    val_data = _generate_graph_dataset(
        val_pref_suff, case_id, act_label, timestamp,
        cat_cols, num_node_cols, suffix_num_cols,
        cardinality_dict, window_size, outcome, act_mapping)

    print("Computing test seq-graph dataset ...")
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

def log_to_seq_graphs(log,
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
    """Same as log_to_graphs() but prefix graphs are directed chains."""
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

    print("Generating Seq-Graph Datasets...")
    train_data, val_data, test_data, *_ = _graph_data_train_test(
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
