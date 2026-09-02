"""Flipped-order variant of the sequence-graph preprocessing pipeline.

Identical to from_log_to_tensors_seq_graph.py except:
  - Only the test set is generated.
  - Within each parallel block (same timestamp), the event order is reversed
    before time features are computed, so all derived fields (ts_prev, edge
    attributes, suffix labels, actLabel) are consistent with the flipped order.
  - Output is saved as test_seqgraphdataset_flip.pt in results_per_log/<log_name>/.
"""

import os
import pickle

import torch

from Preprocessing.from_log_to_tensors_seq_graph import (
    _build_prefix_graph,
    _make_act_label_mapping,
    _generate_graph_dataset,
)
from Preprocessing.dataframes_pipeline import main_dataframe_pipeline


def _graph_data_test_flip(test_pref_suff,
                          case_id, act_label, timestamp,
                          cardinality_dict, num_cols_dict, cat_cols_dict,
                          window_size, outcome):
    cat_cols        = cat_cols_dict['prefix_df']
    suffix_num_cols = num_cols_dict['suffix_df']
    num_node_cols   = [c for c in num_cols_dict['prefix_df'] if c != 'ts_prev']
    act_mapping     = _make_act_label_mapping(cardinality_dict, act_label)

    print("Computing test seq-graph dataset (flipped) ...")
    return _generate_graph_dataset(
        test_pref_suff, case_id, act_label, timestamp,
        cat_cols, num_node_cols, suffix_num_cols,
        cardinality_dict, window_size, outcome, act_mapping)


def log_to_seq_graphs_flip(log,
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
    """Same as log_to_seq_graphs() but only generates the test set with
    within-block event order reversed."""
    log_transformed = False
    print("Generating Dataframes (flipped test) ...")
    (_, _, test_pref_suff,
     cardinality_dict, num_cols_dict, cat_cols_dict,
     _, _) = main_dataframe_pipeline(
        log, log_name, start_date, start_before_date, end_date, max_days,
        test_len_share, val_len_share, window_size, log_transformed, mode,
        case_id, act_label, timestamp,
        cat_casefts, num_casefts, cat_eventfts, num_eventfts, outcome,
        flip_test=True)

    n_test = test_pref_suff[0]['orig_case_id'].nunique()

    print("Generating Seq-Graph Dataset (flipped) ...")
    test_data = _graph_data_test_flip(
        test_pref_suff, case_id, act_label, timestamp,
        cardinality_dict, num_cols_dict, cat_cols_dict,
        window_size, outcome)

    # Prefix-suffix pairs actually retained (after out-of-time split debiasing).
    p_test = len(test_data)
    print(f"Cases – test: {n_test}  pairs: {p_test}")

    output_dir = os.path.join('results_per_log', log_name)
    os.makedirs(output_dir, exist_ok=True)
    torch.save(test_data, os.path.join(output_dir, 'test_seqgraphdataset_flip.pt'))
    print(f"Saved to '{output_dir}/test_seqgraphdataset_flip.pt'")

    return {'n_test': n_test, 'test_pairs': p_test}
