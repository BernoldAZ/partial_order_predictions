"""Module containing the entire pipeline to fit and evaluate the BEST
(Bilaterally Expanding Subtrace Tree) baseline model.

Based on:
    Rauch, S., Frey, C. M. M., Maldonado, A. J., & Seidl, T. (2025).
    BEST: Bilaterally Expanding Subtrace Tree for Event Sequence Prediction.
    In Business Process Management (BPM 2025). Springer, LNCS 16044.
    https://link.springer.com/chapter/10.1007/978-3-032-02867-9_25

BEST is a non-parametric, data-mining-based baseline that predicts activity
suffixes by matching conditional pattern frequencies from the training log.
Unlike the other baselines in this repository, BEST requires no gradient-
based optimisation: the "training" step consists of building an n-gram
pattern tree from the training data.  No epoch loop, no optimizer, and no
checkpoint loading are needed.

Because BEST is control-flow only (NDA), it uses no timestamp or case/event
attributes.  Time metrics (TTNE MAE, RRT MAE) are computed using a constant
predictor equal to the training mean TTNE, consistent with how RRT is derived
in the iterative-feedback SEP-LSTM baseline.
"""

import pandas as pd
import numpy as np
import torch
import os
import pickle
import time
import networkx as nx


def _build_suffix_graph(acts, nb_bits):
    """Build a NetworkX DiGraph for a suffix.

    Mirrors the prefix graph edge structure:
      - intra-block: pairwise bidirectional edges between concurrent events
      - inter-block: all events in block_i -> all events in block_{i+1}

    Position 0 always starts block 0; nb_bits[i] for i >= 1 determines
    whether event i starts a new block (True) or is concurrent with the
    previous event (False).
    """
    n = len(acts)
    G = nx.DiGraph()
    if n == 0:
        return G
    for i in range(n):
        G.add_node(i, act=int(acts[i]))
    blocks = [[0]]
    for i in range(1, n):
        if nb_bits[i]:
            blocks.append([i])
        else:
            blocks[-1].append(i)
    for block in blocks:
        for u in block:
            for v in block:
                if u != v:
                    G.add_edge(u, v)
    for bi in range(len(blocks) - 1):
        for u in blocks[bi]:
            for v in blocks[bi + 1]:
                G.add_edge(u, v)
    return G


def _node_match(n1, n2):
    return n1['act'] == n2['act']


def _graph_edit_similarity(G_pred, G_true):
    """GES = 1 - GED / max(|V|+|E|) using a fast greedy upper bound."""
    np_n, np_e = G_pred.number_of_nodes(), G_pred.number_of_edges()
    nt_n, nt_e = G_true.number_of_nodes(), G_true.number_of_edges()
    if np_n == 0 and nt_n == 0:
        return 1.0
    if np_n == 0 or nt_n == 0:
        return 0.0
    denom = (np_n + np_e) + (nt_n + nt_e)
    ged = next(nx.optimize_graph_edit_distance(G_pred, G_true, node_match=_node_match))
    return 1.0 - ged / denom


def train_eval(log_name, run_id=1):
    """Fit and evaluate the BEST baseline with the parameters used in the
    SuTraN paper.

    Parameters
    ----------
    log_name : str
        Name of the event log.  Must match the ``log_name`` parameter
        passed to ``log_to_tensors()`` in
        ``Preprocessing/from_log_to_tensors.py``.  The preprocessed pickle
        files and tensor datasets are expected to be located in a
        subdirectory called ``log_name`` relative to the current working
        directory.

    Notes
    -----
    Unlike the other baselines (SEP-LSTM, SuTraN NDA), BEST does not require
    a ``tss_index`` parameter.  Those models need it to locate the *time since
    start* / *time since previous* features inside the prefix numerical tensor
    for their iterative feedback loop.  BEST is control-flow only and never
    processes timestamps, so the TTNE / RRT statistics are read directly from
    ``train_means_dict['timeLabel_df']`` at fixed indices 0 (TTNE) and 1 (RRT).
    """

    # -----------------------------------------------------------------------
    # Helper
    # -----------------------------------------------------------------------
    def load_dict(path_name):
        with open(path_name, 'rb') as file:
            return pickle.load(file)

    # -----------------------------------------------------------------------
    # Load preprocessed metadata dictionaries
    # -----------------------------------------------------------------------
    cardinality_dict = load_dict(
        os.path.join('results_per_log', log_name, log_name + '_cardin_dict.pkl')
    )
    num_activities = cardinality_dict['concept:name'] + 2
    print("num_activities:", num_activities)

    num_cols_dict = load_dict(
        os.path.join('results_per_log', log_name, log_name + '_num_cols_dict.pkl')
    )
    cat_cols_dict = load_dict(
        os.path.join('results_per_log', log_name, log_name + '_cat_cols_dict.pkl')
    )
    train_means_dict = load_dict(
        os.path.join('results_per_log', log_name, log_name + '_train_means_dict.pkl')
    )
    train_std_dict = load_dict(
        os.path.join('results_per_log', log_name, log_name + '_train_std_dict.pkl')
    )

    # Standardisation statistics used for de-standardising time predictions
    # and labels during evaluation.
    mean_std_ttne = [
        train_means_dict['timeLabel_df'][0],
        train_std_dict['timeLabel_df'][0]
    ]
    mean_std_rrt = [
        train_means_dict['timeLabel_df'][1],
        train_std_dict['timeLabel_df'][1]
    ]

    num_categoricals_pref = len(cat_cols_dict['prefix_df'])

    # -----------------------------------------------------------------------
    # Create output directory
    # -----------------------------------------------------------------------
    backup_path = os.path.join('results_per_log', log_name, "BEST_results")
    os.makedirs(backup_path, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load datasets
    # -----------------------------------------------------------------------
    train_dataset = torch.load(os.path.join('results_per_log', log_name, 'train_tensordataset.pt'))
    val_dataset   = torch.load(os.path.join('results_per_log', log_name, 'val_tensordataset.pt'))
    test_dataset  = torch.load(os.path.join('results_per_log', log_name, 'test_tensordataset.pt'))

    # -----------------------------------------------------------------------
    # Fit BEST model on training data
    # BEST only needs the activity label prefix tensor, the padding mask,
    # and the activity suffix label tensor -- all present in the original
    # dataset without any conversion.
    # -----------------------------------------------------------------------
    from BEST.best_model import BESTModel

    best_model = BESTModel(
        num_activities=num_activities,
        max_context_length=10
    )

    print("Fitting BEST model on training data ...")
    _train_start = time.time()
    best_model.fit(
        train_dataset=train_dataset,
        num_categoricals_pref=num_categoricals_pref
    )
    print("BEST model fitted successfully.")
    training_time = time.time() - _train_start

    # Optionally persist the fitted model to disk for later reuse
    model_save_path = os.path.join(backup_path, 'best_model.pkl')
    with open(model_save_path, 'wb') as f:
        pickle.dump(best_model, f)
    print("BEST model saved to:", model_save_path)

    # -----------------------------------------------------------------------
    # Run inference on the test set
    # -----------------------------------------------------------------------
    from BEST.inference_procedure_best import inference_loop

    results_path = os.path.join(backup_path, "TEST_SET_RESULTS")
    os.makedirs(results_path, exist_ok=True)

    _test_start = time.time()
    inf_results = inference_loop(
        best_model=best_model,
        inference_dataset=test_dataset,
        num_categoricals_pref=num_categoricals_pref,
        mean_std_ttne=mean_std_ttne,
        mean_std_rrt=mean_std_rrt,
        results_path=results_path,
        dl_batch_size=512
    )
    testing_time = time.time() - _test_start

    # -----------------------------------------------------------------------
    # Unpack and print results
    # -----------------------------------------------------------------------
    avg_dam_lev               = inf_results[0]
    perc_too_early            = inf_results[1]
    perc_too_late             = inf_results[2]
    perc_correct              = inf_results[3]
    mean_absolute_length_diff = inf_results[4]
    mean_too_early            = inf_results[5]
    mean_too_late             = inf_results[6]
    avg_MAE_stand_RRT         = inf_results[7]
    avg_MAE_minutes_RRT       = inf_results[8]
    avg_MAE_ttne_minutes      = inf_results[9]
    results_dict_pref         = inf_results[-2]
    results_dict_suf          = inf_results[-1]

    print("\n=== BEST Test Set Results ===")
    print("Avg 1-(normalised) DL similarity activity suffix: {}".format(avg_dam_lev))
    print(
        "Percentage of suffixes predicted to END: "
        "too early - {} ; right moment - {} ; too late - {}".format(
            perc_too_early, perc_correct, perc_too_late
        )
    )
    print("Too early instances -- avg # events too early: {}".format(mean_too_early))
    print("Too late  instances -- avg # events too late:  {}".format(mean_too_late))
    print("Avg absolute length difference: {}".format(mean_absolute_length_diff))
    print(
        "Avg MAE TTNE (constant-mean predictor): {} (minutes)".format(
            avg_MAE_ttne_minutes
        )
    )
    print(
        "Avg MAE RRT  (constant-mean predictor): {} (minutes)".format(
            avg_MAE_minutes_RRT
        )
    )

    # -----------------------------------------------------------------------
    # Persist aggregated scalar results
    # -----------------------------------------------------------------------
    avg_results_dict = {
        "DL sim"               : avg_dam_lev,
        "MAE TTNE minutes"     : avg_MAE_ttne_minutes,
        "MAE RRT minutes"      : avg_MAE_minutes_RRT,
        "training_time"        : training_time,
        "testing_time"         : testing_time,
        "num_trainable_params" : None,
    }
    # ── Next-act and concurrent-subset metrics ──────────────────────────────
    from sklearn.metrics import accuracy_score, f1_score as _f1_score
    _acts    = torch.load(os.path.join(results_path, 'suffix_acts_decoded.pt'))
    _labels  = torch.load(os.path.join(results_path, 'labels.pt'))
    _act_lbl = _labels[-1]
    def _nap(pred, gt):
        p, g = pred.numpy(), gt.numpy()
        return (float(accuracy_score(g, p)),
                float(_f1_score(g, p, average='weighted', zero_division=0)))
    next_acc, next_f1 = _nap(_acts[:, 0], _act_lbl[:, 0])
    avg_results_dict.update({
        'next_act_accuracy':   round(next_acc, 6),
        'next_act_f1_weighted': round(next_f1,  6),
    })
    _nb_labels = torch.load(
        os.path.join('results_per_log', log_name, 'test_new_block_labels.pt'))
    end_tok = num_activities - 1
    _W = _acts.shape[1]
    _ges_vals = []
    _suf_lens = []
    for _i in range(len(_acts)):
        _end_pos = (_acts[_i] == end_tok).nonzero(as_tuple=True)[0]
        _pl = int(_end_pos[0]) if len(_end_pos) > 0 else _W
        _end_lbl_pos = (_act_lbl[_i] == end_tok).nonzero(as_tuple=True)[0]
        _al = int(_end_lbl_pos[0])
        _suf_lens.append(_al)
        _G_pred = _build_suffix_graph(_acts[_i, :_pl].tolist(), [True] * _pl)
        _G_true = _build_suffix_graph(
            _act_lbl[_i, :_al].tolist(),
            (_nb_labels[_i, :_al] > 0.5).tolist())
        _sim = _graph_edit_similarity(_G_pred, _G_true)
        _ges_vals.append(_sim)
    ges = sum(_ges_vals) / len(_ges_vals) if _ges_vals else 1.0
    avg_results_dict.update({
        'ges_approx': round(ges, 6),
    })
    path_name_average_results = os.path.join(results_path, 'averaged_results.pkl')
    with open(path_name_average_results, 'wb') as f:
        pickle.dump(avg_results_dict, f)

    # -----------------------------------------------------------------------
    # Persist per-length result dictionaries
    # -----------------------------------------------------------------------
    _raw_test = torch.load(os.path.join('results_per_log', log_name, 'test_tensordataset.pt'))
    _pref_lens = (_raw_test[num_categoricals_pref - 1] != 0).sum(dim=1).tolist()
    _ges_by_pref = {}
    _ges_by_suf = {}
    for _plen, _slen, _gval in zip(_pref_lens, _suf_lens, _ges_vals):
        _ges_by_pref.setdefault(int(_plen), []).append(_gval)
        _ges_by_suf.setdefault(_slen, []).append(_gval)
    for _plen, _gvals in _ges_by_pref.items():
        if _plen in results_dict_pref:
            results_dict_pref[_plen].append(sum(_gvals) / len(_gvals))
    for _slen, _gvals in _ges_by_suf.items():
        if _slen in results_dict_suf:
            results_dict_suf[_slen].append(sum(_gvals) / len(_gvals))
    path_name_prefix = os.path.join(results_path, 'prefix_length_results_dict.pkl')
    path_name_suffix = os.path.join(results_path, 'suffix_length_results_dict.pkl')
    with open(path_name_prefix, 'wb') as f:
        pickle.dump(results_dict_pref, f)
    with open(path_name_suffix, 'wb') as f:
        pickle.dump(results_dict_suf, f)
