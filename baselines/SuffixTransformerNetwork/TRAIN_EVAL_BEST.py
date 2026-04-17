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


def train_eval(log_name):
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
    best_model.fit(
        train_dataset=train_dataset,
        num_categoricals_pref=num_categoricals_pref
    )
    print("BEST model fitted successfully.")

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

    inf_results = inference_loop(
        best_model=best_model,
        inference_dataset=test_dataset,
        num_categoricals_pref=num_categoricals_pref,
        mean_std_ttne=mean_std_ttne,
        mean_std_rrt=mean_std_rrt,
        results_path=results_path,
        dl_batch_size=512
    )

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
        "DL sim"          : avg_dam_lev,
        "MAE TTNE minutes": avg_MAE_ttne_minutes,
        "MAE RRT minutes" : avg_MAE_minutes_RRT,
    }
    path_name_average_results = os.path.join(results_path, 'averaged_results.pkl')
    with open(path_name_average_results, 'wb') as f:
        pickle.dump(avg_results_dict, f)

    # -----------------------------------------------------------------------
    # Persist per-length result dictionaries
    # -----------------------------------------------------------------------
    path_name_prefix = os.path.join(results_path, 'prefix_length_results_dict.pkl')
    path_name_suffix = os.path.join(results_path, 'suffix_length_results_dict.pkl')
    with open(path_name_prefix, 'wb') as f:
        pickle.dump(results_dict_pref, f)
    with open(path_name_suffix, 'wb') as f:
        pickle.dump(results_dict_suf, f)
