#!/usr/bin/env python

import os, random, json, itertools
import numpy as np
import pandas as pd
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.obj import EventLog
import pm4py
from pydream.LogWrapper import LogWrapper
from pydream.EnhancedPN import EnhancedPN
from pydream.predictive.nap.NAP import NAP
from definitions import ROOT_DIR
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score
)

random.seed(42)
np.random.seed(42)


def batched_predict(nap, tss_objs, batch_size=512):
    """Batched replacement for NAP.predict() — one model.predict() call instead of N."""
    all_features = []
    for sample in tss_objs:
        exported = sample.export()
        all_features.append(list(itertools.chain(exported["tss"][0], exported["tss"][1], exported["tss"][2])))
    X = nap.stdScaler.transform(np.array(all_features))
    probs = nap.model.predict(X, batch_size=batch_size, verbose=0)
    pred_indices = np.argmax(probs, axis=1)
    # Build index→event mapping from the one_hot_dict
    idx_to_event = {int(np.argmax(v)): k for k, v in nap.one_hot_dict.items()}
    pred_events = [idx_to_event[i] for i in pred_indices]
    return pred_indices.tolist(), pred_events

# -----------------------
# Training parameters.
EPOCHS = 100
BATCH_SIZE = 32

# -----------------------
# Directories.
NA_DIR = os.path.join(ROOT_DIR, "evaluation")
RAW_DATASETS_DIR = os.path.join(NA_DIR, "raw_datasets_that_are_not_evaluated")
SPLIT_DATASETS_DIR = os.path.join(NA_DIR, "split_datasets")
RESULTS_DIR = os.path.join(NA_DIR, "results_pydream")
MODELS_DIR = os.path.join(NA_DIR, "models_pydream")
TSS_DIR = os.path.join(NA_DIR, "tss_pydream")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TSS_DIR, exist_ok=True)

# ----------------------- Single-log endpoint -----------------------
def run_pydream_for_log(
    log_name,
    split_dir=None,
    results_dir=None,
    models_dir=None,
    tss_dir=None,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    random_seed=42,
):
    """Train and evaluate the PyDREAM NAP model for one event log.

    Designed to be called from a Jupyter notebook::

        from next_activity_prediction_pydream import run_pydream_for_log

        result = run_pydream_for_log(
            log_name='bpic2019',
            epochs=100,
        )

    Parameters
    ----------
    log_name : str
        Base name of the event log (without path or extension), matching
        the file-name convention used by the split script.
    split_dir : str or None
        Directory containing pre-split XES files named
        ``train_<log_name>.xes.gz``, ``val_<log_name>.xes.gz``, and
        ``test_<log_name>.xes.gz``.  Defaults to ``split_datasets``.
    results_dir : str or None
        Directory where result files are written.  Defaults to
        ``results_pydream``.
    models_dir : str or None
        Directory where NAP model checkpoints are saved.  Defaults to
        ``models_pydream``.
    tss_dir : str or None
        Directory where timed state sample JSON files are cached.  Defaults
        to ``tss_pydream``.
    epochs : int
        Maximum training epochs.  Default taken from module-level ``EPOCHS``.
    batch_size : int
        Mini-batch size.  Default taken from module-level ``BATCH_SIZE``.
    random_seed : int
        Seed applied to Python ``random`` and NumPy.

    Returns
    -------
    dict
        Dictionary with keys ``log``, ``accuracy``, ``mcc``,
        ``weighted_recall``, ``weighted_precision``, ``f1``.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    # ── Directories ──────────────────────────────────────────────────────────
    _split_dir   = split_dir   or SPLIT_DATASETS_DIR
    _results_dir = results_dir or RESULTS_DIR
    _models_dir  = models_dir  or MODELS_DIR
    _tss_dir     = tss_dir     or TSS_DIR
    os.makedirs(_results_dir, exist_ok=True)
    os.makedirs(_models_dir,  exist_ok=True)
    os.makedirs(_tss_dir,     exist_ok=True)

    # ── Load splits ──────────────────────────────────────────────────────────
    train_path = os.path.join(_split_dir, f"train_{log_name}.xes.gz")
    val_path   = os.path.join(_split_dir, f"val_{log_name}.xes.gz")
    test_path  = os.path.join(_split_dir, f"test_{log_name}.xes.gz")
    print(f"\n{'='*60}")
    print(f"Processing log: {log_name}")
    print(f"  train : {train_path}")
    print(f"  val   : {val_path}")
    print(f"  test  : {test_path}")
    print(f"{'='*60}")

    train_log = xes_importer.apply(train_path, parameters={"timestamp_sort": True, "timestamp_key": "time:timestamp"})
    val_log   = xes_importer.apply(val_path,   parameters={"timestamp_sort": True, "timestamp_key": "time:timestamp"})
    test_log  = xes_importer.apply(test_path,  parameters={"timestamp_sort": True, "timestamp_key": "time:timestamp"})

    # ── Petri net ────────────────────────────────────────────────────────────
    train_val_log = EventLog(list(train_log) + list(val_log))
    train_val_log.extensions.update(train_log.extensions)
    train_val_log.classifiers.update(train_log.classifiers)
    train_val_log.attributes.update(train_log.attributes)

    net, initial_marking, _ = pm4py.discover_petri_net_inductive(train_log)
    print("Petri net discovered from training log.")

    enhanced_pn = EnhancedPN(net, initial_marking)
    enhanced_pn.enhance(LogWrapper(train_val_log))
    print("Petri net enhanced with decay functions from train+val.")

    # ── TSS generation ───────────────────────────────────────────────────────
    tss_train_path = os.path.join(_tss_dir, f"tss_train_{log_name}.json")
    tss_val_path   = os.path.join(_tss_dir, f"tss_val_{log_name}.json")
    tss_test_path  = os.path.join(_tss_dir, f"tss_test_{log_name}.json")

    tss_train_json, _          = enhanced_pn.decay_replay(log_wrapper=LogWrapper(train_log))
    tss_val_json,   _          = enhanced_pn.decay_replay(log_wrapper=LogWrapper(val_log))
    tss_test_json,  tss_test_objs = enhanced_pn.decay_replay(log_wrapper=LogWrapper(test_log))

    with open(tss_train_path, "w") as f:
        json.dump(tss_train_json, f)
    with open(tss_val_path, "w") as f:
        json.dump(tss_val_json, f)
    with open(tss_test_path, "w") as f:
        json.dump(tss_test_json, f)
    print("TSS generated and saved for train, val, test.")

    # ── Train NAP ────────────────────────────────────────────────────────────
    model_dir = os.path.join(_models_dir, log_name)
    os.makedirs(model_dir, exist_ok=True)

    nap = NAP(tss_train_path, tss_test_path, options={"n_epochs": epochs, "n_batch_size": batch_size})
    print(f"Training NAP for log {log_name} ...")
    nap.train(model_dir, log_name, save_results=False)
    nap.loadModel(model_dir, log_name)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    labeled_pairs = [(obj, s["label"]) for obj, s in zip(tss_test_objs, tss_test_json)
                     if s["label"] is not None]
    if not labeled_pairs:
        print(f"No labeled test samples for {log_name}, skipping.")
        return None
    tss_test_labeled_objs, y_true = zip(*labeled_pairs)
    _, y_pred = batched_predict(nap, list(tss_test_labeled_objs))
    y_true = list(y_true)

    acc       = accuracy_score(y_true, y_pred)
    mcc       = matthews_corrcoef(y_true, y_pred)
    w_recall  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    w_prec    = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    w_f1      = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"  Accuracy={acc:.4f}  MCC={mcc:.4f}  F1={w_f1:.4f}")

    # ── Save results ─────────────────────────────────────────────────────────
    result_dict = {
        "log": log_name,
        "accuracy": acc,
        "mcc": mcc,
        "weighted_recall": w_recall,
        "weighted_precision": w_prec,
        "f1": w_f1,
    }
    result_dir = os.path.join(_results_dir, log_name)
    os.makedirs(result_dir, exist_ok=True)
    pd.DataFrame([result_dict]).to_csv(os.path.join(result_dir, f"{log_name}_results.csv"), index=False)
    with open(os.path.join(result_dir, f"{log_name}.txt"), "w") as f:
        f.write(f"Accuracy: {acc}\n")
        f.write(f"MCC: {mcc}\n")
        f.write(f"Weighted recall: {w_recall}\n")
        f.write(f"Weighted precision: {w_prec}\n")
        f.write(f"Weighted f1: {w_f1}\n")
    print(f"Results saved to '{result_dir}'")
    return result_dict


if __name__ == "__main__":
    raw_logs = [f for f in os.listdir(RAW_DATASETS_DIR) if f.endswith(".xes.gz")]
    print(raw_logs)
    results_summary = []

    for raw_log in raw_logs:
        log_name = os.path.splitext(os.path.splitext(raw_log)[0])[0]
        train_path = os.path.join(SPLIT_DATASETS_DIR, f"train_{log_name}.xes.gz")
        val_path = os.path.join(SPLIT_DATASETS_DIR, f"val_{log_name}.xes.gz")
        test_path = os.path.join(SPLIT_DATASETS_DIR, f"test_{log_name}.xes.gz")
        print("\n========== Processing log:", log_name, "==========")
        print("Train:", train_path)
        print("Val:", val_path)
        print("Test:", test_path)

        # Load logs.
        train_log = xes_importer.apply(train_path, parameters={"timestamp_sort": True, "timestamp_key": "time:timestamp"})
        val_log = xes_importer.apply(val_path, parameters={"timestamp_sort": True, "timestamp_key": "time:timestamp"})
        test_log = xes_importer.apply(test_path, parameters={"timestamp_sort": True, "timestamp_key": "time:timestamp"})

        # Combine train+val for Petri net enhancement (no test leakage).
        train_val_log = EventLog(list(train_log) + list(val_log))
        train_val_log.extensions.update(train_log.extensions)
        train_val_log.classifiers.update(train_log.classifiers)
        train_val_log.attributes.update(train_log.attributes)

        # Discover Petri net from training log only (no test leakage).
        net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(train_log)
        print("Petri net discovered from training log.")

        # Enhance Petri net with decay functions from train+val only (no test leakage).
        enhanced_pn = EnhancedPN(net, initial_marking)
        enhanced_pn.enhance(LogWrapper(train_val_log))
        print("Petri net enhanced with decay functions from train+val.")

        # Generate timed state samples (TSS) via decay replay for each split.
        tss_train_path = os.path.join(TSS_DIR, f"tss_train_{log_name}.json")
        tss_val_path = os.path.join(TSS_DIR, f"tss_val_{log_name}.json")
        tss_test_path = os.path.join(TSS_DIR, f"tss_test_{log_name}.json")

        tss_train_json, _ = enhanced_pn.decay_replay(log_wrapper=LogWrapper(train_log))
        tss_val_json, _ = enhanced_pn.decay_replay(log_wrapper=LogWrapper(val_log))
        tss_test_json, tss_test_objs = enhanced_pn.decay_replay(log_wrapper=LogWrapper(test_log))

        with open(tss_train_path, "w") as f:
            json.dump(tss_train_json, f)
        with open(tss_val_path, "w") as f:
            json.dump(tss_val_json, f)
        with open(tss_test_path, "w") as f:
            json.dump(tss_test_json, f)
        print("TSS generated and saved for train, val, test.")

        # Train NAP model on train TSS; test TSS used for held-out evaluation.
        model_dir = os.path.join(MODELS_DIR, log_name)
        os.makedirs(model_dir, exist_ok=True)

        # n_batch_size is the correct option key in NAP.
        nap = NAP(tss_train_path, tss_test_path, options={"n_epochs": EPOCHS, "n_batch_size": BATCH_SIZE})
        print("Training NAP for log", log_name)
        nap.train(model_dir, log_name, save_results=False)

        # Load best model weights.
        nap.loadModel(model_dir, log_name)

        # predict() takes a list of TimedStateSample objects and returns (pred_indices, pred_event_names).
        # Filter to only samples that have a label (last event of a trace has no next activity).
        labeled_pairs = [(obj, s["label"]) for obj, s in zip(tss_test_objs, tss_test_json)
                         if s["label"] is not None]
        if not labeled_pairs:
            print(f"No labeled test samples for {log_name}, skipping.")
            continue
        tss_test_labeled_objs, y_true = zip(*labeled_pairs)
        _, y_pred = batched_predict(nap, list(tss_test_labeled_objs))
        y_true = list(y_true)

        # Compute evaluation metrics.
        acc = accuracy_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        weighted_recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        weighted_precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        print(f"Results for {log_name}: Accuracy={acc:.4f}, MCC={mcc:.4f}, F1={weighted_f1:.4f}")

        # Save per-log results.
        result_dict = {
            "log": log_name,
            "accuracy": acc,
            "mcc": mcc,
            "weighted_recall": weighted_recall,
            "weighted_precision": weighted_precision,
            "f1": weighted_f1,
        }
        results_summary.append(result_dict)

        d = os.path.join(RESULTS_DIR, log_name)
        os.makedirs(d, exist_ok=True)
        pd.DataFrame([result_dict]).to_csv(os.path.join(d, f"{log_name}_results.csv"), index=False)

        with open(os.path.join(d, f"{log_name}.txt"), "w") as f:
            f.write("Accuracy: " + str(acc))
            f.write("\nMCC: " + str(mcc))
            f.write("\nWeighted recall: " + str(weighted_recall))
            f.write("\nWeighted precision: " + str(weighted_precision))
            f.write("\nWeighted f1: " + str(weighted_f1))
        print(f"Results for log {log_name} saved.")

    # Aggregate overall results.
    df_results = pd.DataFrame(results_summary)
    df_results.to_csv(os.path.join(RESULTS_DIR, "all_logs_results.csv"), index=False)
    print("\nOverall results:")
    print(df_results)
    avg_results = df_results.groupby("log").agg({"accuracy": "mean", "f1": "mean"}).reset_index()
    print("\nAverage results per log:")
    print(avg_results)
    avg_results.to_csv(os.path.join(RESULTS_DIR, "average_results.csv"), index=False)

    print("Pipeline evaluation finished.")
