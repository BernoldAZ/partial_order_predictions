import os
import pickle
import numpy as np
import pandas as pd
import glob

RESULT_DIRS = {
    #"SuTraN (DA)":       "SUTRAN_DA_results",
    "SuTraN (NDA)":      "SUTRAN_NDA_results",
    #"CRTP-LSTM (DA)":    "CRTP_LSTM_DA_results",
    #"CRTP-LSTM (NDA)":   "CRTP_LSTM_NDA_results",
    "ED-LSTM":           "ED_LSTM_results",
    "SEP-LSTM":          "SEP_LSTM_results",
    #"BEST":              "BEST_results",
}

METRICS = {
    "DL similarity ↑":    ("DL sim",                    4),
    "MAE TTNE (min) ↓":   ("MAE TTNE minutes",          2),
    "MAE RRT (min) ↓":    ("MAE RRT minutes",           2),
    "Next act acc ↑":     ("next_act_accuracy",         4),
    "Next act F1 (wt) ↑": ("next_act_f1_weighted",      4),
    "Conc N":             ("conc_n_samples",             0),
    "Conc DL sim ↑":      ("conc_dl_similarity",        4),
    "Conc TTNE (min) ↓":  ("conc_ttne_mae_minutes",     2),
    "Conc RRT (min) ↓":   ("conc_rrt_mae_minutes",      2),
    "Conc next acc ↑":    ("conc_next_act_accuracy",    4),
    "Conc next F1 ↑":     ("conc_next_act_f1_weighted", 4),
    "GES ↑":              ("ges_exact",                 4),
    "BOS":                ("bo_combined",               4),
    "BOS order":          ("bo_order_sim",              4),
    "BOS Content":        ("bo_content_f1",             4),
    "Conc GES ↑":         ("conc_ges_approx",           4),
}

# Partial-order metrics replaced by po_ttne_results.pkl when po_source == "ttne"
PO_KEYS = ("ges_exact", "bo_combined", "bo_order_sim", "bo_content_f1")

def _load_runs(log_dir, result_dir_base, run_id=None, po_source="total"):
    """Load averaged_results.pkl for all existing runs (or only `run_id` if given).

    po_source="total": GES/BOS from the total-order suffix (averaged_results.pkl).
    po_source="ttne":  GES/BOS from the TTNE-derived partial order (po_ttne_results.pkl),
                       NaN when that file does not exist.
    """
    pattern = os.path.join(
        log_dir,
        f"{result_dir_base}_run{run_id if run_id is not None else '*'}",
        "TEST_SET_RESULTS",
        "averaged_results.pkl"
    )

    results = []

    for path in sorted(glob.glob(pattern)):
        with open(path, "rb") as f:
            res = pickle.load(f)
        if po_source == "ttne":
            po_path = os.path.join(os.path.dirname(path), "po_ttne_results.pkl")
            po = {}
            if os.path.isfile(po_path):
                with open(po_path, "rb") as f:
                    po = pickle.load(f)
            res.update({k: po.get(k, float("nan")) for k in PO_KEYS})
        results.append(res)

    return results


def get_suffix_baseline_results(LOGS, base_dir="baselines/results_per_log", run_id=None,
                                po_source="total") -> pd.DataFrame:
    rows = []

    for log_name in LOGS:
        log_dir = os.path.join(base_dir, log_name)

        for model_name, result_dir_base in RESULT_DIRS.items():
            runs = _load_runs(log_dir, result_dir_base, run_id, po_source)
            row = {"Log": log_name, "Model": model_name, "Runs": len(runs)}

            for col, (key, decimals) in METRICS.items():
                if runs:
                    raw    = [r.get(key, float("nan")) for r in runs]
                    values = [float(v) if v != '' else float("nan") for v in raw]
                    row[f"{col} mean"] = round(float(np.nanmean(values)), decimals)
                    row[f"{col} std"]  = round(float(np.nanstd(values, ddof=1)), decimals) if len(runs) > 1 else float("nan")
                else:
                    row[f"{col} mean"] = float("nan")
                    row[f"{col} std"]  = float("nan")

            row["# params"] = next(
                (r.get("num_trainable_params") for r in runs if "num_trainable_params" in r),
                None
            )

            for time_col in ("training_time", "testing_time", "inference_time", "evaluation_time"):
                values = [r[time_col] for r in runs if time_col in r]
                row[time_col] = round(float(np.mean(values)), 2) if values else float("nan")

            rows.append(row)

    pd.set_option("display.float_format", "{:.4f}".format)
    return pd.DataFrame(rows).set_index(["Log", "Model"])


def get_po_threshold_comparison(LOGS, base_dir="baselines/results_per_log", run_id=None) -> pd.DataFrame:
    """GES / BOS per run of the TTNE-derived partial order, threshold X (rate-matched,
    po_ttne_results.pkl) vs threshold 0 (po_ttne_zero_results.pkl). Diff = 0 - X.
    Runs where neither file exists are skipped; a missing one gives NaN."""
    rows = []

    for log_name in LOGS:
        log_dir = os.path.join(base_dir, log_name)

        for model_name, result_dir_base in RESULT_DIRS.items():
            pattern = os.path.join(
                log_dir,
                f"{result_dir_base}_run{run_id if run_id is not None else '*'}",
                "TEST_SET_RESULTS"
            )

            for res_path in sorted(glob.glob(pattern)):
                po = {}
                for name, fname in (("X", "po_ttne_results.pkl"), ("0", "po_ttne_zero_results.pkl")):
                    path = os.path.join(res_path, fname)
                    if os.path.isfile(path):
                        with open(path, "rb") as f:
                            po[name] = pickle.load(f)
                if not po:
                    continue

                run = int(os.path.basename(os.path.dirname(res_path)).rsplit("_run", 1)[-1])
                row = {"Log": log_name, "Model": model_name, "Run": run,
                       "X (s)": po.get("X", {}).get("X", float("nan"))}
                for col, key in (("GES", "ges_exact"), ("BOS", "bo_combined")):
                    v_x = po.get("X", {}).get(key, float("nan"))
                    v_0 = po.get("0", {}).get(key, float("nan"))
                    row[f"{col} X"] = round(float(v_x), 4)
                    row[f"{col} 0"] = round(float(v_0), 4)
                    row[f"{col} diff"] = round(float(v_0 - v_x), 4)
                rows.append(row)

    pd.set_option("display.float_format", "{:.4f}".format)
    return pd.DataFrame(rows).set_index(["Log", "Model", "Run"])