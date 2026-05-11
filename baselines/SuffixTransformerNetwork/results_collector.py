import os
import pickle
import numpy as np
import pandas as pd

RESULT_DIRS = {
    "SuTraN (DA)":       "SUTRAN_DA_results",
    "SuTraN (NDA)":      "SUTRAN_NDA_results",
    "CRTP-LSTM (DA)":    "CRTP_LSTM_DA_results",
    "CRTP-LSTM (NDA)":   "CRTP_LSTM_NDA_results",
    "ED-LSTM":           "ED_LSTM_results",
    "SEP-LSTM":          "SEP_LSTM_results",
    "BEST":              "BEST_results",
}

METRICS = {
    "DL similarity ↑":  ("DL sim",              4),
    "MAE TTNE (min) ↓": ("MAE TTNE minutes",    2),
    "MAE RRT (min) ↓":  ("MAE RRT minutes",     2),
}


def _load_runs(log_dir, result_dir_base):
    """Load averaged_results.pkl for every run that exists (run1, run2, ...)."""
    results = []
    run_id = 1
    while True:
        path = os.path.join(
            log_dir, f"{result_dir_base}_run{run_id}",
            "TEST_SET_RESULTS", "averaged_results.pkl"
        )
        if not os.path.exists(path):
            break
        with open(path, "rb") as f:
            results.append(pickle.load(f))
        run_id += 1
    return results


def get_suffix_baseline_results(LOGS, base_dir="baselines/SuffixTransformerNetwork/results_per_log") -> pd.DataFrame:
    rows = []

    for log_name in LOGS:
        log_dir = os.path.join(base_dir, log_name)

        for model_name, result_dir_base in RESULT_DIRS.items():
            runs = _load_runs(log_dir, result_dir_base)
            row = {"Log": log_name, "Model": model_name, "Runs": len(runs)}

            for col, (key, decimals) in METRICS.items():
                if runs:
                    values = [r.get(key, float("nan")) for r in runs]
                    row[f"{col} mean"] = round(float(np.mean(values)), decimals)
                    row[f"{col} std"]  = round(float(np.std(values, ddof=1)), decimals) if len(runs) > 1 else float("nan")
                else:
                    row[f"{col} mean"] = float("nan")
                    row[f"{col} std"]  = float("nan")

            rows.append(row)

    pd.set_option("display.float_format", "{:.4f}".format)
    return pd.DataFrame(rows).set_index(["Log", "Model"])