import os
import pickle
import numpy as np
import pandas as pd
import glob

RESULT_DIRS = {
    #"SuTraN (DA)":       "SUTRAN_DA_results",
    "SuTraN (NDA)":      "SUTRAN_NDA_results",
    #"CRTP-LSTM (DA)":    "CRTP_LSTM_DA_results",
    "CRTP-LSTM (NDA)":   "CRTP_LSTM_NDA_results",
    "ED-LSTM":           "ED_LSTM_results",
    "SEP-LSTM":          "SEP_LSTM_results",
    "BEST":              "BEST_results",
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
    "GES ↑":              ("ges_approx",                4),
    "Conc GES ↑":         ("conc_ges_approx",           4),
}

def _load_runs(log_dir, result_dir_base):
    """Load averaged_results.pkl for all existing runs."""
    pattern = os.path.join(
        log_dir,
        f"{result_dir_base}_run*",
        "TEST_SET_RESULTS",
        "averaged_results.pkl"
    )

    results = []

    for path in sorted(glob.glob(pattern)):
        with open(path, "rb") as f:
            results.append(pickle.load(f))

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