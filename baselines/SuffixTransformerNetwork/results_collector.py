import os
import pickle
import pandas as pd

def get_suffix_baseline_results(LOGS, base_dir="baselines/SuffixTransformerNetwork/results_per_log") -> pd.DataFrame:
    RESULT_DIRS = {
        "SuTraN (DA)":       "SUTRAN_DA_results",
        "SuTraN (NDA)":      "SUTRAN_NDA_results",
        "CRTP-LSTM (DA)":    "CRTP_LSTM_DA_results",
        "CRTP-LSTM (NDA)":   "CRTP_LSTM_NDA_results",
        "ED-LSTM":           "ED_LSTM_results",
        "SEP-LSTM":          "SEP_LSTM_results",
        "BEST":              "BEST_results",
    }

    rows = []

    for cfg in LOGS:
        log = cfg["log_name"]

        for model_name, result_dir in RESULT_DIRS.items():
            path = os.path.join(
                base_dir, log, result_dir,
                "TEST_SET_RESULTS", "averaged_results.pkl"
            )

            if os.path.exists(path):
                with open(path, "rb") as f:
                    res = pickle.load(f)

                row = {
                    "Log": log,
                    "Model": model_name,
                    "DL similarity ↑": round(res.get("DL sim", float("nan")), 4),
                    "MAE TTNE (min) ↓": round(res.get("MAE TTNE minutes", float("nan")), 2),
                    "MAE RRT (min) ↓": round(res.get("MAE RRT minutes", float("nan")), 2),
                }
            else:
                row = {
                    "Log": log,
                    "Model": model_name,
                    "DL similarity ↑": None,
                    "MAE TTNE (min) ↓": None,
                    "MAE RRT (min) ↓": None,
                }

            rows.append(row)

    df = pd.DataFrame(rows).set_index(["Log", "Model"])
    pd.set_option("display.float_format", "{:.4f}".format)

    return df