import os
import numpy as np
import pandas as pd
from baselines.SuffixTransformerNetwork.results_collector import get_suffix_baseline_results

RESULTS_TXT = "approach_suffix_v2/baseline_results.txt"
SCALABILITY_TXT = "approach_suffix_v2/scalability_results.txt"

OWN_CONFIGS = {
    "Graph_v1": {
        "results_sub": "approach_suffix_v2/results_time_gatv2_gru_nb_v1",
        "csv_file":    "results_suffix_time_gnn.csv",
    }
}

COL_MAP = {
    "dl_similarity":       "DL similarity ↑",
    "ges_approx":          "GES ↑",
    "ttne_mae_minutes":    "MAE TTNE (min) ↓",
    "rrt_mae_minutes":     "MAE RRT (min) ↓",
    "first_step_accuracy": "Next act acc ↑",
    "first_step_f1":       "Next act F1 (wt) ↑",
}

METRIC_COLS = ["GES ↑", "DL similarity ↑", "MAE TTNE (min) ↓", "MAE RRT (min) ↓",
               "Next act acc ↑", "Next act F1 (wt) ↑"]


def fmt_mean_std(mean, std):
    if pd.isna(mean):
        return "N/A"
    if pd.isna(std):
        return f"{mean:.2f}"
    return f"{mean:.2f} ± {std:.2f}"


def get_own_results():
    rows = []
    for model_name, cfg in OWN_CONFIGS.items():
        run_id, run_dfs = 1, []
        while True:
            path = os.path.join(cfg["results_sub"], f"run_{run_id}", cfg["csv_file"])
            if not os.path.isfile(path):
                break
            run_dfs.append(pd.read_csv(path).rename(columns=COL_MAP))
            run_id += 1
        if not run_dfs:
            continue

        all_logs = {log for df in run_dfs for log in df["log"].unique()}
        for log_name in all_logs:
            row = {"Log": log_name, "Model": model_name, "Runs": len(run_dfs)}
            for col in METRIC_COLS:
                vals = [
                    float(df.loc[df["log"] == log_name, col].iloc[0])
                    for df in run_dfs
                    if not df.loc[df["log"] == log_name].empty and col in df.columns
                ]
                row[f"{col} mean"] = round(np.mean(vals), 4) if vals else float("nan")
                row[f"{col} std"]  = round(np.std(vals, ddof=1), 4) if len(vals) > 1 else float("nan")
            params_vals = [
                int(df.loc[df["log"] == log_name, "num_trainable_params"].iloc[0])
                for df in run_dfs
                if not df.loc[df["log"] == log_name].empty and "num_trainable_params" in df.columns
            ]
            row["# params"] = params_vals[0] if params_vals else float("nan")

            train_time_vals = [
                float(df.loc[df["log"] == log_name, "training_time_seconds"].iloc[0])
                for df in run_dfs
                if not df.loc[df["log"] == log_name].empty and "training_time_seconds" in df.columns
            ]
            row["training_time"] = round(np.mean(train_time_vals), 2) if train_time_vals else float("nan")

            test_time_vals = [
                float(df.loc[df["log"] == log_name, "testing_time_seconds"].iloc[0])
                for df in run_dfs
                if not df.loc[df["log"] == log_name].empty and "testing_time_seconds" in df.columns
            ]
            row["testing_time"] = round(np.mean(test_time_vals), 2) if test_time_vals else float("nan")

            rows.append(row)

    return pd.DataFrame(rows).set_index(["Log", "Model"]) if rows else pd.DataFrame()


def run():
    logs = [
        d for d in os.listdir("baselines/SuffixTransformerNetwork/results_per_log")
        if os.path.isdir(os.path.join("baselines/SuffixTransformerNetwork/results_per_log", d))
    ]

    df_baselines = get_suffix_baseline_results(logs)
    df_own = get_own_results()
    if df_own.empty:
        print("No own results found.")
        return

    combined = pd.concat([df_baselines, df_own])

    for log_name, group in combined.groupby(level="Log"):
        best_mask = group.index.get_level_values("Model") == "BEST"
        if best_mask.any():
            for idx in group[best_mask].index:
                combined.loc[idx, "MAE TTNE (min) ↓ mean"] = np.nan
                combined.loc[idx, "MAE RRT (min) ↓ mean"]  = np.nan

    for col in METRIC_COLS:
        combined[col] = combined.apply(
            lambda r: fmt_mean_std(r[f"{col} mean"], r[f"{col} std"]), axis=1
        )

    with open(RESULTS_TXT, "w", encoding="utf-8") as f:
        for log_name, group in combined.groupby(level="Log"):
            header = f"\n=== Log: {log_name} ==="
            table_str = (
                group.reset_index()
                .sort_values("GES ↑ mean", ascending=False)
                [["Model",
                  "GES ↑",
                  "DL similarity ↑",
                  "MAE TTNE (min) ↓",
                  "MAE RRT (min) ↓",
                  "Next act acc ↑",
                  "Next act F1 (wt) ↑",
                  "# params"]]
                .to_string(index=False)
            )
            print(header)
            print(table_str)
            f.write(header + "\n")
            f.write(table_str + "\n")

    with open(SCALABILITY_TXT, "w", encoding="utf-8") as f:
        for log_name, group in combined.groupby(level="Log"):
            header = f"\n=== Log: {log_name} ==="
            table_str = (
                group.reset_index()
                .sort_values("GES ↑ mean", ascending=False)
                .rename(columns={"training_time": "Training time (s)",
                                  "testing_time":  "Testing time (s)"})
                [["Model", "Training time (s)", "Testing time (s)", "# params"]]
                .to_string(index=False)
            )
            print(header)
            print(table_str)
            f.write(header + "\n")
            f.write(table_str + "\n")


if __name__ == "__main__":
    run()
