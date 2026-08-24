import os
import numpy as np
import pandas as pd

ABLATION_CONFIGS = {
    "v1 (scheduled sampling)": {
        "results_sub": "approach_suffix_v2/results_time_gatv2_gru_nb_v1",
        "csv_file":    "results_suffix_time_gnn.csv",
    },
    "v2 (teacher forcing)": {
        "results_sub": "approach_suffix_v2/results_time_gatv2_gru_nb_v2",
        "csv_file":    "results_suffix_time_gnn.csv",
    },
    "v3 (next-activity only)": {
        "results_sub": "approach_suffix_v2/results_time_gatv2_gru_nb_v3",
        "csv_file":    "results_suffix_time_gnn.csv",
    },
}

METRICS = [
    ("GES ↑",        "ges_approx",          "↑", lambda v: f"{v:.2f}"),
    ("DL sim ↑",     "dl_similarity",        "↑", lambda v: f"{v:.2f}"),
    ("TTNE (min) ↓", "ttne_mae_minutes",     "↓", lambda v: f"{int(round(v))}"),
    ("RRT (min) ↓",  "rrt_mae_minutes",      "↓", lambda v: f"{int(round(v))}"),
    ("NB F1 ↑",      "nb_f1",               "↑", lambda v: f"{v:.2f}"),
    ("NB acc ↑",     "nb_accuracy",          "↑", lambda v: f"{v:.2f}"),
    ("Step F1 ↑",    "first_step_f1",        "↑", lambda v: f"{v:.2f}"),
    ("Step acc ↑",   "first_step_accuracy",  "↑", lambda v: f"{v:.2f}"),
]

SEP = "─"


def auto_widths(headers, rows):
    widths = [max(3, len(str(h))) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(str(cell)))
    return widths


def print_table(title, headers, aligns, rows):
    widths = auto_widths(headers, rows)
    sep = " │ "
    header_line = sep.join(
        str(h).rjust(w) if a == "right" else str(h).ljust(w)
        for h, w, a in zip(headers, widths, aligns)
    )
    total_w = len(header_line)
    print(f"  {title}")
    print("  " + SEP * total_w)
    print("  " + header_line)
    print("  " + SEP * total_w)
    for row in rows:
        cells = [
            str(row[i] if i < len(row) else "").rjust(w) if a == "right"
            else str(row[i] if i < len(row) else "").ljust(w)
            for i, (w, a) in enumerate(zip(widths, aligns))
        ]
        print("  " + sep.join(cells))
    print("  " + SEP * total_w)
    print()


def load_variant(cfg):
    run_id, run_dfs = 1, []
    while True:
        path = os.path.join(cfg["results_sub"], f"run_{run_id}", cfg["csv_file"])
        if not os.path.isfile(path):
            break
        run_dfs.append(pd.read_csv(path))
        run_id += 1
    return run_dfs


def run():
    variant_data = {}
    for name, cfg in ABLATION_CONFIGS.items():
        dfs = load_variant(cfg)
        if dfs:
            variant_data[name] = dfs

    if not variant_data:
        print("No ablation results found.")
        return

    all_logs = sorted({
        log
        for dfs in variant_data.values()
        for df in dfs
        for log in df["log"].unique()
    })

    bar = "═" * 100
    print(f"\n╔{bar}╗")
    print(f"║  {'ABLATION: Training method (scheduled sampling × teacher forcing × next-activity)':<98}║")
    print(f"╚{bar}╝\n")

    headers = ["Variant", "N"] + [label for label, *_ in METRICS] + ["# params"]
    aligns  = ["left", "right"] + ["right"] * len(METRICS) + ["right"]

    for log_name in all_logs:
        print(f"  === Log: {log_name} ===\n")
        rows = []
        for variant in ABLATION_CONFIGS:
            if variant not in variant_data:
                continue
            dfs = variant_data[variant]
            log_dfs = [df for df in dfs if log_name in df["log"].values]
            if not log_dfs:
                continue

            row = [variant, str(len(log_dfs))]
            for _, col, _, fmt in METRICS:
                vals = [
                    float(df.loc[df["log"] == log_name, col].iloc[0])
                    for df in log_dfs
                    if col in df.columns and not df.loc[df["log"] == log_name].empty
                ]
                if not vals:
                    row.append("N/A")
                elif len(vals) > 1:
                    row.append(f"{fmt(np.mean(vals))} ± {fmt(np.std(vals, ddof=1))}")
                else:
                    row.append(fmt(np.mean(vals)))

            params_vals = [
                int(df.loc[df["log"] == log_name, "num_trainable_params"].iloc[0])
                for df in log_dfs
                if "num_trainable_params" in df.columns
                and not df.loc[df["log"] == log_name].empty
            ]
            row.append(str(params_vals[0]) if params_vals else "N/A")
            rows.append(row)

        if not rows:
            print("  (no data)\n")
            continue

        rows.sort(key=lambda r: float(r[2].split(" ")[0]) if r[2] != "N/A" else -1, reverse=True)
        print_table("Results by training-method variant", headers, aligns, rows)


if __name__ == "__main__":
    run()
