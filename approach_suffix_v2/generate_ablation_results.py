import os
import numpy as np
import pandas as pd

ABLATION_CONFIGS = {
    "input": {
        "title": "ABLATION: Input structure (block-at-a-time v2)",
        "extra_metrics": [
            ("DL sim ↑", "dl_similarity", "↑", lambda v: f"{v:.2f}"),
        ],
        "variants": {
            "Graph": {
                #"results_sub": "approach_suffix_v2/results_time_gatv2_gru_block_v2", # decoder Block by Block
                "results_sub": "approach_suffix_v2/results_time_gatv2_gru_nb_v2_actnb",
                "csv_file":    "results_suffix_time_gnn.csv",
            },
            "Seq": {
                #"results_sub": "approach_suffix_v2/results_time_gatv2_seq_gru_block_v2", # decoder Block by Block
                "results_sub": "approach_suffix_v2/results_time_gatv2_seq_gru_nb_v2_actnb",
                "csv_file":    "results_suffix_time_gnn.csv",
            },
            "Seq reversed": {
                #"results_sub": "approach_suffix_v2/results_time_gatv2_seq_gru_block_v2", # decoder Block by Block
                "results_sub": "approach_suffix_v2/results_time_gatv2_seq_gru_nb_v2_actnb",
                "csv_file":    "results_suffix_time_gnn_prefixflip.csv",
            },
            "Seq random": {
                #"results_sub": "approach_suffix_v2/results_time_gatv2_seq_gru_block_v2", # decoder Block by Block
                "results_sub": "approach_suffix_v2/results_time_gatv2_seq_gru_nb_v2_actnb",
                "csv_file":    "results_suffix_time_gnn_prefixrandom.csv",
            },
        },
    },
    "decoder": {
        "title": "ABLATION: Decoder structure (graph input, v2)",
        "variants": {
            "Block-at-a-time": {
                "results_sub": "approach_suffix_v2/results_time_gatv2_gru_block_v2",
                "csv_file":    "results_suffix_time_gnn.csv",
            },
            "Event-at-a-time": {
                "results_sub": "approach_suffix_v2/results_time_gatv2_gru_nb_v2_actnb",
                "csv_file":    "results_suffix_time_gnn.csv",
            },
        },
    },
}

METRICS = [
    ("GES ↑",        "ges_exact",      "↑", lambda v: f"{v:.2f}"),
    ("BO order ↑",   "bo_order_sim",   "↑", lambda v: f"{v:.2f}"),
    ("BO content ↑", "bo_content_f1",  "↑", lambda v: f"{v:.2f}"),
    ("BO comb ↑",    "bo_combined",    "↑", lambda v: f"{v:.2f}"),
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
    for ablation in ABLATION_CONFIGS.values():
        run_ablation(ablation["title"], ablation["variants"], METRICS + ablation.get("extra_metrics", []))


def run_ablation(title, variants, metrics):
    variant_data = {}
    for name, cfg in variants.items():
        dfs = load_variant(cfg)
        if dfs:
            variant_data[name] = dfs

    if not variant_data:
        print(f"No ablation results found for: {title}")
        return

    all_logs = sorted({
        log
        for dfs in variant_data.values()
        for df in dfs
        for log in df["log"].unique()
    })

    bar = "═" * 100
    print(f"\n╔{bar}╗")
    print(f"║  {title:<98}║")
    print(f"╚{bar}╝\n")

    headers = ["Variant", "N"] + [label for label, *_ in metrics] + ["# params"]
    aligns  = ["left", "right"] + ["right"] * len(metrics) + ["right"]

    for log_name in all_logs:
        print(f"  === Log: {log_name} ===\n")
        rows = []
        for variant in variants:
            if variant not in variant_data:
                continue
            dfs = variant_data[variant]
            log_dfs = [df for df in dfs if log_name in df["log"].values]
            if not log_dfs:
                continue

            row = [variant, str(len(log_dfs))]
            for _, col, _, fmt in metrics:
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
        print_table("Results", headers, aligns, rows)


if __name__ == "__main__":
    run()
