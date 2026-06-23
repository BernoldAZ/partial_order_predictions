import argparse
import os
import re
import numpy as np
import pandas as pd
from baselines.SuffixTransformerNetwork.results_collector import get_suffix_baseline_results

# ── Shared config ─────────────────────────────────────────────────────────────

RESULTS_TXT = "approach_suffix_v2/results.txt"
RESULTS_DIR = "approach_suffix_v2"

# ── Own-model column map & metrics (used by 'own' mode) ───────────────────────

OWN_COL_MAP = {
    "dl_similarity":      "DL similarity ↑",
    "ges_approx":         "GES ↑",
    "ttne_mae_minutes":   "MAE TTNE (min) ↓",
    "rrt_mae_minutes":    "MAE RRT (min) ↓",
    "nb_accuracy":        "NB acc ↑",
    "nb_f1":              "NB F1 ↑",
    "next_act_accuracy":  "Next acc ↑",
    "next_act_f1":        "Next F1 ↑",
}

METRICS = [
    ("GES ↑",            "ges",      "↑"),
    ("DL similarity ↑",  "dl",       "↑"),
    ("MAE TTNE (min) ↓", "ttne",     "↓"),
    ("MAE RRT (min) ↓",  "rrt",      "↓"),
    ("NB acc ↑",         "nb_acc",   "↑"),
    ("NB F1 ↑",          "nb_f1",    "↑"),
    ("Next acc ↑",       "next_acc", "↑"),
    ("Next F1 ↑",        "next_f1",  "↑"),
]
METRIC_KEYS = [k for _, k, _ in METRICS]
METRIC_DIRS = {k: d for _, k, d in METRICS}

# ── Baseline-comparison config (used by 'results' / 'tables' modes) ───────────

METRIC_COLS = [
    "DL similarity ↑", "GES ↑", "MAE TTNE (min) ↓", "MAE RRT (min) ↓",
    "Conc N", "Conc DL sim ↑", "Conc TTNE (min) ↓", "Conc RRT (min) ↓", "Conc GES ↑",
]

OWN_CONFIGS = {
    "GNN_Suffix_new_block": {
        "results_sub": "approach_suffix_v2/results_time_gatv2_gru_new_block",
        "csv_file":    "results_suffix_time_gnn.csv",
        "col_map":     {
            "dl_similarity":    "DL similarity ↑",
            "ttne_mae_minutes": "MAE TTNE (min) ↓",
            "rrt_mae_minutes":  "MAE RRT (min) ↓",
            "conc_count": "Conc N",
            "conc_dl_similarity": "Conc DL sim ↑",
            "conc_ttne_mae_minutes": "Conc TTNE (min) ↓",
            "conc_rrt_mae_minutes": "Conc RRT (min) ↓",
            "conc_next_act_accuracy": "Conc next acc ↑",
            "conc_next_act_f1_weighted": "Conc next F1 ↑",
            "ges_approx":               "GES ↑",
            "conc_ges_approx":          "Conc GES ↑",
        },
    },
    "GNN_Suffix_only": {
        "results_sub": "approach_suffix_v2/results_gatv2_gru_activity_block",
        "csv_file":    "results_suffix_gnn.csv",
        "col_map":     {
            "dl_similarity":    "DL similarity ↑",
            "ttne_mae_minutes": "MAE TTNE (min) ↓",
            "rrt_mae_minutes":  "MAE RRT (min) ↓",
            "conc_count": "Conc N",
            "conc_dl_similarity": "Conc DL sim ↑",
            "conc_ttne_mae_minutes": "Conc TTNE (min) ↓",
            "conc_rrt_mae_minutes": "Conc RRT (min) ↓",
            "conc_next_act_accuracy": "Conc next acc ↑",
            "conc_next_act_f1_weighted": "Conc next F1 ↑",
            "ges_approx":               "GES ↑",
            "conc_ges_approx":          "Conc GES ↑",
        },
    }
}

GNN_MODELS = set(OWN_CONFIGS)
GNN_LIST   = list(OWN_CONFIGS)

# ── Shared formatting helpers ─────────────────────────────────────────────────

SEP = "─"

def fmt_dl(v):  return f"{v:.4f}" if v is not None else "N/A"
def fmt_int(v): return f"{int(round(v))}" if v is not None else "N/A"
def fmt_pct(v):
    if v is None: return "N/A"
    return f"{'+'if v>0 else ''}{v:.1f}%"
def avg(lst):   return sum(lst) / len(lst) if lst else None

def fmt_metric(key, v):
    return fmt_dl(v) if key in ("dl", "ges", "nb_acc", "nb_f1", "next_acc", "next_f1") else fmt_int(v)

def auto_widths(headers, rows, min_w=3):
    n = len(headers)
    widths = [max(min_w, len(str(h))) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < n:
                widths[i] = max(widths[i], len(str(cell)))
    return widths

def fmt_cell(val, w, align):
    s = str(val)
    if align == "right":  return s.rjust(w)
    if align == "center": return s.center(w)
    return s.ljust(w)

def print_sub_table(title, headers, aligns, rows):
    widths = auto_widths(headers, rows)
    sep = " │ "
    header_line = sep.join(fmt_cell(h, w, a) for h, w, a in zip(headers, widths, aligns))
    total_w = len(header_line)
    print(f"  {title}")
    print("  " + SEP * total_w)
    print("  " + header_line)
    print("  " + SEP * total_w)
    for row in rows:
        cells = [fmt_cell(row[i] if i < len(row) else "", widths[i], aligns[i])
                 for i in range(len(headers))]
        print("  " + sep.join(cells))
    print("  " + SEP * total_w)
    print()

# ── 'own' mode: auto-discovery & loading ──────────────────────────────────────

def discover_models():
    models = []
    for entry in sorted(os.listdir(RESULTS_DIR)):
        if not entry.startswith("results_"):
            continue
        folder = os.path.join(RESULTS_DIR, entry)
        if os.path.isdir(folder):
            models.append((entry[len("results_"):], folder))
    return models


def load_runs(folder):
    run_id, dfs = 1, []
    while True:
        run_dir = os.path.join(folder, f"run_{run_id}")
        if not os.path.isdir(run_dir):
            break
        csv_files = [f for f in os.listdir(run_dir) if f.endswith(".csv")]
        if csv_files:
            path = os.path.join(run_dir, csv_files[0])
            dfs.append(pd.read_csv(path).rename(columns=OWN_COL_MAP))
        run_id += 1
    return dfs


def load_all_results():
    models = discover_models()
    if not models:
        return {}, []

    all_data, model_list = {}, []
    for model_name, folder in models:
        run_dfs = load_runs(folder)
        if not run_dfs:
            continue
        model_list.append(model_name)
        model_data = {}
        all_logs = {log for df in run_dfs for log in df["log"].unique()}
        for log_name in all_logs:
            entry = {"runs": len(run_dfs)}
            for col, key, _ in METRICS:
                vals = [
                    float(df.loc[df["log"] == log_name, col].iloc[0])
                    for df in run_dfs
                    if not df.loc[df["log"] == log_name].empty and col in df.columns
                ]
                entry[key] = np.mean(vals) if vals else None
            params_vals = [
                int(df.loc[df["log"] == log_name, "num_trainable_params"].iloc[0])
                for df in run_dfs
                if not df.loc[df["log"] == log_name].empty
                and "num_trainable_params" in df.columns
            ]
            entry["params"] = params_vals[0] if params_vals else None
            model_data[log_name] = entry
        all_data[model_name] = model_data

    return all_data, model_list


def print_per_log(all_data, model_list):
    all_logs = sorted({log for m in all_data.values() for log in m})
    if not all_logs:
        print("  No results found.")
        return

    bar = "═" * 100
    print(f"\n╔{bar}╗")
    print(f"║  {'PER-LOG RESULTS':<98}║")
    print(f"╚{bar}╝\n")

    metric_labels = [col for col, _, _ in METRICS]
    headers = ["Model", "Runs"] + metric_labels + ["# params"]
    aligns  = ["left", "right"] + ["right"] * len(METRICS) + ["right"]

    for log_name in all_logs:
        print(f"  === Log: {log_name} ===\n")
        rows = []
        for model_name in model_list:
            entry = all_data[model_name].get(log_name)
            if entry is None:
                continue
            row = [model_name, str(entry["runs"])]
            for _, key, _ in METRICS:
                row.append(fmt_metric(key, entry[key]))
            row.append(fmt_int(entry["params"]))
            rows.append(row)

        if not rows:
            print("  (no data)\n")
            continue
        rows.sort(key=lambda r: float(r[2]) if r[2] != "N/A" else -1, reverse=True)
        print_sub_table("Results (sorted by GES ↑)", headers, aligns, rows)


def print_head_to_head(all_data, model_list):
    all_logs = sorted({log for m in all_data.values() for log in m})

    bar = "═" * 120
    print(f"\n╔{bar}╗")
    print(f"║  {'HEAD-TO-HEAD SUMMARY  (rank among own models per log, then averaged)':<118}║")
    print(f"╚{bar}╝\n")

    ranks_a = {m: {k: [] for k in METRIC_KEYS} for m in model_list}
    wins_a  = {m: {k: 0  for k in METRIC_KEYS} for m in model_list}
    vals_a  = {m: {k: [] for k in METRIC_KEYS} for m in model_list}

    for log_name in all_logs:
        for key, direction in METRIC_DIRS.items():
            present = [
                (m, all_data[m][log_name][key])
                for m in model_list
                if log_name in all_data[m] and all_data[m][log_name].get(key) is not None
            ]
            if len(present) < 2:
                for m, v in present:
                    vals_a[m][key].append(v)
                continue
            reverse = direction == "↑"
            sorted_p = sorted(present, key=lambda x: x[1], reverse=reverse)
            for rank, (m, v) in enumerate(sorted_p, 1):
                ranks_a[m][key].append(rank)
                vals_a[m][key].append(v)
            wins_a[sorted_p[0][0]][key] += 1

    col_headers = ["Model"]
    col_aligns  = ["left"]
    for col, key, _ in METRICS:
        col_headers += [f"{col} Rank", "Wins", "Avg"]
        col_aligns  += ["right", "right", "right"]

    rows = []
    for m in model_list:
        row = [m]
        for _, key, _ in METRICS:
            avg_rank = avg(ranks_a[m][key])
            row.append(f"{avg_rank:.2f}" if avg_rank is not None else "N/A")
            row.append(str(wins_a[m][key]))
            row.append(fmt_metric(key, avg(vals_a[m][key])))
        rows.append(row)

    print_sub_table("Avg rank per metric across logs (1=best). Wins = #logs where this model was best.",
                    col_headers, col_aligns, rows)

    scores = {
        m: -3 * (avg(ranks_a[m]["ges"])  or 99)
           -2 * (avg(ranks_a[m]["dl"])   or 99)
           -1 * (avg(ranks_a[m]["ttne"]) or 99)
           -1 * (avg(ranks_a[m]["rrt"])  or 99)
        for m in model_list
    }
    best = max(scores, key=scores.get)
    print(f"  ★  Best overall: {best}")
    print()


def run_own():
    all_data, model_list = load_all_results()
    if not model_list:
        print(f"No results found under {RESULTS_DIR}/results_*/")
        return
    print(f"  Found models: {', '.join(model_list)}")
    print_per_log(all_data, model_list)
    if len(model_list) > 1:
        print_head_to_head(all_data, model_list)

# ── 'results' mode helpers ────────────────────────────────────────────────────

def get_own_suffix_results():
    rows = []
    for model_name, cfg in OWN_CONFIGS.items():
        run_id, run_dfs = 1, []
        while True:
            path = os.path.join(cfg["results_sub"], f"run_{run_id}", cfg["csv_file"])
            if not os.path.isfile(path):
                break
            run_dfs.append(pd.read_csv(path).rename(columns=cfg["col_map"]))
            run_id += 1

        all_logs = {log for df in run_dfs for log in df["log"].unique()}
        for log_name in all_logs:
            row = {"Log": log_name, "Model": model_name, "Runs": len(run_dfs)}
            for col in METRIC_COLS:
                vals = [
                    float(df.loc[df["log"] == log_name, col].iloc[0])
                    for df in run_dfs
                    if not df.loc[df["log"] == log_name].empty and col in df.columns
                ]
                row[f"{col} mean"] = round(np.mean(vals),        4) if vals else float("nan")
                row[f"{col} std"]  = round(np.std(vals, ddof=1), 4) if len(vals) > 1 else float("nan")

            params_vals = [
                int(df.loc[df["log"] == log_name, "num_trainable_params"].iloc[0])
                for df in run_dfs
                if not df.loc[df["log"] == log_name].empty
                and "num_trainable_params" in df.columns
            ]
            row["# params"] = params_vals[0] if params_vals else float("nan")
            rows.append(row)

    return pd.DataFrame(rows).set_index(["Log", "Model"])


def run_results():
    logs = [
        d for d in os.listdir("baselines/SuffixTransformerNetwork/results_per_log")
        if os.path.isdir(os.path.join("baselines/SuffixTransformerNetwork/results_per_log", d))
    ]

    df_baselines = get_suffix_baseline_results(logs)
    df_own       = get_own_suffix_results()
    combined     = pd.concat([df_baselines, df_own])

    for log_name, group in combined.groupby(level="Log"):
        best_mask = group.index.get_level_values("Model") == "BEST"
        if best_mask.any():
            for idx in group[best_mask].index:
                combined.loc[idx, "MAE TTNE (min) ↓ mean"] = np.nan
                combined.loc[idx, "MAE RRT (min) ↓ mean"]  = np.nan

    with open(RESULTS_TXT, "w", encoding="utf-8") as f:
        for log_name, group in combined.groupby(level="Log"):
            header = f"\n=== Log: {log_name} ==="
            table_str = (
                group.reset_index()[
                    ["Model",
                     "DL similarity ↑ mean",
                     "GES ↑ mean",
                     "MAE TTNE (min) ↓ mean",
                     "MAE RRT (min) ↓ mean",
                     '# params']
                ]
                .sort_values("GES ↑ mean", ascending=False)
                .to_string(index=False)
            )
            print(header)
            print(table_str)
            f.write(header + "\n")
            f.write(table_str + "\n")

# ── 'tables' mode helpers ─────────────────────────────────────────────────────

def parse_float(s):
    s = s.strip()
    if s in ("NaN", ""):
        return None
    try:
        return float(s)
    except Exception:
        return None


def parse_results(filename):
    logs, log_order, current_log = {}, [], None
    with open(filename, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            m = re.match(r"=== Log: (.+) ===", line.strip())
            if m:
                current_log = m.group(1)
                logs[current_log] = {}
                log_order.append(current_log)
                continue
            if not current_log or not line.strip() or "DL similarity" in line:
                continue
            parts = re.split(r"\s{2,}", line.strip())
            if len(parts) >= 7:
                logs[current_log][parts[0]] = {
                    "dl":     parse_float(parts[1]),
                    "ttne":   parse_float(parts[3]),
                    "rrt":    parse_float(parts[5]),
                    "params": parse_float(parts[7]) if len(parts) > 7 else None,
                }
    return logs, log_order


def non_gnn(log_data):
    return {k: v for k, v in log_data.items() if k not in GNN_MODELS}

def pct(val, ref):
    if val is None or ref is None:
        return None
    return (val - ref) / abs(ref) * 100


def make_detail_rows(logs, log_order, gnn_model):
    rows_dl, rows_ttne, rows_rrt = [], [], []
    for log_name in log_order:
        ld = logs[log_name]
        if gnn_model not in ld:
            continue
        gnn = ld[gnn_model]
        ng  = non_gnn(ld)

        dl_pairs     = [(k, v["dl"]) for k, v in ng.items() if v["dl"] is not None]
        dl_only      = [v for _, v in dl_pairs]
        all_dl_pairs = dl_pairs + ([(gnn_model, gnn["dl"])] if gnn["dl"] is not None else [])
        best_dl_model, best_dl = max(all_dl_pairs, key=lambda x: x[1]) if all_dl_pairs else (None, None)
        mean_dl = avg(dl_only)
        if gnn["dl"] is not None and dl_only:
            all_dl   = dl_only + [gnn["dl"]]
            rank_str = f"{sorted(all_dl, reverse=True).index(gnn['dl']) + 1}/{len(all_dl)}"
        else:
            rank_str = "N/A"
        rows_dl.append([
            log_name, rank_str,
            fmt_dl(gnn["dl"]), fmt_dl(best_dl), best_dl_model or "N/A",
            fmt_pct(pct(gnn["dl"], best_dl)),
            fmt_dl(mean_dl),
            fmt_pct(pct(gnn["dl"], mean_dl)),
        ])

        ttne_pairs     = [(k, v["ttne"]) for k, v in ng.items() if v["ttne"] is not None]
        ttne_only      = [v for _, v in ttne_pairs]
        all_ttne_pairs = ttne_pairs + ([(gnn_model, gnn["ttne"])] if gnn["ttne"] is not None else [])
        best_ttne_model, best_ttne = min(all_ttne_pairs, key=lambda x: x[1]) if all_ttne_pairs else (None, None)
        mean_ttne = avg(ttne_only)
        if gnn["ttne"] is not None and ttne_only:
            all_ttne      = ttne_only + [gnn["ttne"]]
            ttne_rank_str = f"{sorted(all_ttne).index(gnn['ttne']) + 1}/{len(all_ttne)}"
        else:
            ttne_rank_str = "N/A"
        rows_ttne.append([
            log_name, ttne_rank_str,
            fmt_int(gnn["ttne"]), fmt_int(best_ttne), best_ttne_model or "N/A",
            fmt_pct(pct(gnn["ttne"], best_ttne)),
            fmt_int(mean_ttne),
            fmt_pct(pct(gnn["ttne"], mean_ttne)),
        ])

        rrt_pairs     = [(k, v["rrt"]) for k, v in ng.items() if v["rrt"] is not None]
        rrt_only      = [v for _, v in rrt_pairs]
        all_rrt_pairs = rrt_pairs + ([(gnn_model, gnn["rrt"])] if gnn["rrt"] is not None else [])
        best_rrt_model, best_rrt = min(all_rrt_pairs, key=lambda x: x[1]) if all_rrt_pairs else (None, None)
        mean_rrt = avg(rrt_only)
        if gnn["rrt"] is not None and rrt_only:
            all_rrt      = rrt_only + [gnn["rrt"]]
            rrt_rank_str = f"{sorted(all_rrt).index(gnn['rrt']) + 1}/{len(all_rrt)}"
        else:
            rrt_rank_str = "N/A"
        rows_rrt.append([
            log_name, rrt_rank_str,
            fmt_int(gnn["rrt"]), fmt_int(best_rrt), best_rrt_model or "N/A",
            fmt_pct(pct(gnn["rrt"], best_rrt)),
            fmt_int(mean_rrt),
            fmt_pct(pct(gnn["rrt"], mean_rrt)),
        ])

    return rows_dl, rows_ttne, rows_rrt


def make_vs_baselines(logs, log_order, gnn_model):
    baseline_names = sorted({k for ld in logs.values() for k in ld if k not in GNN_MODELS})
    all_models = [gnn_model] + baseline_names

    dl_vals_a    = {m: [] for m in all_models}
    ttne_vals_a  = {m: [] for m in all_models}
    rrt_vals_a   = {m: [] for m in all_models}
    dl_ranks_a   = {m: [] for m in all_models}
    ttne_ranks_a = {m: [] for m in all_models}
    rrt_ranks_a  = {m: [] for m in all_models}
    dl_best_a    = {m: [] for m in all_models}
    dl_mean_a    = {m: [] for m in all_models}
    ttne_best_a  = {m: [] for m in all_models}
    ttne_mean_a  = {m: [] for m in all_models}
    rrt_best_a   = {m: [] for m in all_models}
    rrt_mean_a   = {m: [] for m in all_models}
    params_a     = {m: [] for m in all_models}

    for log_name in log_order:
        ld      = logs[log_name]
        present = [m for m in all_models if m in ld]

        dl_p = [(m, ld[m]["dl"]) for m in present if ld[m]["dl"] is not None]
        if dl_p:
            best_v = max(v for _, v in dl_p)
            mean_v = avg([v for _, v in dl_p])
            for rank, (m, v) in enumerate(sorted(dl_p, key=lambda x: x[1], reverse=True), 1):
                dl_vals_a[m].append(v);  dl_ranks_a[m].append(rank)
                dl_best_a[m].append(pct(v, best_v)); dl_mean_a[m].append(pct(v, mean_v))

        ttne_p = [(m, ld[m]["ttne"]) for m in present if ld[m].get("ttne") is not None]
        if ttne_p:
            best_v = min(v for _, v in ttne_p)
            mean_v = avg([v for _, v in ttne_p])
            for rank, (m, v) in enumerate(sorted(ttne_p, key=lambda x: x[1]), 1):
                ttne_vals_a[m].append(v); ttne_ranks_a[m].append(rank)
                ttne_best_a[m].append(pct(v, best_v)); ttne_mean_a[m].append(pct(v, mean_v))

        rrt_p = [(m, ld[m]["rrt"]) for m in present if ld[m].get("rrt") is not None]
        if rrt_p:
            best_v = min(v for _, v in rrt_p)
            mean_v = avg([v for _, v in rrt_p])
            for rank, (m, v) in enumerate(sorted(rrt_p, key=lambda x: x[1]), 1):
                rrt_vals_a[m].append(v);  rrt_ranks_a[m].append(rank)
                rrt_best_a[m].append(pct(v, best_v)); rrt_mean_a[m].append(pct(v, mean_v))

        for m in present:
            p = ld[m].get("params")
            if p is not None:
                params_a[m].append(p)

    def wins(lst): return sum(1 for r in lst if r == 1)

    rows = []
    for m in all_models:
        marker     = " ◄" if m == gnn_model else ""
        avg_dl_r   = avg(dl_ranks_a[m])
        avg_ttne_r = avg(ttne_ranks_a[m])
        avg_rrt_r  = avg(rrt_ranks_a[m])
        rows.append([
            m + marker,
            str(len(dl_vals_a[m])),
            fmt_dl(avg(dl_vals_a[m])),
            f"{avg_dl_r:.2f}"   if avg_dl_r   is not None else "N/A",
            str(wins(dl_ranks_a[m])),
            fmt_pct(avg(dl_best_a[m])),
            fmt_pct(avg(dl_mean_a[m])),
            fmt_int(avg(ttne_vals_a[m])),
            f"{avg_ttne_r:.2f}" if avg_ttne_r is not None else "N/A",
            str(wins(ttne_ranks_a[m])),
            fmt_pct(avg(ttne_best_a[m])),
            fmt_pct(avg(ttne_mean_a[m])),
            fmt_int(avg(rrt_vals_a[m])),
            f"{avg_rrt_r:.2f}"  if avg_rrt_r  is not None else "N/A",
            str(wins(rrt_ranks_a[m])),
            fmt_pct(avg(rrt_best_a[m])),
            fmt_pct(avg(rrt_mean_a[m])),
            fmt_int(avg(params_a[m])),
        ])
    return rows


def print_gnn_block(logs, log_order, gnn_model, idx):
    print()
    title = f"  TABLE {idx}: {gnn_model}"
    bar = "═" * 160
    print(f"╔{bar}╗")
    print(f"║{title:<160}║")
    print(f"╚{bar}╝")
    print()

    rows_dl, rows_ttne, rows_rrt = make_detail_rows(logs, log_order, gnn_model)

    print_sub_table("DL Similarity ↑  (higher is better)",
        ["Log", "Rank", "DL", "Best DL", "Best Model (DL)", "Δ vs Best", "Mean DL", "Δ vs Mean"],
        ["left", "center", "right", "right", "left", "right", "right", "right"],
        rows_dl)

    print_sub_table("MAE TTNE ↓  (lower is better)",
        ["Log", "Rank", "TTNE (min)", "Best TTNE", "Best Model (TTNE)", "Δ vs Best", "Mean TTNE", "Δ vs Mean"],
        ["left", "center", "right", "right", "left", "right", "right", "right"],
        rows_ttne)

    print_sub_table("MAE RRT ↓  (lower is better)",
        ["Log", "Rank", "RRT (min)", "Best RRT", "Best Model (RRT)", "Δ vs Best", "Mean RRT", "Δ vs Mean"],
        ["left", "center", "right", "right", "left", "right", "right", "right"],
        rows_rrt)

    vs_rows = make_vs_baselines(logs, log_order, gnn_model)
    print_sub_table(
        "Performance vs Baselines  (◄ = this GNN model | rank among GNN+baselines | Δ% vs best/mean per log, averaged)",
        ["Model", "N",
         "Avg DL", "DL Rank", "DL↑ Wins", "Avg Δ Best", "Avg Δ Mean",
         "Avg TTNE", "TTNE Rank", "TTNE↓ Wins", "Avg Δ Best", "Avg Δ Mean",
         "Avg RRT", "RRT Rank", "RRT↓ Wins", "Avg Δ Best", "Avg Δ Mean",
         "Avg # params"],
        ["left", "right",
         "right", "right", "right", "right", "right",
         "right", "right", "right", "right", "right",
         "right", "right", "right", "right", "right",
         "right"],
        vs_rows)


def print_gnn_head_to_head(logs, log_order):
    print()
    bar = "═" * 120
    print(f"╔{bar}╗")
    print(f"║  {'SUMMARY: GNN_Suffix Variants Head-to-Head':<118}║")
    print(f"╚{bar}╝")
    print()

    dl_ranks_a   = {m: [] for m in GNN_LIST}
    ttne_ranks_a = {m: [] for m in GNN_LIST}
    rrt_ranks_a  = {m: [] for m in GNN_LIST}
    dl_wins      = {m: 0  for m in GNN_LIST}
    ttne_wins    = {m: 0  for m in GNN_LIST}
    rrt_wins     = {m: 0  for m in GNN_LIST}
    dl_best_a    = {m: [] for m in GNN_LIST}
    dl_mean_a    = {m: [] for m in GNN_LIST}
    ttne_best_a  = {m: [] for m in GNN_LIST}
    ttne_mean_a  = {m: [] for m in GNN_LIST}
    rrt_best_a   = {m: [] for m in GNN_LIST}
    rrt_mean_a   = {m: [] for m in GNN_LIST}
    params_a     = {m: [] for m in GNN_LIST}

    for log_name in log_order:
        ld      = logs[log_name]
        ng      = non_gnn(ld)
        present = [m for m in GNN_LIST if m in ld]
        if not present:
            continue

        dl_p = [(m, ld[m]["dl"]) for m in present if ld[m]["dl"] is not None]
        if dl_p:
            dl_ng   = [v["dl"] for v in ng.values() if v["dl"] is not None]
            best_ng = max(dl_ng) if dl_ng else None
            mean_ng = avg(dl_ng)
            for rank, (m, v) in enumerate(sorted(dl_p, key=lambda x: x[1], reverse=True), 1):
                dl_ranks_a[m].append(rank)
                dl_best_a[m].append(pct(v, best_ng))
                dl_mean_a[m].append(pct(v, mean_ng))
            dl_wins[sorted(dl_p, key=lambda x: x[1], reverse=True)[0][0]] += 1

        ttne_p = [(m, ld[m]["ttne"]) for m in present if ld[m]["ttne"] is not None]
        if ttne_p:
            ttne_ng = [v["ttne"] for v in ng.values() if v["ttne"] is not None]
            best_ng = min(ttne_ng) if ttne_ng else None
            mean_ng = avg(ttne_ng)
            for rank, (m, v) in enumerate(sorted(ttne_p, key=lambda x: x[1]), 1):
                ttne_ranks_a[m].append(rank)
                ttne_best_a[m].append(pct(v, best_ng))
                ttne_mean_a[m].append(pct(v, mean_ng))
            ttne_wins[sorted(ttne_p, key=lambda x: x[1])[0][0]] += 1

        rrt_p = [(m, ld[m]["rrt"]) for m in present if ld[m]["rrt"] is not None]
        if rrt_p:
            rrt_ng  = [v["rrt"] for v in ng.values() if v["rrt"] is not None]
            best_ng = min(rrt_ng) if rrt_ng else None
            mean_ng = avg(rrt_ng)
            for rank, (m, v) in enumerate(sorted(rrt_p, key=lambda x: x[1]), 1):
                rrt_ranks_a[m].append(rank)
                rrt_best_a[m].append(pct(v, best_ng))
                rrt_mean_a[m].append(pct(v, mean_ng))
            rrt_wins[sorted(rrt_p, key=lambda x: x[1])[0][0]] += 1

        for m in present:
            p = ld[m].get("params")
            if p is not None:
                params_a[m].append(p)

    headers = [
        "Model",
        "DL Rank", "DL↑ Wins", "Avg Δ Best DL", "Avg Δ Mean DL",
        "TTNE Rank", "TTNE↓ Wins", "Avg Δ Best TTNE", "Avg Δ Mean TTNE",
        "RRT Rank", "RRT↓ Wins", "Avg Δ Best RRT", "Avg Δ Mean RRT",
        "Avg # params",
    ]
    aligns = ["left"] + ["right"] * 13

    scores = {}
    rows   = []
    for m in GNN_LIST:
        avg_dl_r   = avg(dl_ranks_a[m])
        avg_ttne_r = avg(ttne_ranks_a[m])
        avg_rrt_r  = avg(rrt_ranks_a[m])
        rows.append([
            m,
            f"{avg_dl_r:.2f}"   if avg_dl_r   is not None else "N/A",
            str(dl_wins[m]),
            fmt_pct(avg(dl_best_a[m])),
            fmt_pct(avg(dl_mean_a[m])),
            f"{avg_ttne_r:.2f}" if avg_ttne_r is not None else "N/A",
            str(ttne_wins[m]),
            fmt_pct(avg(ttne_best_a[m])),
            fmt_pct(avg(ttne_mean_a[m])),
            f"{avg_rrt_r:.2f}"  if avg_rrt_r  is not None else "N/A",
            str(rrt_wins[m]),
            fmt_pct(avg(rrt_best_a[m])),
            fmt_pct(avg(rrt_mean_a[m])),
            fmt_int(avg(params_a[m])),
        ])
        scores[m] = (
            -3 * (avg_dl_r   or 99)
            -2 * (avg_ttne_r or 99)
            -1 * (avg_rrt_r  or 99)
        )

    print_sub_table(
        "Rank among GNN variants only (1=best).  Avg Δ vs best/mean of non-GNN baselines per log, then averaged.",
        headers, aligns, rows)

    best = max(scores, key=scores.get)
    print(f"  ★  Best overall GNN_Suffix variant: {best}")
    print()
    print("  Notes:")
    print("  - DL ↑ higher is better;  TTNE ↓ and RRT ↓ lower is better.")
    print("  - GNN_Suffix and GNN_Suffix_Stop produce no TTNE/RRT output → N/A.")
    print("  - Winner weighted score: 3×DL_rank + 2×TTNE_rank + 1×RRT_rank (lower avg rank = higher score).")
    print()


def run_tables():
    logs, log_order = parse_results(RESULTS_TXT)
    for idx, gnn_model in enumerate(GNN_LIST, 1):
        print_gnn_block(logs, log_order, gnn_model, idx)
    print_gnn_head_to_head(logs, log_order)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["results", "tables", "own"],
                        help="'results': collect CSV results and write results.txt  |  "
                             "'tables': print comparison tables from results.txt  |  "
                             "'own': auto-discover own models and print per-log + head-to-head")
    args = parser.parse_args()

    if args.mode == "results":
        run_results()
    elif args.mode == "tables":
        run_tables()
    else:
        run_own()
