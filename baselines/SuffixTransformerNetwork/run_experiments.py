"""Single-script entry point to reproduce all experiments.

Equivalent to executing every cell of run_experiments.ipynb, but runs as a
plain Python process so it can be launched from the command line or inside a
container without JupyterLab.

Usage examples
--------------
Run everything:
    python run_experiments.py

Skip dataset creation (tensors already on disk):
    python run_experiments.py --skip-data

Run only specific models:
    python run_experiments.py --skip-data --models BEST SuTraN_DA

Run only specific logs:
    python run_experiments.py --skip-data --logs BPIC_17

Available model names (case-insensitive):
    SuTraN_DA  SuTraN_NDA  CRTP_LSTM_DA  CRTP_LSTM_NDA
    ED_LSTM    SEP_LSTM    BEST

Docker commands
---------------
Build the image (run once from the repo root):
    docker build -t ppm-sutran-best .

Run everything (data creation + all 7 models on all 3 logs):
    docker run -it --rm \\
      -v $(pwd):/app \\
      --gpus all \\
      ppm-sutran-best \\
      python run_experiments.py

Skip data creation (tensors already on disk):
    docker run -it --rm -v $(pwd):/app --gpus all ppm-sutran-best python run_experiments.py --skip-data

Run only BEST on all logs:
    docker run -it --rm \\
      -v $(pwd):/app \\
      --gpus all \\
      ppm-sutran-best \\
      python run_experiments.py --skip-data --models BEST

Run specific models on a specific log:
    docker run -it --rm \\
      -v $(pwd):/app \\
      --gpus all \\
      ppm-sutran-best \\
      python run_experiments.py --skip-data --models BEST SuTraN_DA --logs BPIC_17
"""

import argparse
import os
import pickle
import sys
import traceback

# ---------------------------------------------------------------------------
# Shared configuration
# ---------------------------------------------------------------------------
LOGS = [
    #{"log_name": "BPIC_17",    "tss_index": 5},
    #{"log_name": "BPIC_17_DR", "tss_index": 5},
    #{"log_name": "BPIC_19",    "tss_index": 1},
    {"log_name": "Sepsis",    "tss_index": 4},  
]

ALL_MODELS = [
    #"SuTraN_DA",
    #"SuTraN_NDA",
    #"CRTP_LSTM_DA",
    "CRTP_LSTM_NDA",
    #"ED_LSTM",
    #"SEP_LSTM",
    #"BEST",
]

RESULT_DIRS = {
    "SuTraN_DA":    "SUTRAN_DA_results",
    "SuTraN_NDA":   "SUTRAN_NDA_results",
    "CRTP_LSTM_DA": "CRTP_LSTM_DA_results",
    "CRTP_LSTM_NDA":"CRTP_LSTM_NDA_results",
    "ED_LSTM":      "ED_LSTM_results",
    "SEP_LSTM":     "SEP_LSTM_results",
    "BEST":         "BEST_results",
}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run all PPM suffix-prediction experiments."
    )
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip dataset creation (assume tensors are already on disk).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        metavar="MODEL",
        default=None,
        help=(
            "Whitelist of models to train/evaluate (space-separated). "
            "If omitted, all models are run. "
            "Choices (case-insensitive): " + ", ".join(ALL_MODELS)
        ),
    )
    parser.add_argument(
        "--logs",
        nargs="+",
        metavar="LOG",
        default=None,
        help=(
            "Whitelist of log names to process (space-separated). "
            "If omitted, all logs are used. "
            "Choices: BPIC_17, BPIC_17_DR, BPIC_19"
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def section(title):
    bar = "=" * 70
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}\n", flush=True)


def run_step(label, fn):
    """Run *fn* and catch exceptions so one failure does not abort the rest."""
    print(f">>> {label}", flush=True)
    try:
        fn()
        print(f"    OK\n", flush=True)
        return True
    except Exception:
        print(f"    FAILED — traceback below:", flush=True)
        traceback.print_exc()
        print(flush=True)
        return False


# ---------------------------------------------------------------------------
# Part 1 — Dataset creation
# ---------------------------------------------------------------------------

def create_datasets(log_filter):
    section("Part 1 — Creating Datasets")

    steps = [
        ("BPIC_17",    "create_BPIC17_OG_data",  "construct_BPIC17_datasets"),
        ("BPIC_17_DR", "create_BPIC17_DR_data",  "construct_BPIC17_DR_datasets"),
        ("BPIC_19",    "create_BPIC19_data",      "construct_BPIC19_datasets"),
    ]

    for log_name, module_name, fn_name in steps:
        if log_filter and log_name not in log_filter:
            print(f"    Skipping {log_name} (not in --logs filter)\n", flush=True)
            continue
        import importlib
        mod = importlib.import_module(module_name)
        fn  = getattr(mod, fn_name)
        run_step(f"Creating tensors for {log_name}", fn)


# ---------------------------------------------------------------------------
# Part 2 — Model training & evaluation
# ---------------------------------------------------------------------------

def train_eval_all(model_filter, log_filter):
    section("Part 2 — Model Training & Evaluation")

    active_logs = [
        cfg for cfg in LOGS
        if log_filter is None or cfg["log_name"] in log_filter
    ]

    def run_model(label, fn):
        for cfg in active_logs:
            run_step(f"{label} — {cfg['log_name']}", fn(cfg))

    # ---- SuTraN (DA) -------------------------------------------------------
    if "SuTraN_DA" in model_filter:
        import TRAIN_EVAL_SUTRAN_DA as sutran_da
        section("Model 1 — SuTraN (Data-Aware)")
        for cfg in active_logs:
            run_step(
                f"SuTraN (DA) — {cfg['log_name']}",
                lambda c=cfg: sutran_da.train_eval(log_name=c["log_name"]),
            )

    # ---- SuTraN (NDA) -------------------------------------------------------
    if "SuTraN_NDA" in model_filter:
        import TRAIN_EVAL_SUTRAN_NDA as sutran_nda
        section("Model 2 — SuTraN (Non-Data-Aware)")
        for cfg in active_logs:
            run_step(
                f"SuTraN (NDA) — {cfg['log_name']}",
                lambda c=cfg: sutran_nda.train_eval(
                    log_name=c["log_name"], tss_index=c["tss_index"]
                ),
            )

    # ---- CRTP-LSTM (DA) ----------------------------------------------------
    if "CRTP_LSTM_DA" in model_filter:
        import TRAIN_EVAL_CRTP_LSTM_DA as crtp_da
        section("Model 3 — CRTP-LSTM (Data-Aware)")
        for cfg in active_logs:
            run_step(
                f"CRTP-LSTM (DA) — {cfg['log_name']}",
                lambda c=cfg: crtp_da.train_eval(log_name=c["log_name"]),
            )

    # ---- CRTP-LSTM (NDA) ---------------------------------------------------
    if "CRTP_LSTM_NDA" in model_filter:
        import TRAIN_EVAL_CRTP_LSTM_ND as crtp_nda
        section("Model 4 — CRTP-LSTM (Non-Data-Aware)")
        for cfg in active_logs:
            run_step(
                f"CRTP-LSTM (NDA) — {cfg['log_name']}",
                lambda c=cfg: crtp_nda.train_eval(
                    log_name=c["log_name"], tss_index=c["tss_index"]
                ),
            )

    # ---- ED-LSTM ------------------------------------------------------------
    if "ED_LSTM" in model_filter:
        import TRAIN_EVAL_ED_LSTM as ed_lstm
        section("Model 5 — ED-LSTM")
        for cfg in active_logs:
            run_step(
                f"ED-LSTM — {cfg['log_name']}",
                lambda c=cfg: ed_lstm.train_eval(
                    log_name=c["log_name"], tss_index=c["tss_index"]
                ),
            )

    # ---- SEP-LSTM -----------------------------------------------------------
    if "SEP_LSTM" in model_filter:
        import TRAIN_EVAL_SEP_LSTM as sep_lstm
        section("Model 6 — SEP-LSTM")
        for cfg in active_logs:
            run_step(
                f"SEP-LSTM — {cfg['log_name']}",
                lambda c=cfg: sep_lstm.train_eval(
                    log_name=c["log_name"], tss_index=c["tss_index"]
                ),
            )

    # ---- BEST ---------------------------------------------------------------
    if "BEST" in model_filter:
        import TRAIN_EVAL_BEST as best
        section("Model 7 — BEST")
        for cfg in active_logs:
            run_step(
                f"BEST — {cfg['log_name']}",
                lambda c=cfg: best.train_eval(log_name=c["log_name"]),
            )


# ---------------------------------------------------------------------------
# Part 3 — Results summary
# ---------------------------------------------------------------------------

def print_results(log_filter, model_filter):
    section("Part 3 — Results Summary")

    active_logs = [
        cfg["log_name"] for cfg in LOGS
        if log_filter is None or cfg["log_name"] in log_filter
    ]

    col_w = 22
    metric_w = 16

    header = (
        f"{'Log':<14}{'Model':<{col_w}}"
        f"{'DL sim ↑':>{metric_w}}"
        f"{'MAE TTNE (min) ↓':>{metric_w}}"
        f"{'MAE RRT (min) ↓':>{metric_w}}"
    )
    print(header)
    print("-" * len(header))

    for log_name in active_logs:
        for model in ALL_MODELS:
            if model not in model_filter:
                continue
            result_dir = RESULT_DIRS[model]
            path = os.path.join(
                log_name, result_dir, "TEST_SET_RESULTS", "averaged_results.pkl"
            )
            if not os.path.exists(path):
                dl = ttne = rrt = "N/A"
            else:
                with open(path, "rb") as f:
                    res = pickle.load(f)
                dl   = f"{res.get('DL sim', float('nan')):.4f}"
                ttne = f"{res.get('MAE TTNE minutes', float('nan')):.2f}"
                rrt  = f"{res.get('MAE RRT minutes', float('nan')):.2f}"

            print(
                f"{log_name:<14}{model:<{col_w}}"
                f"{dl:>{metric_w}}{ttne:>{metric_w}}{rrt:>{metric_w}}"
            )

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Normalise model filter to uppercase/underscore form
    if args.models is not None:
        normalised = {m.upper().replace("-", "_") for m in args.models}
        model_filter = [m for m in ALL_MODELS if m.upper() in normalised]
        unknown = normalised - {m.upper() for m in model_filter}
        if unknown:
            print(
                f"WARNING: unknown model(s) ignored: {unknown}\n"
                f"  Valid choices: {ALL_MODELS}",
                file=sys.stderr,
            )
    else:
        model_filter = list(ALL_MODELS)

    log_filter = None
    if args.logs is not None:
        log_filter = set(args.logs)
        unknown_logs = log_filter - {cfg["log_name"] for cfg in LOGS}
        if unknown_logs:
            print(
                f"WARNING: unknown log(s) ignored: {unknown_logs}\n"
                f"  Valid choices: {[cfg['log_name'] for cfg in LOGS]}",
                file=sys.stderr,
            )

    section("PPM Suffix Prediction Benchmark — run_experiments.py")
    print(f"Logs    : {[c['log_name'] for c in LOGS if log_filter is None or c['log_name'] in log_filter]}")
    print(f"Models  : {model_filter}")
    print(f"Skip data creation: {args.skip_data}")

    if not args.skip_data:
        create_datasets(log_filter)

    train_eval_all(model_filter, log_filter)

    print_results(log_filter, model_filter)


if __name__ == "__main__":
    main()
