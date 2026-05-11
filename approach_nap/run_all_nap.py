#!/usr/bin/env python
"""Run any GNN NAP variant on all event logs.

Each log runs in a fresh subprocess so the GPU is fully released between runs.
Results from run N are stored in <results_subdir>/run_N/ so all repetitions
are kept on disk — intended for later mean/std computation across 5 runs.

Skips logs where the run-N result already exists on disk.
Safe to re-run after an interruption.

Supported models and their output CSVs
---------------------------------------
  nap          → results/run_N/results_nap_gnn.csv
  nap_time     → results_time/run_N/results_nap_gnn_time.csv
  nap_multiple → results_multiple/run_N/results_nap_multiple_gnn.csv

Usage
-----
    python run_all_nap.py                                   # nap, run 1
    python run_all_nap.py --model nap_time                  # nap_time, run 1
    python run_all_nap.py --model nap_multiple --run-id 2
    python run_all_nap.py --logs-dir /path/to/logs
"""

import argparse
import csv
import os
import subprocess
import sys
import traceback
from datetime import datetime

# ---------------------------------------------------------------------------
# Event logs  (same set as run_all_baselines.py)
# ---------------------------------------------------------------------------

EVENT_LOGS = [
    "RequestForPayment",
    "Sepsis",
    "Hospital_Billing",
    "DomesticDeclarations",
    "PrepaidTravelCost",
    "InternationalDeclarations",
    #"BPI_Challenge_2012",
    "BPI_Challenge_2012_A",
    "BPI_Challenge_2012_O",
    "BPI_Challenge_2012_W",
    "BPIC15_1",
    "BPIC15_2",
    "BPIC15_3",
    "BPIC15_4",
    "BPIC15_5",
    "Road_Traffic_Fine_Management_Process",
    "BPI Challenge 2017"
]

# ---------------------------------------------------------------------------
# Per-model configuration
# Each model writes to its own results subdirectory and CSV file.
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    'nap': {
        'module':      'approach_nap.run_nap',
        'results_sub': 'results',
        'csv_file':    'results_nap_gnn.csv',
    },
    'nap_time': {
        'module':      'approach_nap.run_nap_time',
        'results_sub': 'results_time',
        'csv_file':    'results_nap_gnn_time.csv',
    },
    'nap_multiple': {
        'module':      'approach_nap.run_nap_multiple',
        'results_sub': 'results_multiple',
        'csv_file':    'results_nap_multiple_gnn.csv',
    },
}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)

_DEFAULT_LOGS_DIR = os.path.join(
    _PROJECT_ROOT, "baselines", "next_activity_prediction", "event_logs"
)


def _results_dir(model, run_id):
    return os.path.join(_HERE, MODEL_CONFIGS[model]['results_sub'], f"run_{run_id}")


def _progress_file(model, run_id):
    return os.path.join(_HERE, f"run_all_nap_{model}_progress_run{run_id}.log")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def result_exists(log_name, model, run_id):
    csv_path = os.path.join(_results_dir(model, run_id),
                            MODEL_CONFIGS[model]['csv_file'])
    if not os.path.isfile(csv_path):
        return False
    with open(csv_path, newline='') as f:
        return any(row.get('log') == log_name for row in csv.DictReader(f))


def _find_log_file(log_name, logs_dir):
    for ext in (".xes.gz", ".xes"):
        path = os.path.join(logs_dir, f"{log_name}{ext}")
        if os.path.isfile(path):
            return path
    return None


# ---------------------------------------------------------------------------
# Subprocess
# ---------------------------------------------------------------------------

_PREAMBLE = f"""\
import sys, os
sys.path.insert(0, {_PROJECT_ROOT!r})
os.chdir({_PROJECT_ROOT!r})
"""


def _build_code(log_path, log_name, results_dir, model):
    module = MODEL_CONFIGS[model]['module']
    return (
        _PREAMBLE
        + f"from {module} import run\n"
        + f"try:\n"
        + f"    run(log_path={log_path!r}, log_name={log_name!r}, results_dir={results_dir!r})\n"
        + f"finally:\n"
        + f"    try:\n"
        + f"        import torch, gc; torch.cuda.empty_cache(); gc.collect()\n"
        + f"    except Exception:\n"
        + f"        pass\n"
    )


def _run_subprocess(code):
    result = subprocess.run([sys.executable, "-c", code], cwd=_PROJECT_ROOT)
    if result.returncode != 0:
        sig = -result.returncode if result.returncode < 0 else None
        detail = (
            f"Subprocess exited with code {result.returncode}"
            + (f" (killed by signal {sig})" if sig else "")
        )
        raise RuntimeError(detail)


# ---------------------------------------------------------------------------
# Progress logging
# ---------------------------------------------------------------------------

def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log_progress(progress_file, status, log_name, detail=None):
    header = f"[{_ts()}] {status} | log={log_name}"
    print(header, flush=True)
    with open(progress_file, "a", encoding="utf-8") as f:
        f.write(header + "\n")
        if detail:
            for line in detail.splitlines():
                f.write(f"    {line}\n")
    if detail:
        print(detail, flush=True)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_all(model='nap', run_id=1, progress_file=None, logs_dir=None):
    if model not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model {model!r}. Choose from: "
                         f"{list(MODEL_CONFIGS)}")

    logs_dir      = logs_dir      or _DEFAULT_LOGS_DIR
    progress_file = progress_file or _progress_file(model, run_id)

    print(f"Model              : {model}")
    print(f"Run ID             : {run_id}")
    print(f"Total logs         : {len(EVENT_LOGS)}")
    print(f"Logs directory     : {logs_dir}")
    print(f"Results directory  : {_results_dir(model, run_id)}")
    print(f"Progress file      : {progress_file}\n", flush=True)

    for log_name in EVENT_LOGS:
        if result_exists(log_name, model, run_id):
            print(f"[SKIP] log={log_name}", flush=True)
            continue

        log_file = _find_log_file(log_name, logs_dir)
        if log_file is None:
            print(f"[MISSING] log={log_name} — not found in {logs_dir} "
                  f"(.xes.gz or .xes)", flush=True)
            continue

        _log_progress(progress_file, "RUNNING", log_name)
        try:
            _run_subprocess(
                _build_code(log_file, log_name, _results_dir(model, run_id), model)
            )
            _log_progress(progress_file, "DONE", log_name)
        except Exception:
            _log_progress(progress_file, "ERROR", log_name,
                          detail=traceback.format_exc())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a GNN NAP variant on all event logs."
    )
    parser.add_argument(
        "--model",
        default="nap",
        choices=list(MODEL_CONFIGS),
        help="Which model to run: nap, nap_time, nap_multiple (default: nap)",
    )
    parser.add_argument(
        "--run-id",
        type=int,
        default=1,
        help="Which repetition to run (1-5). Results stored in "
             "<results_subdir>/run_N/. (default: 1)",
    )
    parser.add_argument(
        "--logs-dir",
        default=None,
        help=f"Directory containing event log XES files "
             f"(default: {_DEFAULT_LOGS_DIR})",
    )
    parser.add_argument(
        "--progress-file",
        default=None,
        help="Override the progress log file path (default: auto-derived "
             "from --model and --run-id)",
    )
    args = parser.parse_args()
    run_all(
        model=args.model,
        run_id=args.run_id,
        progress_file=args.progress_file,
        logs_dir=args.logs_dir,
    )
