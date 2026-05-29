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
  nap_time_mlp → results_time_mlp/run_N/results_nap_time_mlp.csv
  nap_multiple → results_multiple/run_N/results_nap_multiple_gnn.csv
  nap_layer    → results_layer/run_N/results_nap_layer_gnn.csv
  nap_gru      → results_gru/run_N/results_nap_gru.csv

Usage
-----
    python run_all_nap.py                                       # nap, run 1, 1 worker
    python run_all_nap.py --workers 4                           # nap, run 1, 4 logs in parallel
    python run_all_nap.py --model nap_layer                  # nap_layer, run 1
    python run_all_nap.py --model nap_multiple --run-id 2 --workers 4
    python run_all_nap.py --logs-dir /path/to/logs

Usage with docker:
    docker run -it --rm -v $(pwd):/workspace --gpus all ml-jupyter-gpu python approach_nap/run_all_nap.py --workers 16 --model nap_gru
"""

import argparse
import concurrent.futures
import csv
import os
import subprocess
import sys
import threading
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
    'nap_time_mlp': {
        'module':      'approach_nap.run_nap_time_mlp',
        'results_sub': 'results_time_mlp',
        'csv_file':    'results_nap_time_mlp.csv',
    },
    'nap_multiple': {
        'module':      'approach_nap.run_nap_multiple',
        'results_sub': 'results_multiple',
        'csv_file':    'results_nap_multiple_gnn.csv',
    },
    'nap_layer': {
        'module':      'approach_nap.run_nap_layer',
        'results_sub': 'results_layer',
        'csv_file':    'results_nap_layer_gnn.csv',
    },
    'nap_gru': {
        'module':      'approach_nap.run_gru',
        'results_sub': 'results_gru',
        'csv_file':    'results_nap_gru.csv',
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

_log_lock = threading.Lock()


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

_OOM_EXIT_CODE = 42


class _OOMError(Exception):
    pass


def _build_code(log_path, log_name, results_dir, model, use_cpu=False):
    module  = MODEL_CONFIGS[model]['module']
    cpu_env = "import os; os.environ['CUDA_VISIBLE_DEVICES'] = ''\n" if use_cpu else ""
    oom_guard = (
        f"except Exception as _e:\n"
        f"    _oom = 'out of memory' in str(_e).lower()\n"
        f"    try:\n"
        f"        import torch; _oom = _oom or isinstance(_e, torch.cuda.OutOfMemoryError)\n"
        f"    except Exception: pass\n"
        f"    if _oom:\n"
        f"        import sys as _sys; _sys.exit({_OOM_EXIT_CODE})\n"
        f"    raise\n"
    ) if not use_cpu else "except Exception: raise\n"
    cleanup = (
        f"finally:\n"
        f"    try:\n"
        f"        import torch, gc; torch.cuda.empty_cache(); gc.collect()\n"
        f"    except Exception: pass\n"
    )
    return (
        _PREAMBLE
        + cpu_env
        + f"from {module} import run\n"
        + f"try:\n"
        + f"    run(log_path={log_path!r}, log_name={log_name!r}, results_dir={results_dir!r})\n"
        + oom_guard
        + cleanup
    )


def _run_subprocess(code):
    result = subprocess.run([sys.executable, "-c", code], cwd=_PROJECT_ROOT)
    if result.returncode == _OOM_EXIT_CODE:
        raise _OOMError()
    if result.returncode != 0:
        sig = -result.returncode if result.returncode < 0 else None
        raise RuntimeError(
            f"Subprocess exited with code {result.returncode}"
            + (f" (killed by signal {sig})" if sig else "")
        )


# ---------------------------------------------------------------------------
# Progress logging
# ---------------------------------------------------------------------------

def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log_progress(progress_file, status, log_name, detail=None):
    header = f"[{_ts()}] {status} | log={log_name}"
    with _log_lock:
        print(header, flush=True)
        with open(progress_file, "a", encoding="utf-8") as f:
            f.write(header + "\n")
            if detail:
                for line in detail.splitlines():
                    f.write(f"    {line}\n")
        if detail:
            print(detail, flush=True)


# ---------------------------------------------------------------------------
# Per-job runner
# ---------------------------------------------------------------------------

def _run_one(log_file, log_name, results_dir, model, progress_file):
    _log_progress(progress_file, "RUNNING", log_name)
    try:
        try:
            _run_subprocess(_build_code(log_file, log_name, results_dir, model, use_cpu=False))
        except _OOMError:
            _log_progress(progress_file, "OOM→CPU", log_name)
            _run_subprocess(_build_code(log_file, log_name, results_dir, model, use_cpu=True))
        _log_progress(progress_file, "DONE", log_name)
    except Exception:
        _log_progress(progress_file, "ERROR", log_name, detail=traceback.format_exc())


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_all(model='nap', run_id=1, progress_file=None, logs_dir=None, workers=1):
    if model not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model {model!r}. Choose from: "
                         f"{list(MODEL_CONFIGS)}")

    logs_dir      = logs_dir      or _DEFAULT_LOGS_DIR
    progress_file = progress_file or _progress_file(model, run_id)
    results_dir   = _results_dir(model, run_id)

    print(f"Model              : {model}")
    print(f"Run ID             : {run_id}")
    print(f"Workers (logs)     : {workers}")
    print(f"Total logs         : {len(EVENT_LOGS)}")
    print(f"Logs directory     : {logs_dir}")
    print(f"Results directory  : {results_dir}")
    print(f"Progress file      : {progress_file}\n", flush=True)

    jobs = []
    for log_name in EVENT_LOGS:
        if result_exists(log_name, model, run_id):
            print(f"[SKIP] log={log_name}", flush=True)
            continue
        log_file = _find_log_file(log_name, logs_dir)
        if log_file is None:
            print(f"[MISSING] log={log_name} — not found in {logs_dir} "
                  f"(.xes.gz or .xes)", flush=True)
            continue
        jobs.append((log_file, log_name))

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_run_one, log_file, log_name, results_dir, model, progress_file): log_name
            for log_file, log_name in jobs
        }
        for fut in concurrent.futures.as_completed(futures):
            try:
                fut.result()
            except Exception:
                log_name = futures[fut]
                print(f"[LOG-FATAL] log={log_name}\n{traceback.format_exc()}", flush=True)


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
        help="Which model to run: nap, nap_time_mlp, nap_multiple, nap_layer, nap_gru (default: nap)",
    )
    parser.add_argument(
        "--run-id",
        type=int,
        default=1,
        help="Which repetition to run (1-5). Results stored in "
             "<results_subdir>/run_N/. (default: 1)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of event logs to train in parallel. Each worker runs its "
             "log in a subprocess. Increase only if GPU memory allows. (default: 1)",
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
        workers=args.workers,
    )
