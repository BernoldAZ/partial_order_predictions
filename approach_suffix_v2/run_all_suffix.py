#!/usr/bin/env python
"""Run any GNN suffix variant on all event logs.

Each log runs in a fresh subprocess so the GPU is fully released between runs.
Results from run N are stored in <results_subdir>/run_N/ so all repetitions
are kept on disk — intended for later mean/std computation across 5 runs.

Skips logs where the run-N result already exists on disk.
Safe to re-run after an interruption.

Supported models and their output CSVs
---------------------------------------
  suffix_time_v1     → results_time_gatv2_gru_nb_v1/run_N/results_suffix_time_gnn.csv
  suffix_time_v1_seq → results_time_gatv2_seq_gru_nb_v1/run_N/results_suffix_time_gnn.csv
  suffix_time_v2     → results_time_gatv2_gru_nb_v2/run_N/results_suffix_time_gnn.csv
  suffix_time_v3     → results_time_gatv2_gru_nb_v3/run_N/results_suffix_time_gnn.csv

Arguments
-----
   --model {suffix_time_v1, suffix_time_v1_seq, suffix_time_v2, suffix_time_v3}
   --run-id N            repetition index (1-5)
   --workers N           event logs trained in parallel
   --logs-dir PATH       override the XES logs directory
   --progress-file PATH  override the progress log path
   --no-train            skip training, load the saved model
   --no-eval             skip evaluation, only time inference

Usage
-----
    python run_all_suffix.py                                                  # suffix_time_v1, run 1, 1 worker
    python run_all_suffix.py --workers 4                                      # suffix_time_v1, run 1, 4 logs in parallel
    python run_all_suffix.py --model suffix_time_v1_seq --run-id 2 --workers 4
    python run_all_suffix.py --logs-dir /path/to/logs

Docker
------
    docker run -it --rm -v $(pwd):/workspace --gpus all ml-jupyter-gpu python approach_suffix_v2/run_all_suffix.py --workers 10 --model suffix_time_v1 --run-id 1 --no-train --no-eval
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

# ─────────────────────────────────────────────
# Event logs  (same set as run_all_nap.py)
# ─────────────────────────────────────────────

EVENT_LOGS = [
    #"RequestForPayment",
    "Sepsis",
    #"Hospital_Billing",
    #"DomesticDeclarations",
    #"PrepaidTravelCost",
    #"InternationalDeclarations",
    "BPI_Challenge_2012_A",
    "BPI_Challenge_2012_O",
    #"BPI_Challenge_2012_W",
    "BPIC15_1",
    "BPIC15_2",
    "BPIC15_3",
    "BPIC15_4",
    "BPIC15_5",
    #"Road_Traffic_Fine_Management_Process",
    #"BPI Challenge 2017",
]

# ─────────────────────────────────────────────
# Per-model configuration
# ─────────────────────────────────────────────

MODEL_CONFIGS = {
    'suffix_time_v1': {
        'module':          'approach_suffix_v2.models_v2.run_suffix_time_v1',
        'version':         None,
        'no_log_path':     True,
        'results_subdir':  'results_time_gatv2_gru_nb_v1',
        'csv_file':        'results_suffix_time_gnn.csv',
    },
    'suffix_time_v1_seq': {
        'module':          'approach_suffix_v2.models_v2.run_suffix_time_v1_seq',
        'version':         None,
        'no_log_path':     True,
        'results_subdir':  'results_time_gatv2_seq_gru_nb_v1',
        'csv_file':        'results_suffix_time_gnn.csv',
    },
    'suffix_time_v2': {
        'module':          'approach_suffix_v2.models_v2.run_suffix_time_v2',
        'version':         None,
        'no_log_path':     True,
        'results_subdir':  'results_time_gatv2_gru_nb_v2',
        'csv_file':        'results_suffix_time_gnn.csv',
    },
    'suffix_time_v3': {
        'module':          'approach_suffix_v2.models_v2.run_suffix_time_v3',
        'version':         None,
        'no_log_path':     True,
        'results_subdir':  'results_time_gatv2_gru_nb_v3',
        'csv_file':        'results_suffix_time_gnn.csv',
    },
}

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────

_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)

_DEFAULT_LOGS_DIR = os.path.join(
    _PROJECT_ROOT, "baselines", "SuffixTransformerNetwork", "Logs"
)


def _results_dir(model, run_id):
    sub = MODEL_CONFIGS[model]['results_subdir']
    return os.path.join(_HERE, sub, f"run_{run_id}")


def _progress_file(model, run_id):
    return os.path.join(_HERE, f"run_all_{model}_progress_run{run_id}.log")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

_log_lock = threading.Lock()
_OOM_EXIT_CODE = 42


class _OOMError(Exception):
    pass


def result_exists(log_name, model, run_id, do_eval=True):
    csv_name = MODEL_CONFIGS[model]['csv_file'] if do_eval else 'inference_times.csv'
    csv_path = os.path.join(_results_dir(model, run_id), csv_name)
    if not os.path.isfile(csv_path):
        return False
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            if row.get('log') == log_name:
                return True
    return False


def _find_log_file(log_name, logs_dir):
    for ext in (".xes.gz", ".xes"):
        path = os.path.join(logs_dir, f"{log_name}{ext}")
        if os.path.isfile(path):
            return path
    return None


# ─────────────────────────────────────────────
# Subprocess
# ─────────────────────────────────────────────

_PREAMBLE = f"""\
import sys, os
sys.path.insert(0, {_PROJECT_ROOT!r})
sys.path.insert(0, {_HERE!r})
sys.path.insert(0, {os.path.join(_HERE, 'models')!r})
sys.path.insert(0, {os.path.join(_HERE, 'models_v2')!r})
os.chdir({_PROJECT_ROOT!r})
"""

_OOM_EXIT_CODE = 42

_OOM_GUARD = (
    f"except Exception as _e:\n"
    f"    _oom = 'out of memory' in str(_e).lower()\n"
    f"    try:\n"
    f"        import torch; _oom = _oom or isinstance(_e, torch.cuda.OutOfMemoryError)\n"
    f"    except Exception: pass\n"
    f"    if _oom:\n"
    f"        import sys as _sys; _sys.exit({_OOM_EXIT_CODE})\n"
    f"    raise\n"
)

_OOM_GUARD_CPU = "except Exception: raise\n"

_CLEANUP = (
    f"finally:\n"
    f"    try:\n"
    f"        import torch, gc; torch.cuda.empty_cache(); gc.collect()\n"
    f"    except Exception: pass\n"
)


def _build_code(log_path, log_name, results_dir, model, use_cpu=False,
                do_train=True, do_eval=True):
    cfg       = MODEL_CONFIGS[model]
    module    = cfg['module']
    version   = cfg['version']
    cpu_env   = "import os; os.environ['CUDA_VISIBLE_DEVICES'] = ''\n" if use_cpu else ""
    oom_guard = _OOM_GUARD_CPU if use_cpu else _OOM_GUARD
    if version is not None:
        call_args = f"log_name={log_name!r}, version={version!r}, results_dir={results_dir!r}"
    elif cfg.get('no_log_path'):
        call_args = f"log_name={log_name!r}, results_dir={results_dir!r}"
    else:
        call_args = f"log_path={log_path!r}, log_name={log_name!r}, results_dir={results_dir!r}"
    if not do_train:
        call_args += ", do_train=False"
    if not do_eval:
        call_args += ", do_eval=False"
    return (
        _PREAMBLE
        + cpu_env
        + f"from {module} import run\n"
        + f"try:\n"
        + f"    run({call_args})\n"
        + oom_guard
        + _CLEANUP
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


# ─────────────────────────────────────────────
# Progress logging
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# Per-job runner
# ─────────────────────────────────────────────

def _run_one(log_file, log_name, results_dir, model, progress_file,
             do_train=True, do_eval=True):
    _log_progress(progress_file, "RUNNING", log_name)
    try:
        try:
            _run_subprocess(_build_code(log_file, log_name, results_dir, model, use_cpu=False,
                                        do_train=do_train, do_eval=do_eval))
        except _OOMError:
            _log_progress(progress_file, "OOM→CPU", log_name)
            _run_subprocess(_build_code(log_file, log_name, results_dir, model, use_cpu=True,
                                        do_train=do_train, do_eval=do_eval))
        _log_progress(progress_file, "DONE", log_name)
    except Exception:
        _log_progress(progress_file, "ERROR", log_name, detail=traceback.format_exc())


# ─────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────

def run_all(model='suffix_time_v1', run_id=1, progress_file=None, logs_dir=None, workers=1,
            do_train=True, do_eval=True):
    if model not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model {model!r}. Choose from: {list(MODEL_CONFIGS)}")

    logs_dir      = logs_dir      or _DEFAULT_LOGS_DIR
    progress_file = progress_file or _progress_file(model, run_id)
    results_dir   = _results_dir(model, run_id)

    print(f"Model              : {model}")
    print(f"Run ID             : {run_id}")
    print(f"Train / Eval       : {do_train} / {do_eval}")
    print(f"Workers (logs)     : {workers}")
    print(f"Total logs         : {len(EVENT_LOGS)}")
    print(f"Logs directory     : {logs_dir}")
    print(f"Results directory  : {results_dir}")
    print(f"Progress file      : {progress_file}\n", flush=True)

    jobs = []
    for log_name in EVENT_LOGS:
        if result_exists(log_name, model, run_id, do_eval):
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
            pool.submit(_run_one, lf, ln, results_dir, model, progress_file,
                        do_train, do_eval): ln
            for lf, ln in jobs
        }
        for fut in concurrent.futures.as_completed(futures):
            try:
                fut.result()
            except Exception:
                ln = futures[fut]
                print(f"[LOG-FATAL] log={ln}\n{traceback.format_exc()}", flush=True)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a GNN suffix variant on all event logs."
    )
    parser.add_argument(
        "--model",
        default="suffix_time_v1",
        choices=sorted(MODEL_CONFIGS.keys()),
        help="Which model to run (default: suffix_time_v1)",
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
        help="Number of event logs to train in parallel. (default: 1)",
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
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Skip training; load the saved model for each log / run_id",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip evaluation; only run inference and report its time",
    )
    args = parser.parse_args()
    run_all(
        model=args.model,
        run_id=args.run_id,
        progress_file=args.progress_file,
        logs_dir=args.logs_dir,
        workers=args.workers,
        do_train=not args.no_train,
        do_eval=not args.no_eval,
    )
