#!/usr/bin/env python
"""Run all SuffixTransformerNetwork models on all event logs.

Each combination runs in a fresh subprocess so the GPU is fully released
between runs (process exit guarantees CUDA context cleanup).

Results from run N are stored in <model>_results_runN/ so all repetitions
are kept on disk — intended for later mean/std computation across 5 runs.

Skips combinations where the run-N result already exists on disk.
Safe to re-run after an interruption.

Usage:
    python run_all_suffix_baselines.py                          # run 1, 1 worker (default)
    python run_all_suffix_baselines.py --workers 12             # run 1, 4 logs in parallel
    python run_all_suffix_baselines.py --run-id 2               # run 2
    python run_all_suffix_baselines.py --run-id 2 --workers 4 --progress-file /path/to/custom.log

Usage with docker:
    docker run -it --rm -v $(pwd):/app --gpus all ppm-sutran-best python3 run_all_suffix_baselines.py --workers 16 --run-id 2
"""

import argparse
import concurrent.futures
import os
import shutil
import subprocess
import sys
import threading
import traceback
from datetime import datetime

# ---------------------------------------------------------------------------
# Models and logs
# ---------------------------------------------------------------------------

MODELS = [
    "SuTraN_DA",
    "SuTraN_NDA",
    "CRTP_LSTM_DA",
    "CRTP_LSTM_NDA",
    "ED_LSTM",
    "SEP_LSTM",
    "BEST",
]

# Models that require a tss_index parameter (non-data-aware variants).
_NDA_MODELS = {"SuTraN_NDA", "CRTP_LSTM_NDA", "ED_LSTM", "SEP_LSTM"}

EVENT_LOGS = [
    "BPIC15_1",
    "BPIC15_2",
    "BPIC15_3",
    "BPIC15_4",
    "BPIC15_5",
    "RequestForPayment",
    "Sepsis",
    "DomesticDeclarations",
    "PrepaidTravelCost",
    "InternationalDeclarations",
    #"BPI_Challenge_2012", # It have been splitted in 3 files A,O,W. Many papers do this.
    "BPI_Challenge_2012_A",
    "BPI_Challenge_2012_O",
    "BPI_Challenge_2012_W",
    "Hospital_Billing",
    "Road_Traffic_Fine_Management_Process",
    #"BPI Challenge 2017"
]

# Maps model name → the directory name train_eval writes results into.
_RESULT_DIRS = {
    "SuTraN_DA":     "SUTRAN_DA_results",
    "SuTraN_NDA":    "SUTRAN_NDA_results",
    "CRTP_LSTM_DA":  "CRTP_LSTM_DA_results",
    "CRTP_LSTM_NDA": "CRTP_LSTM_NDA_results",
    "ED_LSTM":       "ED_LSTM_results",
    "SEP_LSTM":      "SEP_LSTM_results",
    "BEST":          "BEST_results",
}

# Maps model name → the TRAIN_EVAL_*.py module to import.
_MODULE_MAP = {
    "SuTraN_DA":     "TRAIN_EVAL_SUTRAN_DA",
    "SuTraN_NDA":    "TRAIN_EVAL_SUTRAN_NDA",
    "CRTP_LSTM_DA":  "TRAIN_EVAL_CRTP_LSTM_DA",
    "CRTP_LSTM_NDA": "TRAIN_EVAL_CRTP_LSTM_ND",
    "ED_LSTM":       "TRAIN_EVAL_ED_LSTM",
    "SEP_LSTM":      "TRAIN_EVAL_SEP_LSTM",
    "BEST":          "TRAIN_EVAL_BEST",
}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_RESULTS_BASE = os.path.join(_HERE, "results_per_log")


def _run_result_dir(model, log_name, run_id):
    return os.path.join(_RESULTS_BASE, log_name, f"{_RESULT_DIRS[model]}_run{run_id}")


def result_exists(model, log_name, run_id):
    path = os.path.join(_run_result_dir(model, log_name, run_id),
                        "TEST_SET_RESULTS", "averaged_results.pkl")
    return os.path.isfile(path)


def data_exists(log_name):
    return os.path.isfile(os.path.join(_RESULTS_BASE, log_name, "train_tensordataset.pt"))


def _read_tss_index(log_name):
    path = os.path.join(_RESULTS_BASE, log_name, "tss_index.txt")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return int(f.read().strip())


def _progress_file_for_run(run_id):
    return os.path.join(_HERE, f"run_all_suffix_progress_run{run_id}.log")


def _find_log_file(log_name):
    """Return the path to the log file, trying .xes.gz then .xes. None if missing."""
    for ext in (".xes.gz", ".xes"):
        path = os.path.join(_HERE, "Logs", f"{log_name}{ext}")
        if os.path.isfile(path):
            return path
    return None


# ---------------------------------------------------------------------------
# Progress logging
# ---------------------------------------------------------------------------

_log_lock = threading.Lock()


def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_progress(progress_file, status, model, log_name, detail=None):
    header = f"[{_ts()}] {status} | model={model} | log={log_name}"
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
# Subprocess code builders
#
# All subprocesses chdir to _HERE so that relative paths inside the
# TRAIN_EVAL_*.py files (e.g. "results_per_log/...") resolve correctly.
# ---------------------------------------------------------------------------

_PREAMBLE = f"""\
import sys, os
sys.path.insert(0, {_HERE!r})
os.chdir({_HERE!r})
"""

_GPU_CLEANUP = """\
try:
    import torch, gc
    torch.cuda.empty_cache()
    gc.collect()
except Exception:
    pass
"""


def _build_data_code(log_name, log_file):
    log_path = os.path.relpath(log_file, _HERE)
    return (
        _PREAMBLE
        + f"from create_general_data import construct_datasets\n"
        + f"construct_datasets(log_path={log_path!r}, log_name={log_name!r})\n"
        + _GPU_CLEANUP
    )


_OOM_EXIT_CODE = 42


class _OOMError(Exception):
    pass


def _build_model_code(model, log_name, tss_index, use_cpu=False):
    module = _MODULE_MAP[model]
    if model in _NDA_MODELS:
        call = f"m.train_eval(log_name={log_name!r}, tss_index={tss_index})"
    else:
        call = f"m.train_eval(log_name={log_name!r})"

    cpu_env = "import os; os.environ['CUDA_VISIBLE_DEVICES'] = ''\n" if use_cpu else ""

    # When running on GPU, catch OOM and exit with sentinel so the orchestrator
    # can retry the same job on CPU without counting it as a hard failure.
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

    return (
        _PREAMBLE
        + cpu_env
        + f"import {module} as m\n"
        + f"try:\n"
        + f"    {call}\n"
        + oom_guard
        + _GPU_CLEANUP
    )


def _run_subprocess(code):
    result = subprocess.run([sys.executable, "-c", code], cwd=_HERE)
    if result.returncode == _OOM_EXIT_CODE:
        raise _OOMError()
    if result.returncode != 0:
        sig = -result.returncode if result.returncode < 0 else None
        raise RuntimeError(
            f"Subprocess exited with code {result.returncode}"
            + (f" (killed by signal {sig})" if sig else "")
        )


# ---------------------------------------------------------------------------
# Per-job runner (one model × one log)
# ---------------------------------------------------------------------------

def _train_one(model, log_name, tss, run_id, progress_file):
    log_progress(progress_file, "RUNNING", model, log_name)
    try:
        try:
            _run_subprocess(_build_model_code(model, log_name, tss, use_cpu=False))
        except _OOMError:
            log_progress(progress_file, "OOM→CPU", model, log_name)
            _run_subprocess(_build_model_code(model, log_name, tss, use_cpu=True))

        src = os.path.join(_RESULTS_BASE, log_name, _RESULT_DIRS[model])
        dst = _run_result_dir(model, log_name, run_id)
        if os.path.isdir(src):
            shutil.move(src, dst)

        log_progress(progress_file, "DONE", model, log_name)
    except Exception:
        log_progress(progress_file, "ERROR", model, log_name,
                     detail=traceback.format_exc())


def _run_combination(model, log_name, run_id, progress_file):
    """Run one model × log combination. Called from thread pool."""
    if not data_exists(log_name):
        print(f"[SKIP-NO-DATA] model={model} | log={log_name}", flush=True)
        return
    if result_exists(model, log_name, run_id):
        print(f"[SKIP] model={model} | log={log_name}", flush=True)
        return
    tss = None
    if model in _NDA_MODELS:
        tss = _read_tss_index(log_name)
        if tss is None:
            print(f"[SKIP-NO-TSS] model={model} | log={log_name} "
                  f"— tss_index.txt missing, re-run preprocessing", flush=True)
            return
    _train_one(model, log_name, tss, run_id, progress_file)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_all_suffix(run_id=1, progress_file=None, workers=1):
    progress_file = progress_file or _progress_file_for_run(run_id)
    os.makedirs(_RESULTS_BASE, exist_ok=True)

    print(f"Run ID             : {run_id}")
    print(f"Workers (combinations) : {workers}")
    print(f"Total combinations     : {len(EVENT_LOGS) * len(MODELS)}")
    print(f"Progress file      : {progress_file}\n", flush=True)

    # ── Step 1: preprocessing (sequential — typically already cached) ──────
    for log_name in EVENT_LOGS:
        if data_exists(log_name):
            print(f"[DATA-SKIP] {log_name}", flush=True)
            continue
        log_file = _find_log_file(log_name)
        if log_file is None:
            print(f"[DATA-MISSING] {log_name} — not found in Logs/ (.xes.gz or .xes)", flush=True)
            continue
        print(f"[DATA-RUNNING] {log_name}", flush=True)
        try:
            _run_subprocess(_build_data_code(log_name, log_file))
            print(f"[DATA-DONE] {log_name}", flush=True)
        except Exception:
            print(f"[DATA-ERROR] {log_name}\n{traceback.format_exc()}", flush=True)

    # ── Step 2: model training — parallelised at the combination level ────────
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_run_combination, model, log_name, run_id, progress_file): (model, log_name)
            for log_name in EVENT_LOGS
            for model in MODELS
        }
        for fut in concurrent.futures.as_completed(futures):
            model, log_name = futures[fut]
            try:
                fut.result()
            except Exception:
                print(f"[COMBO-FATAL] model={model} | log={log_name}\n{traceback.format_exc()}",
                      flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run all SuffixTransformerNetwork models on all event logs."
    )
    parser.add_argument(
        "--run-id",
        type=int,
        default=1,
        help="Which repetition to run (1-5). Results are stored in "
             "<model>_results_runN/ so all runs are kept. (default: 1)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of event logs to train in parallel. Each worker runs all "
             "models for its log sequentially. Increase only if GPU memory allows. "
             "(default: 1)",
    )
    parser.add_argument(
        "--progress-file",
        default=None,
        help="Override the progress log file path (default: auto-derived from --run-id)",
    )
    args = parser.parse_args()
    run_all_suffix(run_id=args.run_id, progress_file=args.progress_file,
                   workers=args.workers)
