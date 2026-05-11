#!/usr/bin/env python
"""Run all baseline model/method/log combinations and log progress to a file.

Each combination runs in a fresh subprocess so that TensorFlow and PyTorch
never share a CUDA context (mixing them in one process causes segfaults).
Subprocess exit also guarantees full GPU memory release between runs.

Skips combinations already marked DONE so the script is safe to re-run after
an interruption.

Usage:
    python run_all_baselines.py                          # run 1, 1 worker (default)
    python run_all_baselines.py --workers 4              # run 1, 4 logs in parallel
    python run_all_baselines.py --run-id 2
    python run_all_baselines.py --run-id 2 --workers 4

    Usage with docker:
    docker run -it --rm -v $(pwd):/app --gpus all ppm-baseline-one python3 run_all_baselines.py --workers 4
"""

import argparse
import concurrent.futures
import os
import subprocess
import sys
import threading
import traceback
from datetime import datetime

# ---------------------------------------------------------------------------
# Combinations to run
# ---------------------------------------------------------------------------

TASKS = [
    # (model, method)
    # For pydream "NAP" is just a progress-log label — run_pydream_for_log
    # has no method parameter.
    ("everman", "Activity-Context Bag Of Words PPMI w_5"),
    ("everman", "Activity-Context N-Grams PPMI w_5"),
    ("everman", "Bose 2009 Substitution Scores"),
    ("everman", "De Koninck 2018 act2vec CBOW w_3"),
    ("everman", "Gamallo Fernandez 2023 Context Based w_3"),
    ("everman", "one_hot"),
    ("pydream", "NAP"),
    ("tax",     "one_hot"),
]

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
    "BPI Challenge 2017"
]
# ---------------------------------------------------------------------------
# Default progress file location (same evaluation/ folder the models use)
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_EVAL_DIR = os.path.join(_HERE, "evaluation")
DEFAULT_PROGRESS_FILE = os.path.join(_EVAL_DIR, "run_all_baselines_progress.log")


def _results_dirs_for_run(run_id):
    """Return {model: results_dir} for the given run_id (1-based).

    Run 1 uses the default directory names (already completed).
    Runs 2+ get a '_runN' suffix so each run's CSVs are kept separately.
    """
    suffix = "" if run_id == 1 else f"_run{run_id}"
    return {
        "everman": os.path.join(_EVAL_DIR, f"results_everman{suffix}"),
        "pydream":  os.path.join(_EVAL_DIR, f"results_pydream{suffix}"),
        "tax":      os.path.join(_EVAL_DIR, f"results_tax{suffix}"),
    }


def _progress_file_for_run(run_id):
    if run_id == 1:
        return DEFAULT_PROGRESS_FILE
    return os.path.join(_EVAL_DIR, f"run_all_baselines_progress_run{run_id}.log")

# ---------------------------------------------------------------------------
# Code injected at the top of every subprocess.
#
# The psutil patch fixes a Docker/pm4py incompatibility: pm4py reads the
# parent process name at import time; inside a container the parent of PID 1
# is PID 0, which has no /proc/0/stat entry, causing NoSuchProcess to abort
# the import.  Falling back to the current process lets the import succeed.
# ---------------------------------------------------------------------------

_SUBPROCESS_PREAMBLE = f"""\
try:
    import psutil as _psutil
    _Orig = _psutil.Process
    class _Safe(_Orig):
        def __init__(self, pid=None):
            try:
                super().__init__(pid)
            except _psutil.NoSuchProcess:
                super().__init__(None)
    _psutil.Process = _Safe
except ImportError:
    pass
import sys
sys.path.insert(0, {_HERE!r})
"""

# ---------------------------------------------------------------------------
# Progress logging helpers
# ---------------------------------------------------------------------------

_log_lock = threading.Lock()


def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_progress(progress_file, status, model, method, log_name, detail=None):
    header = f"[{_ts()}] {status} | model={model} | method={method} | log={log_name}"
    with _log_lock:
        print(header, flush=True)
        with open(progress_file, "a", encoding="utf-8") as f:
            f.write(header + "\n")
            if detail:
                for line in detail.splitlines():
                    f.write(f"    {line}\n")
        if detail:
            print(detail, flush=True)


def result_exists(model, method, log_name, results_dir):
    """Return True if the result CSV for this combination already exists on disk."""
    if model == "pydream":
        path = os.path.join(results_dir, log_name, f"{log_name}_results.csv")
    else:  # everman, tax
        path = os.path.join(results_dir, log_name, method, f"{log_name}_{method}_results.csv")
    return os.path.isfile(path)

# ---------------------------------------------------------------------------
# Per-combination subprocess runner
# ---------------------------------------------------------------------------

_OOM_EXIT_CODE = 42


class _OOMError(Exception):
    pass


def _build_subprocess_code(model, method, log_name, results_dir, use_cpu=False):
    preamble = _SUBPROCESS_PREAMBLE
    cpu_env = "import os; os.environ['CUDA_VISIBLE_DEVICES'] = ''\n" if use_cpu else ""

    # Gamallo Fernandez uses PyTorch Lightning for embeddings, which is called
    # from inside run_everman_for_log.  TF is imported at module level in
    # next_activity_prediction_everman, so by the time Lightning tries to
    # initialize CUDA, TF already owns it — causing a segfault.
    #
    # Fix: (1) tell TF not to pre-allocate the whole GPU, and (2) force
    # PyTorch to initialize its CUDA context *before* TF does, so both
    # frameworks can coexist on the same device.
    if model == "everman" and method.startswith("Gamallo Fernandez"):
        preamble += (
            "import os\n"
            "os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'\n"
            "try:\n"
            "    import torch\n"
            "    if torch.cuda.is_available():\n"
            "        torch.zeros(1, device='cuda')  # init PyTorch CUDA before TF\n"
            "except Exception:\n"
            "    pass\n"
        )

    if model == "everman":
        call = f"run_everman_for_log(log_name={log_name!r}, method={method!r}, results_dir={results_dir!r})"
        body = "from next_activity_prediction_everman import run_everman_for_log\n"
    elif model == "pydream":
        call = f"run_pydream_for_log(log_name={log_name!r}, results_dir={results_dir!r})"
        body = "from next_activity_prediction_pydream import run_pydream_for_log\n"
    elif model == "tax":
        call = f"run_tax_for_log(log_name={log_name!r}, encoding_methods_list=[{method!r}], results_dir={results_dir!r})"
        body = "from next_activity_prediction_tax import run_tax_for_log\n"
    else:
        raise ValueError(f"Unknown model: {model}")

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
        preamble
        + cpu_env
        + body
        + f"try:\n"
        + f"    {call}\n"
        + oom_guard
    )


def run_combination(model, method, log_name, results_dir, use_cpu=False):
    """Run one combination in a fresh subprocess; raise on non-zero exit."""
    code = _build_subprocess_code(model, method, log_name, results_dir, use_cpu=use_cpu)
    result = subprocess.run([sys.executable, "-c", code], cwd=_HERE)
    if result.returncode == _OOM_EXIT_CODE:
        raise _OOMError()
    if result.returncode != 0:
        sig = -result.returncode if result.returncode < 0 else None
        raise RuntimeError(
            f"Subprocess exited with code {result.returncode}"
            + (f" (killed by signal {sig} — likely segfault)" if sig else "")
        )


# ---------------------------------------------------------------------------
# Per-job and per-log runners
# ---------------------------------------------------------------------------

def _run_task(model, method, log_name, results_dir, progress_file):
    log_progress(progress_file, "RUNNING", model, method, log_name)
    try:
        try:
            run_combination(model, method, log_name, results_dir, use_cpu=False)
        except _OOMError:
            log_progress(progress_file, "OOM→CPU", model, method, log_name)
            run_combination(model, method, log_name, results_dir, use_cpu=True)
        log_progress(progress_file, "DONE", model, method, log_name)
    except Exception:
        log_progress(progress_file, "ERROR", model, method, log_name,
                     detail=traceback.format_exc())


def _run_log(log_name, progress_file, results_dirs):
    """Run all (model, method) tasks sequentially for one log. Called from thread pool."""
    for model, method in TASKS:
        if result_exists(model, method, log_name, results_dirs[model]):
            print(f"[SKIP] model={model} | method={method} | log={log_name}", flush=True)
            continue
        _run_task(model, method, log_name, results_dirs[model], progress_file)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_all_baselines(run_id=1, progress_file=None, workers=1):
    progress_file = progress_file or _progress_file_for_run(run_id)
    results_dirs = _results_dirs_for_run(run_id)
    os.makedirs(os.path.dirname(progress_file), exist_ok=True)

    print(f"Run ID             : {run_id}")
    print(f"Workers (logs)     : {workers}")
    print(f"Total logs         : {len(EVENT_LOGS)}")
    print(f"Progress file      : {progress_file}\n", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_run_log, log_name, progress_file, results_dirs): log_name
            for log_name in EVENT_LOGS
        }
        for fut in concurrent.futures.as_completed(futures):
            log_name = futures[fut]
            try:
                fut.result()
            except Exception:
                print(f"[LOG-FATAL] log={log_name}\n{traceback.format_exc()}",
                      flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run all baseline model/method/log combinations."
    )
    parser.add_argument(
        "--run-id",
        type=int,
        default=1,
        help="Which repetition to run (1-5). Run 1 uses the default result dirs; "
             "runs 2-5 write to results_<model>_runN. (default: 1)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of event logs to train in parallel. Each worker runs all "
             "tasks for its log sequentially. Increase only if GPU memory allows. "
             "(default: 1)",
    )
    parser.add_argument(
        "--progress-file",
        default=None,
        help="Override the progress log file path (default: auto-derived from --run-id)",
    )
    args = parser.parse_args()
    run_all_baselines(run_id=args.run_id, progress_file=args.progress_file,
                      workers=args.workers)
