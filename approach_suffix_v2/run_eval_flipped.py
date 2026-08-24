#!/usr/bin/env python
"""Evaluate trained seq-graph models on the flipped test set.

For each run_X folder found under the model's results directory, loads the
trained checkpoint for every event log that has one and runs inference on
test_seqgraphdataset_flip.pt.  Results are written to
<results_subdir>/run_X/results_suffix_time_gnn_flip.csv.

Skips logs where the flip result already exists or where no checkpoint is found.
Safe to re-run after an interruption.

Supported models
----------------
  suffix_time_v1_seq  →  results_time_gatv2_seq_gru_nb_v1/

Usage
-----
    python run_eval_flipped.py --model suffix_time_v1_seq
    docker run -it --rm -v $(pwd):/workspace --gpus all ml-jupyter-gpu python approach_suffix_v2/run_eval_flipped.py --model suffix_time_v1_seq --workers 4
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
# Event logs
# ─────────────────────────────────────────────

EVENT_LOGS = [
    "Sepsis",
    "BPI_Challenge_2012_A",
    "BPI_Challenge_2012_O",
    "BPIC15_1",
    "BPIC15_2",
    "BPIC15_3",
    "BPIC15_4",
    "BPIC15_5",
]

# ─────────────────────────────────────────────
# Per-model configuration
# ─────────────────────────────────────────────

MODEL_CONFIGS = {
    'suffix_time_v1_seq': {
        'module':         'approach_suffix_v2.models_v2.run_suffix_time_v1_seq',
        'results_subdir': 'results_time_gatv2_seq_gru_nb_v1',
        'method_name':    'gatv2_seq_gru_nb_v1',
    },
}

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────

_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)

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

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

_log_lock = threading.Lock()


class _OOMError(Exception):
    pass


def _find_run_dirs(model):
    subdir = os.path.join(_HERE, MODEL_CONFIGS[model]['results_subdir'])
    if not os.path.isdir(subdir):
        return []
    return sorted(
        d for d in os.listdir(subdir)
        if d.startswith('run_') and os.path.isdir(os.path.join(subdir, d))
    )


def _run_dir_path(model, run_name):
    return os.path.join(_HERE, MODEL_CONFIGS[model]['results_subdir'], run_name)


def checkpoint_exists(log_name, model, run_dir):
    method = MODEL_CONFIGS[model]['method_name']
    return os.path.isfile(os.path.join(run_dir, f'{log_name}_{method}.pt'))


def flip_result_exists(log_name, model, run_dir):
    csv_path = os.path.join(run_dir, 'results_suffix_time_gnn_flip.csv')
    if not os.path.isfile(csv_path):
        return False
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            if row.get('log') == log_name:
                return True
    return False


def flip_data_exists(log_name):
    path = os.path.join(_HERE, 'results_per_log', log_name, 'test_seqgraphdataset_flip.pt')
    return os.path.isfile(path)


# ─────────────────────────────────────────────
# Subprocess
# ─────────────────────────────────────────────

def _build_eval_code(log_name, results_dir, model, use_cpu=False):
    module    = MODEL_CONFIGS[model]['module']
    cpu_env   = "import os; os.environ['CUDA_VISIBLE_DEVICES'] = ''\n" if use_cpu else ""
    oom_guard = _OOM_GUARD_CPU if use_cpu else _OOM_GUARD
    call_args = f"log_name={log_name!r}, results_dir={results_dir!r}"
    return (
        _PREAMBLE
        + cpu_env
        + f"from {module} import run_eval_flip\n"
        + f"try:\n"
        + f"    run_eval_flip({call_args})\n"
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


def _log_progress(progress_file, status, log_name, run_name, detail=None):
    header = f"[{_ts()}] {status} | log={log_name} run={run_name}"
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

def _run_one(log_name, run_dir, run_name, model, progress_file):
    _log_progress(progress_file, "RUNNING", log_name, run_name)
    try:
        try:
            _run_subprocess(_build_eval_code(log_name, run_dir, model, use_cpu=False))
        except _OOMError:
            _log_progress(progress_file, "OOM→CPU", log_name, run_name)
            _run_subprocess(_build_eval_code(log_name, run_dir, model, use_cpu=True))
        _log_progress(progress_file, "DONE", log_name, run_name)
    except Exception:
        _log_progress(progress_file, "ERROR", log_name, run_name, detail=traceback.format_exc())


# ─────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────

def run_all(model, workers=1, progress_file=None):
    if model not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model {model!r}. Choose from: {list(MODEL_CONFIGS)}")

    run_names = _find_run_dirs(model)
    if not run_names:
        print(f"No run_X folders found under "
              f"'{MODEL_CONFIGS[model]['results_subdir']}'. Nothing to do.")
        return

    progress_file = progress_file or os.path.join(
        _HERE, f"run_eval_flip_{model}.log")

    print(f"Model      : {model}")
    print(f"Runs found : {run_names}")
    print(f"Workers    : {workers}")
    print(f"Progress   : {progress_file}\n", flush=True)

    jobs = []
    for run_name in run_names:
        run_dir = _run_dir_path(model, run_name)
        for log_name in EVENT_LOGS:
            if not flip_data_exists(log_name):
                print(f"[SKIP-NO-FLIP-DATA] log={log_name}  "
                      f"(test_seqgraphdataset_flip.pt missing)", flush=True)
                continue
            if not checkpoint_exists(log_name, model, run_dir):
                print(f"[SKIP-NO-MODEL] log={log_name}  run={run_name}", flush=True)
                continue
            if flip_result_exists(log_name, model, run_dir):
                print(f"[SKIP] log={log_name}  run={run_name}", flush=True)
                continue
            jobs.append((log_name, run_dir, run_name))

    if not jobs:
        print("All results already exist or no checkpoints found. Nothing to do.")
        return

    print(f"Jobs to run: {len(jobs)}\n", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_run_one, ln, rd, rn, model, progress_file): (ln, rn)
            for ln, rd, rn in jobs
        }
        for fut in concurrent.futures.as_completed(futures):
            try:
                fut.result()
            except Exception:
                ln, rn = futures[fut]
                print(f"[LOG-FATAL] log={ln} run={rn}\n{traceback.format_exc()}",
                      flush=True)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained seq-graph models on the flipped test set."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(MODEL_CONFIGS.keys()),
        help="Which model to evaluate",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of jobs to run in parallel (default: 1)",
    )
    parser.add_argument(
        "--progress-file",
        default=None,
        help="Override the progress log file path",
    )
    args = parser.parse_args()
    run_all(
        model=args.model,
        workers=args.workers,
        progress_file=args.progress_file,
    )
