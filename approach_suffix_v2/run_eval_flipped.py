#!/usr/bin/env python
"""Evaluate trained graph / seq-graph models on the prefix-flipped test set.

The prefix-flipped test set is built by performing the prefix/suffix split
first (identical to the normal run) and only then reversing the event order
within each concurrent block of the already-cut prefix (see
dataframes_pipeline._flip_prefix_blocks), so labels are identical to the
normal test set.

For each run_X folder found under the model's results directory, loads the
trained checkpoint for every event log that has one and runs inference on
the model's prefix-flipped test-set file (see MODEL_CONFIGS).  Results are
written to <results_subdir>/run_X/results_suffix_time_gnn_prefixflip.csv.

Skips logs where the result already exists (unless --force) or where no
checkpoint is found.
Safe to re-run after an interruption.

Supported models
----------------
  suffix_time_v1_seq_prefixflip  →  results_time_gatv2_seq_gru_nb_v1/   (test_seqgraphdataset_prefixflip.pt)
  suffix_time_v1_prefixflip      →  results_time_gatv2_gru_nb_v1/       (test_graphdataset_prefixflip.pt)

Usage
-----
    python run_eval_flipped.py --model suffix_time_v1_prefixflip
    python run_eval_flipped.py --model suffix_time_v1_seq_prefixflip --workers 4
    python run_eval_flipped.py --model suffix_time_v1_prefixflip --run-ids 1 2 3   # only these run_X folders

Arguments
-----
   --model {suffix_time_v1_seq_prefixflip,suffix_time_v1_prefixflip}
                                   which model to evaluate (required)
   --run-ids N [N ...]            only evaluate these run_X folders (default: all found)
   --workers N                    jobs to run in parallel
   --progress-file PATH           override the progress log path
   --force                        re-run the flip eval even if the result exists

Docker
------
   docker run -it --rm -v $(pwd):/workspace --gpus all ml-jupyter-gpu python approach_suffix_v2/run_eval_flipped.py --workers 2 --model suffix_time_v1_seq_prefixflip --run-ids 1

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
    'suffix_time_v1_seq_prefixflip': {
        'module':          'approach_suffix_v2.models_v2.run_suffix_time_v1_seq',
        'results_subdir':  'results_time_gatv2_seq_gru_nb_v1',
        'method_name':     'gatv2_seq_gru_nb_v1',
        'flip_data_file':  'test_seqgraphdataset_prefixflip.pt',
        'eval_func_name':  'run_eval_prefix_flip',
        'result_csv_file': 'results_suffix_time_gnn_prefixflip.csv',
    },
    'suffix_time_v1_prefixflip': {
        'module':          'approach_suffix_v2.models_v2.run_suffix_time_v1',
        'results_subdir':  'results_time_gatv2_gru_nb_v1',
        'method_name':     'gatv2_gru_nb_v1',
        'flip_data_file':  'test_graphdataset_prefixflip.pt',
        'eval_func_name':  'run_eval_prefix_flip',
        'result_csv_file': 'results_suffix_time_gnn_prefixflip.csv',
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
    csv_path = os.path.join(run_dir, MODEL_CONFIGS[model]['result_csv_file'])
    if not os.path.isfile(csv_path):
        return False
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            if row.get('log') == log_name:
                return True
    return False


def flip_data_exists(log_name, model):
    path = os.path.join(_HERE, 'results_per_log', log_name, MODEL_CONFIGS[model]['flip_data_file'])
    return os.path.isfile(path)


# ─────────────────────────────────────────────
# Subprocess
# ─────────────────────────────────────────────

def _build_eval_code(log_name, results_dir, model, use_cpu=False):
    module      = MODEL_CONFIGS[model]['module']
    func_name   = MODEL_CONFIGS[model]['eval_func_name']
    cpu_env     = "import os; os.environ['CUDA_VISIBLE_DEVICES'] = ''\n" if use_cpu else ""
    oom_guard   = _OOM_GUARD_CPU if use_cpu else _OOM_GUARD
    call_args   = f"log_name={log_name!r}, results_dir={results_dir!r}"
    return (
        _PREAMBLE
        + cpu_env
        + f"from {module} import {func_name}\n"
        + f"try:\n"
        + f"    {func_name}({call_args})\n"
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

def run_all(model, workers=1, progress_file=None, force=False, run_ids=None):
    if model not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model {model!r}. Choose from: {list(MODEL_CONFIGS)}")

    run_names = _find_run_dirs(model)
    if run_ids:
        wanted = {f"run_{i}" for i in run_ids}
        run_names = [rn for rn in run_names if rn in wanted]
    if not run_names:
        print(f"No run_X folders found under "
              f"'{MODEL_CONFIGS[model]['results_subdir']}'"
              + (f" matching --run-ids {list(run_ids)}" if run_ids else "")
              + ". Nothing to do.")
        return

    progress_file = progress_file or os.path.join(
        _HERE, f"run_eval_flip_{model}.log")

    print(f"Model      : {model}")
    print(f"Runs found : {run_names}")
    print(f"Force      : {force}")
    print(f"Workers    : {workers}")
    print(f"Progress   : {progress_file}\n", flush=True)

    jobs = []
    for run_name in run_names:
        run_dir = _run_dir_path(model, run_name)
        for log_name in EVENT_LOGS:
            if not flip_data_exists(log_name, model):
                print(f"[SKIP-NO-FLIP-DATA] log={log_name}  "
                      f"({MODEL_CONFIGS[model]['flip_data_file']} missing)", flush=True)
                continue
            if not checkpoint_exists(log_name, model, run_dir):
                print(f"[SKIP-NO-MODEL] log={log_name}  run={run_name}", flush=True)
                continue
            if not force and flip_result_exists(log_name, model, run_dir):
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
        description="Evaluate trained graph / seq-graph models on the prefix-flipped test set."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(MODEL_CONFIGS.keys()),
        help="Which model to evaluate",
    )
    parser.add_argument(
        "--run-ids",
        type=int,
        nargs="+",
        default=None,
        help="Only evaluate these run_X folders (e.g. --run-ids 1 2 3). "
             "Default: every run_X folder found.",
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
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run the flip evaluation even if the result already exists "
             "(bypasses the flip_result_exists skip). Overwrites the existing "
             "row in results_suffix_time_gnn_prefixflip.csv.",
    )
    args = parser.parse_args()
    run_all(
        model=args.model,
        workers=args.workers,
        progress_file=args.progress_file,
        force=args.force,
        run_ids=args.run_ids,
    )
