"""
Multi-window expected counts for uncertain logs (single realization enumeration pass).

Goal
----
If the bottleneck is generating deterministic trace realizations, we should not
re-enumerate realizations separately for each (window_size, method) combination.

This module enumerates each uncertain trace realization ONCE and, for that realized
trace, updates expected counts for multiple window sizes and both context
interpretations (sequential and multiset) in the same pass.

Output counts are later reused to build:
- AC (activity-context) matrices for each (window_size, context_kind)
- AA (activity-activity) matrices derived from AC counts for each (window_size, context_kind)
and then PMI/PPMI post-processing is applied as usual.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from math import ceil, floor
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from distances.uncertain_activity_distances.data_util.uncertain_expected_counts import (
    PAD_TOKEN,
    DEFAULT_NA_LABEL,
    Context,
    ContextSeq,
    ProgressFn,
    _estimate_upper_bound_realizations,
    iter_trace_realizations,
    prune_distribution,
)
from uncertain_utils.uncertain_xes_reader import UncertainEventLog


CountsByWindow = Dict[int, Dict[str, Dict[Context, float]]]


def _merge_counts_by_window(target: CountsByWindow, src: CountsByWindow) -> None:
    for w, act_map in src.items():
        t_act_map = target.setdefault(w, {})
        for a, ctx_map in act_map.items():
            t_ctx_map = t_act_map.setdefault(a, {})
            for c, v in ctx_map.items():
                t_ctx_map[c] = t_ctx_map.get(c, 0.0) + float(v)


def _merge_counts(target: Dict[str, float], src: Dict[str, float]) -> None:
    for k, v in src.items():
        target[k] = target.get(k, 0.0) + float(v)


def _worker_multiwindow_for_trace(args):
    (
        trace_event_probs,
        window_sizes,
        top_k,
        min_prob,
        pad_token,
        na_label,
        max_realizations_per_trace,
        exclude,
    ) = args

    # Prepare per-event options (possibly pruned)
    event_opts: List[List[Tuple[str, float]]] = []
    for d in trace_event_probs:
        opts = prune_distribution(d, top_k=top_k, min_prob=min_prob)
        if not opts:
            return {}, {}, {}
        event_opts.append(opts)

    # Precompute padding per window size
    meta = {}
    for w in window_sizes:
        mid = w // 2
        pad_left = floor((w - 1) / 2)
        pad_right = ceil((w - 1) / 2)
        meta[w] = (mid, pad_left, pad_right)

    counts_seq: CountsByWindow = {w: defaultdict(lambda: defaultdict(float)) for w in window_sizes}  # type: ignore[assignment]
    counts_mset: CountsByWindow = {w: defaultdict(lambda: defaultdict(float)) for w in window_sizes}  # type: ignore[assignment]
    activity_freq: Dict[str, float] = defaultdict(float)

    for seq, p_tr in iter_trace_realizations(event_opts, na_label=na_label, max_realizations=max_realizations_per_trace):
        if not seq:
            continue

        # expected #(a)
        for a in seq:
            if a in exclude:
                continue
            activity_freq[a] += p_tr

        # contexts for each window size
        for w in window_sizes:
            mid, pad_left, pad_right = meta[w]
            padded = [pad_token] * pad_left + list(seq) + [pad_token] * pad_right
            for i in range(len(seq)):
                window = padded[i : i + w]
                center = window[mid]
                if center in exclude:
                    continue
                left = window[:mid]
                right = window[mid + 1 :]
                seq_ctx: ContextSeq = tuple(left + right)

                counts_seq[w][center][seq_ctx] += p_tr
                mset_ctx: Context = frozenset(Counter(seq_ctx).items())
                counts_mset[w][center][mset_ctx] += p_tr

    # convert nested defaultdicts to dicts
    counts_seq_out: CountsByWindow = {w: {a: dict(cmap) for a, cmap in act.items()} for w, act in counts_seq.items()}
    counts_mset_out: CountsByWindow = {w: {a: dict(cmap) for a, cmap in act.items()} for w, act in counts_mset.items()}
    return counts_seq_out, counts_mset_out, dict(activity_freq)


def compute_multiwindow_expected_counts(
    log: UncertainEventLog,
    *,
    window_sizes: Sequence[int],
    top_k: Optional[int] = None,
    min_prob: float = 0.0,
    pad_token: str = PAD_TOKEN,
    na_label: str = DEFAULT_NA_LABEL,
    max_realizations_per_trace: Optional[int] = None,
    exclude_activities: Optional[set[str]] = None,
    progress: Optional[ProgressFn] = None,
    n_jobs: int = 1,
    mp_start_method: str = "fork",
) -> Tuple[CountsByWindow, CountsByWindow, Dict[str, float]]:
    """
    Returns
    -------
    counts_seq_by_w:
        w -> activity -> seq_context(tuple) -> expected count
    counts_mset_by_w:
        w -> activity -> mset_context(frozenset(Counter(...).items())) -> expected count
    activity_freq:
        activity -> expected count (# occurrences)
    """
    ws = list(window_sizes)
    if not ws or any(w < 1 or w % 2 == 0 for w in ws):
        raise ValueError("window_sizes must be odd positive integers, e.g. [3,5,9].")

    exclude = exclude_activities or set()
    exclude.add(pad_token)

    if n_jobs is not None and n_jobs > 1:
        import multiprocessing as mp

        ctx = mp.get_context(mp_start_method)
        tasks = []
        for tr in log.traces:
            trace_event_probs = [ev.activity_probs for ev in tr.events]
            tasks.append((trace_event_probs, ws, top_k, min_prob, pad_token, na_label, max_realizations_per_trace, exclude))

        if progress is not None:
            progress(f"[uncertain-mw-mp] starting pool: n_jobs={n_jobs} start_method={mp_start_method} traces={len(tasks)}")

        global_seq: CountsByWindow = {}
        global_mset: CountsByWindow = {}
        global_freq: Dict[str, float] = {}

        done = 0
        with ctx.Pool(processes=n_jobs) as pool:
            for cseq, cmset, af in pool.imap_unordered(_worker_multiwindow_for_trace, tasks, chunksize=1):
                _merge_counts_by_window(global_seq, cseq)
                _merge_counts_by_window(global_mset, cmset)
                _merge_counts(global_freq, af)
                done += 1
                if progress is not None and done % 1 == 0:
                    progress(f"[uncertain-mw-mp] completed {done}/{len(tasks)} traces")
        return global_seq, global_mset, global_freq

    # single-process
    global_seq: CountsByWindow = {}
    global_mset: CountsByWindow = {}
    global_freq: Dict[str, float] = {}

    for trace_idx, tr in enumerate(log.traces, start=1):
        trace_event_probs = [ev.activity_probs for ev in tr.events]
        event_opts = [prune_distribution(d, top_k=top_k, min_prob=min_prob) for d in trace_event_probs]
        if any(not opts for opts in event_opts):
            continue

        if progress is not None:
            ub = _estimate_upper_bound_realizations(event_opts)
            progress(f"[uncertain-mw] trace {trace_idx}/{len(log.traces)} events={len(tr.events)} ub={ub} cap={max_realizations_per_trace}")

        cseq, cmset, af = _worker_multiwindow_for_trace(
            (trace_event_probs, ws, top_k, min_prob, pad_token, na_label, max_realizations_per_trace, exclude)
        )
        _merge_counts_by_window(global_seq, cseq)
        _merge_counts_by_window(global_mset, cmset)
        _merge_counts(global_freq, af)

    return global_seq, global_mset, global_freq



