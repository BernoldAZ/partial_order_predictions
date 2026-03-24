"""
Multi-window expected counts using TOP-K most probable trace realizations per uncertain trace.

This is an alternative to full realization enumeration:
- For each uncertain trace, generate up to K most probable deterministic realizations
  (with A* search), and accumulate expected counts from those realizations only.
- Also report the cumulative probability mass covered by those K realizations.

This is useful when:
- exact enumeration is infeasible
- you want a principled truncation by probability mass
"""

from __future__ import annotations

from collections import Counter, defaultdict
from math import ceil, floor
from typing import Dict, List, Optional, Sequence, Tuple

from distances.uncertain_activity_distances.data_util.uncertain_expected_counts import (
    PAD_TOKEN,
    DEFAULT_NA_LABEL,
    Context,
    ContextSeq,
    ProgressFn,
    prune_distribution,
)
from distances.uncertain_activity_distances.data_util.uncertain_topk_trace_realizations import (
    iter_topk_trace_realizations,
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

def _threshold_and_renormalize_event_options(
    probs: Dict[str, float],
    *,
    prob_threshold: float,
) -> List[Tuple[str, float]]:
    """
    Keep only labels with p >= prob_threshold and renormalize to sum to 1.

    If threshold removes all labels, keep the original argmax label with prob 1.0.
    """
    items = [(str(a), float(p)) for a, p in probs.items() if p is not None and float(p) > 0.0 and float(p) >= prob_threshold]
    if not items:
        # fallback: keep argmax of original distribution
        if not probs:
            return []
        a_max = max(probs.items(), key=lambda kv: float(kv[1]))[0]
        return [(str(a_max), 1.0)]
    s = sum(p for _, p in items)
    if s > 0:
        items = [(a, p / s) for a, p in items]
    items.sort(key=lambda kv: kv[1], reverse=True)
    return items


def _worker_topk_for_trace(args):
    (
        trace_idx,
        trace_event_probs,
        window_sizes,
        top_k_traces,
        pad_token,
        na_label,
        exclude,
        prob_threshold,
    ) = args

    ws = list(window_sizes)
    meta = {}
    for w in ws:
        mid = w // 2
        pad_left = floor((w - 1) / 2)
        pad_right = ceil((w - 1) / 2)
        meta[w] = (mid, pad_left, pad_right)

    # Prepare per-event options (sorted by prob desc to help A*)
    event_opts: List[List[Tuple[str, float]]] = []
    for d in trace_event_probs:
        opts = _threshold_and_renormalize_event_options(d, prob_threshold=prob_threshold)
        if not opts:
            return {}, {}, 0, 0.0
        event_opts.append(opts)

    counts_seq: CountsByWindow = {w: defaultdict(lambda: defaultdict(float)) for w in ws}  # type: ignore[assignment]
    counts_mset: CountsByWindow = {w: defaultdict(lambda: defaultdict(float)) for w in ws}  # type: ignore[assignment]
    activity_freq: Dict[str, float] = defaultdict(float)

    cum_mass = 0.0
    generated = 0

    for seq, p_tr in iter_topk_trace_realizations(event_opts, k=top_k_traces, na_label=na_label):
        generated += 1
        cum_mass += p_tr
        if not seq:
            continue

        for a in seq:
            if a in exclude:
                continue
            activity_freq[a] += p_tr

        for w in ws:
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

    counts_seq_out: CountsByWindow = {w: {a: dict(cmap) for a, cmap in act.items()} for w, act in counts_seq.items()}
    counts_mset_out: CountsByWindow = {w: {a: dict(cmap) for a, cmap in act.items()} for w, act in counts_mset.items()}
    return trace_idx, counts_seq_out, counts_mset_out, dict(activity_freq), generated, cum_mass


def compute_multiwindow_expected_counts_topk(
    log: UncertainEventLog,
    *,
    window_sizes: Sequence[int],
    top_k_traces: int,
    prob_threshold: float = 0.0,
    pad_token: str = PAD_TOKEN,
    na_label: str = DEFAULT_NA_LABEL,
    exclude_activities: Optional[set[str]] = None,
    progress: Optional[ProgressFn] = None,
    n_jobs: int = 1,
    mp_start_method: str = "fork",
) -> Tuple[CountsByWindow, CountsByWindow, Dict[str, float], List[Tuple[int, int, float]]]:
    """
    Returns
    -------
    counts_seq_by_w, counts_mset_by_w, activity_freq
    per_trace_mass:
        list of tuples (trace_index_1based, generated, cumulative_mass)
    """
    ws = list(window_sizes)
    if not ws or any(w < 1 or w % 2 == 0 for w in ws):
        raise ValueError("window_sizes must be odd positive integers, e.g. [3,5,9].")
    if top_k_traces <= 0:
        raise ValueError("top_k_traces must be > 0")
    if prob_threshold < 0.0 or prob_threshold >= 1.0:
        raise ValueError("prob_threshold must be in [0.0, 1.0). Use 0.10 for 10%.")

    exclude = exclude_activities or set()
    exclude.add(pad_token)

    import multiprocessing as mp

    ctx = mp.get_context(mp_start_method)
    tasks = []
    for trace_idx, tr in enumerate(log.traces, start=1):
        trace_event_probs = [ev.activity_probs for ev in tr.events]
        tasks.append((trace_idx, trace_event_probs, ws, top_k_traces, pad_token, na_label, exclude, prob_threshold))

    if progress is not None:
        progress(
            f"[uncertain-topk-mp] starting pool: n_jobs={n_jobs} start_method={mp_start_method} "
            f"traces={len(tasks)} top_k_traces={top_k_traces} prob_threshold={prob_threshold}"
        )

    global_seq: CountsByWindow = {}
    global_mset: CountsByWindow = {}
    global_freq: Dict[str, float] = {}
    per_trace_mass: List[Tuple[int, int, float]] = []

    done = 0
    with ctx.Pool(processes=n_jobs) as pool:
        for trace_idx, cseq, cmset, af, generated, mass in pool.imap_unordered(_worker_topk_for_trace, tasks, chunksize=1):
            _merge_counts_by_window(global_seq, cseq)
            _merge_counts_by_window(global_mset, cmset)
            _merge_counts(global_freq, af)
            per_trace_mass.append((trace_idx, generated, mass))
            done += 1
            if progress is not None:
                progress(f"[uncertain-topk-mp] completed {done}/{len(tasks)} traces (trace={trace_idx} mass={mass:.6f})")

    return global_seq, global_mset, global_freq, per_trace_mass


