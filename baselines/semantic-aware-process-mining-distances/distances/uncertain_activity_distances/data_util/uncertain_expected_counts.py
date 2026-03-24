"""
Expected-count extraction for uncertain event logs.

This file implements the core idea:
replace deterministic integer counts with *expected counts* computed from
per-event probability distributions.

For the count-based methods in the journal extension, we follow the formalisation:
for each uncertain trace, enumerate all deterministic trace realizations and weight
their contributions by the realization probability.

Important: We support an "absence" label (typically "NA"). If an event is realized
as NA, then the event does *not* occur in that deterministic trace, i.e., it is
removed and does not contribute to contexts. This implies variable-length realized
traces, which is why we enumerate trace realizations (rather than using fixed-position
windows).

Because real-world distributions can be dense (many activities with tiny
probabilities), the helper `prune_distribution()` supports top-k / min-prob pruning.
If you set `top_k=None` and `min_prob=0.0`, you will enumerate the full support of
each event distribution (which can be extremely expensive).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from math import ceil, floor
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

from uncertain_utils.uncertain_xes_reader import UncertainEventLog


PAD_TOKEN = "."  # keep consistent with deterministic code (see distances/activity_distances/data_util/algorithm.py)
DEFAULT_NA_LABEL = "NA"

ContextSeq = Tuple[str, ...]
ContextMSet = frozenset  # frozenset(Counter(context).items())  (hashable multiset)
Context = Union[ContextSeq, ContextMSet]

ProgressFn = Callable[[str], None]

def _merge_nested_counts(
    target: Dict[str, Dict[Context, float]],
    src: Dict[str, Dict[Context, float]],
) -> None:
    for a, cmap in src.items():
        t = target.setdefault(a, {})
        for c, v in cmap.items():
            t[c] = t.get(c, 0.0) + float(v)


def _merge_counts(target: Dict[str, float], src: Dict[str, float]) -> None:
    for k, v in src.items():
        target[k] = target.get(k, 0.0) + float(v)


def _worker_expected_counts_for_trace(args):
    """
    Worker for multiprocessing: compute expected #(a,c) and #(a) for a single trace.

    Parameters passed in args to keep it picklable.
    """
    (
        trace_event_probs,
        ngram_size,
        context_kind,
        top_k,
        min_prob,
        pad_token,
        na_label,
        max_realizations_per_trace,
        exclude,
    ) = args

    mid = ngram_size // 2
    pad_left = floor((ngram_size - 1) / 2)
    pad_right = ceil((ngram_size - 1) / 2)

    # Build per-event options
    event_opts: List[List[Tuple[str, float]]] = []
    for d in trace_event_probs:
        opts = prune_distribution(d, top_k=top_k, min_prob=min_prob)
        if not opts:
            return {}, {}
        event_opts.append(opts)

    context_dict: Dict[str, Dict[Context, float]] = defaultdict(lambda: defaultdict(float))
    activity_freq: Dict[str, float] = defaultdict(float)

    for seq, p_tr in iter_trace_realizations(event_opts, na_label=na_label, max_realizations=max_realizations_per_trace):
        if not seq:
            continue

        for a in seq:
            if a in exclude:
                continue
            activity_freq[a] += p_tr

        padded_seq = [pad_token] * pad_left + list(seq) + [pad_token] * pad_right
        for i in range(len(seq)):
            window = padded_seq[i : i + ngram_size]
            center = window[mid]
            if center in exclude:
                continue

            left = window[:mid]
            right = window[mid + 1 :]
            seq_context: ContextSeq = tuple(left + right)
            if context_kind == "seq":
                ctx: Context = seq_context
            else:
                ctx = frozenset(Counter(seq_context).items())

            context_dict[center][ctx] += p_tr

    return {a: dict(cmap) for a, cmap in context_dict.items()}, dict(activity_freq)


def _estimate_upper_bound_realizations(event_distributions: Sequence[List[Tuple[str, float]]]) -> int:
    """
    Upper bound on the number of trace realizations as the cartesian-product size.

    Note: with NA-skip semantics, different label choices can collapse to the same
    realized sequence; we still use the product as a simple upper bound.
    """
    prod = 1
    for opts in event_distributions:
        prod *= max(1, len(opts))
    return prod


def prune_distribution(
    dist: Dict[str, float],
    *,
    top_k: Optional[int] = None,
    min_prob: float = 0.0,
) -> List[Tuple[str, float]]:
    """
    Turn a dict activity->prob into a list of (activity, prob), optionally pruned.
    """
    items = [(a, float(p)) for a, p in dist.items() if p is not None and float(p) > 0.0 and float(p) >= min_prob]
    if not items:
        return []

    items.sort(key=lambda kv: kv[1], reverse=True)
    if top_k is not None:
        items = items[:top_k]

    return items


def iter_trace_realizations(
    event_distributions: Sequence[List[Tuple[str, float]]],
    *,
    na_label: str = DEFAULT_NA_LABEL,
    max_realizations: Optional[int] = None,
) -> Iterator[Tuple[List[str], float]]:
    """
    Enumerate all deterministic realizations of an uncertain trace.

    Each input position is one uncertain event distribution (list of (activity, prob)).
    The realization probability is the product of chosen probabilities.

    NA semantics:
      If the chosen activity equals `na_label`, the event is treated as "did not happen"
      and is therefore omitted from the realized trace (variable-length realizations).
    """
    # Important performance note:
    # A naive cartesian-product materialization creates |A|^|trace| sequences in memory
    # before we can apply `max_realizations`. That is infeasible for long traces.
    #
    # We therefore use a depth-first generator that can stop early once enough
    # realizations have been produced.

    emitted = 0

    def dfs(i: int, seq: List[str], prob: float) -> Iterator[Tuple[List[str], float]]:
        nonlocal emitted
        if max_realizations is not None and emitted >= max_realizations:
            return
        if i >= len(event_distributions):
            if prob > 0.0:
                emitted += 1
                # Copy `seq` because we mutate it during backtracking
                yield (list(seq), prob)
            return

        options = event_distributions[i]
        for a, p in options:
            if p <= 0.0:
                continue
            new_prob = prob * p
            if new_prob <= 0.0:
                continue

            if a == na_label:
                yield from dfs(i + 1, seq, new_prob)
            else:
                seq.append(a)
                yield from dfs(i + 1, seq, new_prob)
                seq.pop()

            if max_realizations is not None and emitted >= max_realizations:
                return

    yield from dfs(0, [], 1.0)


def compute_expected_activity_frequencies(
    log: UncertainEventLog,
    *,
    top_k: Optional[int] = None,
    min_prob: float = 0.0,
    na_label: str = DEFAULT_NA_LABEL,
    max_realizations_per_trace: Optional[int] = None,
    exclude_activities: Optional[set[str]] = None,
    progress: Optional[ProgressFn] = None,
    progress_every_realizations: int = 50_000,
) -> Dict[str, float]:
    """
    Expected number of occurrences of each activity in the entire log.

    This is used by PMI/PPMI post-processing as p(a) = #(a) / N.
    """
    exclude = exclude_activities or set()
    freq: Dict[str, float] = defaultdict(float)
    for trace_idx, tr in enumerate(log.traces, start=1):
        event_opts: List[List[Tuple[str, float]]] = []
        for ev in tr.events:
            opts = prune_distribution(ev.activity_probs, top_k=top_k, min_prob=min_prob)
            if not opts:
                # no probability mass left for this event -> no realizations
                event_opts = []
                break
            event_opts.append(opts)

        if not event_opts:
            continue

        if progress is not None:
            ub = _estimate_upper_bound_realizations(event_opts)
            cap = max_realizations_per_trace
            progress(
                f"[uncertain] trace {trace_idx}/{len(log.traces)} events={len(tr.events)} "
                f"upper_bound_realizations={ub} cap={cap}"
            )

        realized = 0
        for seq, p_tr in iter_trace_realizations(event_opts, na_label=na_label, max_realizations=max_realizations_per_trace):
            if not seq:
                continue
            counts = Counter(seq)
            for a, c in counts.items():
                if a in exclude:
                    continue
                freq[a] += float(c) * p_tr
            realized += 1
            if progress is not None and realized % progress_every_realizations == 0:
                progress(f"[uncertain]  trace {trace_idx}: realized {realized} traces so far...")
    return dict(freq)

def compute_expected_context_counts_and_activity_frequencies(
    log: UncertainEventLog,
    *,
    ngram_size: int,
    context_kind: str,  # "seq" | "mset"
    top_k: Optional[int] = None,
    min_prob: float = 0.0,
    pad_token: str = PAD_TOKEN,
    na_label: str = DEFAULT_NA_LABEL,
    max_realizations_per_trace: Optional[int] = None,
    exclude_activities: Optional[set[str]] = None,
    progress: Optional[ProgressFn] = None,
    progress_every_realizations: int = 50_000,
    n_jobs: int = 1,
    mp_start_method: str = "spawn",
) -> Tuple[Dict[str, Dict[Context, float]], Dict[str, float]]:
    """
    Combined pass: compute expected #(a,c) and expected #(a) with one realization enumeration.

    This is a major runtime optimization because the AC/AA pipelines require both:
    - expected activity-context counts #(a,c)
    - expected activity frequencies #(a) (used by PMI/PPMI)

    If computed separately, we'd enumerate all deterministic trace realizations twice.
    """
    if ngram_size < 1 or ngram_size % 2 == 0:
        raise ValueError("ngram_size must be an odd positive integer (e.g., 3, 5, 9).")
    if context_kind not in ("seq", "mset"):
        raise ValueError("context_kind must be 'seq' or 'mset'.")

    exclude = exclude_activities or set()

    context_dict: Dict[str, Dict[Context, float]] = defaultdict(lambda: defaultdict(float))
    activity_freq: Dict[str, float] = defaultdict(float)

    mid = ngram_size // 2
    pad_left = floor((ngram_size - 1) / 2)
    pad_right = ceil((ngram_size - 1) / 2)

    if n_jobs is not None and n_jobs > 1:
        import multiprocessing as mp

        exclude = exclude_activities or set()
        ctx = mp.get_context(mp_start_method)
        tasks = []
        for tr in log.traces:
            trace_event_probs = [ev.activity_probs for ev in tr.events]
            tasks.append(
                (
                    trace_event_probs,
                    ngram_size,
                    context_kind,
                    top_k,
                    min_prob,
                    pad_token,
                    na_label,
                    max_realizations_per_trace,
                    exclude,
                )
            )

        total = len(tasks)
        global_context: Dict[str, Dict[Context, float]] = {}
        global_freq: Dict[str, float] = {}

        if progress is not None:
            progress(f"[uncertain-mp] starting pool: n_jobs={n_jobs} start_method={mp_start_method} traces={total}")

        done = 0
        with ctx.Pool(processes=n_jobs) as pool:
            for ctx_part, freq_part in pool.imap_unordered(_worker_expected_counts_for_trace, tasks, chunksize=1):
                _merge_nested_counts(global_context, ctx_part)
                _merge_counts(global_freq, freq_part)
                done += 1
                if progress is not None and (done % 1 == 0):
                    progress(f"[uncertain-mp] completed {done}/{total} traces")

        return global_context, global_freq

    for trace_idx, tr in enumerate(log.traces, start=1):
        # Build per-event option lists (possibly pruned)
        event_opts: List[List[Tuple[str, float]]] = []
        for ev in tr.events:
            opts = prune_distribution(ev.activity_probs, top_k=top_k, min_prob=min_prob)
            if not opts:
                event_opts = []
                break
            event_opts.append(opts)

        if not event_opts:
            continue

        if progress is not None:
            ub = _estimate_upper_bound_realizations(event_opts)
            cap = max_realizations_per_trace
            progress(
                f"[uncertain] trace {trace_idx}/{len(log.traces)} events={len(tr.events)} "
                f"upper_bound_realizations={ub} cap={cap}"
            )

        realized = 0
        for seq, p_tr in iter_trace_realizations(
            event_opts, na_label=na_label, max_realizations=max_realizations_per_trace
        ):
            if not seq:
                continue

            # Update expected activity frequencies #(a)
            for a in seq:
                if a in exclude:
                    continue
                activity_freq[a] += p_tr

            # Update expected activity-context counts #(a,c) via sliding windows on realized trace
            padded_seq = [pad_token] * pad_left + list(seq) + [pad_token] * pad_right
            for i in range(len(seq)):
                window = padded_seq[i : i + ngram_size]
                center = window[mid]
                if center in exclude:
                    continue

                left = window[:mid]
                right = window[mid + 1 :]
                seq_context: ContextSeq = tuple(left + right)
                if context_kind == "seq":
                    ctx: Context = seq_context
                else:
                    ctx = frozenset(Counter(seq_context).items())

                context_dict[center][ctx] += p_tr

            realized += 1
            if progress is not None and realized % progress_every_realizations == 0:
                progress(f"[uncertain]  trace {trace_idx}: realized {realized} traces so far...")

    return {a: dict(cmap) for a, cmap in context_dict.items()}, dict(activity_freq)


def compute_expected_context_counts(
    log: UncertainEventLog,
    *,
    ngram_size: int,
    context_kind: str,  # "seq" | "mset"
    top_k: Optional[int] = None,
    min_prob: float = 0.0,
    pad_token: str = PAD_TOKEN,
    na_label: str = DEFAULT_NA_LABEL,
    max_realizations_per_trace: Optional[int] = None,
    exclude_activities: Optional[set[str]] = None,
    progress: Optional[ProgressFn] = None,
    progress_every_realizations: int = 50_000,
) -> Dict[str, Dict[Context, float]]:
    """
    Compute the expected counts mapping #(a, c) for uncertain data.

    Returns
    -------
    context_dict:
        activity -> { context -> expected_count }
        Where `context` is:
        - tuple[str, ...] for sequential context ("seq")
        - frozenset(Counter(tuple).items()) for multiset context ("mset")
    """
    if ngram_size < 1 or ngram_size % 2 == 0:
        raise ValueError("ngram_size must be an odd positive integer (e.g., 3, 5, 9).")
    if context_kind not in ("seq", "mset"):
        raise ValueError("context_kind must be 'seq' or 'mset'.")

    exclude = exclude_activities or set()

    context_dict: Dict[str, Dict[Context, float]] = defaultdict(lambda: defaultdict(float))
    mid = ngram_size // 2
    pad_left = floor((ngram_size - 1) / 2)
    pad_right = ceil((ngram_size - 1) / 2)

    for trace_idx, tr in enumerate(log.traces, start=1):
        # Build pruned per-event distributions once
        event_opts: List[List[Tuple[str, float]]] = []
        for ev in tr.events:
            opts = prune_distribution(ev.activity_probs, top_k=top_k, min_prob=min_prob)
            if not opts:
                event_opts = []
                break
            event_opts.append(opts)

        if not event_opts:
            continue

        if progress is not None:
            ub = _estimate_upper_bound_realizations(event_opts)
            cap = max_realizations_per_trace
            progress(
                f"[uncertain] trace {trace_idx}/{len(log.traces)} events={len(tr.events)} "
                f"upper_bound_realizations={ub} cap={cap}"
            )

        # Enumerate deterministic trace realizations and count windows on realized trace
        realized = 0
        for seq, p_tr in iter_trace_realizations(event_opts, na_label=na_label, max_realizations=max_realizations_per_trace):
            if not seq:
                continue

            padded_seq = [pad_token] * pad_left + list(seq) + [pad_token] * pad_right
            # Iterate over positions of the *realized* trace (excluding padding-only)
            for i in range(len(seq)):
                window = padded_seq[i : i + ngram_size]
                center = window[mid]
                if center in exclude:
                    continue

                left = window[:mid]
                right = window[mid + 1 :]
                seq_context: ContextSeq = tuple(left + right)
                if context_kind == "seq":
                    ctx: Context = seq_context
                else:
                    ctx = frozenset(Counter(seq_context).items())

                context_dict[center][ctx] += p_tr
            realized += 1
            if progress is not None and realized % progress_every_realizations == 0:
                progress(f"[uncertain]  trace {trace_idx}: realized {realized} traces so far...")

    # Convert nested defaultdicts to dicts for stability / serialization
    return {a: dict(cmap) for a, cmap in context_dict.items()}


def compute_context_frequencies_from_expected_counts(
    expected_counts: Dict[str, Dict[Context, float]]
) -> Dict[Context, float]:
    """
    Compute #(c) = sum_a #(a,c) from the expected count map.
    """
    cf: Dict[Context, float] = defaultdict(float)
    for cmap in expected_counts.values():
        for c, v in cmap.items():
            cf[c] += float(v)
    return dict(cf)


