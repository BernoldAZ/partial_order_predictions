"""
Multi-window window-based expected counts (no full trace realization enumeration).

This is the window-based analogue of `uncertain_multiwindow_expected_counts.py`:
we avoid building full deterministic traces and instead enumerate all possible windows
around each center, with NA-skip semantics.

We compute counts for multiple window sizes in one run over the log to minimize overhead.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from math import ceil, floor
from typing import Dict, List, Optional, Sequence, Tuple

from uncertain_utils.uncertain_xes_reader import UncertainEventLog
from distances.uncertain_activity_distances.data_util.uncertain_window_based_expected_counts import (
    PAD_TOKEN,
    DEFAULT_NA_LABEL,
    _enumerate_side_contexts,
    _options,
    _pad_trace,
    Context,
)


CountsByWindow = Dict[int, Dict[str, Dict[Context, float]]]
ContextFreqByWindow = Dict[int, Dict[Context, float]]


def compute_expected_counts_window_based_multiwindow(
    log: UncertainEventLog,
    *,
    window_sizes: Sequence[int],
    context_kind: str,  # "seq" | "mset"
    na_label: str = DEFAULT_NA_LABEL,
    pad_token: str = PAD_TOKEN,
    prob_threshold: float = 0.0,
    exclude_activities: Optional[set[str]] = None,
    progress: Optional[callable] = None,
    label_map: Optional[Dict[str, str]] = None,
) -> Tuple[CountsByWindow, ContextFreqByWindow, Dict[str, float]]:
    """
    Returns
    -------
    expected_counts_by_w:
        w -> activity -> context -> expected count
    activity_freq:
        expected #(a) independent of window size
    """
    ws = list(window_sizes)
    if not ws or any(w < 1 or w % 2 == 0 for w in ws):
        raise ValueError("window_sizes must be odd positive integers, e.g. [3,5,9].")
    if context_kind not in ("seq", "mset"):
        raise ValueError("context_kind must be 'seq' or 'mset'")

    exclude = exclude_activities or set()
    exclude.add(pad_token)

    # We can pad with max k so all windows have enough room
    max_k = max(w // 2 for w in ws)
    pad_left = max_k
    pad_right = max_k

    expected_counts_by_w: CountsByWindow = {w: defaultdict(lambda: defaultdict(float)) for w in ws}  # type: ignore[assignment]
    context_freq_by_w: ContextFreqByWindow = {w: defaultdict(float) for w in ws}  # type: ignore[assignment]
    activity_freq: Dict[str, float] = defaultdict(float)

    # Precompute needed lengths per window size
    needed_by_w = {w: w // 2 for w in ws}

    for trace_idx, tr in enumerate(log.traces, start=1):
        if progress is not None:
            progress(f"[uncertain-window-mw] trace {trace_idx}/{len(log.traces)} events={len(tr.events)}")

        trace_probs = [ev.activity_probs for ev in tr.events]
        padded = _pad_trace(trace_probs, pad_left=pad_left, pad_right=pad_right, pad_token=pad_token)
        padded_opts = [_options(d, prob_threshold=prob_threshold, renormalize=True, label_map=label_map) for d in padded]

        for center_idx in range(pad_left, pad_left + len(tr.events)):
            center_opts = padded_opts[center_idx]

            left_outward = [padded_opts[center_idx - i] for i in range(1, center_idx + 1)]
            right_outward = [padded_opts[center_idx + i] for i in range(1, len(padded_opts) - center_idx)]

            # For each window size compute left/right context distributions
            left_dists = {w: _enumerate_side_contexts(left_outward, needed=needed_by_w[w], na_label=na_label, direction="left") for w in ws}
            right_dists = {w: _enumerate_side_contexts(right_outward, needed=needed_by_w[w], na_label=na_label, direction="right") for w in ws}

            for a, p_a in center_opts:
                if p_a <= 0.0:
                    continue
                if a == na_label or a in exclude:
                    continue

                activity_freq[a] += p_a

                for w in ws:
                    left_dist = left_dists[w]
                    right_dist = right_dists[w]
                    for l_seq, p_l in left_dist.items():
                        for r_seq, p_r in right_dist.items():
                            p = p_a * p_l * p_r
                            if p <= 0.0:
                                continue
                            seq_ctx = tuple(l_seq + r_seq)
                            if context_kind == "seq":
                                ctx: Context = seq_ctx
                            else:
                                ctx = frozenset(Counter(seq_ctx).items())
                            expected_counts_by_w[w][a][ctx] += p
                            context_freq_by_w[w][ctx] += p

    # Convert nested defaultdicts to dicts for stability
    out: CountsByWindow = {w: {a: dict(cmap) for a, cmap in act.items()} for w, act in expected_counts_by_w.items()}
    out_cf: ContextFreqByWindow = {w: dict(cf) for w, cf in context_freq_by_w.items()}
    return out, out_cf, dict(activity_freq)


