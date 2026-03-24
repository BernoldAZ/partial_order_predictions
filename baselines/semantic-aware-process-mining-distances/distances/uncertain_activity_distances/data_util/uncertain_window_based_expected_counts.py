"""
Window-based expected counts for uncertain logs (avoid full trace realization enumeration).

Key idea
--------
Old (trace-based) method:
  enumerate all deterministic *trace* realizations (exponential), then slide windows.

New (window-based) method (this module):
  for each center position, enumerate all possible *window* realizations and their probabilities
  directly, without building full traces.

NA semantics
-----------
We treat `NA` as "event absent". When building a window around a center:
- If the center resolves to NA, we ignore that realization (no window).
- For left/right context, NA events are skipped and we extend outward until we collected
  the required number of non-NA activities.

This matches the intuition: if you removed NA events from the realized trace and then took
the fixed-size window in the realized trace, you'd get the same contexts.

Practical note
--------------
This still enumerates combinations, but only for the *window neighborhood* rather than
the full trace, which is typically far smaller.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import ceil, floor
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from uncertain_utils.uncertain_xes_reader import UncertainEventLog

PAD_TOKEN = "."
DEFAULT_NA_LABEL = "NA"

ContextSeq = Tuple[str, ...]
ContextMSet = frozenset
Context = ContextSeq | ContextMSet


def _pad_trace(
    trace_event_probs: List[Dict[str, float]],
    *,
    pad_left: int,
    pad_right: int,
    pad_token: str = PAD_TOKEN,
) -> List[Dict[str, float]]:
    pad_dist = {pad_token: 1.0}
    return [pad_dist] * pad_left + trace_event_probs + [pad_dist] * pad_right


def _options(
    dist: Dict[str, float],
    *,
    prob_threshold: float = 0.0,
    renormalize: bool = True,
    label_map: Optional[Dict[str, str]] = None,
) -> List[Tuple[str, float]]:
    """
    Convert an event distribution dict into a sorted option list, optionally thresholded.

    - If `prob_threshold > 0`, drop labels with p < prob_threshold.
    - If `renormalize` is True, renormalize remaining probabilities to sum to 1.
    - If thresholding removes everything, fall back to keeping the original argmax with prob 1.0.
    """
    if prob_threshold < 0.0 or prob_threshold >= 1.0:
        raise ValueError("prob_threshold must be in [0.0, 1.0). Use 0.05 for 5%.")

    items = []
    for a, p in dist.items():
        if p is None:
            continue
        pf = float(p)
        if pf <= 0.0 or pf < prob_threshold:
            continue
        aa = str(a)
        if label_map is not None:
            aa = label_map.get(aa, aa)
        items.append((aa, pf))
    if not items:
        if not dist:
            return []
        a_max = max(dist.items(), key=lambda kv: float(kv[1]))[0]
        a_max = str(a_max)
        if label_map is not None:
            a_max = label_map.get(a_max, a_max)
        return [(a_max, 1.0)]

    if renormalize:
        s = sum(p for _, p in items)
        if s > 0:
            items = [(a, p / s) for a, p in items]

    items.sort(key=lambda kv: kv[1], reverse=True)
    return items


@dataclass(frozen=True)
class _SideResult:
    """One side (left or right) context realization and its probability."""
    seq: Tuple[str, ...]
    prob: float


def _enumerate_side_contexts(
    padded_opts: Sequence[List[Tuple[str, float]]],
    *,
    needed: int,
    na_label: str,
    direction: str,  # "left" or "right"
) -> Dict[Tuple[str, ...], float]:
    """
    Enumerate distribution over sequences of length `needed` obtained by scanning outward.

    `padded_opts` is the list of per-position options starting adjacent to center and going outward.

    For each position we either:
    - pick NA (skip, does not contribute to collected activities)
    - pick a non-NA label (contributes as next activity in the collected sequence)

    We stop collecting once we have `needed` activities; remaining outward events are irrelevant.
    """
    if needed == 0:
        return {tuple(): 1.0}

    # Dynamic programming over outward offsets from the center.
    #
    # `frontier_mass`:
    #   maps a *partially collected* context sequence (length < needed)
    #   to the probability mass of all ways to obtain it after processing
    #   the outward events seen so far.
    #
    # `completed_mass`:
    #   maps a *completed* context sequence (length == needed)
    #   to the probability mass of all ways to obtain it. Once completed,
    #   it is kept here and does not need further expansion.
    frontier_mass: Dict[Tuple[str, ...], float] = {tuple(): 1.0}
    completed_mass: Dict[Tuple[str, ...], float] = defaultdict(float)

    # Iterate outward events one by one. Each `event_options` is the list of (label, prob)
    # for that outward event.
    for event_options in padded_opts:
        new_dp: Dict[Tuple[str, ...], float] = defaultdict(float)
        for seq, mass in frontier_mass.items():
            # already collected enough: carry to done
            if len(seq) >= needed:
                completed_mass[seq] += mass
                continue

            for label, p in event_options:
                if p <= 0.0:
                    continue
                if label == na_label:
                    new_dp[seq] += mass * p
                else:
                    if direction == "left":
                        new_seq = (label,) + seq
                    else:
                        new_seq = seq + (label,)
                    if len(new_seq) >= needed:
                        completed_mass[new_seq] += mass * p
                    else:
                        new_dp[new_seq] += mass * p
        frontier_mass = dict(new_dp)
        if not frontier_mass:
            break

        # If all mass is already in done (no active partials), we can stop early.
        # (Note: if `frontier_mass` is empty, we already broke above.)

    # Merge any leftover dp that already reached needed (or didn’t due to too short padding)
    for seq, mass in frontier_mass.items():
        if len(seq) >= needed:
            completed_mass[seq] += mass

    # In properly padded traces, we should always be able to fill `needed` using PAD tokens.
    return dict(completed_mass)


def compute_expected_counts_window_based(
    log: UncertainEventLog,
    *,
    window_size: int,
    context_kind: str,  # "seq" | "mset"
    na_label: str = DEFAULT_NA_LABEL,
    pad_token: str = PAD_TOKEN,
    exclude_activities: Optional[set[str]] = None,
    progress: Optional[callable] = None,
) -> Tuple[Dict[str, Dict[Context, float]], Dict[str, float]]:
    """
    Compute expected #(a,c) and expected #(a) using window-based enumeration.
    """
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be odd (3,5,9,...)")
    if context_kind not in ("seq", "mset"):
        raise ValueError("context_kind must be 'seq' or 'mset'")

    exclude = exclude_activities or set()
    exclude.add(pad_token)

    k = window_size // 2  # needed on each side
    pad_left = k
    pad_right = k

    expected_counts: Dict[str, Dict[Context, float]] = defaultdict(lambda: defaultdict(float))
    activity_freq: Dict[str, float] = defaultdict(float)

    for trace_idx, tr in enumerate(log.traces, start=1):
        if progress is not None:
            progress(f"[uncertain-window] trace {trace_idx}/{len(log.traces)} events={len(tr.events)}")

        trace_probs = [ev.activity_probs for ev in tr.events]
        padded = _pad_trace(trace_probs, pad_left=pad_left, pad_right=pad_right, pad_token=pad_token)
        padded_opts = [_options(d) for d in padded]

        # iterate over centers in the *original* trace positions (offset by pad_left in padded list)
        for center_idx in range(pad_left, pad_left + len(tr.events)):
            center_opts = padded_opts[center_idx]

            # Prebuild outward option lists (adjacent outward)
            left_outward = [padded_opts[center_idx - i] for i in range(1, center_idx + 1)]
            right_outward = [padded_opts[center_idx + i] for i in range(1, len(padded_opts) - center_idx)]

            # distributions over left/right sequences (length k) accounting for NA-skips
            left_dist = _enumerate_side_contexts(left_outward, needed=k, na_label=na_label, direction="left")
            right_dist = _enumerate_side_contexts(right_outward, needed=k, na_label=na_label, direction="right")

            for a, p_a in center_opts:
                if p_a <= 0.0:
                    continue
                if a == na_label or a in exclude:
                    continue

                # expected #(a): probability center is activity a
                activity_freq[a] += p_a

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
                        expected_counts[a][ctx] += p

    return {a: dict(cmap) for a, cmap in expected_counts.items()}, dict(activity_freq)


