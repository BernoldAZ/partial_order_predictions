"""
Top-K most probable deterministic trace realizations for an uncertain trace.

We assume an uncertain trace is represented by per-event independent categorical distributions.
To avoid enumerating the full cartesian product, we enumerate only the K most probable
realizations using A* (best-first search) with an admissible upper-bound heuristic.

NA semantics
-----------
If a chosen label equals `na_label`, the event is treated as "did not happen" and is
omitted from the realized trace sequence (variable-length realized traces).

Output
------
Yields (realized_sequence, probability) in non-increasing probability order.
Also supports computing the cumulative probability mass covered by the first K.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class _Node:
    pos: int
    logp: float
    parent: Optional["_Node"]
    label: Optional[str]  # None means NA-skip for this position


def _reconstruct_sequence(node: _Node) -> List[str]:
    out: List[str] = []
    cur = node
    while cur.parent is not None:
        if cur.label is not None:
            out.append(cur.label)
        cur = cur.parent
    out.reverse()
    return out


def iter_topk_trace_realizations(
    event_options: Sequence[List[Tuple[str, float]]],
    *,
    k: int,
    na_label: str = "NA",
) -> Iterator[Tuple[List[str], float]]:
    """
    Enumerate up to K most probable realizations for one uncertain trace.

    Parameters
    ----------
    event_options:
        List of per-position options, each a list of (label, prob). Probabilities should sum to 1,
        but we do not require exact normalization.
    k:
        Max number of realizations to yield.
    """
    if k <= 0:
        return
        yield  # pragma: no cover

    m = len(event_options)
    if m == 0:
        return

    # Precompute best possible remaining log-prob mass from each position (admissible heuristic)
    best_log_per_pos: List[float] = []
    for opts in event_options:
        best = 0.0
        for _, p in opts:
            if p is None:
                continue
            pf = float(p)
            if pf <= 0.0:
                continue
            if pf > best:
                best = pf
        if best <= 0.0:
            # No valid continuation; this trace has no realizations.
            return
        best_log_per_pos.append(math.log(best))

    suffix_best: List[float] = [0.0] * (m + 1)
    for i in range(m - 1, -1, -1):
        suffix_best[i] = suffix_best[i + 1] + best_log_per_pos[i]

    # Max-heap on bound = logp(prefix) + suffix_best[pos]
    # Python heapq is min-heap, so we store (-bound, node_id, node).
    heap: List[Tuple[float, int, _Node]] = []
    uid = 0

    root = _Node(pos=0, logp=0.0, parent=None, label=None)
    heapq.heappush(heap, (-(0.0 + suffix_best[0]), uid, root))
    uid += 1

    yielded = 0
    while heap and yielded < k:
        _, _, node = heapq.heappop(heap)
        if node.pos == m:
            # Complete realization
            seq = _reconstruct_sequence(node)
            p = math.exp(node.logp)
            yielded += 1
            yield seq, p
            continue

        opts = event_options[node.pos]
        next_pos = node.pos + 1
        for label, prob in opts:
            pf = float(prob)
            if pf <= 0.0:
                continue
            child_logp = node.logp + math.log(pf)
            child_label = None if label == na_label else str(label)
            child = _Node(pos=next_pos, logp=child_logp, parent=node, label=child_label)
            bound = child_logp + suffix_best[next_pos]
            heapq.heappush(heap, (-bound, uid, child))
            uid += 1


def topk_probability_mass(
    event_options: Sequence[List[Tuple[str, float]]],
    *,
    k: int,
    na_label: str = "NA",
) -> Tuple[int, float]:
    """
    Return (num_generated, cumulative_probability_mass) for top-K realizations.
    """
    s = 0.0
    n = 0
    for _, p in iter_topk_trace_realizations(event_options, k=k, na_label=na_label):
        s += p
        n += 1
    return n, s


