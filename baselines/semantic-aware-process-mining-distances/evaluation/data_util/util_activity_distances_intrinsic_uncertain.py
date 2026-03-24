"""
Utilities for the *uncertain* intrinsic evaluation benchmark.

Design goal
-----------
Mirror the structure and naming of the deterministic intrinsic evaluation utilities
in `evaluation/data_util/util_activity_distances_intrinsic.py`, but operate on
`UncertainEventLog` objects and support uncertainty-level truncation (top-k labels).

Key responsibilities
-------------------
- Create uncertain ground truth logs via trace-consistent activity replacement
- Create uncertainty levels (k=1..5) by truncating each event distribution to top-k
- Compute normalized Shannon entropy statistics per uncertainty level
- Save/load ground truth logs and per-method results (parallel directory structure)
"""

from __future__ import annotations

import math
import os
import pickle
import random
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from definitions import ROOT_DIR
from uncertain_utils.uncertain_xes_reader import UncertainEvent, UncertainEventLog, UncertainTrace


# ---------------------------------------------------------------------------
# Uncertainty levels + statistics
# ---------------------------------------------------------------------------


def topk_distribution(dist: Dict[str, float], *, k: int, include_na: bool = True, na_label: str = "NA") -> Dict[str, float]:
    """
    Keep the top-k labels by probability and renormalize.

    include_na=True means NA is treated like any other label (if it is among the top-k, it is kept).
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    items = [(str(a), float(p)) for a, p in dist.items() if p is not None and float(p) > 0.0]
    if not items:
        return {}
    items.sort(key=lambda kv: kv[1], reverse=True)
    kept = items[:k]
    out = {a: p for a, p in kept}
    s = sum(out.values())
    if s > 0:
        out = {a: p / s for a, p in out.items()}
    return out


def normalized_shannon_entropy(dist: Dict[str, float]) -> float:
    """
    Normalized Shannon entropy in [0,1] computed as:
      H(p) / log(|supp(p)|)

    - If |supp| <= 1, return 0.0.
    - Uses natural log (base cancels due to normalization).
    """
    probs = [float(p) for p in dist.values() if p is not None and float(p) > 0.0]
    m = len(probs)
    if m <= 1:
        return 0.0
    h = -sum(p * math.log(p) for p in probs)
    return float(h / math.log(m))


def uncertainty_stats(log: UncertainEventLog, *, na_label: str = "NA") -> Dict[str, float]:
    """
    Aggregate uncertainty statistics over all events.
    """
    ent = []
    maxp = []
    supp = []
    nap = []
    for tr in log.traces:
        for ev in tr.events:
            d = ev.activity_probs
            if not d:
                continue
            ent.append(normalized_shannon_entropy(d))
            maxp.append(max(float(p) for p in d.values() if p is not None))
            supp.append(float(sum(1 for p in d.values() if p is not None and float(p) > 0.0)))
            nap.append(float(d.get(na_label, 0.0)))
    if not ent:
        return {"avg_norm_entropy": 0.0, "avg_max_prob": 0.0, "avg_support": 0.0, "avg_na_prob": 0.0}
    return {
        "avg_norm_entropy": float(sum(ent) / len(ent)),
        "avg_max_prob": float(sum(maxp) / len(maxp)),
        "avg_support": float(sum(supp) / len(supp)),
        "avg_na_prob": float(sum(nap) / len(nap)),
    }


def apply_uncertainty_level(
    log: UncertainEventLog, *, k: int, na_label: str = "NA"
) -> UncertainEventLog:
    """
    Return a new log where each event distribution is truncated to top-k and renormalized.
    """
    new_traces: List[UncertainTrace] = []
    for tr in log.traces:
        new_events: List[UncertainEvent] = []
        for ev in tr.events:
            new_events.append(
                UncertainEvent(activity_probs=topk_distribution(ev.activity_probs, k=k, na_label=na_label), attributes=ev.attributes)
            )
        new_traces.append(
            UncertainTrace(
                trace_id=tr.trace_id,
                case_name=tr.case_name,
                events=new_events,
                attributes=tr.attributes,
            )
        )
    return UncertainEventLog(traces=new_traces, log_name=log.log_name)


# ---------------------------------------------------------------------------
# Ground truth generation (uncertain)
# ---------------------------------------------------------------------------


def activities_in_uncertain_log(log: UncertainEventLog, *, exclude: Optional[set[str]] = None) -> List[str]:
    ex = exclude or set()
    acts: set[str] = set()
    for tr in log.traces:
        for ev in tr.events:
            for a, p in ev.activity_probs.items():
                if a in ex:
                    continue
                if p is None or float(p) <= 0.0:
                    continue
                acts.add(str(a))
    return sorted(acts)


def get_uncertain_logs_with_replaced_activities_dict(
    activities_to_replace_in_each_run_list: Sequence[Tuple[str, ...]],
    base_uncertain_log: UncertainEventLog,
    *,
    different_activities_to_replace_count: int,
    activities_to_replace_with_count: int,
    na_label: str = "NA",
) -> Dict[Tuple[str, ...], UncertainEventLog]:
    """
    Uncertain analogue of `get_logs_with_replaced_activities_dict(...)`.

    For each run (a tuple of `different_activities_to_replace_count` activities), return an
    uncertain log where probability mass of those activities is replaced by new labels.

    Trace-consistent replacement rule:
      Within each trace, for each replaced activity a, we pick one replacement a:i and
      map all probability mass of a in that trace to a:i.
    """
    out: Dict[Tuple[str, ...], UncertainEventLog] = {}

    for activities_to_replace_tuple in activities_to_replace_in_each_run_list:
        # pool indices per original activity (balanced reuse across the log, as in deterministic version)
        pools: Dict[str, set[int]] = {a: set(range(activities_to_replace_with_count)) for a in activities_to_replace_tuple}

        new_traces: List[UncertainTrace] = []
        for tr in base_uncertain_log.traces:
            # per-trace chosen replacement label for each replaced activity
            chosen: Dict[str, str] = {}

            new_events: List[UncertainEvent] = []
            for ev in tr.events:
                d = ev.activity_probs
                new_d: Dict[str, float] = {}

                for a, p in d.items():
                    if p is None or float(p) <= 0.0:
                        continue
                    aa = str(a)
                    pp = float(p)
                    if aa == na_label:
                        new_d[na_label] = new_d.get(na_label, 0.0) + pp
                        continue
                    if aa in pools:
                        if aa not in chosen:
                            if len(pools[aa]) == 0:
                                pools[aa] = set(range(activities_to_replace_with_count))
                            chosen_idx = pools[aa].pop()
                            chosen[aa] = f"{aa}:{chosen_idx}"
                        rep = chosen[aa]
                        new_d[rep] = new_d.get(rep, 0.0) + pp
                    else:
                        new_d[aa] = new_d.get(aa, 0.0) + pp

                # NOTE: no renormalization needed here because the mapping is injective if replacement pools are disjoint.
                new_events.append(UncertainEvent(activity_probs=new_d, attributes=ev.attributes))

            new_traces.append(
                UncertainTrace(
                    trace_id=tr.trace_id,
                    case_name=tr.case_name,
                    events=new_events,
                    attributes=tr.attributes,
                )
            )

        out[tuple(activities_to_replace_tuple)] = UncertainEventLog(traces=new_traces, log_name=base_uncertain_log.log_name)

    return out


# ---------------------------------------------------------------------------
# Save / load (mirror deterministic structure)
# ---------------------------------------------------------------------------


def gt_uncertain_dir(log_name: str) -> str:
    return os.path.join(
        ROOT_DIR, "evaluation", "evaluation_of_activity_distances", "intrinsic_evaluation_uncertain", "newly_created_logs", log_name
    )


def results_uncertain_dir(log_name: str) -> str:
    return os.path.join(
        ROOT_DIR, "evaluation", "evaluation_of_activity_distances", "intrinsic_evaluation_uncertain", "results", log_name
    )


def save_uncertain_ground_truth_logs(
    log_name: str,
    *,
    different_activities_to_replace_count: int,
    activities_to_replace_with_count: int,
    sampling_size: int,
    logs_with_replaced_activities_dict: Dict[Tuple[str, ...], UncertainEventLog],
) -> str:
    os.makedirs(gt_uncertain_dir(log_name), exist_ok=True)
    file_name = f"r_{different_activities_to_replace_count}_w_{activities_to_replace_with_count}_s_{sampling_size}.pkl"
    path = os.path.join(gt_uncertain_dir(log_name), file_name)
    with open(path, "wb") as f:
        pickle.dump(logs_with_replaced_activities_dict, f)
    return path


def load_uncertain_ground_truth_logs(
    log_name: str,
    *,
    different_activities_to_replace_count: int,
    activities_to_replace_with_count: int,
    sampling_size: int,
) -> Optional[Dict[Tuple[str, ...], UncertainEventLog]]:
    file_name = f"r_{different_activities_to_replace_count}_w_{activities_to_replace_with_count}_s_{sampling_size}.pkl"
    path = os.path.join(gt_uncertain_dir(log_name), file_name)
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def save_uncertain_result(
    results: Tuple,
    *,
    log_name: str,
    activity_distance_function: str,
    different_activities_to_replace_count: int,
    activities_to_replace_with_count: int,
    sampling_size: int,
    uncertainty_level: int,
) -> str:
    """
    Mirror deterministic per-method result caching, with an additional uncertainty level.
    """
    base = os.path.join(results_uncertain_dir(log_name), activity_distance_function)
    os.makedirs(base, exist_ok=True)
    file_name = f"r_{different_activities_to_replace_count}_w_{activities_to_replace_with_count}_s_{sampling_size}_u_{uncertainty_level}.pkl"
    path = os.path.join(base, file_name)
    with open(path, "wb") as f:
        pickle.dump(results, f)
    return path


def load_uncertain_result(
    *,
    log_name: str,
    activity_distance_function: str,
    different_activities_to_replace_count: int,
    activities_to_replace_with_count: int,
    sampling_size: int,
    uncertainty_level: int,
) -> Optional[Tuple]:
    base = os.path.join(results_uncertain_dir(log_name), activity_distance_function)
    file_name = f"r_{different_activities_to_replace_count}_w_{activities_to_replace_with_count}_s_{sampling_size}_u_{uncertainty_level}.pkl"
    path = os.path.join(base, file_name)
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


