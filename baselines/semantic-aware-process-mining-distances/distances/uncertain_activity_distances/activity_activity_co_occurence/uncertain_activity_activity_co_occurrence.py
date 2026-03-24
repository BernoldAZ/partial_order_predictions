"""
Uncertain Activity-Activity (AA) co-occurrence embeddings using expected counts.

This mirrors `distances/activity_distances/activity_activity_co_occurence/activity_activity_co_occurrence.py`,
but instead of integer counts from deterministic n-grams, we:

1) compute expected counts #(a,c) for activity-context pairs from an `UncertainEventLog`
2) aggregate contexts into an AA matrix using the journal-extension definition:
      AA(a,a') = sum_{c where #(a,c)>0 and #(a',c)>0} ( #(a,c) + #(a',c) )

Then we interpret each row AA(a,·) as an embedding and compute cosine distances.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from distances.activity_distances.data_util.algorithm import get_cosine_distance_dict
from distances.uncertain_activity_distances.data_util.uncertain_expected_counts import (
    PAD_TOKEN,
    Context,
    compute_expected_context_counts_and_activity_frequencies,
)
from uncertain_utils.uncertain_xes_reader import UncertainEventLog


def get_uncertain_activity_activity_co_occurrence_matrix(
    log: UncertainEventLog,
    *,
    ngram_size: int,
    context_kind: str,  # "seq" | "mset"
    top_k: int | None = None,
    min_prob: float = 0.0,
    na_label: str = "NA",
    max_realizations_per_trace: int | None = None,
    exclude_activities: set[str] | None = None,
    progress=None,
    progress_every_realizations: int = 50_000,
    n_jobs: int = 1,
    mp_start_method: str = "spawn",
) -> Tuple[Dict[Tuple[str, str], float], Dict[str, np.ndarray], Dict[str, float], Dict[str, int]]:
    """
    Build AA embeddings for uncertain logs and return cosine distances.

    Returns
    -------
    distance_matrix:
        dict[(a,a')] -> cosine distance between AA-row embeddings
    embeddings:
        dict[a] -> vector over activities (AA row for activity a)
    activity_freq_dict:
        expected #(a) (used for PMI/PPMI outside this function)
    activity_index:
        activity -> column index
    """
    exclude = exclude_activities or set()
    exclude.add(PAD_TOKEN)

    # Compute expected #(a,c) and #(a) in one pass
    expected_counts, activity_freq_dict = compute_expected_context_counts_and_activity_frequencies(
        log,
        ngram_size=ngram_size,
        context_kind=context_kind,
        top_k=top_k,
        min_prob=min_prob,
        na_label=na_label,
        max_realizations_per_trace=max_realizations_per_trace,
        exclude_activities=exclude,
        progress=progress,
        progress_every_realizations=progress_every_realizations,
        n_jobs=n_jobs,
        mp_start_method=mp_start_method,
    )

    alphabet = sorted(activity_freq_dict.keys())
    activity_index = {a: i for i, a in enumerate(alphabet)}

    # Initialize AA embeddings
    embeddings: Dict[str, np.ndarray] = {a: np.zeros(len(alphabet), dtype=float) for a in alphabet}

    # Invert expected_counts into context -> {activity -> #(a,c)}
    context_to_counts: Dict[Context, Dict[str, float]] = defaultdict(dict)
    for a, cmap in expected_counts.items():
        for ctx, v in cmap.items():
            context_to_counts[ctx][a] = float(v)

    # Aggregate contexts into AA
    for ctx, a_counts in context_to_counts.items():
        activities_in_ctx = list(a_counts.keys())
        for a in activities_in_ctx:
            row = embeddings.get(a)
            if row is None:
                continue
            for b in activities_in_ctx:
                j = activity_index[b]
                row[j] += a_counts[a] + a_counts[b]

    distance_matrix = get_cosine_distance_dict(embeddings)
    return distance_matrix, embeddings, activity_freq_dict, activity_index


