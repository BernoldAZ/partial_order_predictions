"""
Uncertain Activity-Context (AC) expected-count embeddings.

This mirrors `distances/activity_distances/activity_context_frequency/activity_contex_frequency.py`
but operates on an `UncertainEventLog` and uses expected counts #(a,c) instead of integer counts.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from distances.activity_distances.data_util.algorithm import get_cosine_distance_dict
from distances.uncertain_activity_distances.data_util.uncertain_expected_counts import (
    PAD_TOKEN,
    Context,
    compute_context_frequencies_from_expected_counts,
    compute_expected_context_counts_and_activity_frequencies,
)
from uncertain_utils.uncertain_xes_reader import UncertainEventLog


def get_uncertain_activity_context_frequency_matrix(
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
) -> Tuple[Dict[Tuple[str, str], float], Dict[str, np.ndarray], Dict[str, float], Dict[Context, float], Dict[Context, int]]:
    """
    Build AC embeddings for uncertain logs and return cosine distances.

    Returns
    -------
    distance_matrix:
        dict[(a,a')] -> cosine distance between embeddings
    embeddings:
        dict[a] -> vector over contexts
    activity_freq_dict:
        expected #(a)
    context_freq_dict:
        expected #(c)
    context_index:
        mapping context -> column index
    """
    exclude = exclude_activities or set()
    exclude.add(PAD_TOKEN)

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

    # Collect all contexts used by any activity
    all_contexts = {ctx for cmap in expected_counts.values() for ctx in cmap.keys()}
    context_index = {ctx: i for i, ctx in enumerate(all_contexts)}

    context_freq_dict = compute_context_frequencies_from_expected_counts(expected_counts)

    # Build embeddings
    alphabet = sorted(activity_freq_dict.keys())
    embeddings: Dict[str, np.ndarray] = {}
    for a in alphabet:
        embeddings[a] = np.zeros(len(all_contexts), dtype=float)

    for a, cmap in expected_counts.items():
        if a not in embeddings:
            # activity could be excluded or pruned out
            continue
        for ctx, v in cmap.items():
            embeddings[a][context_index[ctx]] += float(v)

    distance_matrix = get_cosine_distance_dict(embeddings)
    return distance_matrix, embeddings, activity_freq_dict, context_freq_dict, context_index


