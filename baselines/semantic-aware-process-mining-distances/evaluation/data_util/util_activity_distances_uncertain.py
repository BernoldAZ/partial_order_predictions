"""
Utilities for computing activity distances from *uncertain* event logs.

This is a parallel entry point to `evaluation/data_util/util_activity_distances.py`
but intentionally does NOT modify the deterministic evaluation code.

Current scope
-------------
Primarily the *count-based* family (AA/AC with None/PMI/PPMI) is implemented here.

Additionally, we provide an *uncertain* variant of Act2Vec (CBOW and Skip-gram)
trained directly on event-level probability distributions (soft labels).

The uncertain log input is expected to be an `UncertainEventLog` created by:
    `uncertain_utils.uncertain_xes_reader.read_uncertain_xes()`
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

from distances.activity_distances.pmi.pmi import (
    get_activity_activity_frequency_matrix_pmi,
    get_activity_context_frequency_matrix_pmi,
)
from distances.uncertain_activity_distances.activity_activity_co_occurence.uncertain_activity_activity_co_occurrence import (
    get_uncertain_activity_activity_co_occurrence_matrix,
)
from distances.uncertain_activity_distances.activity_context_frequency.uncertain_activity_context_frequency import (
    get_uncertain_activity_context_frequency_matrix,
)
from distances.uncertain_activity_distances.uncertain_act2vec.uncertain_act2vec import (
    UncertainAct2VecConfig,
    get_uncertain_act2vec_distance_matrix,
)
from uncertain_utils.uncertain_xes_reader import UncertainEventLog


UNCERTAIN_COUNT_BASED_METHODS: List[str] = [
    # AA (activity-activity)
    "Uncertain AA Seq",
    "Uncertain AA Seq PMI",
    "Uncertain AA Seq PPMI",
    "Uncertain AA MSet",
    "Uncertain AA MSet PMI",
    "Uncertain AA MSet PPMI",
    # AC (activity-context)
    "Uncertain AC Seq",
    "Uncertain AC Seq PMI",
    "Uncertain AC Seq PPMI",
    "Uncertain AC MSet",
    "Uncertain AC MSet PMI",
    "Uncertain AC MSet PPMI",
]

UNCERTAIN_NEURAL_METHODS: List[str] = [
    "Uncertain act2vec CBOW",
    "Uncertain act2vec Skip-gram",
]


def get_uncertain_activity_distance_matrix(
    log: UncertainEventLog,
    *,
    method_name: str,
    window_size: int,
    top_k: int | None = None,
    min_prob: float = 0.0,
    na_label: str = "NA",
    max_realizations_per_trace: int | None = None,
    exclude_activities: set[str] | None = None,
    progress=None,
    progress_every_realizations: int = 50_000,
    n_jobs: int = 1,
    mp_start_method: str = "spawn",
) -> Tuple[Dict[Tuple[str, str], float], dict]:
    """
    Compute an activity distance matrix for one uncertain count-based method.

    Returns
    -------
    distance_matrix:
        dict[(a,a')] -> cosine distance
    debug:
        small dict containing embeddings and frequency dicts (useful for inspection)
    """
    if method_name in UNCERTAIN_NEURAL_METHODS:
        sg = 0 if method_name.endswith("CBOW") else 1
        cfg = UncertainAct2VecConfig(
            window_size=window_size,
            embedding_dim=16,
            epochs=10,
            batch_size=256,
            start_alpha=0.025,
            alpha_decay_per_epoch=0.002,
            min_alpha=0.0001,
            seed=0,
            device="cpu",
            training_mode="window_realizations",
            prob_threshold=0.0,
            top_k=top_k,
            min_prob=min_prob,
            na_label=na_label,
            pad_token=".",
            drop_na_and_renormalize=False,
            exclude_activities=exclude_activities,
            progress_every_samples=50_000,
            negative=5,
            negative_sampling_exponent=0.75,
        )
        dist, emb, dbg = get_uncertain_act2vec_distance_matrix(log, sg=sg, config=cfg, progress=progress)
        return dist, {"embeddings": emb, **dbg}

    if method_name not in UNCERTAIN_COUNT_BASED_METHODS:
        raise ValueError(f"Unknown uncertain method: {method_name!r}")

    # Determine design choices
    if " AA " in f" {method_name} ":
        matrix_type = "AA"
    else:
        matrix_type = "AC"

    if " Seq" in method_name:
        context_kind = "seq"
    else:
        context_kind = "mset"

    postproc = "none"
    if method_name.endswith(" PPMI"):
        postproc = "ppmi"
    elif method_name.endswith(" PMI"):
        postproc = "pmi"

    if matrix_type == "AA":
        dist, emb, act_freq, act_index = get_uncertain_activity_activity_co_occurrence_matrix(
            log,
            ngram_size=window_size,
            context_kind=context_kind,
            top_k=top_k,
            min_prob=min_prob,
            na_label=na_label,
            max_realizations_per_trace=max_realizations_per_trace,
            exclude_activities=exclude_activities,
            progress=progress,
            progress_every_realizations=progress_every_realizations,
            n_jobs=n_jobs,
            mp_start_method=mp_start_method,
        )
        if postproc == "pmi":
            dist, emb = get_activity_activity_frequency_matrix_pmi(emb, act_freq, act_index, 0)
        elif postproc == "ppmi":
            dist, emb = get_activity_activity_frequency_matrix_pmi(emb, act_freq, act_index, 1)
        return dist, {"embeddings": emb, "activity_freq": act_freq, "activity_index": act_index}

    # AC
    dist, emb, act_freq, ctx_freq, ctx_index = get_uncertain_activity_context_frequency_matrix(
        log,
        ngram_size=window_size,
        context_kind=context_kind,
        top_k=top_k,
        min_prob=min_prob,
        na_label=na_label,
        max_realizations_per_trace=max_realizations_per_trace,
        exclude_activities=exclude_activities,
        progress=progress,
        progress_every_realizations=progress_every_realizations,
        n_jobs=n_jobs,
        mp_start_method=mp_start_method,
    )
    if postproc == "pmi":
        dist, emb = get_activity_context_frequency_matrix_pmi(emb, act_freq, ctx_freq, ctx_index, 0)
    elif postproc == "ppmi":
        dist, emb = get_activity_context_frequency_matrix_pmi(emb, act_freq, ctx_freq, ctx_index, 1)
    return dist, {"embeddings": emb, "activity_freq": act_freq, "context_freq": ctx_freq, "context_index": ctx_index}


