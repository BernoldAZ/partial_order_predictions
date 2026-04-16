"""
Compute activity distance matrices for the *uncertain* intrinsic benchmark.

This mirrors the role of `evaluation/data_util/util_activity_distances.py` and the
deterministic intrinsic evaluation calling pattern, but:
- input logs are `UncertainEventLog`
- methods are the uncertain window-based count methods and uncertain act2vec

Implementation note
-------------------
For the intrinsic benchmark we prefer window-based exact counting (with top-k capped
event distributions, typically <=5) to keep runtime feasible.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Tuple

import numpy as np

from distances.activity_distances.data_util.algorithm import get_cosine_distance_dict
from distances.activity_distances.pmi.pmi import (
    get_activity_activity_frequency_matrix_pmi,
)
from distances.uncertain_activity_distances.data_util.uncertain_sparse_math import cosine_distance_matrix_sparse
from distances.uncertain_activity_distances.data_util.uncertain_sparse_pmi import ac_to_pmi_sparse
from distances.uncertain_activity_distances.data_util.uncertain_window_based_multiwindow_expected_counts import (
    compute_expected_counts_window_based_multiwindow,
)
from distances.uncertain_activity_distances.uncertain_act2vec.uncertain_act2vec import (
    UncertainAct2VecConfig,
    get_uncertain_act2vec_distance_matrix,
)
from evaluation.data_util.uncertain_evaluation_helpers import extract_window_size
from uncertain_utils.uncertain_xes_reader import UncertainEventLog


_COUNTS_CACHE: dict[tuple[int, int, str], tuple[dict, dict, dict]] = {}

def _pick_torch_device() -> str:
    """
    Pick the best available PyTorch device for act2vec training/inference.

    Priority:
    - "cuda" if CUDA is available
    - "mps" if Apple Metal (MPS) is available (macOS)
    - "cpu" otherwise
    """
    try:
        import torch  # local import: only needed for act2vec
    except Exception:
        return "cpu"

    try:
        if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass

    try:
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and getattr(mps, "is_available", None) and mps.is_available():
            return "mps"
    except Exception:
        pass

    return "cpu"


def clear_counts_cache() -> None:
    """
    Clear cached window-based expected counts.

    Use this between uncertainty levels `u` to keep memory bounded while still
    avoiding recomputation within the same `u` (where many methods share the same counts).
    """
    _COUNTS_CACHE.clear()


def _strip_window_suffix(method: str) -> str:
    # deterministic code appends " w_3" etc; keep the base name here
    if " w_" in method:
        return method.split(" w_")[0].strip()
    return method.strip()


def get_uncertain_activity_distance_matrix_one_method(
    log: UncertainEventLog,
    *,
    method_name_with_window: str,
    na_label: str = "NA",
    progress=None,
) -> Tuple[Dict[Tuple[str, str], float], dict]:
    """
    Return distance dict[(a,b)] -> distance and a debug dict.
    """
    window_size = extract_window_size(method_name_with_window)
    method_name = _strip_window_suffix(method_name_with_window)

    # ---------------------------------------------------------------------
    # Uncertain act2vec (window-realizations expected loss)
    # ---------------------------------------------------------------------
    if method_name in ("Uncertain act2vec CBOW", "Uncertain act2vec Skip-gram"):
        sg = 0 if method_name.endswith("CBOW") else 1
        device = _pick_torch_device()
        cfg = UncertainAct2VecConfig(
            window_size=window_size,
            embedding_dim=16,
            epochs=10,
            batch_size=256,
            start_alpha=0.025,
            alpha_decay_per_epoch=0.002,
            min_alpha=0.0001,
            seed=0,
            device=device,
            training_mode="window_realizations",
            prob_threshold=0.0,
            na_label=na_label,
            pad_token=".",
            negative=5,
            negative_sampling_exponent=0.75,
            progress_every_samples=50_000,
        )
        dist, emb, dbg = get_uncertain_act2vec_distance_matrix(log, sg=sg, config=cfg, progress=progress)
        if isinstance(dbg, dict):
            dbg.setdefault("device", device)
        return dist, dbg

    # ---------------------------------------------------------------------
    # Window-based uncertain count-based methods (exact under naive independence)
    # ---------------------------------------------------------------------
    # Parse choices from method string (keeps naming consistent with `util_activity_distances_uncertain.py`)
    if not method_name.startswith("Uncertain "):
        raise ValueError(f"Unknown uncertain method: {method_name!r}")

    matrix_type = "AA" if " AA " in f" {method_name} " else "AC"
    context_kind = "seq" if " Seq" in method_name else "mset"
    post = "none"
    if method_name.endswith(" PPMI"):
        post = "ppmi"
    elif method_name.endswith(" PMI"):
        post = "pmi"

    cache_key = (id(log), window_size, context_kind)
    cached = _COUNTS_CACHE.get(cache_key)
    if cached is None:
        expected_counts_by_w, ctx_freq_by_w, act_freq = compute_expected_counts_window_based_multiwindow(
            log,
            window_sizes=[window_size],
            context_kind=context_kind,
            na_label=na_label,
            prob_threshold=0.0,  # exact (event distributions already capped upstream)
            progress=progress,
        )
        _COUNTS_CACHE[cache_key] = (expected_counts_by_w, ctx_freq_by_w, act_freq)
    else:
        expected_counts_by_w, ctx_freq_by_w, act_freq = cached
    counts = expected_counts_by_w[window_size]  # activity -> context -> expected count

    if matrix_type == "AC":
        # sparse dict embeddings
        ac = {a: dict(cmap) for a, cmap in counts.items()}
        if post == "pmi":
            ac = ac_to_pmi_sparse(ac, activity_freq=act_freq, context_freq=ctx_freq_by_w[window_size], ppmi=False)
        elif post == "ppmi":
            ac = ac_to_pmi_sparse(ac, activity_freq=act_freq, context_freq=ctx_freq_by_w[window_size], ppmi=True)
        dist = cosine_distance_matrix_sparse(ac)
        return dist, {"activity_freq": act_freq, "context_kind": context_kind, "matrix_type": matrix_type, "post": post}

    # AA: build dense AA rows over activities
    alphabet = sorted(act_freq.keys())
    idx = {a: i for i, a in enumerate(alphabet)}
    aa_emb = {a: np.zeros(len(alphabet), dtype=float) for a in alphabet}

    context_to_counts: Dict[object, Dict[str, float]] = defaultdict(dict)
    for a, cmap in counts.items():
        for ctx, v in cmap.items():
            context_to_counts[ctx][a] = float(v)

    for ctx, a_counts in context_to_counts.items():
        acts = list(a_counts.keys())
        for a in acts:
            row = aa_emb.get(a)
            if row is None:
                continue
            for b in acts:
                row[idx[b]] += a_counts[a] + a_counts[b]

    if post == "pmi":
        dist, _ = get_activity_activity_frequency_matrix_pmi(aa_emb, act_freq, idx, 0)
        return dist, {"activity_freq": act_freq, "context_kind": context_kind, "matrix_type": matrix_type, "post": post}
    if post == "ppmi":
        dist, _ = get_activity_activity_frequency_matrix_pmi(aa_emb, act_freq, idx, 1)
        return dist, {"activity_freq": act_freq, "context_kind": context_kind, "matrix_type": matrix_type, "post": post}

    dist = get_cosine_distance_dict(aa_emb)
    return dist, {"activity_freq": act_freq, "context_kind": context_kind, "matrix_type": matrix_type, "post": post}


def get_uncertain_activity_distance_matrix_dict(
    activity_distance_function_list,
    logs_with_replaced_activities: Dict[Tuple[str, ...], UncertainEventLog],
    *,
    na_label: str = "NA",
    progress=None,
) -> Dict[str, Dict[Tuple[str, ...], Dict[Tuple[str, str], float]]]:
    """
    Mirror deterministic `get_activity_distance_matrix_dict` but for uncertain logs.

    Returns:
      dict[method_name] -> dict[activities_to_replace_tuple] -> distance_matrix_dict
    """
    out = defaultdict(dict)
    for method in activity_distance_function_list:
        for activities_to_replace, log in logs_with_replaced_activities.items():
            dist, _dbg = get_uncertain_activity_distance_matrix_one_method(
                log, method_name_with_window=method, na_label=na_label, progress=progress
            )
            out[method][activities_to_replace] = dist
    return dict(out)


