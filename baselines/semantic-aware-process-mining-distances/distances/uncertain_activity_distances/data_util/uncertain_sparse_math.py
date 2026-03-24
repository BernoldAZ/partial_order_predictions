"""
Sparse math utilities for uncertain embeddings.

Why
---
Window-based counting can produce *huge* context spaces (millions of distinct contexts).
Building dense numpy vectors of length |C| (as done in the deterministic baseline) can
explode RAM.

These helpers keep embeddings in sparse form:
  embedding[a] = {context: weight}
and compute cosine distances directly on sparse dicts.
"""

from __future__ import annotations

import math
from typing import Dict, Hashable, Tuple


def cosine_distance_sparse(a: Dict[Hashable, float], b: Dict[Hashable, float]) -> float:
    """
    Cosine distance = 1 - cosine similarity for sparse dict vectors.
    """
    if not a and not b:
        return 0.0

    # dot product: iterate over smaller dict
    if len(a) > len(b):
        a, b = b, a

    dot = 0.0
    for k, va in a.items():
        vb = b.get(k)
        if vb is not None:
            dot += float(va) * float(vb)

    norm_a = math.sqrt(sum(float(v) * float(v) for v in a.values()))
    norm_b = math.sqrt(sum(float(v) * float(v) for v in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0

    cos_sim = dot / (norm_a * norm_b)
    return 1.0 - cos_sim


def cosine_distance_matrix_sparse(embeddings: Dict[str, Dict[Hashable, float]]) -> Dict[Tuple[str, str], float]:
    """
    Return dict[(a,b)] -> cosine distance for all pairs of activities.
    """
    acts = list(embeddings.keys())
    out: Dict[Tuple[str, str], float] = {}
    for a in acts:
        for b in acts:
            out[(a, b)] = cosine_distance_sparse(embeddings[a], embeddings[b])
    return out



