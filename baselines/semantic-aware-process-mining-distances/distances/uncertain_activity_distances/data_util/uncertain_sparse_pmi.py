"""
Sparse PMI/PPMI transforms for sparse embeddings.

We mirror the deterministic PMI idea but operate on sparse dict embeddings:
  emb[a][c] = AC(a,c)   (expected count)

We compute:
  PMI(a,c) = log( (AC(a,c)/N) / (p(a) p(c)) )
with p(a)=#(a)/N and p(c)=#(c)/N, and then optionally clamp to 0 (PPMI).
"""

from __future__ import annotations

import math
from typing import Dict, Hashable


def ac_to_pmi_sparse(
    ac_embeddings: Dict[str, Dict[Hashable, float]],
    *,
    activity_freq: Dict[str, float],
    context_freq: Dict[Hashable, float],
    ppmi: bool,
) -> Dict[str, Dict[Hashable, float]]:
    """
    Transform sparse AC counts into sparse PMI/PPMI embeddings.
    """
    N = float(sum(activity_freq.values()))
    if N <= 0.0:
        return {a: {} for a in ac_embeddings.keys()}

    out: Dict[str, Dict[Hashable, float]] = {}
    for a, cmap in ac_embeddings.items():
        pa = float(activity_freq.get(a, 0.0)) / N
        if pa <= 0.0:
            out[a] = {}
            continue

        row: Dict[Hashable, float] = {}
        for c, v in cmap.items():
            vc = float(v)
            if vc <= 0.0:
                continue
            pc = float(context_freq.get(c, 0.0)) / N
            if pc <= 0.0:
                continue
            pij = vc / N
            pmi = math.log(pij / (pa * pc))
            if ppmi:
                if pmi > 0.0:
                    row[c] = pmi
            else:
                row[c] = pmi
        out[a] = row

    return out



