"""
Small helpers for uncertain evaluation scripts.

We intentionally keep this module dependency-light (no tensorflow, no pm4py) so
uncertain evaluation scripts can run in minimal environments.
"""

from __future__ import annotations

import re
from typing import List, Sequence


def extract_window_size(s: str) -> int:
    match = re.search(r"w_(\d+)", s)
    return int(match.group(1)) if match else 3


def add_window_size_evaluation(methods: Sequence[str], window_size_list: Sequence[int]) -> List[str]:
    """
    Mirror the deterministic convention: append `w_<n>` suffix to methods that use a window size.
    """
    out: List[str] = []
    for m in methods:
        # If already has w_ suffix, keep as-is
        if re.search(r"w_\d+", m):
            out.append(m)
            continue
        for w in window_size_list:
            out.append(f"{m} w_{w}")
    return out


