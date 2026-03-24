"""
Exact, low-RAM backend for window-based uncertain counting using SQLite.

Why
---
The window-based method can generate millions of distinct contexts, and storing
`expected_counts[a][ctx]` in Python dicts can exceed 25GB RAM.

This module keeps the computation exact while bounding RAM by:
1) Streaming window contributions into a SQLite table `counts(ctx, act, val)`
2) Streaming the table back grouped by ctx to compute cosine/PMI/PPMI distances
   without materializing the full context space in memory.
"""

from __future__ import annotations

import math
import os
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from math import ceil, floor
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from uncertain_utils.uncertain_xes_reader import UncertainEventLog
from distances.activity_distances.data_util.algorithm import get_cosine_distance_dict
from distances.activity_distances.pmi.pmi import get_activity_activity_frequency_matrix_pmi


PAD_TOKEN = "."
DEFAULT_NA_LABEL = "NA"


def _options_threshold_renorm(dist: Dict[str, float], *, prob_threshold: float, label_map: Dict[str, str]) -> List[Tuple[str, float]]:
    items = []
    for a, p in dist.items():
        if p is None:
            continue
        pf = float(p)
        if pf <= 0.0 or pf < prob_threshold:
            continue
        aa = label_map.get(str(a), str(a))
        items.append((aa, pf))
    if not items:
        if not dist:
            return []
        a_max = max(dist.items(), key=lambda kv: float(kv[1]))[0]
        a_max = label_map.get(str(a_max), str(a_max))
        return [(a_max, 1.0)]
    s = sum(p for _, p in items)
    if s > 0:
        items = [(a, p / s) for a, p in items]
    items.sort(key=lambda kv: kv[1], reverse=True)
    return items


def _serialize_context_seq(seq: Tuple[str, ...]) -> str:
    # Use a separator unlikely to appear in IDs.
    return "\x1f".join(seq)


def _serialize_context_mset(counter_items: Iterable[Tuple[str, int]]) -> str:
    # Deterministic ordering
    parts = [f"{a}:{c}" for a, c in sorted(counter_items)]
    return "\x1f".join(parts)


@dataclass(frozen=True)
class SqliteCountsConfig:
    prob_threshold: float = 0.05
    na_label: str = DEFAULT_NA_LABEL
    pad_token: str = PAD_TOKEN
    flush_every: int = 200_000  # number of buffered updates before flushing


def init_counts_db(conn: sqlite3.Connection) -> None:
    """
    Initialize pragmas + schema on an existing connection.

    This supports both on-disk DBs (Path) and in-memory DBs (":memory:").
    """
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=OFF;")
    cur.execute("PRAGMA temp_store=MEMORY;")
    cur.execute("PRAGMA cache_size=-200000;")  # ~200MB cache
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS counts (
            ctx TEXT NOT NULL,
            act TEXT NOT NULL,
            val REAL NOT NULL,
            PRIMARY KEY (ctx, act)
        );
        """
    )
    conn.commit()


def create_counts_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    init_counts_db(conn)
    return conn


def upsert_counts(conn: sqlite3.Connection, rows: List[Tuple[str, str, float]]) -> None:
    cur = conn.cursor()
    cur.executemany(
        """
        INSERT INTO counts(ctx, act, val) VALUES (?, ?, ?)
        ON CONFLICT(ctx, act) DO UPDATE SET val = val + excluded.val
        """,
        rows,
    )
    conn.commit()


def window_count_to_sqlite(
    log: UncertainEventLog,
    *,
    window_size: int,
    context_kind: str,  # "seq" | "mset"
    label_map: Dict[str, str],
    cfg: SqliteCountsConfig,
    db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
    progress: Optional[callable] = None,
) -> Dict[str, float]:
    """
    Stream expected counts #(a,c) into SQLite and return activity_freq #(a).

    Provide either:
    - db_path: creates/closes an on-disk DB, or
    - conn: uses an existing connection (e.g. sqlite3.connect(":memory:")) and does NOT close it.
    """
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be odd.")
    if context_kind not in ("seq", "mset"):
        raise ValueError("context_kind must be 'seq' or 'mset'")

    k = window_size // 2
    pad_left = k
    pad_right = k

    activity_freq: Dict[str, float] = defaultdict(float)

    owns_conn = False
    if conn is None:
        if db_path is None:
            raise ValueError("Provide either db_path or conn.")
        conn = create_counts_db(db_path)
        owns_conn = True
    else:
        init_counts_db(conn)
    buf: List[Tuple[str, str, float]] = []

    def flush():
        nonlocal buf
        if buf:
            upsert_counts(conn, buf)
            buf = []

    # Precompute per-trace
    for trace_idx, tr in enumerate(log.traces, start=1):
        if progress is not None:
            progress(f"[sqlite-count] trace {trace_idx}/{len(log.traces)} events={len(tr.events)}")

        trace_probs = [ev.activity_probs for ev in tr.events]
        pad_dist = {cfg.pad_token: 1.0}
        padded = [pad_dist] * pad_left + trace_probs + [pad_dist] * pad_right
        padded_opts = [_options_threshold_renorm(d, prob_threshold=cfg.prob_threshold, label_map=label_map) for d in padded]

        # Iterate centers (original positions)
        for center_idx in range(pad_left, pad_left + len(tr.events)):
            center_opts = padded_opts[center_idx]

            left_outward = [padded_opts[center_idx - i] for i in range(1, center_idx + 1)]
            right_outward = [padded_opts[center_idx + i] for i in range(1, len(padded_opts) - center_idx)]

            # Enumerate side contexts (DP) using compressed labels
            left_dist = _enumerate_side(left_outward, needed=k, na_label=cfg.na_label, direction="left")
            right_dist = _enumerate_side(right_outward, needed=k, na_label=cfg.na_label, direction="right")

            for a, p_a in center_opts:
                if p_a <= 0.0:
                    continue
                if a == cfg.na_label or a == cfg.pad_token:
                    continue
                activity_freq[a] += p_a

                for l_seq, p_l in left_dist.items():
                    for r_seq, p_r in right_dist.items():
                        p = p_a * p_l * p_r
                        if p <= 0.0:
                            continue
                        seq_ctx = tuple(l_seq + r_seq)
                        if context_kind == "seq":
                            ctx_key = _serialize_context_seq(seq_ctx)
                        else:
                            ctx_key = _serialize_context_mset(Counter(seq_ctx).items())
                        buf.append((ctx_key, a, float(p)))
                        if len(buf) >= cfg.flush_every:
                            flush()

    flush()
    if owns_conn:
        conn.close()
    return dict(activity_freq)


def _enumerate_side(
    padded_opts: Sequence[List[Tuple[str, float]]],
    *,
    needed: int,
    na_label: str,
    direction: str,
) -> Dict[Tuple[str, ...], float]:
    if needed == 0:
        return {tuple(): 1.0}
    frontier: Dict[Tuple[str, ...], float] = {tuple(): 1.0}
    done: Dict[Tuple[str, ...], float] = defaultdict(float)
    for event_options in padded_opts:
        nxt: Dict[Tuple[str, ...], float] = defaultdict(float)
        for seq, mass in frontier.items():
            if len(seq) >= needed:
                done[seq] += mass
                continue
            for label, p in event_options:
                if p <= 0.0:
                    continue
                if label == na_label:
                    nxt[seq] += mass * p
                else:
                    new_seq = (label,) + seq if direction == "left" else seq + (label,)
                    if len(new_seq) >= needed:
                        done[new_seq] += mass * p
                    else:
                        nxt[new_seq] += mass * p
        frontier = dict(nxt)
        if not frontier:
            break
    for seq, mass in frontier.items():
        if len(seq) >= needed:
            done[seq] += mass
    return dict(done)


def compute_ac_distances_from_sqlite(
    conn: sqlite3.Connection,
    *,
    activity_freq: Dict[str, float],
) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]]:
    """
    Stream counts grouped by ctx and compute cosine distances for:
    - raw AC
    - PMI-weighted AC
    - PPMI-weighted AC
    """
    acts = sorted(activity_freq.keys())
    idx = {a: i for i, a in enumerate(acts)}
    n = len(acts)

    N = float(sum(activity_freq.values()))
    if N <= 0.0:
        zero = {(a, b): 0.0 for a in acts for b in acts}
        return zero, zero, zero

    # dot/norm accumulators for 3 variants
    dot_raw = np.zeros((n, n), dtype=float)
    dot_pmi = np.zeros((n, n), dtype=float)
    dot_ppmi = np.zeros((n, n), dtype=float)
    norm_raw = np.zeros(n, dtype=float)
    norm_pmi = np.zeros(n, dtype=float)
    norm_ppmi = np.zeros(n, dtype=float)

    cur = conn.cursor()
    cur.execute("SELECT ctx, act, val FROM counts ORDER BY ctx")

    current_ctx = None
    items: List[Tuple[str, float]] = []

    def flush_ctx():
        nonlocal items
        if not items:
            return
        # ctx_sum = #(c)
        ctx_sum = sum(v for _, v in items)
        if ctx_sum <= 0.0:
            items = []
            return

        # Precompute per-activity weights for each variant for this context
        raw_w = []
        pmi_w = []
        ppmi_w = []
        for a, v in items:
            ia = idx[a]
            raw = float(v)
            raw_w.append((ia, raw))
            norm_raw[ia] += raw * raw

            # PMI = log( (v/N) / ( (#a/N)*(#c/N) ) ) = log( v*N / (#a*#c) )
            pa_cnt = float(activity_freq.get(a, 0.0))
            if pa_cnt > 0.0:
                pmi = math.log((raw * N) / (pa_cnt * ctx_sum))
            else:
                pmi = 0.0
            pmi_w.append((ia, pmi))
            norm_pmi[ia] += pmi * pmi

            ppmi = pmi if pmi > 0.0 else 0.0
            ppmi_w.append((ia, ppmi))
            norm_ppmi[ia] += ppmi * ppmi

        # Update dot products (k^2 where k=activities in this context)
        for i_idx, wi in raw_w:
            for j_idx, wj in raw_w:
                dot_raw[i_idx, j_idx] += wi * wj
        for i_idx, wi in pmi_w:
            for j_idx, wj in pmi_w:
                dot_pmi[i_idx, j_idx] += wi * wj
        for i_idx, wi in ppmi_w:
            for j_idx, wj in ppmi_w:
                dot_ppmi[i_idx, j_idx] += wi * wj

        items = []

    for ctx_key, act, val in cur:
        ctx_key = str(ctx_key)
        act = str(act)
        val = float(val)
        if current_ctx is None:
            current_ctx = ctx_key
        if ctx_key != current_ctx:
            flush_ctx()
            current_ctx = ctx_key
        items.append((act, val))
    flush_ctx()

    def to_dist(dot, norm):
        out: Dict[Tuple[str, str], float] = {}
        for a in acts:
            ia = idx[a]
            for b in acts:
                ib = idx[b]
                na = math.sqrt(norm[ia])
                nb = math.sqrt(norm[ib])
                if na == 0.0 or nb == 0.0:
                    out[(a, b)] = 1.0 if a != b else 0.0
                else:
                    cos_sim = dot[ia, ib] / (na * nb)
                    out[(a, b)] = 1.0 - cos_sim
        return out

    return to_dist(dot_raw, norm_raw), to_dist(dot_pmi, norm_pmi), to_dist(dot_ppmi, norm_ppmi)


def compute_aa_from_sqlite(
    conn: sqlite3.Connection,
    *,
    activity_freq: Dict[str, float],
) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]]:
    """
    Compute AA / AA-PMI / AA-PPMI distance matrices by streaming contexts and building an |A|x|A| matrix.
    """
    acts = sorted(activity_freq.keys())
    idx = {a: i for i, a in enumerate(acts)}
    n = len(acts)
    aa = np.zeros((n, n), dtype=float)

    cur = conn.cursor()
    cur.execute("SELECT ctx, act, val FROM counts ORDER BY ctx")
    current_ctx = None
    items: List[Tuple[str, float]] = []

    def flush_ctx():
        nonlocal items
        if not items:
            return
        # for each pair in this context: AA[a,b] += v_a + v_b
        for a, va in items:
            ia = idx[a]
            for b, vb in items:
                ib = idx[b]
                aa[ia, ib] += float(va) + float(vb)
        items = []

    for ctx_key, act, val in cur:
        ctx_key = str(ctx_key)
        act = str(act)
        val = float(val)
        if current_ctx is None:
            current_ctx = ctx_key
        if ctx_key != current_ctx:
            flush_ctx()
            current_ctx = ctx_key
        items.append((act, val))
    flush_ctx()

    # Build embeddings dict for deterministic cosine utility
    embeddings = {a: aa[idx[a], :].copy() for a in acts}
    dist_aa = get_cosine_distance_dict(embeddings)
    act_index = {a: i for i, a in enumerate(acts)}
    dist_pmi, _ = get_activity_activity_frequency_matrix_pmi(embeddings, activity_freq, act_index, 0)
    dist_ppmi, _ = get_activity_activity_frequency_matrix_pmi(embeddings, activity_freq, act_index, 1)
    return dist_aa, dist_pmi, dist_ppmi


