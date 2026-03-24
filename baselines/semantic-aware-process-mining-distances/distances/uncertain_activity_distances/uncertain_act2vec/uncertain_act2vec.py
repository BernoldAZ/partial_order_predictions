"""
Uncertain Act2Vec (CBOW + Skip-gram) with soft labels for uncertain event logs.

Motivation
----------
The deterministic act2vec implementation in this repo uses `gensim.Word2Vec`, which expects
discrete tokens (one activity label per event). For uncertain event logs, each event is a
probability distribution over activities.

This module implements a *direct probabilistic* variant:
  - each uncertain event e_t is represented by a probability vector p_t over activities
  - the input embedding lookup becomes an expectation: h_t = p_t @ W_in
  - training uses cross-entropy against the *soft* target distribution(s)

Important (critical) note about "expected embeddings"
-----------------------------------------------------
Linearity makes the projection step exact: p_t @ W_in equals E[W_in[a]] if a ~ p_t.

However, the overall Word2Vec objective includes *non-linearities* (softmax / log-softmax),
so training on expected inputs is not generally identical to:
  "train on all deterministic realizations and average the loss"
because log-softmax(E[x]) != E[log-softmax(x)] in general (Jensen).

So this is a principled "learn from uncertainty directly" approach, but it is not guaranteed
to reproduce the same result as exact enumeration of all realizations.

Scope / NA semantics
--------------------
This implementation supports uncertain logs with no NA, i.e. fixed-position event sequences.
If your XES contains an NA label that means "event absent", then correctly modeling variable-
length realizations requires window-shift semantics (as implemented for the count-based methods).
Here, we provide a pragmatic handling:
  - NA mass contributes a zero vector to h_t (i.e. drops signal proportionally)
  - we can optionally drop NA and renormalize to condition on "event happened"
This is *not* equivalent to the count-based NA shifting semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import time

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Uncertain act2vec requires PyTorch. Install torch (see README CUDA note if needed)."
    ) from e

from uncertain_utils.uncertain_xes_reader import UncertainEventLog
from distances.uncertain_activity_distances.data_util.uncertain_expected_counts import prune_distribution
from distances.uncertain_activity_distances.data_util.uncertain_window_based_expected_counts import (
    PAD_TOKEN as DEFAULT_PAD_TOKEN,
    _enumerate_side_contexts,
    _options,
    _pad_trace,
)


def _cosine_distance(u: np.ndarray, v: np.ndarray) -> float:
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return 1.0
    return float(1.0 - (u @ v) / (nu * nv))


def _alphabet_from_log(
    log: UncertainEventLog,
    *,
    na_label: str,
    exclude_activities: Optional[set[str]],
) -> List[str]:
    acts: set[str] = set()
    ex = exclude_activities or set()
    for tr in log.traces:
        for ev in tr.events:
            for a in ev.activity_probs.keys():
                if a == na_label:
                    continue
                if a in ex:
                    continue
                acts.add(a)
    return sorted(acts)


def _probs_dict_to_dense(
    probs: Dict[str, float],
    *,
    act_to_idx: Dict[str, int],
    top_k: Optional[int],
    min_prob: float,
    na_label: str,
    drop_na_and_renormalize: bool,
) -> np.ndarray:
    """
    Convert a sparse distribution dict into a dense probability vector aligned with act_to_idx.

    We also support pruning (top_k / min_prob) to speed up training when distributions are dense.
    """
    pruned_list = prune_distribution(probs, top_k=top_k, min_prob=min_prob)
    if not pruned_list:
        return np.zeros((len(act_to_idx),), dtype=np.float32)

    # Optionally drop NA and renormalize to condition on "event happened"
    if drop_na_and_renormalize:
        pruned_list = [(a, p) for (a, p) in pruned_list if a != na_label]
        s = float(sum(p for (_a, p) in pruned_list))
        if s > 0:
            pruned_list = [(a, float(p) / s) for (a, p) in pruned_list]

    vec = np.zeros((len(act_to_idx),), dtype=np.float32)
    for a, p in pruned_list:
        if a == na_label:
            # If not renormalizing, NA simply contributes no embedding mass.
            continue
        idx = act_to_idx.get(a)
        if idx is None:
            continue
        try:
            vec[idx] = float(p)
        except Exception:
            continue
    return vec


@dataclass(frozen=True)
class UncertainAct2VecConfig:
    window_size: int = 3
    embedding_dim: int = 16
    epochs: int = 10
    batch_size: int = 256
    # Match deterministic act2vec in this repo (`distances/activity_distances/de_koninck_2018_act2vec/algorithm.py`):
    # start_alpha=0.025, manual decrement 0.002 per epoch, min_alpha=0.0001.
    start_alpha: float = 0.025
    alpha_decay_per_epoch: float = 0.002
    min_alpha: float = 0.0001
    seed: int = 0
    device: str = "cpu"  # "cpu" or "cuda"

    # Uncertain-specific knobs
    # If `training_mode == "window_realizations"`, these are used to prune dense per-event distributions.
    # Set prob_threshold=0.0 for exact enumeration (no pruning).
    training_mode: str = "window_realizations"  # "window_realizations" (exact NA semantics) | "soft_labels"
    prob_threshold: float = 0.0  # drop labels with p < prob_threshold, then renormalize (0.0 = exact)
    top_k: int | None = None
    min_prob: float = 0.0
    na_label: str = "NA"
    pad_token: str = DEFAULT_PAD_TOKEN
    drop_na_and_renormalize: bool = False
    exclude_activities: set[str] | None = None

    # Word2Vec objective: negative sampling (gensim default negative=5, ns exponent ~0.75)
    negative: int = 5
    negative_sampling_exponent: float = 0.75

    # Progress
    progress_every_samples: int = 50_000


class _SoftmaxWord2Vec(nn.Module):
    """
    Two-matrix Word2Vec-style model with full softmax output:
      - W_in: |A| x D
      - W_out: |A| x D
    """

    def __init__(self, vocab_size: int, dim: int) -> None:
        super().__init__()
        self.W_in = nn.Parameter(torch.randn(vocab_size, dim) * 0.01)
        self.W_out = nn.Parameter(torch.randn(vocab_size, dim) * 0.01)

    def logits_from_hidden(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, D) -> logits: (B, |A|)
        return h @ self.W_out.t()


def _soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy with soft targets (targets are distributions that sum to 1, but we don't enforce it).
    Returns a scalar mean loss over batch.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -(targets * log_probs).sum(dim=-1)
    return loss.mean()


def _weighted_cross_entropy_with_indices(
    logits: torch.Tensor, target_idx: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """
    Weighted cross-entropy where targets are class indices (one-hot labels),
    and `weights` gives a non-negative weight per example.
    """
    per = F.cross_entropy(logits, target_idx, reduction="none")
    w = weights.clamp_min(0.0)
    denom = w.sum().clamp_min(1e-12)
    return (per * w).sum() / denom


def _weighted_negative_sampling_loss(
    *,
    h: torch.Tensor,  # (B,D)
    pos_idx: torch.Tensor,  # (B,)
    neg_idx: torch.Tensor,  # (B,K)
    W_out: torch.Tensor,  # (V,D)
    weights: torch.Tensor,  # (B,)
) -> torch.Tensor:
    """
    Weighted SGNS-style objective:
      -log σ(h·v_pos) - Σ_k log σ(-h·v_negk)
    averaged with probability weights.
    """
    # (B,D)
    v_pos = W_out.index_select(0, pos_idx)
    pos_score = (h * v_pos).sum(dim=-1)  # (B,)
    pos_loss = F.softplus(-pos_score)  # -log(sigmoid(pos_score))

    # (B,K,D)
    v_neg = W_out.index_select(0, neg_idx.reshape(-1)).reshape(neg_idx.shape[0], neg_idx.shape[1], -1)
    neg_score = (h.unsqueeze(1) * v_neg).sum(dim=-1)  # (B,K)
    neg_loss = F.softplus(neg_score).sum(dim=-1)  # Σ -log(sigmoid(-neg_score))

    per = pos_loss + neg_loss
    w = weights.clamp_min(0.0)
    denom = w.sum().clamp_min(1e-12)
    return (per * w).sum() / denom


def _expected_activity_unigram(
    log: UncertainEventLog,
    *,
    act_to_idx: Dict[str, int],
    cfg: UncertainAct2VecConfig,
) -> np.ndarray:
    """
    Compute an expected activity frequency vector for negative-sampling.

    For each event distribution, we optionally threshold+renormalize (prob_threshold),
    then add probabilities to expected counts.
    """
    counts = np.zeros((len(act_to_idx),), dtype=np.float64)
    exclude = set(cfg.exclude_activities or set())
    exclude.add(cfg.pad_token)

    for tr in log.traces:
        for ev in tr.events:
            opts = _options(ev.activity_probs, prob_threshold=cfg.prob_threshold, renormalize=True, label_map=None)
            for a, p in opts:
                if a == cfg.na_label or a in exclude:
                    continue
                i = act_to_idx.get(a)
                if i is None:
                    continue
                counts[i] += float(p)

    # Smooth: if everything is zero, fall back to uniform
    s = float(counts.sum())
    if s <= 0.0:
        return np.ones_like(counts) / max(1, len(counts))
    return counts / s


def _iter_weighted_window_realizations(
    log: UncertainEventLog,
    *,
    act_to_idx: Dict[str, int],
    cfg: UncertainAct2VecConfig,
) -> Iterator[Tuple[int, List[int], float]]:
    """
    Yield (center_idx, context_indices, weight) for all window realizations in the log.

    This is the *exact* NA-semantics analogue of the count-based window enumeration:
    - center resolving to NA produces no sample
    - NA events in context are skipped, and we extend outward until we collected k activities
    - PAD is used internally to guarantee k activities at boundaries but is excluded from vocabulary
      (thus PAD positions simply don't contribute to samples).

    Weight equals the probability mass of that realized window (under per-event independence).
    """
    if cfg.window_size < 1 or cfg.window_size % 2 == 0:
        raise ValueError("window_size must be odd (3,5,9,...)")
    k = cfg.window_size // 2

    exclude = set(cfg.exclude_activities or set())
    exclude.add(cfg.pad_token)

    for tr in log.traces:
        trace_probs = [ev.activity_probs for ev in tr.events]
        padded = _pad_trace(trace_probs, pad_left=k, pad_right=k, pad_token=cfg.pad_token)
        padded_opts = [
            _options(d, prob_threshold=cfg.prob_threshold, renormalize=True, label_map=None) for d in padded
        ]

        for center_pos in range(k, k + len(tr.events)):
            center_opts = padded_opts[center_pos]
            left_outward = [padded_opts[center_pos - i] for i in range(1, center_pos + 1)]
            right_outward = [padded_opts[center_pos + i] for i in range(1, len(padded_opts) - center_pos)]

            left_dist = _enumerate_side_contexts(left_outward, needed=k, na_label=cfg.na_label, direction="left")
            right_dist = _enumerate_side_contexts(right_outward, needed=k, na_label=cfg.na_label, direction="right")

            for a, p_a in center_opts:
                if p_a <= 0.0:
                    continue
                if a == cfg.na_label or a in exclude:
                    continue
                a_idx = act_to_idx.get(a)
                if a_idx is None:
                    continue

                for l_seq, p_l in left_dist.items():
                    for r_seq, p_r in right_dist.items():
                        w = float(p_a * p_l * p_r)
                        if w <= 0.0:
                            continue
                        ctx = []
                        for tok in (l_seq + r_seq):
                            if tok == cfg.pad_token:
                                continue
                            if tok == cfg.na_label:
                                # should not occur in completed sequences, but keep defensive
                                continue
                            if tok in exclude:
                                continue
                            ti = act_to_idx.get(tok)
                            if ti is None:
                                continue
                            ctx.append(ti)
                        if not ctx:
                            continue
                        yield a_idx, ctx, w


def _precompute_window_distributions(
    log: UncertainEventLog,
    *,
    act_to_idx: Dict[str, int],
    cfg: UncertainAct2VecConfig,
    progress: Optional[callable],
) -> List[List[Tuple[List[Tuple[int, float]], Dict[Tuple[str, ...], float], Dict[Tuple[str, ...], float]]]]:
    """
    Precompute per-trace per-center distributions to avoid recomputing them each epoch.

    Returns for each trace a list of centers. Each center entry is:
      (center_options, left_dist, right_dist)
    where center_options is a list of (activity_index, prob).
    """
    if cfg.window_size < 1 or cfg.window_size % 2 == 0:
        raise ValueError("window_size must be odd (3,5,9,...)")
    k = cfg.window_size // 2

    exclude = set(cfg.exclude_activities or set())
    exclude.add(cfg.pad_token)

    all_traces: List[List[Tuple[List[Tuple[int, float]], Dict[Tuple[str, ...], float], Dict[Tuple[str, ...], float]]]] = []
    for trace_idx, tr in enumerate(log.traces, start=1):
        if progress is not None:
            progress(f"[uncertain-act2vec] precompute trace {trace_idx}/{len(log.traces)} events={len(tr.events)}")

        trace_probs = [ev.activity_probs for ev in tr.events]
        padded = _pad_trace(trace_probs, pad_left=k, pad_right=k, pad_token=cfg.pad_token)
        padded_opts = [
            _options(d, prob_threshold=cfg.prob_threshold, renormalize=True, label_map=None) for d in padded
        ]

        centers: List[Tuple[List[Tuple[int, float]], Dict[Tuple[str, ...], float], Dict[Tuple[str, ...], float]]] = []
        for center_pos in range(k, k + len(tr.events)):
            center_opts_raw = padded_opts[center_pos]
            center_opts: List[Tuple[int, float]] = []
            for a, p_a in center_opts_raw:
                if p_a <= 0.0:
                    continue
                if a == cfg.na_label or a in exclude:
                    continue
                ai = act_to_idx.get(a)
                if ai is None:
                    continue
                center_opts.append((ai, float(p_a)))
            if not center_opts:
                continue

            left_outward = [padded_opts[center_pos - i] for i in range(1, center_pos + 1)]
            right_outward = [padded_opts[center_pos + i] for i in range(1, len(padded_opts) - center_pos)]
            left_dist = _enumerate_side_contexts(left_outward, needed=k, na_label=cfg.na_label, direction="left")
            right_dist = _enumerate_side_contexts(right_outward, needed=k, na_label=cfg.na_label, direction="right")
            centers.append((center_opts, left_dist, right_dist))

        all_traces.append(centers)
    return all_traces


def _build_dense_traces(
    log: UncertainEventLog,
    act_to_idx: Dict[str, int],
    *,
    cfg: UncertainAct2VecConfig,
) -> List[List[np.ndarray]]:
    dense_traces: List[List[np.ndarray]] = []
    for tr in log.traces:
        seq: List[np.ndarray] = []
        for ev in tr.events:
            vec = _probs_dict_to_dense(
                ev.activity_probs,
                act_to_idx=act_to_idx,
                top_k=cfg.top_k,
                min_prob=cfg.min_prob,
                na_label=cfg.na_label,
                drop_na_and_renormalize=cfg.drop_na_and_renormalize,
            )
            # Skip "empty" events (e.g. if everything got pruned away)
            if float(vec.sum()) <= 0.0:
                continue
            seq.append(vec)
        if seq:
            dense_traces.append(seq)
    return dense_traces


def _iter_cbow_samples(
    dense_traces: Sequence[Sequence[np.ndarray]],
    k: int,
) -> Iterable[Tuple[np.ndarray, List[np.ndarray]]]:
    for seq in dense_traces:
        n = len(seq)
        for t in range(n):
            ctx: List[np.ndarray] = []
            for j in range(-k, k + 1):
                if j == 0:
                    continue
                u = t + j
                if 0 <= u < n:
                    ctx.append(seq[u])
            if not ctx:
                continue
            yield seq[t], ctx


def _iter_sg_samples(
    dense_traces: Sequence[Sequence[np.ndarray]],
    k: int,
) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    for seq in dense_traces:
        n = len(seq)
        for t in range(n):
            center = seq[t]
            for j in range(-k, k + 1):
                if j == 0:
                    continue
                u = t + j
                if 0 <= u < n:
                    yield center, seq[u]


def _train_cbow(
    model: _SoftmaxWord2Vec,
    dense_traces: Sequence[Sequence[np.ndarray]],
    *,
    cfg: UncertainAct2VecConfig,
) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device(cfg.device)
    model.to(device)

    samples = list(_iter_cbow_samples(dense_traces, k=cfg.window_size))
    if not samples:
        return

    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    for _epoch in range(cfg.epochs):
        np.random.shuffle(samples)
        for i in range(0, len(samples), cfg.batch_size):
            batch = samples[i : i + cfg.batch_size]
            centers = np.stack([c for (c, _ctx) in batch], axis=0)  # (B, A)
            ctx_lists = [ctx for (_c, ctx) in batch]
            max_c = max(len(ctx) for ctx in ctx_lists)
            # pad contexts to (B, C, A)
            ctx = np.zeros((len(batch), max_c, centers.shape[1]), dtype=np.float32)
            mask = np.zeros((len(batch), max_c), dtype=np.float32)
            for bi, lst in enumerate(ctx_lists):
                for ci, vec in enumerate(lst):
                    ctx[bi, ci, :] = vec
                    mask[bi, ci] = 1.0

            centers_t = torch.tensor(centers, device=device)
            ctx_t = torch.tensor(ctx, device=device)
            mask_t = torch.tensor(mask, device=device)  # (B,C)

            # expected embeddings of each context event: (B,C,A) @ (A,D) = (B,C,D)
            h_ctx = ctx_t @ model.W_in
            # average over real context positions
            denom = mask_t.sum(dim=1, keepdim=True).clamp_min(1.0)  # (B,1)
            H = (h_ctx * mask_t.unsqueeze(-1)).sum(dim=1) / denom  # (B,D)

            logits = model.logits_from_hidden(H)
            loss = _soft_cross_entropy(logits, centers_t)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()


def _train_sg(
    model: _SoftmaxWord2Vec,
    dense_traces: Sequence[Sequence[np.ndarray]],
    *,
    cfg: UncertainAct2VecConfig,
) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device(cfg.device)
    model.to(device)

    samples = list(_iter_sg_samples(dense_traces, k=cfg.window_size))
    if not samples:
        return

    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    for _epoch in range(cfg.epochs):
        np.random.shuffle(samples)
        for i in range(0, len(samples), cfg.batch_size):
            batch = samples[i : i + cfg.batch_size]
            centers = np.stack([c for (c, _o) in batch], axis=0)  # (B, A)
            outs = np.stack([o for (_c, o) in batch], axis=0)  # (B, A)

            centers_t = torch.tensor(centers, device=device)
            outs_t = torch.tensor(outs, device=device)

            # center expected embedding: (B,A) @ (A,D) -> (B,D)
            h = centers_t @ model.W_in
            logits = model.logits_from_hidden(h)
            loss = _soft_cross_entropy(logits, outs_t)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()


def get_uncertain_act2vec_distance_matrix(
    log: UncertainEventLog,
    *,
    sg: int,
    config: UncertainAct2VecConfig,
    progress: Optional[callable] = None,
) -> Tuple[Dict[Tuple[str, str], float], Dict[str, np.ndarray], dict]:
    """
    Train uncertain act2vec and return pairwise cosine distances + embedding dict.

    Parameters
    ----------
    sg:
        0 = CBOW, 1 = Skip-gram (same convention as gensim).
    config:
        Training and preprocessing configuration.

    Returns
    -------
    distances:
        dict[(a,a')] -> cosine distance
    embeddings:
        dict[a] -> np.ndarray embedding (W_in row)
    debug:
        small dict with metadata (alphabet, act_to_idx, etc.)
    """
    if sg not in (0, 1):
        raise ValueError("sg must be 0 (CBOW) or 1 (Skip-gram)")

    alphabet = _alphabet_from_log(log, na_label=config.na_label, exclude_activities=config.exclude_activities)
    if not alphabet:
        return {}, {}, {"alphabet": [], "reason": "empty alphabet"}

    act_to_idx = {a: i for i, a in enumerate(alphabet)}
    model = _SoftmaxWord2Vec(vocab_size=len(alphabet), dim=config.embedding_dim)

    # Default: exact window-realization training (matches NA shifting semantics conceptually)
    if config.training_mode == "window_realizations":
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        device = torch.device(config.device)
        model.to(device)
        # Match deterministic act2vec in this repo: SGD with epoch-wise alpha schedule.
        opt = torch.optim.SGD(model.parameters(), lr=config.start_alpha)

        t_pre0 = time.time()
        pre = _precompute_window_distributions(log, act_to_idx=act_to_idx, cfg=config, progress=progress)
        if progress is not None:
            progress(
                f"[uncertain-act2vec] precompute done in {time.time()-t_pre0:.2f}s "
                f"(traces={len(pre)}, epochs={config.epochs})"
            )

        # Negative sampling distribution (Word2Vec default style)
        unigram = _expected_activity_unigram(log, act_to_idx=act_to_idx, cfg=config)
        # ns distribution ∝ unigram^{0.75}
        ns = unigram ** float(config.negative_sampling_exponent)
        ns = ns / max(1e-12, float(ns.sum()))
        ns_t = torch.tensor(ns, device=device, dtype=torch.float32)

        # Rough denominator for per-epoch progress (exact enough for progress reporting)
        # For CBOW: #windows ≈ Σ centers (|center_opts| * |L| * |R|)
        # For SG: multiply by (2k) since each window expands to up to 2k targets
        approx_windows = 0
        for centers in pre:
            for center_opts, left_dist, right_dist in centers:
                approx_windows += len(center_opts) * len(left_dist) * len(right_dist)
        approx_samples_per_epoch = int(approx_windows if sg == 0 else approx_windows * max(1, (config.window_size - 1)))
        if progress is not None:
            progress(
                f"[uncertain-act2vec] approx samples/epoch={approx_samples_per_epoch:,} "
                f"(windows≈{approx_windows:,}, sg={sg})"
            )

        # Accumulate into batches.
        # For CBOW: batch sample is (center_idx, context_indices, weight).
        # For Skip-gram: we expand each realized window into multiple (center_idx, out_idx, weight) pairs.
        batch_centers: List[int] = []
        batch_ctx: List[List[int]] = []
        batch_w: List[float] = []
        batch_out: List[int] = []
        sample_counter = 0
        t0 = time.time()
        epoch_sample_counter = 0

        def flush_batch() -> None:
            nonlocal sample_counter
            nonlocal epoch_sample_counter
            if not batch_centers:
                return
            B = len(batch_centers)
            weights_t = torch.tensor(batch_w, device=device, dtype=torch.float32)

            if sg == 0:
                max_c = max(len(x) for x in batch_ctx)
                ctx_idx = np.full((B, max_c), fill_value=-1, dtype=np.int64)
                mask = np.zeros((B, max_c), dtype=np.float32)
                for i, lst in enumerate(batch_ctx):
                    for j, tok in enumerate(lst):
                        ctx_idx[i, j] = tok
                        mask[i, j] = 1.0

                centers_t = torch.tensor(batch_centers, device=device, dtype=torch.long)
                ctx_idx_t = torch.tensor(ctx_idx, device=device, dtype=torch.long)
                mask_t = torch.tensor(mask, device=device, dtype=torch.float32)

                safe_idx = torch.clamp(ctx_idx_t, min=0)
                emb = model.W_in.index_select(0, safe_idx.reshape(-1)).reshape(B, max_c, -1)
                emb = emb * mask_t.unsqueeze(-1)
                denom = mask_t.sum(dim=1, keepdim=True).clamp_min(1.0)
                H = emb.sum(dim=1) / denom

                if config.negative and config.negative > 0:
                    neg_idx = torch.multinomial(ns_t, num_samples=B * int(config.negative), replacement=True).reshape(
                        B, int(config.negative)
                    )
                    loss = _weighted_negative_sampling_loss(
                        h=H, pos_idx=centers_t, neg_idx=neg_idx, W_out=model.W_out, weights=weights_t
                    )
                else:
                    logits = model.logits_from_hidden(H)
                    loss = _weighted_cross_entropy_with_indices(logits, centers_t, weights_t)
            else:
                # Skip-gram: center embedding predicts a single output token per sample
                centers_t = torch.tensor(batch_centers, device=device, dtype=torch.long)
                outs_t = torch.tensor(batch_out, device=device, dtype=torch.long)
                h = model.W_in.index_select(0, centers_t)
                if config.negative and config.negative > 0:
                    neg_idx = torch.multinomial(ns_t, num_samples=B * int(config.negative), replacement=True).reshape(
                        B, int(config.negative)
                    )
                    loss = _weighted_negative_sampling_loss(
                        h=h, pos_idx=outs_t, neg_idx=neg_idx, W_out=model.W_out, weights=weights_t
                    )
                else:
                    logits = model.logits_from_hidden(h)
                    loss = _weighted_cross_entropy_with_indices(logits, outs_t, weights_t)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            epoch_sample_counter += B
            batch_centers.clear()
            batch_ctx.clear()
            batch_w.clear()
            batch_out.clear()

        for _epoch in range(config.epochs):
            if progress is not None:
                progress(f"[uncertain-act2vec] epoch {_epoch+1}/{config.epochs} (alpha={opt.param_groups[0]['lr']:.4f}) ...")
            epoch_sample_counter = 0

            for centers in pre:
                for center_opts, left_dist, right_dist in centers:
                    for a_idx, p_a in center_opts:
                        for l_seq, p_l in left_dist.items():
                            for r_seq, p_r in right_dist.items():
                                w = float(p_a * p_l * p_r)
                                if w <= 0.0:
                                    continue
                                ctx = []
                                for tok in (l_seq + r_seq):
                                    if tok == config.pad_token or tok == config.na_label:
                                        continue
                                    ti = act_to_idx.get(tok)
                                    if ti is not None:
                                        ctx.append(ti)
                                if not ctx:
                                    continue

                                if sg == 0:
                                    batch_centers.append(a_idx)
                                    batch_ctx.append(ctx)
                                    batch_w.append(w)
                                    sample_counter += 1
                                else:
                                    # Expand to multiple targets (one per context token)
                                    for out_idx in ctx:
                                        batch_centers.append(a_idx)
                                        batch_out.append(out_idx)
                                        batch_w.append(w)
                                        sample_counter += 1

                                if len(batch_centers) >= config.batch_size:
                                    flush_batch()

                                if progress is not None and config.progress_every_samples > 0:
                                    if sample_counter % config.progress_every_samples == 0:
                                        dt = max(1e-9, time.time() - t0)
                                        # Relative epoch progress (approx denominator)
                                        rel = (
                                            min(1.0, float(epoch_sample_counter) / float(max(1, approx_samples_per_epoch)))
                                            if approx_samples_per_epoch > 0
                                            else 0.0
                                        )
                                        progress(
                                            f"[uncertain-act2vec] samples={sample_counter:,} "
                                            f"rate={sample_counter/dt:,.1f}/s "
                                            f"epoch={_epoch+1}/{config.epochs} "
                                            f"epoch_progress≈{rel*100:.1f}% "
                                            f"(sg={sg}, |A|={len(alphabet)}, dim={config.embedding_dim})"
                                        )
            flush_batch()

            # End-of-epoch progress update
            if progress is not None and approx_samples_per_epoch > 0:
                progress(
                    f"[uncertain-act2vec] epoch {_epoch+1}/{config.epochs} done "
                    f"(epoch_samples={epoch_sample_counter:,}, approx_total={approx_samples_per_epoch:,})"
                )

            # Manual alpha schedule to match deterministic implementation
            new_lr = max(config.min_alpha, opt.param_groups[0]["lr"] - config.alpha_decay_per_epoch)
            opt.param_groups[0]["lr"] = float(new_lr)

    # Optional alternative: direct soft-label training on fixed-position event distributions (not NA-shift exact)
    elif config.training_mode == "soft_labels":
        dense_traces = _build_dense_traces(log, act_to_idx, cfg=config)
        if not dense_traces:
            return {}, {}, {"alphabet": alphabet, "reason": "no usable traces after pruning"}
        if sg == 0:
            _train_cbow(model, dense_traces, cfg=config)
        else:
            _train_sg(model, dense_traces, cfg=config)
    else:
        raise ValueError("training_mode must be 'window_realizations' or 'soft_labels'")

    W_in = model.W_in.detach().cpu().numpy()
    embeddings = {a: W_in[act_to_idx[a]].copy() for a in alphabet}

    distances: Dict[Tuple[str, str], float] = {}
    for a1 in alphabet:
        for a2 in alphabet:
            distances[(a1, a2)] = _cosine_distance(embeddings[a1], embeddings[a2])

    debug = {
        "alphabet": alphabet,
        "act_to_idx": act_to_idx,
        "sg": sg,
        "training_mode": config.training_mode,
        "embedding_dim": config.embedding_dim,
        "window_size": config.window_size,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "prob_threshold": config.prob_threshold,
        "top_k": config.top_k,
        "min_prob": config.min_prob,
        "na_label": config.na_label,
        "pad_token": config.pad_token,
        "drop_na_and_renormalize": config.drop_na_and_renormalize,
    }
    return distances, embeddings, debug


