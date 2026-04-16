"""Inference and evaluation procedure for the BEST baseline model.

Since BEST only predicts activity suffixes (control-flow), timestamps are
approximated with a constant predictor equal to the training mean TTNE for
every suffix step.  The Remaining Runtime (RRT) prediction for each instance
is then derived as the sum of those constant per-step TTNE values up to
(but not including) the predicted END token step, matching the convention
used by the other baselines in this repository.

Returned metric format (matching the CRTP-LSTM convention):
    [0]  avg_dam_lev               -- avg 1-normalised DL similarity
    [1]  perc_too_early            -- fraction of instances END predicted too early
    [2]  perc_too_late             -- fraction of instances END predicted too late
    [3]  perc_correct              -- fraction of instances END predicted at right step
    [4]  mean_absolute_length_diff -- avg |predicted_len - actual_len|
    [5]  mean_too_early            -- avg # steps too early (for too-early instances)
    [6]  mean_too_late             -- avg # steps too late  (for too-late  instances)
    [7]  avg_MAE_stand_RRT         -- 0.0  (not applicable: no standardised RRT pred)
    [8]  avg_MAE_minutes_RRT       -- avg MAE RRT in minutes (constant-mean predictor)
    [9]  avg_MAE_ttne_minutes      -- avg MAE TTNE in minutes (constant-mean predictor)
    [-2] results_dict_pref         -- per-prefix-length {k: [avg_dl, avg_mae_rrt, n]}
    [-1] results_dict_suf          -- per-suffix-length {k: [avg_dl, avg_mae_rrt, n]}
"""

import os
import torch
from tqdm import tqdm

# Use GPU for the batched DL computation if available; everything else is CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def inference_loop(best_model,
                   inference_dataset,
                   num_categoricals_pref,
                   mean_std_ttne,
                   mean_std_rrt,
                   results_path=None,
                   dl_batch_size=512):
    """Run BEST inference on *inference_dataset* and compute all standard
    evaluation metrics used in the SuTraN benchmark paper.

    Parameters
    ----------
    best_model : BESTModel
        A fitted instance of :class:`BEST.best_model.BESTModel`.
    inference_dataset : tuple of torch.Tensor
        Dataset tuple in the format produced by the preprocessing pipeline.
        The activity label prefix tensor is at index
        ``num_categoricals_pref - 1``, the boolean padding mask at index
        ``num_categoricals_pref + 1``, and labels are stored in the last
        three positions:
        ``[-3]`` TTNE labels ``(N, W, 1)``,
        ``[-2]`` RRT  labels ``(N, W, 1)``,
        ``[-1]`` activity suffix labels ``(N, W)``.
    num_categoricals_pref : int
        Number of categorical features in each prefix event token
        (including the activity label).
    mean_std_ttne : list of float
        ``[mean, std]`` of the TTNE labels in the training set (seconds).
        Used as a constant per-step TTNE predictor and for MAE evaluation.
    mean_std_rrt : list of float
        ``[mean, std]`` of the RRT labels in the training set (seconds).
        Used to de-standardise the RRT ground-truth labels for MAE evaluation.
    results_path : str or None, optional
        Directory in which prediction tensors and metric dictionaries are
        saved.  When ``None`` (default), nothing is written to disk.
    dl_batch_size : int, optional
        Number of instances processed at once in the batched DL computation.
        Reduce if GPU memory is limited.  By default 512.

    Returns
    -------
    list
        Results in the format described in the module docstring.
    """
    # -----------------------------------------------------------------------
    # 1.  Unpack tensors from the dataset
    # -----------------------------------------------------------------------
    pref_act_tensor  = inference_dataset[num_categoricals_pref - 1]  # (N, W)
    padding_mask     = inference_dataset[num_categoricals_pref + 1]  # (N, W) bool
    ttne_labels_raw  = inference_dataset[-3][:, :, 0]                # (N, W)
    rrt_labels_raw   = inference_dataset[-2][:, 0, 0]                # (N,)
    act_labels       = inference_dataset[-1]                          # (N, W)

    N, W = pref_act_tensor.shape
    num_classes  = int(act_labels.max().item()) + 1
    end_token    = num_classes - 1

    # Prefix lengths: number of non-padded positions (padding_mask True = pad)
    pref_lengths = (~padding_mask).sum(dim=1).tolist()  # list[int]

    # 0-based index of the END token in the ground-truth suffix
    actual_lengths = torch.argmax(
        (act_labels == end_token).to(torch.int64), dim=1
    )  # (N,)  0-based

    # -----------------------------------------------------------------------
    # 2.  Generate activity suffix predictions using BEST
    # -----------------------------------------------------------------------
    suffix_acts_pred = torch.zeros(N, W, dtype=torch.int64)

    for i in tqdm(range(N), desc="BEST suffix prediction"):
        pref_len = pref_lengths[i]
        prefix   = pref_act_tensor[i, :pref_len].tolist()
        suffix   = best_model.predict_suffix(prefix, W)
        suffix_acts_pred[i] = torch.tensor(suffix, dtype=torch.int64)

    # 0-based index of the predicted END token for each instance
    pred_end_mask = (suffix_acts_pred == end_token)          # (N, W)
    pred_lengths  = torch.argmax(
        pred_end_mask.to(torch.int64), dim=1
    )  # (N,)
    # If END was never predicted, treat the last position as the END step
    no_end_pred = ~pred_end_mask.any(dim=1)                  # (N,)
    pred_lengths[no_end_pred] = W - 1

    # -----------------------------------------------------------------------
    # 3.  Time predictions: constant predictor = training mean TTNE
    #     BEST is NDA and does not produce timestamp predictions.
    #     We approximate TTNE at every step with the training mean, then
    #     derive the RRT prediction as the sum over non-END steps.
    # -----------------------------------------------------------------------
    ttne_mean_sec = float(mean_std_ttne[0])
    ttne_std_sec  = float(mean_std_ttne[1])

    # Constant TTNE prediction in seconds (all positions = training mean)
    suffix_ttne_preds_sec = torch.full(
        (N, W), fill_value=ttne_mean_sec, dtype=torch.float32
    )

    # Zero out positions after the predicted END token
    counting = torch.arange(W, dtype=torch.int64).unsqueeze(0).expand(N, -1)  # (N, W)
    after_pred_end = counting > pred_lengths.unsqueeze(-1)   # (N, W)
    suffix_ttne_preds_sec[after_pred_end] = 0.0

    # Also zero the END token step itself (convention: do not include TTNE
    # at END position in the RRT sum, matching existing benchmarks)
    batch_idx = torch.arange(N, dtype=torch.int64)
    suffix_ttne_preds_sec[batch_idx, pred_lengths] = 0.0

    # RRT prediction in seconds = sum of per-step TTNE predictions
    rrt_pred_sec = suffix_ttne_preds_sec.sum(dim=1)           # (N,)

    # -----------------------------------------------------------------------
    # 4.  TTNE MAE (constant-mean predictor vs ground-truth labels)
    # -----------------------------------------------------------------------
    ttne_labels_sec = ttne_labels_raw.clone() * ttne_std_sec + ttne_mean_sec
    ttne_labels_sec = ttne_labels_sec.clamp(min=0.0)

    # Evaluate only at actual (ground-truth) suffix positions
    at_or_before_end = counting <= actual_lengths.unsqueeze(-1)  # (N, W)
    MAE_ttne_sec = torch.abs(suffix_ttne_preds_sec - ttne_labels_sec)
    avg_MAE_ttne_minutes = (
        MAE_ttne_sec[at_or_before_end].mean().item() / 60.0
    )

    # -----------------------------------------------------------------------
    # 5.  RRT MAE
    # -----------------------------------------------------------------------
    rrt_std_sec  = float(mean_std_rrt[1])
    rrt_mean_sec = float(mean_std_rrt[0])
    rrt_labels_sec = rrt_labels_raw.clone() * rrt_std_sec + rrt_mean_sec
    rrt_labels_sec = rrt_labels_sec.clamp(min=0.0)

    MAE_rrt_sec      = torch.abs(rrt_pred_sec - rrt_labels_sec)  # (N,)
    avg_MAE_minutes_RRT = MAE_rrt_sec.mean().item() / 60.0
    avg_MAE_stand_RRT   = 0.0  # no meaningful standardised prediction for BEST

    # -----------------------------------------------------------------------
    # 6.  Normalised Damerau-Levenshtein similarity (batched, GPU-accelerated)
    # -----------------------------------------------------------------------
    dam_lev_similarity = _compute_dl_similarity_batched(
        suffix_acts_pred=suffix_acts_pred,
        act_labels=act_labels,
        pred_lengths=pred_lengths,
        actual_lengths=actual_lengths,
        window_size=W,
        batch_size=dl_batch_size
    )  # (N,) float32, CPU

    avg_dam_lev = dam_lev_similarity.mean().item()

    # -----------------------------------------------------------------------
    # 7.  Suffix length statistics
    # -----------------------------------------------------------------------
    length_diff         = pred_lengths - actual_lengths           # (N,)
    too_early_bool      = pred_lengths < actual_lengths
    too_late_bool       = pred_lengths > actual_lengths
    length_diff_early   = length_diff[too_early_bool]
    length_diff_late    = length_diff[too_late_bool]

    total_n        = N
    n_too_early    = too_early_bool.sum().item()
    n_too_late     = too_late_bool.sum().item()
    n_correct      = (length_diff == 0).sum().item()

    perc_too_early           = n_too_early / total_n
    perc_too_late            = n_too_late  / total_n
    perc_correct             = n_correct   / total_n
    mean_absolute_length_diff = length_diff.abs().float().mean().item()
    mean_too_early = (
        length_diff_early.abs().float().mean().item() if n_too_early > 0 else 0.0
    )
    mean_too_late = (
        length_diff_late.abs().float().mean().item() if n_too_late > 0 else 0.0
    )

    # -----------------------------------------------------------------------
    # 8.  Per-prefix-length and per-suffix-length result dictionaries
    # -----------------------------------------------------------------------
    pref_len_tensor = torch.tensor(pref_lengths, dtype=torch.int64)  # (N,)
    suf_len_tensor  = actual_lengths + 1                               # (N,) 1-based
    MAE_rrt_minutes = MAE_rrt_sec / 60.0                               # (N,)

    results_dict_pref = {}
    for k in range(1, W + 1):
        mask = pref_len_tensor == k
        dl_k  = dam_lev_similarity[mask]
        rrt_k = MAE_rrt_minutes[mask]
        n_k   = dl_k.shape[0]
        if n_k > 0:
            results_dict_pref[k] = [
                dl_k.mean().item(),
                rrt_k.mean().item(),
                n_k
            ]

    results_dict_suf = {}
    for k in range(1, W + 1):
        mask = suf_len_tensor == k
        dl_k  = dam_lev_similarity[mask]
        rrt_k = MAE_rrt_minutes[mask]
        n_k   = dl_k.shape[0]
        if n_k > 0:
            results_dict_suf[k] = [
                dl_k.mean().item(),
                rrt_k.mean().item(),
                n_k
            ]

    # -----------------------------------------------------------------------
    # 9.  Optionally persist predictions and metrics to disk
    # -----------------------------------------------------------------------
    if results_path:
        os.makedirs(results_path, exist_ok=True)

        torch.save(
            suffix_acts_pred,
            os.path.join(results_path, 'suffix_acts_decoded.pt')
        )
        torch.save(
            suffix_ttne_preds_sec,
            os.path.join(results_path, 'suffix_ttne_preds.pt')
        )
        torch.save(
            pref_len_tensor.cpu(),
            os.path.join(results_path, 'pref_len.pt')
        )
        torch.save(
            suf_len_tensor.cpu(),
            os.path.join(results_path, 'suf_len.pt')
        )
        torch.save(
            inference_dataset[-3:],
            os.path.join(results_path, 'labels.pt')
        )
        torch.save(
            dam_lev_similarity,
            os.path.join(results_path, 'dam_lev_similarity.pt')
        )
        torch.save(
            MAE_rrt_minutes,
            os.path.join(results_path, 'MAE_rrt_minutes.pt')
        )

    # -----------------------------------------------------------------------
    # 10.  Assemble and return results list
    # -----------------------------------------------------------------------
    return_list = [
        avg_dam_lev,                # [0]
        perc_too_early,             # [1]
        perc_too_late,              # [2]
        perc_correct,               # [3]
        mean_absolute_length_diff,  # [4]
        mean_too_early,             # [5]
        mean_too_late,              # [6]
        avg_MAE_stand_RRT,          # [7]  0.0 for BEST
        avg_MAE_minutes_RRT,        # [8]
        avg_MAE_ttne_minutes,       # [9]
        results_dict_pref,          # [-2]
        results_dict_suf,           # [-1]
    ]
    return return_list


# ---------------------------------------------------------------------------
# Internal helper: batched normalised Damerau-Levenshtein similarity
# ---------------------------------------------------------------------------

def _compute_dl_similarity_batched(suffix_acts_pred,
                                   act_labels,
                                   pred_lengths,
                                   actual_lengths,
                                   window_size,
                                   batch_size=512):
    """Compute the normalised Damerau-Levenshtein similarity for every
    instance using the same batched DP algorithm as
    ``OneStepAheadBenchmarks.inference_environment.BatchInference``.

    Parameters
    ----------
    suffix_acts_pred : torch.Tensor
        Predicted activity suffix, shape ``(N, W)``, dtype int64.
    act_labels : torch.Tensor
        Ground-truth activity suffix labels, shape ``(N, W)``, dtype int64.
    pred_lengths : torch.Tensor
        0-based index of the predicted END token, shape ``(N,)``, int64.
    actual_lengths : torch.Tensor
        0-based index of the ground-truth END token, shape ``(N,)``, int64.
    window_size : int
        ``W`` -- the maximum suffix length.
    batch_size : int
        Number of instances processed per GPU batch.

    Returns
    -------
    torch.Tensor
        DL similarity values, shape ``(N,)``, dtype float32, on CPU.
    """
    N = suffix_acts_pred.shape[0]
    dl_sim_all = torch.zeros(N, dtype=torch.float32)

    for start in tqdm(range(0, N, batch_size), desc="Computing DL similarity"):
        end = min(start + batch_size, N)
        bs  = end - start

        preds_b      = suffix_acts_pred[start:end].to(device)   # (bs, W)
        labels_b     = act_labels[start:end].to(device)          # (bs, W)
        pred_len_b   = (pred_lengths[start:end]   + 1).to(device)  # 1-based (bs,)
        actual_len_b = (actual_lengths[start:end] + 1).to(device)  # 1-based (bs,)
        max_len_b    = torch.maximum(pred_len_b, actual_len_b)   # (bs,)

        # Initialise DP matrix
        d = torch.zeros(
            bs, window_size + 1, window_size + 1,
            dtype=torch.int64, device=device
        )
        arange_t = torch.arange(
            window_size + 1, dtype=torch.int64, device=device
        ).unsqueeze(0)                                           # (1, W+1)
        d[:, 0, :] = arange_t
        d[:, :, 0] = arange_t

        # Fill DP table
        for i in range(1, window_size + 1):
            for j in range(1, window_size + 1):
                cost         = torch.where(
                    preds_b[:, i - 1] == labels_b[:, j - 1], 0, 1
                )
                deletion     = d[:, i - 1, j] + 1
                insertion    = d[:, i, j - 1] + 1
                substitution = d[:, i - 1, j - 1] + cost
                d[:, i, j]   = torch.minimum(
                    torch.minimum(deletion, insertion), substitution
                )
                # Transposition (Damerau extension)
                if i > 1 and j > 1:
                    tpos_possible = (
                        (preds_b[:, i - 1] == labels_b[:, j - 2]) &
                        (preds_b[:, i - 2] == labels_b[:, j - 1])
                    )
                    with_tpos = torch.minimum(
                        d[:, i, j],
                        d[:, i - 2, j - 2] + cost
                    )
                    d[:, i, j] = torch.where(tpos_possible, with_tpos, d[:, i, j])

        # Extract DL distance and normalise
        batch_arange = torch.arange(bs, dtype=torch.int64, device=device)
        dl_dist_b    = (
            d[batch_arange, pred_len_b, actual_len_b].float()
            / max_len_b.float()
        )
        dl_sim_all[start:end] = (1.0 - dl_dist_b).cpu()

    return dl_sim_all
