"""Train and evaluate GATv2EncoderGRUDecoderNewBlockV3 for activity
suffix + TTNE prediction.  Same architecture as v1; the only difference
is that training is next-activity prediction only: a single decoder step
supervised against the first suffix position (no scheduled sampling, no
full-window teacher forcing). Test-time evaluation is unchanged: full
suffix autoregressive rollout, same metric suite as v1/v2.

Usage
-----
    python run_suffix_time_v3.py <log_name> [results_dir]

Loads pre-built datasets from results_per_log/<log_name>/:
    train_graphdataset.pt, val_graphdataset.pt, test_graphdataset.pt
    <log_name>_train_means_dict.pkl, <log_name>_train_std_dict.pkl

Run create_general_data.py first to generate these files.
"""
import argparse
import csv
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from model_suffix_time_v3 import GATv2EncoderGRUDecoderNewBlockV3
from partial_order_metrics import ges_compute_sample, bucket_order_similarity

# ─── Hyperparameters ──────────────────────────────────────────────────────────
D_MODEL     = 64
DROPOUT     = 0.4
N_GRU_LAYERS = 1
N_GNN_LAYERS = 2
LR          = 0.002
MAX_EPOCHS  = 200
PATIENCE    = 24
LR_PATIENCE = 10
MAX_NORM    = 2.0
BATCH_SIZE  = 128

# ─── New-block loss weight ────────────────────────────────────────────────────
NB_WEIGHT = 1.0   # scalar multiplier on the new-block BCE loss term

METHOD_NAME = 'gatv2_gru_nb_v3'
GES_WORKERS = 4


# ─── Loss ─────────────────────────────────────────────────────────────────────

def _masked_ce(act_logits, act_targets, num_activities):
    return F.cross_entropy(
        act_logits.view(-1, num_activities),
        act_targets.view(-1),
        ignore_index=0,
    )


def _masked_mae(ttne_preds, ttne_targets):
    p    = ttne_preds.view(-1)
    t    = ttne_targets.view(-1)
    mask = (t != -100).float()
    return (torch.abs(p - t) * mask).sum() / mask.sum().clamp(min=1)


def _new_block_bce(nb_logits, act_targets, nb_targets, num_activities, nb_pos_weight):
    """BCE over real (non-padding, non-EOS) positions; target from new_block_label."""
    logits  = nb_logits.view(-1)
    targets = nb_targets.view(-1)
    acts    = act_targets.view(-1)
    is_real    = (acts != 0) & (acts != num_activities - 1)
    pos_weight = torch.tensor([nb_pos_weight], device=logits.device)
    raw = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction='none')
    return raw[is_real].mean()


def _loss(act_logits, ttne_preds, nb_logits,
          data, num_activities, nb_pos_weight, window_size):
    """act_logits/ttne_preds/nb_logits cover only suffix position 0 (next-activity
    prediction), so targets are sliced to position 0 to match."""
    B = data.num_graphs
    act_target0  = data.act_label_seq.view(B, window_size)[:, :1]
    ttne_target0 = data.ttnext_label.view(B, window_size, -1)[:, :1, :]
    nb_target0   = data.new_block_label.view(B, window_size)[:, :1]

    ce  = _masked_ce(act_logits, act_target0, num_activities)
    mae = _masked_mae(ttne_preds, ttne_target0)
    nb  = _new_block_bce(nb_logits, act_target0, nb_target0,
                         num_activities, nb_pos_weight)
    return ce + mae + NB_WEIGHT * nb


# ─── Metrics ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def _compute_metrics(suffix_acts, suffix_ttne, suffix_nb,
                     act_labels, ttne_labels, rrt_labels, nb_labels,
                     num_activities, mean_std_ttne, mean_std_rrt,
                     compute_ges=False, compute_first_step=False):
    device  = suffix_acts.device
    N, W    = suffix_acts.shape
    end_tok = num_activities - 1
    ttne_mean, ttne_std = mean_std_ttne
    rrt_mean,  rrt_std  = mean_std_rrt

    actual_length = (act_labels == end_tok).to(torch.int64).argmax(dim=-1)

    has_end     = (suffix_acts == end_tok).any(dim=-1)
    first_end   = (suffix_acts == end_tok).to(torch.int64).argmax(dim=-1)
    pred_length = torch.where(has_end, first_end, torch.full_like(first_end, W - 1))

    counting    = torch.arange(W, device=device).unsqueeze(0)
    batch_range = torch.arange(N, device=device)

    # ── Damerau-Levenshtein similarity ────────────────────────────────────────
    len_pred   = pred_length      # activities only, EOS excluded
    len_actual = actual_length
    max_len    = torch.maximum(len_pred, len_actual).clamp(min=1).float()

    d  = torch.full((N, W + 1, W + 1), fill_value=0, dtype=torch.int64, device=device)
    ar = torch.arange(W + 1, device=device).unsqueeze(0)
    d[:, 0, :] = ar
    d[:, :, 0] = ar
    for i in range(1, W + 1):
        for j in range(1, W + 1):
            cost         = torch.where(suffix_acts[:, i-1] == act_labels[:, j-1], 0, 1)
            deletion     = d[:, i-1, j]   + 1
            insertion    = d[:, i, j-1]   + 1
            substitution = d[:, i-1, j-1] + cost
            d[:, i, j]  = torch.minimum(torch.minimum(deletion, insertion), substitution)
            if i > 1 and j > 1:
                tpos_true   = (
                    (suffix_acts[:, i-1] == act_labels[:, j-2]) &
                    (suffix_acts[:, i-2] == act_labels[:, j-1])
                )
                min_og_tpos = torch.minimum(d[:, i, j], d[:, i-2, j-2] + cost)
                d[:, i, j]  = torch.where(tpos_true, min_og_tpos, d[:, i, j])
    dl_per_inst = 1.0 - d[batch_range, len_pred, len_actual].float() / max_len
    dl_sim = dl_per_inst.mean().item()

    # ── TTNE MAE ──────────────────────────────────────────────────────────────
    ttne_preds_sec  = (suffix_ttne.clone() * ttne_std + ttne_mean).clamp(min=0)
    ttne_labels_sec = (ttne_labels.clone() * ttne_std + ttne_mean).clamp(min=0)
    ttne_preds_sec[counting > pred_length.unsqueeze(-1)] = 0.0

    before_end   = counting <= actual_length.unsqueeze(-1)
    ttne_abs     = torch.abs(ttne_preds_sec - ttne_labels_sec)
    ttne_mae_sec = ttne_abs[before_end].mean().item()
    ttne_mae_per_inst_min = (
        (ttne_abs * before_end.float()).sum(dim=-1)
        / before_end.sum(dim=-1).clamp(min=1)
    ) / 60.0

    # ── RRT MAE ───────────────────────────────────────────────────────────────
    rrt_preds_sec = ttne_preds_sec.clone()
    rrt_preds_sec[batch_range, pred_length] = 0.0
    rrt_preds_sec = rrt_preds_sec.sum(dim=-1)

    rrt_label_sec = (rrt_labels[:, 0] * rrt_std + rrt_mean).clamp(min=0)
    rrt_mae_per_inst = torch.abs(rrt_preds_sec - rrt_label_sec)
    rrt_mae_sec      = rrt_mae_per_inst.mean().item()

    mean_pred_len   = (pred_length   + 1).float().mean().item()
    mean_actual_len = (actual_length + 1).float().mean().item()

    # ── New-block F1 and accuracy ──────────────────────────────────────────────
    is_real = (act_labels != 0) & (act_labels != end_tok)              # (N, W)

    after_pred_eos = counting > pred_length.unsqueeze(-1)              # (N, W)
    nb_clean       = suffix_nb.clone()
    nb_clean[after_pred_eos] = 0.0

    nb_pred = (nb_clean[is_real] > 0.5).long()
    nb_true = (nb_labels[is_real] > 0.5).long()

    tp = ((nb_pred == 1) & (nb_true == 1)).sum().item()
    fp = ((nb_pred == 1) & (nb_true == 0)).sum().item()
    fn = ((nb_pred == 0) & (nb_true == 1)).sum().item()
    tn = ((nb_pred == 0) & (nb_true == 0)).sum().item()
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    nb_f1     = 2 * precision * recall / max(precision + recall, 1e-8)
    nb_acc    = (nb_pred == nb_true).float().mean().item() if nb_pred.numel() > 0 else 0.0

    # ── First-step accuracy and weighted F1 ──────────────────────────────────
    if compute_first_step:
        fp0   = suffix_acts[:, 0]
        ft0   = act_labels[:, 0]
        mask0 = ft0 != 0
        fp0, ft0  = fp0[mask0], ft0[mask0]
        first_acc = (fp0 == ft0).float().mean().item() if fp0.numel() > 0 else 0.0
        total     = ft0.numel()
        wf1_sum   = 0.0
        for c in ft0.unique():
            support = (ft0 == c).sum().item()
            tp_c    = ((fp0 == c) & (ft0 == c)).sum().item()
            fp_c    = ((fp0 == c) & (ft0 != c)).sum().item()
            fn_c    = ((fp0 != c) & (ft0 == c)).sum().item()
            pr      = tp_c / max(tp_c + fp_c, 1)
            re      = tp_c / max(tp_c + fn_c, 1)
            wf1_sum += (2 * pr * re / max(pr + re, 1e-8)) * support
        first_f1 = wf1_sum / max(total, 1)
    else:
        first_acc = first_f1 = None

    ges = None
    bo  = None
    if compute_ges:
        pred_len_ges = torch.where(has_end, pred_length, torch.full_like(pred_length, W))
        sa_cpu, la_cpu   = suffix_acts.cpu(), act_labels.cpu()
        nbp_cpu, nbl_cpu = suffix_nb.cpu(), nb_labels.cpu()
        pl_cpu, al_cpu   = pred_len_ges.cpu(), actual_length.cpu()
        args_list = [
            (sa_cpu[i, :int(pl_cpu[i])].tolist(),
             (nbp_cpu[i, :int(pl_cpu[i])] > 0.5).tolist(),
             la_cpu[i, :int(al_cpu[i])].tolist(),
             (nbl_cpu[i, :int(al_cpu[i])] > 0.5).tolist())
            for i in range(N)
        ]
        try:
            with ProcessPoolExecutor(max_workers=GES_WORKERS) as pool:
                ges_vals = list(pool.map(ges_compute_sample, args_list))
        except Exception:
            ges_vals = [ges_compute_sample(a) for a in args_list]
        ges = sum(ges_vals) / len(ges_vals) if ges_vals else 1.0
        bo_vals = [bucket_order_similarity(*a) for a in args_list]
        bo = {}
        for k, name in enumerate(('order', 'content', 'combined')):
            vals = [v[k] for v in bo_vals]
            bo[f'{name}_vals'] = vals
            bo[name] = sum(vals) / len(vals) if vals else 1.0

    return (dl_sim, ttne_mae_sec / 60.0, rrt_mae_sec / 60.0,
            mean_pred_len, mean_actual_len, nb_f1, nb_acc, first_acc, first_f1, ges,
            tp, fp, fn, tn,
            actual_length.tolist(), dl_per_inst.tolist(),
            (rrt_mae_per_inst / 60.0).tolist(), ges_vals if compute_ges else [],
            ttne_mae_per_inst_min.tolist(), bo)


# ─── Evaluation pass ──────────────────────────────────────────────────────────

@torch.no_grad()
def _evaluate(model, loader, device, num_activities, window_size,
              mean_std_ttne, mean_std_tss, mean_std_tsp, mean_std_rrt,
              compute_ges=False, compute_first_step=False, do_eval=True):
    model.eval()
    all_acts, all_ttne, all_nb = [], [], []
    all_lbl_acts, all_lbl_ttne, all_lbl_rrt, all_lbl_nb = [], [], [], []

    inference_start = time.time()
    for data in loader:
        data = data.to(device)
        B    = data.num_graphs
        acts, ttne, nb = model(
            data,
            window_size=window_size,
            mean_std_ttne=mean_std_ttne,
            mean_std_tss=mean_std_tss,
            mean_std_tsp=mean_std_tsp,
        )
        all_acts.append(acts.cpu())
        all_ttne.append(ttne.cpu())
        all_nb.append(nb.cpu())
        all_lbl_acts.append(data.act_label_seq.view(B, window_size).cpu())
        all_lbl_ttne.append(data.ttnext_label.squeeze(-1).view(B, window_size).cpu())
        all_lbl_rrt.append(data.rtime_label.squeeze(-1).view(B, window_size).cpu())
        all_lbl_nb.append(data.new_block_label.view(B, window_size).cpu())
    inference_time = time.time() - inference_start

    if not do_eval:
        return inference_time

    evaluation_start = time.time()
    sa  = torch.cat(all_acts,     dim=0)
    st  = torch.cat(all_ttne,     dim=0)
    snb = torch.cat(all_nb,       dim=0)
    la  = torch.cat(all_lbl_acts, dim=0)
    lt  = torch.cat(all_lbl_ttne, dim=0)
    lr  = torch.cat(all_lbl_rrt,  dim=0)
    lnb = torch.cat(all_lbl_nb,   dim=0)

    result = _compute_metrics(sa, st, snb, la, lt, lr, lnb,
                              num_activities, mean_std_ttne, mean_std_rrt,
                              compute_ges=compute_ges,
                              compute_first_step=compute_first_step)
    evaluation_time = time.time() - evaluation_start
    return (*result, inference_time, evaluation_time)


# ─── Main ─────────────────────────────────────────────────────────────────────

def run(log_name: str, results_dir: str = None, run_id: int = 0,
        do_train: bool = True, do_eval: bool = True):
    torch.manual_seed(run_id)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_id)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    results_dir = results_dir or f'results_time_{METHOD_NAME}'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}\nLog : {log_name}\nDevice : {device}\n{'='*60}")

    # ── Load pre-built datasets ───────────────────────────────────────────────
    data_dir   = os.path.join("approach_suffix_v2", 'results_per_log', log_name)
    train_data = torch.load(os.path.join(data_dir, 'train_graphdataset.pt'), weights_only=False)
    val_data   = torch.load(os.path.join(data_dir, 'val_graphdataset.pt'),   weights_only=False)
    test_data  = torch.load(os.path.join(data_dir, 'test_graphdataset.pt'),  weights_only=False)

    # ── Load normalization stats ──────────────────────────────────────────────
    with open(os.path.join(data_dir, f'{log_name}_train_means_dict.pkl'), 'rb') as f:
        means = pickle.load(f)
    with open(os.path.join(data_dir, f'{log_name}_train_std_dict.pkl'), 'rb') as f:
        stds  = pickle.load(f)

    mean_std_ttne = [means['timeLabel_df'][0], stds['timeLabel_df'][0]]
    mean_std_rrt  = [means['timeLabel_df'][1], stds['timeLabel_df'][1]]
    mean_std_tss  = [means['suffix_df'][0],    stds['suffix_df'][0]]
    mean_std_tsp  = [means['suffix_df'][1],    stds['suffix_df'][1]]

    with open(os.path.join(data_dir, f'{log_name}_cardin_list_prefix.pkl'), 'rb') as f:
        pref_cat_cars = pickle.load(f)
    window_size    = train_data[0].suffix_act.shape[0]
    num_activities = pref_cat_cars[-1] + 2

    # Class weight for new-block BCE: ratio of 0s to 1s among real (non-pad, non-EOS)
    # position-0 labels only, since training only ever supervises position 0.
    eos_tok     = num_activities - 1
    nb_ones = nb_zeros = 0
    for s in train_data:
        act0 = s.act_label_seq[0]
        if (act0 != 0) and (act0 != eos_tok):
            if s.new_block_label[0] > 0.5:
                nb_ones += 1
            else:
                nb_zeros += 1
    nb_pos_weight = nb_zeros / max(nb_ones, 1)

    print(f"num_activities={num_activities}  window_size={window_size}  "
          f"nb_pos_weight={nb_pos_weight:.2f}")

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False)

    # ── Build model ───────────────────────────────────────────────────────────
    model = GATv2EncoderGRUDecoderNewBlockV3(
        num_activities=num_activities,
        d_model=D_MODEL,
        dropout=DROPOUT,
        n_gru_layers=N_GRU_LAYERS,
        n_gnn_layers=N_GNN_LAYERS,
    ).to(device)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {num_trainable_params:,}")

    os.makedirs(results_dir, exist_ok=True)
    best_model_path = os.path.join(results_dir, f'{log_name}_{METHOD_NAME}.pt')

    if do_train:
        optimizer    = torch.optim.NAdam(model.parameters(), lr=LR)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=LR_PATIENCE, threshold=1e-4, min_lr=0)

        best_dl_sim   = -1.0
        best_ttne_mae =  1e9
        best_rrt_mae  =  1e9
        best_nb_f1    = -1.0
        patience_count = 0
        train_start    = time.time()

        # ── Training loop ─────────────────────────────────────────────────────────
        for epoch in range(MAX_EPOCHS):
            torch.manual_seed(epoch)
            model.train()
            total_loss, n_batches = 0.0, 0

            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                act_logits, ttne_preds, nb_logits = model(data)
                loss = _loss(act_logits, ttne_preds, nb_logits,
                             data, num_activities, nb_pos_weight, window_size)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
                optimizer.step()
                total_loss += loss.item()
                n_batches  += 1

            train_loss = total_loss / max(n_batches, 1)

            (dl_sim, ttne_mae_min, rrt_mae_min,
             mean_pred_len, mean_actual_len, nb_f1, nb_acc, _, _, _, _, _, _, _,
             _, _, _, _, _, _, _, _) = _evaluate(
                model, val_loader, device, num_activities, window_size,
                mean_std_ttne, mean_std_tss, mean_std_tsp, mean_std_rrt)

            lr_scheduler.step(1.0 - dl_sim)
            lr = optimizer.param_groups[0]['lr']
            print(f"[{log_name}] Epoch {epoch+1:4d}  loss={train_loss:.4f}  "
                  f"DL={dl_sim:.4f}  TTNE={ttne_mae_min:.2f}min  "
                  f"RRT={rrt_mae_min:.2f}min  NB_F1={nb_f1:.4f}  NB_acc={nb_acc:.4f}  "
                  f"lr={lr:.2e}  len_pred={mean_pred_len:.1f}  len_gt={mean_actual_len:.1f}")

            if dl_sim > best_dl_sim:
                torch.save(model.state_dict(), best_model_path)

            better = (dl_sim > best_dl_sim or
                      ttne_mae_min < best_ttne_mae or
                      rrt_mae_min  < best_rrt_mae or
                      nb_f1 > best_nb_f1)
            if better:
                best_dl_sim   = max(best_dl_sim,   dl_sim)
                best_ttne_mae = min(best_ttne_mae, ttne_mae_min)
                best_rrt_mae  = min(best_rrt_mae,  rrt_mae_min)
                best_nb_f1    = max(best_nb_f1,    nb_f1)
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= PATIENCE:
                    print(f"Early stopping at epoch {epoch+1}.")
                    break

        training_time = time.time() - train_start
    else:
        if not os.path.isfile(best_model_path):
            raise FileNotFoundError(
                f"do_train=False but no saved model found at {best_model_path}")
        print(f"Skipping training — loading existing model → {best_model_path}")
        training_time = 0.0

    # ── Test ──────────────────────────────────────────────────────────────────
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    model.to(device)

    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    if not do_eval:
        inference_time = _evaluate(
            model, test_loader, device, num_activities, window_size,
            mean_std_ttne, mean_std_tss, mean_std_tsp, mean_std_rrt,
            do_eval=False)
        print(f"\nTraining time  : {training_time:.1f}s")
        print(f"Inference time : {inference_time:.1f}s")

        inf_csv    = os.path.join(results_dir, 'inference_times.csv')
        inf_fields = ['log', 'method', 'training_time_seconds', 'inference_time_seconds']
        inf_row    = {
            'log':                    log_name,
            'method':                 METHOD_NAME,
            'training_time_seconds':  round(training_time,  2),
            'inference_time_seconds': round(inference_time, 2),
        }
        lock_path = inf_csv + '.lock'
        while True:
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                break
            except FileExistsError:
                time.sleep(0.05)
        try:
            rows = []
            if os.path.isfile(inf_csv):
                with open(inf_csv, newline='') as f:
                    rows = list(csv.DictReader(f))
            updated = False
            for row in rows:
                if row['log'] == log_name and row['method'] == METHOD_NAME:
                    row.update(inf_row)
                    updated = True
                    break
            if not updated:
                rows.append(inf_row)
            with open(inf_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=inf_fields)
                writer.writeheader()
                writer.writerows(rows)
        finally:
            os.remove(lock_path)
        print(f"Inference times saved → {inf_csv}")

        return {'training_time_seconds': training_time,
                'inference_time_seconds': inference_time}

    (dl_sim, ttne_mae_min, rrt_mae_min, _, _,
     nb_f1, nb_acc, first_acc, first_f1, ges,
     nb_tp, nb_fp, nb_fn, nb_tn,
     suf_lens, dl_per_inst_list, rrt_per_inst_list, ges_vals,
     ttne_per_inst_list,
     bo,
     inference_time, evaluation_time) = _evaluate(
        model, test_loader, device, num_activities, window_size,
        mean_std_ttne, mean_std_tss, mean_std_tsp, mean_std_rrt,
        compute_ges=True, compute_first_step=True)
    testing_time = inference_time + evaluation_time

    print(f"\n{'─'*60}")
    print(f"DL similarity  : {dl_sim:.4f}")
    print(f"GES            : {ges:.4f}")
    print(f"BO order sim   : {bo['order']:.4f}")
    print(f"BO content F1  : {bo['content']:.4f}")
    print(f"BO combined    : {bo['combined']:.4f}")
    print(f"TTNE MAE       : {ttne_mae_min:.2f} min")
    print(f"RRT MAE        : {rrt_mae_min:.2f} min")
    print(f"NB F1          : {nb_f1:.4f}")
    print(f"NB accuracy    : {nb_acc:.4f}")
    print(f"NB CM          : TP={nb_tp}  FP={nb_fp}  FN={nb_fn}  TN={nb_tn}")
    print(f"First-step F1  : {first_f1:.4f}")
    print(f"First-step acc : {first_acc:.4f}")
    print(f"Training time  : {training_time:.1f}s")
    print(f"Inference time : {inference_time:.1f}s")
    print(f"Evaluation time: {evaluation_time:.1f}s")
    print(f"Testing time   : {testing_time:.1f}s")

    # ── Save results ──────────────────────────────────────────────────────────
    csv_path   = os.path.join(results_dir, 'results_suffix_time_gnn.csv')
    fieldnames = ['log', 'model', 'method',
                  'dl_similarity', 'ges_approx', 'bo_order_sim', 'bo_content_f1', 'bo_combined',
                  'ttne_mae_minutes', 'rrt_mae_minutes',
                  'nb_f1', 'nb_accuracy', 'nb_tp', 'nb_fp', 'nb_fn', 'nb_tn',
                  'first_step_f1', 'first_step_accuracy',
                  'training_time_seconds', 'inference_time_seconds',
                  'evaluation_time_seconds', 'testing_time_seconds',
                  'num_trainable_params']
    new_row = {
        'log':                   log_name,
        'model':                 'suffix_time_nb_v3',
        'method':                METHOD_NAME,
        'dl_similarity':         round(dl_sim,        6),
        'ges_approx':            round(ges,           6),
        'bo_order_sim':          round(bo['order'],       6),
        'bo_content_f1':         round(bo['content'],     6),
        'bo_combined':           round(bo['combined'],    6),
        'ttne_mae_minutes':      round(ttne_mae_min,  6),
        'rrt_mae_minutes':       round(rrt_mae_min,   6),
        'nb_f1':                 round(nb_f1,         6),
        'nb_accuracy':           round(nb_acc,        6),
        'nb_tp':                 nb_tp,
        'nb_fp':                 nb_fp,
        'nb_fn':                 nb_fn,
        'nb_tn':                 nb_tn,
        'first_step_f1':         round(first_f1,      6),
        'first_step_accuracy':   round(first_acc,     6),
        'training_time_seconds':   round(training_time,   2),
        'inference_time_seconds':  round(inference_time,  2),
        'evaluation_time_seconds': round(evaluation_time, 2),
        'testing_time_seconds':    round(testing_time,    2),
        'num_trainable_params':  num_trainable_params,
    }

    lock_path = csv_path + '.lock'
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            time.sleep(0.05)
    try:
        rows = []
        if os.path.isfile(csv_path):
            with open(csv_path, newline='') as f:
                rows = list(csv.DictReader(f))
        updated = False
        for row in rows:
            if row['log'] == log_name and row['method'] == METHOD_NAME:
                row.update(new_row)
                updated = True
                break
        if not updated:
            rows.append(new_row)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    finally:
        os.remove(lock_path)

    print(f"Results saved → {csv_path}")

    pref_lens = [test_data[i].num_nodes for i in range(len(test_data))]
    _by_pref = {}
    _by_suf  = {}
    for plen, slen, dl_val, rrt_val, gval, bo_o, bo_c, bo_x in zip(
            pref_lens, suf_lens, dl_per_inst_list, rrt_per_inst_list, ges_vals,
            bo['order_vals'], bo['content_vals'], bo['combined_vals']):
        for d, key in ((_by_pref, int(plen)), (_by_suf, int(slen))):
            d.setdefault(key, {'dl': [], 'rrt': [], 'ges': [],
                               'bo_order': [], 'bo_content': [], 'bo_combined': []})
            d[key]['dl'].append(dl_val)
            d[key]['rrt'].append(rrt_val)
            d[key]['ges'].append(gval)
            d[key]['bo_order'].append(bo_o)
            d[key]['bo_content'].append(bo_c)
            d[key]['bo_combined'].append(bo_x)
    # [dl, rrt, count, ges, bo_order, bo_content, bo_combined]
    pref_len_dict = {k: [sum(v['dl'])/len(v['dl']), sum(v['rrt'])/len(v['rrt']),
                         len(v['dl']), sum(v['ges'])/len(v['ges']),
                         sum(v['bo_order'])/len(v['bo_order']),
                         sum(v['bo_content'])/len(v['bo_content']),
                         sum(v['bo_combined'])/len(v['bo_combined'])]
                     for k, v in _by_pref.items()}
    suf_len_dict  = {k: [sum(v['dl'])/len(v['dl']), sum(v['rrt'])/len(v['rrt']),
                         len(v['dl']), sum(v['ges'])/len(v['ges']),
                         sum(v['bo_order'])/len(v['bo_order']),
                         sum(v['bo_content'])/len(v['bo_content']),
                         sum(v['bo_combined'])/len(v['bo_combined'])]
                     for k, v in _by_suf.items()}
    pref_pkl = os.path.join(results_dir, f'{log_name}_prefix_length_results_dict.pkl')
    suf_pkl  = os.path.join(results_dir, f'{log_name}_suffix_length_results_dict.pkl')
    with open(pref_pkl, 'wb') as f:
        pickle.dump(pref_len_dict, f)
    with open(suf_pkl, 'wb') as f:
        pickle.dump(suf_len_dict, f)
    print(f"Per-length result dicts saved → {results_dir}")

    per_sample_path = os.path.join(results_dir, f'{log_name}_per_sample_metrics.pt')
    torch.save({
        'dl_similarity':    torch.tensor(dl_per_inst_list,   dtype=torch.float32),
        'ges_approx':       torch.tensor(ges_vals,           dtype=torch.float32),
        'bo_order_sim':     torch.tensor(bo['order_vals'], dtype=torch.float32),
        'bo_content_f1':    torch.tensor(bo['content_vals'], dtype=torch.float32),
        'bo_combined':      torch.tensor(bo['combined_vals'], dtype=torch.float32),
        'ttne_mae_minutes': torch.tensor(ttne_per_inst_list, dtype=torch.float32),
        'rrt_mae_minutes':  torch.tensor(rrt_per_inst_list,  dtype=torch.float32),
        'pref_len':         torch.tensor(pref_lens,          dtype=torch.int64),
        'suf_len':          torch.tensor([s + 1 for s in suf_lens], dtype=torch.int64),
    }, per_sample_path)
    print(f"Per-sample metrics saved → {per_sample_path}")

    return {'dl_similarity': dl_sim, 'ges_approx': ges,
            'ttne_mae_minutes': ttne_mae_min, 'rrt_mae_minutes': rrt_mae_min,
            'nb_f1': nb_f1, 'nb_accuracy': nb_acc}


# ─── Entry point ──────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description='Train GRU+new-block v3 model (next-activity prediction only) and evaluate for suffix + RRT')
    p.add_argument('log_name',    help='Log name (must match results_per_log/<log_name>/)')
    p.add_argument('results_dir', nargs='?', default=None)
    p.add_argument('--run_id',    type=int, default=0)
    p.add_argument('--no_train', action='store_true',
                   help='Skip training and load the saved model for this log / run_id')
    p.add_argument('--no_eval', action='store_true',
                   help='Skip evaluation; only run inference and report its time')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    run(args.log_name, args.results_dir, args.run_id,
        do_train=not args.no_train, do_eval=not args.no_eval)
