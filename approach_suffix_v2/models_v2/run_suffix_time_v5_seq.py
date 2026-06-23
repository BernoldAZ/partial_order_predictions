"""Train and evaluate GATv2EncoderGRUDecoderNewBlock (v5) for activity
suffix + TTNE prediction.  v5 is identical to v4 but removes the stop head
(fc_stop); the END token is selected purely from activity logits.

Usage
-----
    python run_suffix_time_v5_seq.py <log_name> [results_dir]

Loads pre-built datasets from results_per_log/<log_name>/:
    train_seqgraphdataset.pt, val_seqgraphdataset.pt, test_seqgraphdataset.pt
    <log_name>_train_means_dict.pkl, <log_name>_train_std_dict.pkl

Run create_seq_graph_data.py first to generate these files.
"""
import argparse
import csv
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from model_suffix_time_v5 import GATv2EncoderGRUDecoderNewBlock

# ─── Hyperparameters ──────────────────────────────────────────────────────────
D_MODEL     = 64
DROPOUT     = 0.4
N_LAYERS    = 1
LR          = 0.002
MAX_EPOCHS  = 200
PATIENCE    = 24
LR_PATIENCE = 10
MAX_NORM    = 2.0
BATCH_SIZE  = 128
SEED        = 24

# ─── Scheduled sampling ───────────────────────────────────────────────────────
USE_SCHEDULED_SAMPLING = True
SS_P_TEACHER_START     = 1.0
SS_P_TEACHER_END       = 0.0
SS_ANNEAL_EPOCHS       = MAX_EPOCHS

# ─── New-block loss weight ────────────────────────────────────────────────────
NB_WEIGHT = 1.0   # scalar multiplier on the new-block BCE loss term

METHOD_NAME = 'gatv2_seq_gru_nb_v5'
GES_WORKERS = 2


# ─── GES helpers ──────────────────────────────────────────────────────────────

def _build_suffix_graph(acts, nb_bits):
    n = len(acts)
    G = nx.DiGraph()
    if n == 0:
        return G
    for i in range(n):
        G.add_node(i, act=int(acts[i]))
    blocks = [[0]]
    for i in range(1, n):
        if nb_bits[i]:
            blocks.append([i])
        else:
            blocks[-1].append(i)
    for block in blocks:
        for u in block:
            for v in block:
                if u != v:
                    G.add_edge(u, v)
    for bi in range(len(blocks) - 1):
        for u in blocks[bi]:
            for v in blocks[bi + 1]:
                G.add_edge(u, v)
    return G


def _node_match(n1, n2):
    return n1['act'] == n2['act']


def _graph_edit_similarity(G_pred, G_true):
    np_n, np_e = G_pred.number_of_nodes(), G_pred.number_of_edges()
    nt_n, nt_e = G_true.number_of_nodes(), G_true.number_of_edges()
    if np_n == 0 and nt_n == 0:
        return 1.0
    if np_n == 0 or nt_n == 0:
        return 0.0
    denom = (np_n + np_e) + (nt_n + nt_e)
    ged   = next(nx.optimize_graph_edit_distance(G_pred, G_true, node_match=_node_match))
    return 1.0 - ged / denom


def _ges_compute_sample(args):
    sa_list, nbp_list, la_list, nbl_list = args
    return _graph_edit_similarity(
        _build_suffix_graph(sa_list, nbp_list),
        _build_suffix_graph(la_list, nbl_list),
    )


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
          data, num_activities, nb_pos_weight):
    ce  = _masked_ce(act_logits, data.act_label_seq, num_activities)
    mae = _masked_mae(ttne_preds, data.ttnext_label)
    nb  = _new_block_bce(nb_logits, data.act_label_seq, data.new_block_label,
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
    len_pred   = pred_length   + 1
    len_actual = actual_length + 1
    max_len    = torch.maximum(len_pred, len_actual).float()

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
    dl_sim = (1.0 - d[batch_range, len_pred, len_actual].float() / max_len).mean().item()

    # ── TTNE MAE ──────────────────────────────────────────────────────────────
    ttne_preds_sec  = (suffix_ttne.clone() * ttne_std + ttne_mean).clamp(min=0)
    ttne_labels_sec = (ttne_labels.clone() * ttne_std + ttne_mean).clamp(min=0)
    ttne_preds_sec[counting > pred_length.unsqueeze(-1)] = 0.0

    before_end   = counting <= actual_length.unsqueeze(-1)
    ttne_mae_sec = torch.abs(ttne_preds_sec - ttne_labels_sec)[before_end].mean().item()

    # ── RRT MAE ───────────────────────────────────────────────────────────────
    rrt_preds_sec = ttne_preds_sec.clone()
    rrt_preds_sec[batch_range, pred_length] = 0.0
    rrt_preds_sec = rrt_preds_sec.sum(dim=-1)

    rrt_label_sec = (rrt_labels[:, 0] * rrt_std + rrt_mean).clamp(min=0)
    rrt_mae_sec   = torch.abs(rrt_preds_sec - rrt_label_sec).mean().item()

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
                ges_vals = list(pool.map(_ges_compute_sample, args_list))
        except Exception:
            ges_vals = [_ges_compute_sample(a) for a in args_list]
        ges = sum(ges_vals) / len(ges_vals) if ges_vals else 1.0

    return (dl_sim, ttne_mae_sec / 60.0, rrt_mae_sec / 60.0,
            mean_pred_len, mean_actual_len, nb_f1, nb_acc, first_acc, first_f1, ges)


# ─── Evaluation pass ──────────────────────────────────────────────────────────

@torch.no_grad()
def _evaluate(model, loader, device, num_activities, window_size,
              mean_std_ttne, mean_std_tss, mean_std_tsp, mean_std_rrt,
              compute_ges=False, compute_first_step=False):
    model.eval()
    all_acts, all_ttne, all_nb = [], [], []
    all_lbl_acts, all_lbl_ttne, all_lbl_rrt, all_lbl_nb = [], [], [], []

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

    sa  = torch.cat(all_acts,     dim=0)
    st  = torch.cat(all_ttne,     dim=0)
    snb = torch.cat(all_nb,       dim=0)
    la  = torch.cat(all_lbl_acts, dim=0)
    lt  = torch.cat(all_lbl_ttne, dim=0)
    lr  = torch.cat(all_lbl_rrt,  dim=0)
    lnb = torch.cat(all_lbl_nb,   dim=0)

    return _compute_metrics(sa, st, snb, la, lt, lr, lnb,
                            num_activities, mean_std_ttne, mean_std_rrt,
                            compute_ges=compute_ges,
                            compute_first_step=compute_first_step)


# ─── Main ─────────────────────────────────────────────────────────────────────

def run(log_name: str, results_dir: str = None):
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    results_dir = results_dir or f'results_time_{METHOD_NAME}'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}\nLog : {log_name}\nDevice : {device}\n{'='*60}")

    # ── Load pre-built datasets ───────────────────────────────────────────────
    data_dir   = os.path.join("approach_suffix_v2", 'results_per_log', log_name)
    train_data = torch.load(os.path.join(data_dir, 'train_seqgraphdataset.pt'), weights_only=False)
    val_data   = torch.load(os.path.join(data_dir, 'val_seqgraphdataset.pt'),   weights_only=False)
    test_data  = torch.load(os.path.join(data_dir, 'test_seqgraphdataset.pt'),  weights_only=False)

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

    # Class weight for new-block BCE: ratio of 0s to 1s among real (non-pad, non-EOS) positions.
    eos_tok     = num_activities - 1
    nb_ones = nb_zeros = 0
    for s in train_data:
        is_real = (s.act_label_seq != 0) & (s.act_label_seq != eos_tok)
        nb_real = s.new_block_label[is_real]
        nb_ones  += (nb_real > 0.5).sum().item()
        nb_zeros += (nb_real < 0.5).sum().item()
    nb_pos_weight = nb_zeros / max(nb_ones, 1)

    print(f"num_activities={num_activities}  window_size={window_size}  "
          f"nb_pos_weight={nb_pos_weight:.2f}")

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False)

    # ── Build model ───────────────────────────────────────────────────────────
    model = GATv2EncoderGRUDecoderNewBlock(
        num_activities=num_activities,
        d_model=D_MODEL,
        dropout=DROPOUT,
        n_layers=N_LAYERS,
        use_scheduled_sampling=USE_SCHEDULED_SAMPLING,
    ).to(device)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {num_trainable_params:,}")

    optimizer    = torch.optim.NAdam(model.parameters(), lr=LR)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=LR_PATIENCE, threshold=1e-4, min_lr=0)

    os.makedirs(results_dir, exist_ok=True)
    best_model_path = os.path.join(results_dir, f'{log_name}_{METHOD_NAME}.pt')

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

        if USE_SCHEDULED_SAMPLING:
            progress  = epoch / max(SS_ANNEAL_EPOCHS - 1, 1)
            p_teacher = max(SS_P_TEACHER_END,
                            SS_P_TEACHER_START - (SS_P_TEACHER_START - SS_P_TEACHER_END) * progress)
        else:
            p_teacher = 1.0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            act_logits, ttne_preds, nb_logits = model(
                data,
                p_teacher=p_teacher,
                mean_std_ttne=mean_std_ttne if USE_SCHEDULED_SAMPLING else None,
                mean_std_tss=mean_std_tss   if USE_SCHEDULED_SAMPLING else None,
                mean_std_tsp=mean_std_tsp   if USE_SCHEDULED_SAMPLING else None,
            )
            loss = _loss(act_logits, ttne_preds, nb_logits,
                         data, num_activities, nb_pos_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        train_loss = total_loss / max(n_batches, 1)

        (dl_sim, ttne_mae_min, rrt_mae_min,
         mean_pred_len, mean_actual_len, nb_f1, nb_acc, _, _, _) = _evaluate(
            model, val_loader, device, num_activities, window_size,
            mean_std_ttne, mean_std_tss, mean_std_tsp, mean_std_rrt)

        lr_scheduler.step(1.0 - dl_sim)
        lr = optimizer.param_groups[0]['lr']
        ss_info = f"  p_teacher={p_teacher:.3f}" if USE_SCHEDULED_SAMPLING else ""
        print(f"[{log_name}] Epoch {epoch+1:4d}  loss={train_loss:.4f}  "
              f"DL={dl_sim:.4f}  TTNE={ttne_mae_min:.2f}min  "
              f"RRT={rrt_mae_min:.2f}min  NB_F1={nb_f1:.4f}  NB_acc={nb_acc:.4f}  "
              f"lr={lr:.2e}  len_pred={mean_pred_len:.1f}  len_gt={mean_actual_len:.1f}{ss_info}")

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

    # ── Test ──────────────────────────────────────────────────────────────────
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    model.to(device)

    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    test_start  = time.time()
    (dl_sim, ttne_mae_min, rrt_mae_min, _, _,
     nb_f1, nb_acc, first_acc, first_f1, ges) = _evaluate(
        model, test_loader, device, num_activities, window_size,
        mean_std_ttne, mean_std_tss, mean_std_tsp, mean_std_rrt,
        compute_ges=True, compute_first_step=True)
    testing_time = time.time() - test_start

    print(f"\n{'─'*60}")
    print(f"DL similarity  : {dl_sim:.4f}")
    print(f"GES            : {ges:.4f}")
    print(f"TTNE MAE       : {ttne_mae_min:.2f} min")
    print(f"RRT MAE        : {rrt_mae_min:.2f} min")
    print(f"NB F1          : {nb_f1:.4f}")
    print(f"NB accuracy    : {nb_acc:.4f}")
    print(f"First-step F1  : {first_f1:.4f}")
    print(f"First-step acc : {first_acc:.4f}")
    print(f"Training time  : {training_time:.1f}s")
    print(f"Testing time   : {testing_time:.1f}s")

    # ── Save results ──────────────────────────────────────────────────────────
    csv_path   = os.path.join(results_dir, 'results_suffix_time_gnn.csv')
    fieldnames = ['log', 'model', 'method',
                  'dl_similarity', 'ges_approx', 'ttne_mae_minutes', 'rrt_mae_minutes',
                  'nb_f1', 'nb_accuracy', 'first_step_f1', 'first_step_accuracy',
                  'training_time_seconds', 'testing_time_seconds',
                  'num_trainable_params']
    new_row = {
        'log':                   log_name,
        'model':                 'suffix_time_nb_v5',
        'method':                METHOD_NAME,
        'dl_similarity':         round(dl_sim,        6),
        'ges_approx':            round(ges,           6),
        'ttne_mae_minutes':      round(ttne_mae_min,  6),
        'rrt_mae_minutes':       round(rrt_mae_min,   6),
        'nb_f1':                 round(nb_f1,         6),
        'nb_accuracy':           round(nb_acc,        6),
        'first_step_f1':         round(first_f1,      6),
        'first_step_accuracy':   round(first_acc,     6),
        'training_time_seconds': round(training_time, 2),
        'testing_time_seconds':  round(testing_time,  2),
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
    return {'dl_similarity': dl_sim, 'ges_approx': ges,
            'ttne_mae_minutes': ttne_mae_min, 'rrt_mae_minutes': rrt_mae_min,
            'nb_f1': nb_f1, 'nb_accuracy': nb_acc}


# ─── Entry point ──────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description='Train GRU+new-block v5 model (seq graph data) and evaluate for suffix + RRT')
    p.add_argument('log_name',    help='Log name (must match results_per_log/<log_name>/)')
    p.add_argument('results_dir', nargs='?', default=None)
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    run(args.log_name, args.results_dir)
