"""Train and evaluate GATv2EncoderGRUDecoderBlockV2: the decoder predicts the
activity suffix one whole concurrent block per step (multi-hot set), ending
with the block {END}.  No new-block head, no time outputs.
Training is teacher-forced; inference is greedy autoregressive.

DL similarity is not reported: the order inside a predicted block is
undefined.  Metrics are GES (exact) and bucket-order similarity, both
order-invariant inside blocks.

Usage
-----
    python run_suffix_v2_block.py <log_name> [results_dir]

Loads pre-built datasets from results_per_log/<log_name>/:
    train_graphdataset.pt, val_graphdataset.pt, test_graphdataset.pt

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

from model_suffix_v2_block import GATv2EncoderGRUDecoderBlockV2, build_block_targets
from partial_order_metrics import ges_exact_compute_sample, bucket_order_similarity

# ─── Hyperparameters ──────────────────────────────────────────────────────────
D_MODEL      = 64
DROPOUT      = 0.4
N_GRU_LAYERS = 1
N_GNN_LAYERS = 2
LR           = 0.002
MAX_EPOCHS   = 200
PATIENCE     = 24
LR_PATIENCE  = 10
MAX_NORM     = 2.0
BATCH_SIZE   = 128

METHOD_NAME = 'gatv2_gru_block_v2'
GES_WORKERS = 4
COMPUTE_GES = True   # False → skip GES (reported as nan)


# ─── Loss ─────────────────────────────────────────────────────────────────────

def _loss(logits, targets, step_mask):
    """Multi-hot BCE over classes 1..C-1 (padding excluded), summed over
    classes and averaged over valid block steps (real blocks + END step)."""
    raw = F.binary_cross_entropy_with_logits(
        logits[..., 1:], targets[..., 1:], reduction='none').sum(dim=-1)   # (B, W)
    return raw[step_mask].mean()


# ─── Metrics ──────────────────────────────────────────────────────────────────

def _flatten_blocks(blocks, n):
    """Predicted multi-hot blocks (W, C) → (acts, nb_bits). Activities inside a
    block are sorted by id; the first opens the block (nb=1)."""
    acts, nbs = [], []
    for t in range(min(int(n), blocks.shape[0])):
        ids = blocks[t].nonzero(as_tuple=True)[0].tolist()
        acts += ids
        nbs  += [True] + [False] * (len(ids) - 1)
    return acts, nbs


@torch.no_grad()
def _compute_metrics(pred_blocks, n_blocks, act_labels, nb_labels,
                     num_activities, compute_ges=False):
    N, W, C = pred_blocks.shape
    end_tok = num_activities - 1

    actual_length = (act_labels == end_tok).to(torch.int64).argmax(dim=-1)

    targets, _, true_n_blocks = build_block_targets(act_labels, nb_labels, num_activities)

    # ── First-block set F1 (END counts as a class) ────────────────────────────
    p0 = pred_blocks[:, 0].clone()
    p0[n_blocks == 0, end_tok] = 1.0
    t0 = targets[:, 0]
    inter = (p0 * t0).sum(dim=-1)
    first_block_f1 = (2 * inter / (p0.sum(dim=-1) + t0.sum(dim=-1)).clamp(min=1)).mean().item()

    mean_pred_blocks = n_blocks.clamp(max=W).float().mean().item()
    mean_true_blocks = true_n_blocks.float().mean().item()

    # ── Partial-order metrics on flattened suffixes ───────────────────────────
    args_list = []
    for i in range(N):
        L = int(actual_length[i])
        acts_p, nb_p = _flatten_blocks(pred_blocks[i], n_blocks[i])
        args_list.append((acts_p, nb_p,
                          act_labels[i, :L].tolist(),
                          (nb_labels[i, :L] > 0.5).tolist()))

    bo_vals = [bucket_order_similarity(*a) for a in args_list]
    bo = {}
    for k, name in enumerate(('order', 'content', 'combined')):
        vals = [v[k] for v in bo_vals]
        bo[f'{name}_vals'] = vals
        bo[name] = sum(vals) / len(vals) if vals else 1.0

    ges, ges_vals = None, []
    if compute_ges:
        if COMPUTE_GES:
            try:
                with ProcessPoolExecutor(max_workers=GES_WORKERS) as pool:
                    ges_vals = list(pool.map(ges_exact_compute_sample, args_list))
            except Exception:
                ges_vals = [ges_exact_compute_sample(a) for a in args_list]
        else:
            ges_vals = [float('nan')] * len(args_list)
        ges = sum(ges_vals) / len(ges_vals) if ges_vals else 1.0

    return {'bo': bo, 'ges': ges, 'ges_vals': ges_vals,
            'first_block_f1': first_block_f1,
            'mean_pred_blocks': mean_pred_blocks, 'mean_true_blocks': mean_true_blocks,
            'suf_lens': actual_length.tolist()}


# ─── Evaluation pass ──────────────────────────────────────────────────────────

@torch.no_grad()
def _evaluate(model, loader, device, num_activities, window_size,
              compute_ges=False, do_eval=True):
    model.eval()
    all_blocks, all_n = [], []
    all_lbl_acts, all_lbl_nb = [], []

    inference_start = time.time()
    for data in loader:
        data = data.to(device)
        B    = data.num_graphs
        blocks, n = model(data, window_size=window_size)
        all_blocks.append(blocks.cpu())
        all_n.append(n.cpu())
        all_lbl_acts.append(data.act_label_seq.view(B, window_size).cpu())
        all_lbl_nb.append(data.new_block_label.view(B, window_size).cpu())
    inference_time = time.time() - inference_start

    if not do_eval:
        return inference_time

    evaluation_start = time.time()
    result = _compute_metrics(torch.cat(all_blocks,   dim=0),
                              torch.cat(all_n,        dim=0),
                              torch.cat(all_lbl_acts, dim=0),
                              torch.cat(all_lbl_nb,   dim=0),
                              num_activities, compute_ges=compute_ges)
    evaluation_time = time.time() - evaluation_start
    return result, inference_time, evaluation_time


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

    with open(os.path.join(data_dir, f'{log_name}_cardin_list_prefix.pkl'), 'rb') as f:
        pref_cat_cars = pickle.load(f)
    window_size    = train_data[0].suffix_act.shape[0]
    num_activities = pref_cat_cars[-1] + 2

    print(f"num_activities={num_activities}  window_size={window_size}")

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False)

    # ── Build model ───────────────────────────────────────────────────────────
    model = GATv2EncoderGRUDecoderBlockV2(
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

        best_bo        = -1.0
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
                logits, targets, step_mask = model(data, window_size=window_size)
                loss = _loss(logits, targets, step_mask)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
                optimizer.step()
                total_loss += loss.item()
                n_batches  += 1

            train_loss = total_loss / max(n_batches, 1)

            m, _, _ = _evaluate(model, val_loader, device, num_activities, window_size)
            bo_comb = m['bo']['combined']

            lr_scheduler.step(1.0 - bo_comb)
            lr = optimizer.param_groups[0]['lr']
            print(f"[{log_name}] Epoch {epoch+1:4d}  loss={train_loss:.4f}  "
                  f"BO={bo_comb:.4f}  BO_ord={m['bo']['order']:.4f}  "
                  f"BO_cont={m['bo']['content']:.4f}  B0_F1={m['first_block_f1']:.4f}  "
                  f"lr={lr:.2e}  blocks_pred={m['mean_pred_blocks']:.1f}  "
                  f"blocks_gt={m['mean_true_blocks']:.1f}")

            if bo_comb > best_bo:
                torch.save(model.state_dict(), best_model_path)
                best_bo        = bo_comb
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
            model, test_loader, device, num_activities, window_size, do_eval=False)
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

    m, inference_time, evaluation_time = _evaluate(
        model, test_loader, device, num_activities, window_size, compute_ges=True)
    testing_time = inference_time + evaluation_time
    bo, ges, ges_vals, suf_lens = m['bo'], m['ges'], m['ges_vals'], m['suf_lens']

    print(f"\n{'─'*60}")
    print(f"GES            : {ges:.4f}")
    print(f"BO order sim   : {bo['order']:.4f}")
    print(f"BO content F1  : {bo['content']:.4f}")
    print(f"BO combined    : {bo['combined']:.4f}")
    print(f"First-block F1 : {m['first_block_f1']:.4f}")
    print(f"Blocks pred/gt : {m['mean_pred_blocks']:.2f} / {m['mean_true_blocks']:.2f}")
    print(f"Training time  : {training_time:.1f}s")
    print(f"Inference time : {inference_time:.1f}s")
    print(f"Evaluation time: {evaluation_time:.1f}s")
    print(f"Testing time   : {testing_time:.1f}s")

    # ── Save results ──────────────────────────────────────────────────────────
    # Same columns as v2; DL, time and new-block columns are left empty
    # (not reported / no such outputs).
    csv_path   = os.path.join(results_dir, 'results_suffix_time_gnn.csv')
    fieldnames = ['log', 'model', 'method',
                  'dl_similarity', 'ges_exact', 'bo_order_sim', 'bo_content_f1', 'bo_combined',
                  'ttne_mae_minutes', 'rrt_mae_minutes',
                  'nb_f1', 'nb_accuracy', 'nb_tp', 'nb_fp', 'nb_fn', 'nb_tn',
                  'nb_macro_precision', 'nb_macro_recall', 'nb_macro_f1',
                  'nb_majority_class', 'nb_majority_share',
                  'first_step_f1', 'first_step_accuracy',
                  'first_block_f1', 'mean_pred_blocks', 'mean_true_blocks',
                  'training_time_seconds', 'inference_time_seconds',
                  'evaluation_time_seconds', 'testing_time_seconds',
                  'num_trainable_params']
    new_row = {k: '' for k in fieldnames}
    new_row.update({
        'log':                   log_name,
        'model':                 'suffix_gru_block_v2',
        'method':                METHOD_NAME,
        'ges_exact':             round(ges,           6),
        'bo_order_sim':          round(bo['order'],       6),
        'bo_content_f1':         round(bo['content'],     6),
        'bo_combined':           round(bo['combined'],    6),
        'first_block_f1':        round(m['first_block_f1'],   6),
        'mean_pred_blocks':      round(m['mean_pred_blocks'], 4),
        'mean_true_blocks':      round(m['mean_true_blocks'], 4),
        'training_time_seconds':   round(training_time,   2),
        'inference_time_seconds':  round(inference_time,  2),
        'evaluation_time_seconds': round(evaluation_time, 2),
        'testing_time_seconds':    round(testing_time,    2),
        'num_trainable_params':  num_trainable_params,
    })

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
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
    finally:
        os.remove(lock_path)

    print(f"Results saved → {csv_path}")

    pref_lens = [test_data[i].num_nodes for i in range(len(test_data))]
    _by_pref = {}
    _by_suf  = {}
    for plen, slen, gval, bo_o, bo_c, bo_x in zip(
            pref_lens, suf_lens, ges_vals,
            bo['order_vals'], bo['content_vals'], bo['combined_vals']):
        for d, key in ((_by_pref, int(plen)), (_by_suf, int(slen))):
            d.setdefault(key, {'ges': [], 'bo_order': [], 'bo_content': [], 'bo_combined': []})
            d[key]['ges'].append(gval)
            d[key]['bo_order'].append(bo_o)
            d[key]['bo_content'].append(bo_c)
            d[key]['bo_combined'].append(bo_x)
    # [dl, rrt, count, ges, bo_order, bo_content, bo_combined]; dl and rrt are NaN (not reported)
    pref_len_dict = {k: [float('nan'), float('nan'),
                         len(v['ges']), sum(v['ges'])/len(v['ges']),
                         sum(v['bo_order'])/len(v['bo_order']),
                         sum(v['bo_content'])/len(v['bo_content']),
                         sum(v['bo_combined'])/len(v['bo_combined'])]
                     for k, v in _by_pref.items()}
    suf_len_dict  = {k: [float('nan'), float('nan'),
                         len(v['ges']), sum(v['ges'])/len(v['ges']),
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
        'ges_exact':        torch.tensor(ges_vals,           dtype=torch.float32),
        'bo_order_sim':     torch.tensor(bo['order_vals'], dtype=torch.float32),
        'bo_content_f1':    torch.tensor(bo['content_vals'], dtype=torch.float32),
        'bo_combined':      torch.tensor(bo['combined_vals'], dtype=torch.float32),
        'pref_len':         torch.tensor(pref_lens,          dtype=torch.int64),
        'suf_len':          torch.tensor([s + 1 for s in suf_lens], dtype=torch.int64),
    }, per_sample_path)
    print(f"Per-sample metrics saved → {per_sample_path}")

    return {'ges_exact': ges, 'bo_combined': bo['combined'],
            'first_block_f1': m['first_block_f1']}


# ─── Entry point ──────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description='Train GATv2 + GRU-decoder full-block v2 model (no time outputs) and evaluate')
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
