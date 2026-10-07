"""Module containing the pipeline to train and evaluate the ASTON baseline.

Based on:
    Rama-Maneiro, E., Vidal, J. C., Lama, M., & Monteagudo-Lago, P. (2024).
    Exploiting recurrent graph neural networks for suffix prediction in
    predictive monitoring. Computing, 106, 3085-3111.
    https://doi.org/10.1007/s00607-024-01315-9

The model (GRU + GRNN encoder, attention GRU decoder) and the length-
normalised beam search are taken unchanged from the original repository
(``ASTON/model/``, ``ASTON/predicter/``). Training follows the hyperparameters reported
in the paper (Sec. 5.1): 150 epochs, batch size 64, embedding / hidden size
32, Adam with lr 0.005, beam width 5, and the model with the lowest
validation loss is kept.

Inputs are built by ``baselines/create_aston_data.py`` from the SuTraN splits, and
evaluation reuses the BEST baseline's metric code so that the output files
(``TEST_SET_RESULTS/averaged_results.pkl`` etc.) match the other baselines.
ASTON only predicts activities, so - as for BEST - the time metrics come
from a constant training-mean TTNE predictor.
"""

import os
import pickle
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "ASTON"))

from datasets.datasets import TracesDatasetSeqTime
from model.encoder_decoder.encoders import EncoderGRNNHiddenState
from model.encoder_decoder.decoders import DecoderLastGRNNState
from model.encoder_decoder.interfaces import EncoderDecoder
from model.loss import SoftmaxCELoss
from predicter.predicter import batched_beam_decode_optimized
from utils.utils import Utils

RESULTS_BASE = os.path.join(_HERE, "results_per_log")

# Paper, Sec. 5.1
NUM_EPOCHS = 150
BATCH_SIZE = 64
EMBED_SIZE = 32
NUM_HIDDENS = 32
LR = 0.005
BEAM_WIDTH = 5
POSTPROCESSING = "beam_length_normalized"
# Original code (aston.py) - not specified in the paper
NUM_LAYERS = 2
DROPOUT = 0.1
TIME_FEATURES = 6


class _NumpyCompatUnpickler(pickle.Unpickler):
    """Lets numpy 1.x (ppm-sutran-best) read aston_data.pkl written with numpy 2
    (ml-jupyter-gpu): numpy 2 renamed ``numpy.core`` to ``numpy._core``."""

    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except ModuleNotFoundError:
            if not module.startswith('numpy._core'):
                raise
            return super().find_class('numpy.core' + module[len('numpy._core'):], name)


class _PrecomputedSuffixes:
    """Feeds the ASTON beam-search predictions to BEST's ``inference_loop``,
    which asks for one suffix per test instance, in dataset order."""

    def __init__(self, suffixes):
        self._rows = iter(suffixes.tolist())

    def predict_suffix(self, prefix, max_len):
        return next(self._rows)


def _make_dataset(split, traces, meta):
    grnn = [traces[t][:k] for t, k in zip(split['trace_idx'], split['X_valid_len'])]
    # TracesDatasetSeqTime expects X[idx] as (num_features, max_len)
    return TracesDatasetSeqTime(split['X'].transpose(0, 2, 1), split['X_valid_len'],
                                split['Y'], split['Y_valid_len'], grnn,
                                meta['N_places'], meta['F'], meta['window_size'])


def _build_model(meta):
    encoder = EncoderGRNNHiddenState(meta['num_activities'], meta['num_resources'], EMBED_SIZE,
                                     NUM_HIDDENS, TIME_FEATURES, NUM_LAYERS, meta['N_places'],
                                     meta['F'], None, meta['adjacency'], DROPOUT)
    decoder = DecoderLastGRNNState(meta['num_activities'], EMBED_SIZE, NUM_HIDDENS, NUM_LAYERS,
                                   DROPOUT)
    return EncoderDecoder(encoder, decoder)


def _xavier_init_weights(m):
    # trainer.train_seq2seq_mixed
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
    if type(m) == torch.nn.GRU:
        for param in m._flat_weights_names:
            if "weight" in param:
                torch.nn.init.xavier_uniform_(m._parameters[param])


def _batch_loss(net, batch, loss_fn, device):
    """Per-instance teacher-forced loss, as in trainer.train_seq2seq_mixed."""
    X, X_valid_len, Y, Y_valid_len, X_grnn = [x.to(device) for x in batch]
    dec_input = torch.cat([X[:, -1, 0].reshape(-1, 1), Y[:, :-1]], 1)
    Y_hat, _ = net(X, dec_input, X_valid_len, X_grnn)
    return loss_fn(Y_hat, Y)


def train_eval(log_name, run_id=1, results_dir=None, do_train=True, do_eval=True):
    """Train ASTON and evaluate it on the SuTraN test set of ``log_name``.

    Parameters
    ----------
    log_name : str
        Event log name; ``create_aston_data.py`` must have been run for it.
    run_id : int
        Repetition index, used as the random seed.
    results_dir : str or None
        Output directory. Defaults to
        ``results_per_log/<log_name>/ASTON_results``.
    do_train : bool
        If False, load ``trained_model.pt`` from ``results_dir`` instead.
    do_eval : bool
        If False, only time the inference and write ``inference_time.pkl``.
    """
    random.seed(run_id)
    np.random.seed(run_id)
    torch.manual_seed(run_id)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_id)
    device = Utils.try_gpu()

    with open(os.path.join(RESULTS_BASE, log_name, 'aston_data.pkl'), 'rb') as f:
        data = _NumpyCompatUnpickler(f).load()
    meta = data['meta']
    W = meta['window_size']
    eos = meta['num_activities'] - 1
    datasets = {s: _make_dataset(data[s], data['traces'], meta) for s in ('train', 'val', 'test')}

    backup_path = results_dir or os.path.join(RESULTS_BASE, log_name, "ASTON_results")
    os.makedirs(backup_path, exist_ok=True)
    model_path = os.path.join(backup_path, 'trained_model.pt')

    net = _build_model(meta)
    net.apply(_xavier_init_weights)
    net.to(device)
    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

    # -----------------------------------------------------------------------
    # Training: keep the epoch with the lowest validation loss
    # -----------------------------------------------------------------------
    if do_train:
        train_iter = DataLoader(datasets['train'], BATCH_SIZE, shuffle=True)
        val_iter = DataLoader(datasets['val'], BATCH_SIZE)
        loss_fn = SoftmaxCELoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=LR)
        best_val_loss = float('inf')

        _train_start = time.time()
        for epoch in range(NUM_EPOCHS):
            net.train()
            train_loss = 0.0
            for batch in train_iter:
                optimizer.zero_grad()
                l = _batch_loss(net, batch, loss_fn, device)
                l.sum().backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1)  # d2l.grad_clipping(net, 1)
                optimizer.step()
                train_loss += l.sum().item()

            net.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_iter:
                    val_loss += _batch_loss(net, batch, loss_fn, device).sum().item()
            train_loss /= len(datasets['train'])
            val_loss /= len(datasets['val'])

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(net.state_dict(), model_path)
            print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - train loss {train_loss:.4f} - "
                  f"val loss {val_loss:.4f} (best {best_val_loss:.4f})", flush=True)
        training_time = time.time() - _train_start
    else:
        training_time = 0.0
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"do_train=False but no saved model at {model_path}")
    net.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    net.eval()

    # -----------------------------------------------------------------------
    # Inference: length-normalised beam search (original predicter)
    # -----------------------------------------------------------------------
    test_iter = DataLoader(datasets['test'], BATCH_SIZE)
    _test_start = time.time()
    with torch.no_grad():
        preds = batched_beam_decode_optimized(net, test_iter, W, BEAM_WIDTH, eos, device, None,
                                              True, postprocessing_type=POSTPROCESSING,
                                              max_len=W, load_net=False)
    suffix_acts_pred = torch.cat(preds).to(torch.int64)  # (N, W)
    inference_time = time.time() - _test_start

    results_path = os.path.join(backup_path, "TEST_SET_RESULTS")
    os.makedirs(results_path, exist_ok=True)

    if not do_eval:
        with open(os.path.join(results_path, 'inference_time.pkl'), 'wb') as _f:
            pickle.dump({"log": log_name, "model": "ASTON", "run_id": run_id,
                         "training_time": training_time,
                         "inference_time": inference_time}, _f)
        print("Inference time (s): {} -- written to {}".format(inference_time, results_path))
        return

    # -----------------------------------------------------------------------
    # Evaluation: identical to TRAIN_EVAL_BEST
    # -----------------------------------------------------------------------
    from BEST.inference_procedure_best import inference_loop
    from TRAIN_EVAL_BEST import _build_suffix_graph, _graph_edit_similarity

    def load_dict(path_name):
        with open(path_name, 'rb') as file:
            return pickle.load(file)

    sutran_dir = os.path.join(RESULTS_BASE, log_name)
    cat_cols_dict = load_dict(os.path.join(sutran_dir, log_name + '_cat_cols_dict.pkl'))
    train_means_dict = load_dict(os.path.join(sutran_dir, log_name + '_train_means_dict.pkl'))
    train_std_dict = load_dict(os.path.join(sutran_dir, log_name + '_train_std_dict.pkl'))
    mean_std_ttne = [train_means_dict['timeLabel_df'][0], train_std_dict['timeLabel_df'][0]]
    mean_std_rrt = [train_means_dict['timeLabel_df'][1], train_std_dict['timeLabel_df'][1]]
    num_categoricals_pref = len(cat_cols_dict['prefix_df'])
    test_dataset = torch.load(os.path.join(sutran_dir, 'test_tensordataset.pt'), weights_only=False)

    inf_results, _, evaluation_time = inference_loop(
        best_model=_PrecomputedSuffixes(suffix_acts_pred),
        inference_dataset=test_dataset,
        num_categoricals_pref=num_categoricals_pref,
        mean_std_ttne=mean_std_ttne,
        mean_std_rrt=mean_std_rrt,
        results_path=results_path,
        dl_batch_size=512,
        do_eval=True,
        return_timing=True
    )
    testing_time = time.time() - _test_start

    avg_dam_lev          = inf_results[0]
    avg_MAE_minutes_RRT  = inf_results[8]
    avg_MAE_ttne_minutes = inf_results[9]
    results_dict_pref    = inf_results[-2]
    results_dict_suf     = inf_results[-1]

    print("\n=== ASTON Test Set Results ===")
    print("Avg 1-(normalised) DL similarity activity suffix: {}".format(avg_dam_lev))
    print("Percentage of suffixes predicted to END: too early - {} ; right moment - {} ; "
          "too late - {}".format(inf_results[1], inf_results[3], inf_results[2]))
    print("Avg absolute length difference: {}".format(inf_results[4]))
    print("Avg MAE TTNE (constant-mean predictor): {} (minutes)".format(avg_MAE_ttne_minutes))
    print("Avg MAE RRT  (constant-mean predictor): {} (minutes)".format(avg_MAE_minutes_RRT))

    avg_results_dict = {
        "DL sim"               : avg_dam_lev,
        "MAE TTNE minutes"     : avg_MAE_ttne_minutes,
        "MAE RRT minutes"      : avg_MAE_minutes_RRT,
        "training_time"        : training_time,
        "testing_time"         : testing_time,
        "inference_time"       : inference_time,
        "evaluation_time"      : evaluation_time,
        "num_trainable_params" : num_params,
    }
    # ── Next-act and GES metrics (as in TRAIN_EVAL_BEST) ────────────────────
    from sklearn.metrics import accuracy_score, f1_score as _f1_score
    _acts    = suffix_acts_pred
    _act_lbl = test_dataset[-1]
    next_acc = float(accuracy_score(_act_lbl[:, 0].numpy(), _acts[:, 0].numpy()))
    next_f1  = float(_f1_score(_act_lbl[:, 0].numpy(), _acts[:, 0].numpy(),
                               average='weighted', zero_division=0))
    avg_results_dict.update({
        'next_act_accuracy':    round(next_acc, 6),
        'next_act_f1_weighted': round(next_f1, 6),
    })
    _nb_labels = torch.load(os.path.join(sutran_dir, 'test_new_block_labels.pt'), weights_only=False)
    _ges_vals = []
    _suf_lens = []
    for _i in range(len(_acts)):
        _end_pos = (_acts[_i] == eos).nonzero(as_tuple=True)[0]
        _pl = int(_end_pos[0]) if len(_end_pos) > 0 else W
        _al = int((_act_lbl[_i] == eos).nonzero(as_tuple=True)[0][0])
        _suf_lens.append(_al)
        _G_pred = _build_suffix_graph(_acts[_i, :_pl].tolist(), [True] * _pl)
        _G_true = _build_suffix_graph(_act_lbl[_i, :_al].tolist(),
                                      (_nb_labels[_i, :_al] > 0.5).tolist())
        _ges_vals.append(_graph_edit_similarity(_G_pred, _G_true))
    ges = sum(_ges_vals) / len(_ges_vals) if _ges_vals else 1.0
    torch.save(torch.tensor(_ges_vals, dtype=torch.float32),
               os.path.join(results_path, 'ges_per_sample.pt'))
    avg_results_dict.update({'ges_approx': round(ges, 6)})
    with open(os.path.join(results_path, 'averaged_results.pkl'), 'wb') as f:
        pickle.dump(avg_results_dict, f)

    # ── Per-length result dictionaries (GES appended, as in TRAIN_EVAL_BEST) ─
    _pref_lens = (test_dataset[num_categoricals_pref - 1] != 0).sum(dim=1).tolist()
    _ges_by_pref = {}
    _ges_by_suf = {}
    for _plen, _slen, _gval in zip(_pref_lens, _suf_lens, _ges_vals):
        _ges_by_pref.setdefault(int(_plen), []).append(_gval)
        _ges_by_suf.setdefault(_slen, []).append(_gval)
    for _plen, _gvals in _ges_by_pref.items():
        if _plen in results_dict_pref:
            results_dict_pref[_plen].append(sum(_gvals) / len(_gvals))
    for _slen, _gvals in _ges_by_suf.items():
        if _slen in results_dict_suf:
            results_dict_suf[_slen].append(sum(_gvals) / len(_gvals))
    with open(os.path.join(results_path, 'prefix_length_results_dict.pkl'), 'wb') as f:
        pickle.dump(results_dict_pref, f)
    with open(os.path.join(results_path, 'suffix_length_results_dict.pkl'), 'wb') as f:
        pickle.dump(results_dict_suf, f)
