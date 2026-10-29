"""
GATv2 encoder + GRU decoder that predicts the suffix one whole concurrent
block per step.  No new-block head and no time outputs.

Predicts, at each decoder step, a multi-hot set over activities (fc_out_act,
sigmoid / BCE-trained).  Class 0 is padding (never used); class
num_activities-1 is END.  The suffix ends with the block {END}.
Duplicate activities inside one block collapse to a single label.

Time is still used as encoder INPUT (node tss, edge tsp), never predicted.

Decoder input at step t: mean activity embedding over the previous block
    step 0   the prefix's last block (same as v2)
    step t   block t-1 of the suffix (ground truth during training,
             predicted set at inference)

The GRU is initialised from h_global (pooled prefix).

Encoder: identical to model_suffix_v2 (stacked GATv2Conv + attention pooling).

Trained via teacher forcing only.
"""

import torch
import torch.nn as nn
import torch.nn.init as init
from torch_geometric.nn import GATv2Conv, global_mean_pool, AttentionalAggregation


def build_block_targets(act_labels, nb_labels, num_activities):
    """Multi-hot block targets from the per-event suffix labels.

    act_labels : (B, W) int64   act_label_seq (0 = pad, num_activities-1 = END)
    nb_labels  : (B, W) float   new_block_label

    Returns
    -------
    targets   : (B, W, C) float  block t's activities; step n_blocks holds {END}
    step_mask : (B, W)    bool   steps 0..n_blocks (real blocks + END step)
    n_blocks  : (B,)      int64  number of real blocks

    The first suffix event always opens block 0 (as in partial_order_metrics),
    even when it is concurrent with the prefix's last event.
    """
    B, W    = act_labels.shape
    C       = num_activities
    end_tok = C - 1
    device  = act_labels.device

    real = (act_labels != 0) & (act_labels != end_tok)                # (B, W)
    nb   = (nb_labels > 0.5) & real
    nb[:, 0] = real[:, 0]
    block_id = nb.long().cumsum(dim=1) - 1                            # (B, W)
    n_blocks = nb.long().sum(dim=1)                                   # (B,)

    targets = torch.zeros(B, W, C, device=device)
    b_idx   = torch.arange(B, device=device).unsqueeze(1).expand(B, W)
    targets[b_idx[real], block_id[real], act_labels[real]] = 1.0
    targets[torch.arange(B, device=device), n_blocks, end_tok] = 1.0

    step_mask = torch.arange(W, device=device).unsqueeze(0) <= n_blocks.unsqueeze(1)
    return targets, step_mask, n_blocks


class GATv2EncoderGRUDecoderBlockV2(nn.Module):
    """
    Standalone GATv2 encoder + GRU decoder with a multi-hot block head.

    Parameters
    ----------
    num_activities : int
        Total classes incl. padding (0) and END (num_activities-1).
    d_model : int
        Hidden size for GNN and GRU.
    dropout : float
    n_gru_layers : int
        Number of GRU layers in the decoder.
    n_gnn_layers : int
        Number of stacked GATv2Conv layers in the encoder.
    nhead : int
        Number of GATv2Conv heads (concat=False → GATv2 output is d_model
        regardless of nhead).
    """

    def __init__(self, num_activities: int, d_model: int = 64,
                 dropout: float = 0.2, n_gru_layers: int = 1, nhead: int = 4,
                 n_gnn_layers: int = 2):
        super().__init__()
        self.num_activities = num_activities
        self.d_model        = d_model
        self.n_gru_layers   = n_gru_layers

        emb_size      = min(600, round(1.6 * (num_activities - 2) ** 0.56))
        self.emb_size = emb_size
        self.act_emb  = nn.Embedding(num_activities - 1, emb_size, padding_idx=0)
        self.dropout  = nn.Dropout(dropout)

        # GNN encoder — concat=False keeps output at d_model regardless of nhead
        self.gnn_layers = nn.ModuleList(
            [GATv2Conv(emb_size + 1, d_model, heads=nhead, concat=False, edge_dim=1)]  # + 1 because of tss
            + [GATv2Conv(d_model, d_model, heads=nhead, concat=False, edge_dim=1)
               for _ in range(n_gnn_layers - 1)]
        )
        self.att_pool = AttentionalAggregation(gate_nn=nn.Linear(d_model, 1))
        self.bn_enc   = nn.BatchNorm1d(d_model)
        self.enc_to_h = nn.Linear(d_model, d_model * n_gru_layers)

        # GRU decoder: block embedding only
        dec_dropout  = dropout if n_gru_layers > 1 else 0.0
        self.decoder = nn.GRU(
            input_size=emb_size,
            hidden_size=d_model,
            num_layers=n_gru_layers,
            batch_first=True,
            dropout=dec_dropout,
        )

        self.fc_out_act = nn.Linear(d_model, num_activities)

        self._init_weights()

    def _init_weights(self):
        init.xavier_uniform_(self.fc_out_act.weight)
        for name, param in self.decoder.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                init.orthogonal_(param.data)

    def _encode(self, data):
        h = torch.cat([self.act_emb(data.cat_x[:, -1]), data.x[:, [0]]], dim=-1)
        h = self.dropout(h)
        for gnn in self.gnn_layers:
            h = gnn(h, data.edge_index, data.edge_attr).relu()
            h = self.dropout(h)
        h_global = self.att_pool(h, data.batch)             # (B, d_model)
        h_global = self.bn_enc(h_global)
        return h_global

    def _start_emb(self, data):
        """Mean activity embedding over the prefix's last block (order invariant)."""
        lb = data.last_block_mask                                           # (N,) bool
        act_emb = self.act_emb(data.cat_x[:, -1])                           # (N, emb)
        return global_mean_pool(act_emb[lb], data.batch[lb], size=data.num_graphs)  # (B, emb)

    def _block_emb(self, multi_hot):
        """Mean activity embedding over a multi-hot block (..., C) → (..., emb).
        Same aggregation as _start_emb; padding and END are ignored."""
        mh = multi_hot[..., 1:-1]                                           # real activities
        total = mh @ self.act_emb.weight[1:]
        return total / mh.sum(dim=-1, keepdim=True).clamp(min=1.0)

    def _init_gru_state(self, c):
        B = c.shape[0]
        return (self.enc_to_h(c)
                .view(B, self.n_gru_layers, self.d_model)
                .permute(1, 0, 2).contiguous())       # (n_gru_layers, B, d_model)

    def forward(self, data, window_size=None):
        c  = self._encode(data)
        h0 = self._init_gru_state(c)
        start_emb = self._start_emb(data)
        B = data.num_graphs
        W = window_size if window_size is not None else (data.suffix_act.shape[0] // B)
        if self.training:
            return self._teacher_forcing(data, h0, start_emb, W)
        return self._autoregressive(h0, start_emb, W)

    # ── Teacher forcing ────────────────────────────────────────────────────────

    def _teacher_forcing(self, data, h0, start_emb, W):
        B = start_emb.shape[0]
        targets, step_mask, _ = build_block_targets(
            data.act_label_seq.view(B, W), data.new_block_label.view(B, W),
            self.num_activities)

        prev_blocks = self._block_emb(targets[:, :-1])                          # inputs of steps 1..W-1
        dec_in      = torch.cat([start_emb.unsqueeze(1), prev_blocks], dim=1)   # (B, W, emb)

        output, _ = self.decoder(dec_in, h0)                                    # (B, W, d_model)
        return self.fc_out_act(output), targets, step_mask                     # (B,W,C), (B,W,C), (B,W)

    # ── Autoregressive inference ───────────────────────────────────────────────

    def _autoregressive(self, h0, start_emb, W):
        B      = start_emb.shape[0]
        C      = self.num_activities
        device = start_emb.device

        pred_blocks = torch.zeros(B, W, C, device=device)
        n_blocks    = torch.full((B,), W, dtype=torch.long, device=device)
        done        = torch.zeros(B, dtype=torch.bool, device=device)

        dec_in = start_emb
        h = h0
        for t in range(W):
            out, h = self.decoder(dec_in.unsqueeze(1), h)
            probs  = torch.sigmoid(self.fc_out_act(out.squeeze(1)))        # (B, C)

            is_end = (probs[:, -1] > 0.5) & ~done
            n_blocks[is_end] = t
            done = done | is_end

            # Real activities above threshold; fall back to the argmax so no block is empty
            real_p = probs[:, 1:-1]
            block  = (real_p > 0.5).float()
            empty  = (block.sum(dim=-1) == 0).nonzero(as_tuple=True)[0]
            block[empty, real_p[empty].argmax(dim=-1)] = 1.0
            block[done] = 0.0

            pred_blocks[:, t, 1:-1] = block
            if done.all():
                break
            dec_in = self._block_emb(pred_blocks[:, t])

        return pred_blocks, n_blocks
