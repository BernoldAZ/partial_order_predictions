"""
GATv2 encoder + GRU decoder with new-block head only, no time outputs.
Standalone implementation — no inheritance from v1/v3.

Predicts:
  - activity suffix      (fc_out_act)
  - new-block label      (fc_new_block) BCE-trained
      target=1  event starts a new concurrent block
      target=0  event is concurrent with the previous one

Time is still used as encoder INPUT (node tss, edge tsp), never predicted.

Decoder input at step t: [act_emb(a_{t-1}), nb_{t-1}]
    step 0   mean activity embedding over all nodes of the prefix's last block
             (order invariant), nb = 1
    step t   embedding of the single previous event and its new-block bit
             (ground truth during training, predicted (logit > 0) at inference)

The GRU is initialised from h_global (pooled prefix), which is the decoder's
only view of the encoded prefix.

Encoder: n_gnn_layers stacked GATv2Conv layers (no residual), followed by
attention pooling (AttentionalAggregation) into h_global.

Trained via teacher forcing only.
"""

import torch
import torch.nn as nn
import torch.nn.init as init
from torch_geometric.nn import GATv2Conv, global_mean_pool, AttentionalAggregation


class GATv2EncoderGRUDecoderNewBlockV2NoTime(nn.Module):
    """
    Standalone GATv2 encoder + GRU decoder with activity and new-block heads.

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

        # GRU decoder: act_emb + nb
        dec_dropout  = dropout if n_gru_layers > 1 else 0.0
        self.decoder = nn.GRU(
            input_size=emb_size + 1,
            hidden_size=d_model,
            num_layers=n_gru_layers,
            batch_first=True,
            dropout=dec_dropout,
        )

        self.fc_out_act   = nn.Linear(d_model, num_activities)
        self.fc_new_block = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self):
        init.xavier_uniform_(self.fc_out_act.weight)
        init.xavier_uniform_(self.fc_new_block.weight)
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
        suffix_act = data.suffix_act.view(B, W)
        nb_label   = data.new_block_label.view(B, W).float()

        dec_acts = suffix_act[:, :-1].clamp(max=self.num_activities - 2)        # inputs of steps 1..W-1
        act_emb  = torch.cat([start_emb.unsqueeze(1), self.act_emb(dec_acts)], dim=1)  # (B, W, emb)
        nb_in    = torch.cat([nb_label.new_ones(B, 1), nb_label[:, :-1]], dim=1).unsqueeze(-1)  # (B, W, 1)
        dec_in   = torch.cat([act_emb, nb_in], dim=-1)                          # (B, W, emb+1)

        output, _ = self.decoder(dec_in, h0)                                    # (B, W, d_model)
        return self.fc_out_act(output), self.fc_new_block(output).squeeze(-1)   # (B,W,C), (B,W)

    # ── Autoregressive inference ───────────────────────────────────────────────

    def _autoregressive(self, h0, start_emb, W):
        B      = start_emb.shape[0]
        device = start_emb.device

        suffix_acts = torch.zeros(B, W, dtype=torch.long,  device=device)
        suffix_nb   = torch.zeros(B, W, dtype=torch.float, device=device)

        nb_curr = torch.ones(B, device=device)
        h = h0
        for t in range(W):
            emb    = start_emb if t == 0 else self.act_emb(act_input)
            dec_in = torch.cat([emb, nb_curr.unsqueeze(-1)], dim=-1).unsqueeze(1)

            out, h = self.decoder(dec_in, h)
            out    = out.squeeze(1)                                 # (B, d_model)

            act_logits = self.fc_out_act(out)                       # (B, C)
            nb_logit   = self.fc_new_block(out).squeeze(-1)         # (B,)

            act_logits[:, 0] = -1e9
            act_selected = act_logits.argmax(dim=-1)                # (B,)

            suffix_acts[:, t] = act_selected
            nb_curr           = (nb_logit > 0).float()
            suffix_nb[:, t]   = nb_curr

            act_input = act_selected.clamp(max=self.num_activities - 2)

        return suffix_acts, suffix_nb
