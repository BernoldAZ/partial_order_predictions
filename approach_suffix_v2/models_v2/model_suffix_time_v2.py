"""
GATv2 encoder + GRU decoder with new-block head only.
Standalone implementation — no inheritance from v1/v3.

Predicts:
  - activity suffix      (fc_out_act)
  - time to next event   (fc_out_ttne)
  - new-block label      (fc_new_block) BCE-trained
      target=1  event starts a new concurrent block
      target=0  event is concurrent with the previous one

tsp/tss feedback during decoding (gap = TTNE if new_block=1 else 0):
    new_block=1  →  tsp[t+1] = TTNE-derived, tss[t+1] = tss[t] + TTNE (sequential event)
    new_block=0  →  tsp[t+1] = standardised(0), tss[t+1] = tss[t] (concurrent, no time gap)

The GRU is initialised from h_global (pooled prefix), which is the decoder's
only view of the encoded prefix.

Encoder: n_gnn_layers stacked GATv2Conv layers (no residual), followed by
attention pooling (AttentionalAggregation) into h_global.

First decoder step is invariant to the order of events within the prefix's
last concurrent block:
    activity  mean activity embedding over all nodes of the last block
    tss       last_prefix_num[:, 0] (identical for every node of the block)
    tsp       edge_attr of the inter-block edge entering the last node of the
              prefix (identical for every edge between the two blocks);
              falls back to last_prefix_num[:, 1] for single-block prefixes

Difference from v1: trained via teacher forcing only (no scheduled sampling).
"""

import torch
import torch.nn as nn
import torch.nn.init as init
from torch_geometric.nn import GATv2Conv, global_mean_pool, AttentionalAggregation


class GATv2EncoderGRUDecoderNewBlockV2(nn.Module):
    """
    Standalone GATv2 encoder + GRU decoder with new-block head (no stop head).
    No _EdgeAttnBias residual; encoder context is h_global only.

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

        # GRU decoder: act_emb + tss + tsp
        dec_dropout  = dropout if n_gru_layers > 1 else 0.0
        self.decoder = nn.GRU(
            input_size=emb_size + 2,
            hidden_size=d_model,
            num_layers=n_gru_layers,
            batch_first=True,
            dropout=dec_dropout,
        )

        self.fc_out_act   = nn.Linear(d_model, num_activities)
        self.fc_out_ttne  = nn.Linear(d_model, 1)
        self.fc_new_block = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self):
        init.xavier_uniform_(self.fc_out_act.weight)
        init.xavier_uniform_(self.fc_out_ttne.weight)
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

    def _start_step(self, data):
        """Order-invariant input for the first decoder step (see module docstring)."""
        B  = data.num_graphs
        lb = data.last_block_mask                                           # (N,) bool
        act_emb   = self.act_emb(data.cat_x[:, -1])                         # (N, emb)
        start_emb = global_mean_pool(act_emb[lb], data.batch[lb], size=B)   # (B, emb)

        # Inter-block edges entering the last node of each prefix: source outside the last block
        src, dst = data.edge_index
        is_last_node = torch.zeros_like(lb)
        is_last_node[data.ptr[1:] - 1] = True
        in_edge = is_last_node[dst] & ~lb[src]
        tsp_0 = data.last_prefix_num[:, 1].clone()                          # (B,) fallback: single-block prefix
        tsp_0[data.batch[dst[in_edge]]] = data.edge_attr[in_edge, 0]
        return start_emb, tsp_0

    def _init_gru_state(self, c):
        B = c.shape[0]
        return (self.enc_to_h(c)
                .view(B, self.n_gru_layers, self.d_model)
                .permute(1, 0, 2).contiguous())       # (n_gru_layers, B, d_model)

    def forward(self, data, window_size=None, mean_std_ttne=None,
                mean_std_tss=None, mean_std_tsp=None):
        c  = self._encode(data)
        h0 = self._init_gru_state(c)
        start_emb, tsp_0 = self._start_step(data)
        if self.training:
            return self._teacher_forcing(data, h0, start_emb, tsp_0)
        else:
            return self._autoregressive(data, h0, start_emb, tsp_0, window_size,
                                        mean_std_ttne, mean_std_tss, mean_std_tsp)

    # ── Parallel decoding helpers ──────────────────────────────────────────────

    def _gt_inputs(self, data, tsp_0):
        """Ground-truth decoder inputs; step 0 (start_emb) is added in _decode."""
        B  = data.num_graphs
        W  = data.suffix_act.shape[0] // B
        suffix_act = data.suffix_act.view(B, W)
        suffix_num = data.suffix_num.view(B, W, 2)

        dec_acts = suffix_act[:, :-1].clamp(max=self.num_activities - 2)  # (B, W-1) inputs of steps 1..W-1
        tss_0    = data.last_prefix_num[:, [0]].unsqueeze(1)                         # (B, 1, 1)
        tss      = torch.cat([tss_0, suffix_num[:, :-1, [0]]], dim=1)                # (B, W, 1)
        tsp      = torch.cat([tsp_0.view(B, 1, 1), suffix_num[:, :-1, [1]]], dim=1)  # (B, W, 1)
        return dec_acts, tss, tsp

    def _decode(self, h0, start_emb, dec_acts, tss, tsp):
        """One parallel decoder pass over all W steps."""
        act_emb = torch.cat([start_emb.unsqueeze(1), self.act_emb(dec_acts)], dim=1)  # (B, W, emb)
        dec_in  = torch.cat([act_emb, tss, tsp], dim=-1)   # (B, W, emb+2)

        output, _ = self.decoder(dec_in, h0)               # (B, W, d_model)
        nb_logits = self.fc_new_block(output).squeeze(-1)  # (B, W)
        return self.fc_out_act(output), self.fc_out_ttne(output), nb_logits

    # ── Teacher forcing ────────────────────────────────────────────────────────

    def _teacher_forcing(self, data, h0, start_emb, tsp_0):
        return self._decode(h0, start_emb, *self._gt_inputs(data, tsp_0))

    # ── Autoregressive inference ───────────────────────────────────────────────

    def _autoregressive(self, data, h0, start_emb, tsp_0, window_size,
                        mean_std_ttne, mean_std_tss, mean_std_tsp):
        B      = h0.shape[1]
        device = h0.device
        ttne_mean, ttne_std = mean_std_ttne
        tss_mean,  tss_std  = mean_std_tss
        tsp_mean,  tsp_std  = mean_std_tsp

        W = window_size if window_size is not None else (data.suffix_num.shape[0] // B)

        suffix_acts = torch.zeros(B, W, dtype=torch.long,  device=device)
        suffix_ttne = torch.zeros(B, W, dtype=torch.float, device=device)
        suffix_nb   = torch.zeros(B, W, dtype=torch.float, device=device)

        tss_curr  = data.last_prefix_num[:, 0]            # last prefix block ts_start
        tsp_curr  = tsp_0                                 # inter-block gap into last prefix block

        h = h0
        for t in range(W):
            emb    = start_emb if t == 0 else self.act_emb(act_input)
            dec_in = torch.cat([emb,
                                 tss_curr.unsqueeze(-1),
                                 tsp_curr.unsqueeze(-1)], dim=-1).unsqueeze(1)

            out, h = self.decoder(dec_in, h)
            out    = out.squeeze(1)                                 # (B, d_model)

            act_logits = self.fc_out_act(out)                         # (B, C)
            ttne_pred  = self.fc_out_ttne(out)                        # (B, 1)
            nb_logit   = self.fc_new_block(out).squeeze(-1)           # (B,)

            act_logits[:, 0] = -1e9
            act_selected = act_logits.argmax(dim=-1)                  # (B,)

            suffix_acts[:, t] = act_selected
            suffix_ttne[:, t] = torch.where(nb_logit > 0, ttne_pred[:, 0],
                                            torch.full((B,), -ttne_mean / ttne_std, device=device))
            suffix_nb[:, t]   = (nb_logit > 0).float()

            ttne_secs = (ttne_pred[:, 0] * ttne_std + ttne_mean).clamp(min=0)
            ttne_secs = torch.where(nb_logit > 0, ttne_secs, torch.zeros_like(ttne_secs))
            tss_secs  = (tss_curr * tss_std + tss_mean).clamp(min=0)
            tss_curr  = (tss_secs + ttne_secs - tss_mean) / tss_std
            tsp_curr  = (ttne_secs - tsp_mean) / tsp_std
            act_input = act_selected.clamp(max=self.num_activities - 2)

        return suffix_acts, suffix_ttne, suffix_nb
