"""
GATv2 encoder + GRU decoder with new-block head only.
Standalone implementation — no inheritance from v1/v2/v3/v4.

Predicts:
  - activity suffix      (fc_out_act)
  - time to next event   (fc_out_ttne)
  - new-block label      (fc_new_block) BCE-trained
      target=1  event starts a new concurrent block
      target=0  event is concurrent with the previous one

tsp feedback during decoding:
    new_block=1  →  tsp[t+1] = TTNE-derived (sequential event)
    new_block=0  →  tsp[t+1] = standardised(0) (concurrent, no time gap)

Difference from v1: trained as next-activity prediction only. Training
runs a single GRU decoder step (using the last prefix event as input) and
is supervised only against the first suffix position — the rest of the
suffix window is never touched during training. Test-time inference is
unchanged: full-suffix autoregressive rollout via _autoregressive.
"""

import torch
import torch.nn as nn
import torch.nn.init as init
from torch_geometric.nn import GATv2Conv, global_mean_pool


class GATv2EncoderGRUDecoderNewBlockV3(nn.Module):
    """
    Standalone GATv2 encoder + GRU decoder with new-block head (no stop head).
    No _EdgeAttnBias residual; no h_last: encoder context is h_global only.

    Parameters
    ----------
    num_activities : int
        Total classes incl. padding (0) and END (num_activities-1).
    d_model : int
        Hidden size for GNN and GRU.
    dropout : float
    n_layers : int
        Number of GRU layers in the decoder.
    nhead : int
        Number of GATv2Conv attention heads (concat=False → output is d_model).
    """

    def __init__(self, num_activities: int, d_model: int = 64,
                 dropout: float = 0.2, n_layers: int = 1, nhead: int = 4):
        super().__init__()
        self.num_activities = num_activities
        self.d_model        = d_model
        self.n_layers       = n_layers

        emb_size      = min(600, round(1.6 * (num_activities - 2) ** 0.56))
        self.emb_size = emb_size
        self.act_emb  = nn.Embedding(num_activities - 1, emb_size, padding_idx=0)
        self.dropout  = nn.Dropout(dropout)

        # GNN encoder — concat=False keeps output at d_model regardless of nhead
        self.gatv2    = GATv2Conv(emb_size + 1, d_model, heads=nhead, concat=False, edge_dim=1) # + 1 because of tss
        self.bn_enc   = nn.BatchNorm1d(d_model)
        self.enc_to_h = nn.Linear(d_model, d_model * n_layers)

        # GRU decoder: act_emb + tss + tsp
        dec_dropout  = dropout if n_layers > 1 else 0.0
        self.decoder = nn.GRU(
            input_size=emb_size + 2,
            hidden_size=d_model,
            num_layers=n_layers,
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
        h = self.gatv2(h, data.edge_index, data.edge_attr).relu()
        h = self.dropout(h)
        h_global = global_mean_pool(h, data.batch)   # (B, d_model)
        return self.bn_enc(h_global)

    def _init_gru_state(self, c):
        B = c.shape[0]
        return (self.enc_to_h(c)
                .view(B, self.n_layers, self.d_model)
                .permute(1, 0, 2).contiguous())       # (n_layers, B, d_model)

    def forward(self, data, window_size=None, mean_std_ttne=None,
                mean_std_tss=None, mean_std_tsp=None):
        c  = self._encode(data)
        h0 = self._init_gru_state(c)
        if self.training:
            return self._next_step(data, h0)
        else:
            return self._autoregressive(data, h0, window_size,
                                        mean_std_ttne, mean_std_tss, mean_std_tsp)

    # ── Next-activity prediction (single decoder step) ─────────────────────────

    def _next_step(self, data, h0):
        dec_start_act = data.cat_x[data.ptr[1:] - 1, -1].clamp(max=self.num_activities - 2)
        act_emb = self.act_emb(dec_start_act)                          # (B, emb)
        tss     = data.last_prefix_num[:, [0]]                         # (B, 1)
        tsp     = data.last_prefix_num[:, [1]]                         # (B, 1)
        dec_in  = torch.cat([act_emb, tss, tsp], dim=-1).unsqueeze(1)  # (B, 1, emb+2)

        output, _ = self.decoder(dec_in, h0)                # (B, 1, d_model)
        nb_logits = self.fc_new_block(output).squeeze(-1)   # (B, 1)
        return self.fc_out_act(output), self.fc_out_ttne(output), nb_logits

    # ── Autoregressive inference ───────────────────────────────────────────────

    def _autoregressive(self, data, h0, window_size, mean_std_ttne, mean_std_tss, mean_std_tsp):
        B      = h0.shape[1]
        device = h0.device
        ttne_mean, ttne_std = mean_std_ttne
        tss_mean,  tss_std  = mean_std_tss
        tsp_mean,  tsp_std  = mean_std_tsp

        W = window_size if window_size is not None else (data.suffix_num.shape[0] // B)

        suffix_acts = torch.zeros(B, W, dtype=torch.long,  device=device)
        suffix_ttne = torch.zeros(B, W, dtype=torch.float, device=device)
        suffix_nb   = torch.zeros(B, W, dtype=torch.float, device=device)

        act_input = data.cat_x[data.ptr[1:] - 1, -1].clamp(max=self.num_activities - 2)
        tss_curr  = data.last_prefix_num[:, 0]            # last prefix ts_start
        tsp_curr  = data.last_prefix_num[:, 1]            # last prefix ts_prev

        h = h0
        for t in range(W):
            emb    = self.act_emb(act_input)
            dec_in = torch.cat([emb,
                                 tss_curr.unsqueeze(-1),
                                 tsp_curr.unsqueeze(-1)], dim=-1).unsqueeze(1)

            out, h = self.decoder(dec_in, h)
            out    = out.squeeze(1)                                   # (B, d_model)

            act_logits = self.fc_out_act(out)                         # (B, C)
            ttne_pred  = self.fc_out_ttne(out)                        # (B, 1)
            nb_logit   = self.fc_new_block(out).squeeze(-1)           # (B,)

            act_logits[:, 0] = -1e9
            act_selected = act_logits.argmax(dim=-1)                  # (B,)

            suffix_acts[:, t] = act_selected
            suffix_ttne[:, t] = ttne_pred[:, 0]
            suffix_nb[:, t]   = (nb_logit > 0).float()

            ttne_secs = (ttne_pred[:, 0] * ttne_std + ttne_mean).clamp(min=0)
            tss_secs  = (tss_curr * tss_std + tss_mean).clamp(min=0)
            tss_curr  = (tss_secs + ttne_secs - tss_mean) / tss_std
            tsp_curr  = torch.where(
                nb_logit > 0,
                (ttne_secs - tsp_mean) / tsp_std,
                torch.full((B,), -tsp_mean / tsp_std, device=device),
            )
            act_input = act_selected.clamp(max=self.num_activities - 2)

        return suffix_acts, suffix_ttne, suffix_nb
