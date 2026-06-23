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
"""

import torch
import torch.nn as nn
import torch.nn.init as init
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.utils import softmax as pyg_softmax, scatter


class _EdgeAttnBias(nn.Module):
    """
    Residual edge-attention aggregation.

    For each target node i, computes softmax attention weights from edge_attr
    over the neighbourhood, then returns a weighted sum of (linearly projected)
    source-node features. Added as a residual to the GATv2 output so the
    time-delta on each edge can directly bias aggregation strength.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.attn_lin = nn.Linear(1, 1)
        self.val_lin  = nn.Linear(d_model, d_model)

    def forward(self, h, edge_index, edge_attr):
        src, dst = edge_index
        attn = pyg_softmax(self.attn_lin(edge_attr), dst, num_nodes=h.size(0))
        msg  = attn * self.val_lin(h[src])
        return scatter(msg, dst, dim=0, dim_size=h.size(0), reduce='sum')


class GATv2EncoderGRUDecoderNewBlock(nn.Module):
    """
    Standalone GATv2 encoder + GRU decoder with new-block head (no stop head).

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
    use_scheduled_sampling : bool
        If True, _scheduled_sampling is used during training instead of teacher forcing.
    """

    def __init__(self, num_activities: int, d_model: int = 64,
                 dropout: float = 0.2, n_layers: int = 1, nhead: int = 4,
                 use_scheduled_sampling: bool = False):
        super().__init__()
        self.num_activities         = num_activities
        self.d_model                = d_model
        self.n_layers               = n_layers
        self.use_scheduled_sampling = use_scheduled_sampling

        emb_size      = min(600, round(1.6 * (num_activities - 2) ** 0.56))
        self.emb_size = emb_size
        self.act_emb  = nn.Embedding(num_activities - 1, emb_size, padding_idx=0)
        self.dropout  = nn.Dropout(dropout)

        # GNN encoder — concat=False keeps output at d_model regardless of nhead
        self.gatv2     = GATv2Conv(emb_size + 1, d_model, heads=nhead, concat=False, edge_dim=1) # + 1 because of tss
        self.edge_bias = _EdgeAttnBias(d_model)
        self.bn_enc    = nn.BatchNorm1d(d_model * 2)
        self.enc_to_h  = nn.Linear(d_model * 2, d_model * n_layers)

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
        h = h + self.edge_bias(h, data.edge_index, data.edge_attr)
        h = self.dropout(h)
        h_global = global_mean_pool(h, data.batch)   # (B, d_model)
        h_last   = global_mean_pool(h[data.last_block_mask], data.batch[data.last_block_mask])  # (B, d_model)
        h = torch.cat([h_global, h_last], dim=-1)    # (B, 2*d_model)
        return self.bn_enc(h)

    def _init_gru_state(self, c):
        B = c.shape[0]
        return (self.enc_to_h(c)
                .view(B, self.n_layers, self.d_model)
                .permute(1, 0, 2).contiguous())       # (n_layers, B, d_model)

    def forward(self, data, window_size=None, mean_std_ttne=None,
                mean_std_tss=None, mean_std_tsp=None, p_teacher=1.0):
        c  = self._encode(data)
        h0 = self._init_gru_state(c)
        if self.training:
            if self.use_scheduled_sampling:
                return self._scheduled_sampling(data, h0, p_teacher,
                                                mean_std_ttne, mean_std_tss, mean_std_tsp)
            else:
                return self._teacher_forcing(data, h0)
        else:
            return self._autoregressive(data, h0, window_size,
                                        mean_std_ttne, mean_std_tss, mean_std_tsp)

    # ── Teacher forcing ────────────────────────────────────────────────────────

    def _teacher_forcing(self, data, h0):
        B  = data.num_graphs
        W  = data.suffix_act.shape[0] // B
        suffix_act = data.suffix_act.view(B, W)
        suffix_num = data.suffix_num.view(B, W, 2)

        dec_start_act = data.cat_x[data.ptr[1:] - 1, -1]
        dec_acts = torch.cat(
            [dec_start_act.unsqueeze(-1), suffix_act[:, :-1]], dim=-1
        ).clamp(max=self.num_activities - 2)              # (B, W)

        act_emb = self.act_emb(dec_acts)                  # (B, W, emb)
        tss_0   = data.last_prefix_num[:, [0]].unsqueeze(1)           # (B, 1, 1)
        tsp_0   = data.last_prefix_num[:, [1]].unsqueeze(1)           # (B, 1, 1)
        tss     = torch.cat([tss_0, suffix_num[:, :-1, [0]]], dim=1)  # (B, W, 1)
        tsp     = torch.cat([tsp_0, suffix_num[:, :-1, [1]]], dim=1)  # (B, W, 1)
        dec_in  = torch.cat([act_emb, tss, tsp], dim=-1)  # (B, W, emb+2)

        output, _ = self.decoder(dec_in, h0)               # (B, W, d_model)
        nb_logits = self.fc_new_block(output).squeeze(-1)  # (B, W)
        return self.fc_out_act(output), self.fc_out_ttne(output), nb_logits

    # ── Scheduled sampling ─────────────────────────────────────────────────────

    def _scheduled_sampling(self, data, h0, p_teacher, mean_std_ttne, mean_std_tss, mean_std_tsp):
        B  = data.num_graphs
        W  = data.suffix_act.shape[0] // B
        suffix_act = data.suffix_act.view(B, W)
        suffix_num = data.suffix_num.view(B, W, 2)

        dec_start_act = data.cat_x[data.ptr[1:] - 1, -1]
        gt_acts = torch.cat(
            [dec_start_act.unsqueeze(-1), suffix_act[:, :-1]], dim=-1
        ).clamp(max=self.num_activities - 2)              # (B, W)

        act_input = gt_acts[:, 0]                         # (B,)
        tss_curr  = data.last_prefix_num[:, 0]            # (B,)  last prefix ts_start
        tsp_curr  = data.last_prefix_num[:, 1]            # (B,)  last prefix ts_prev
        h = h0

        has_norm = mean_std_ttne is not None
        if has_norm:
            ttne_mean, ttne_std = mean_std_ttne
            tss_mean,  tss_std  = mean_std_tss
            tsp_mean,  tsp_std  = mean_std_tsp

        all_act_logits, all_ttne, all_nb = [], [], []

        for t in range(W):
            emb    = self.act_emb(act_input)
            dec_in = torch.cat([emb,
                                 tss_curr.unsqueeze(-1),
                                 tsp_curr.unsqueeze(-1)], dim=-1).unsqueeze(1)  # (B,1,emb+2)

            out, h = self.decoder(dec_in, h)
            out    = out.squeeze(1)                        # (B, d_model)

            act_logits = self.fc_out_act(out)              # (B, C)
            ttne_pred  = self.fc_out_ttne(out)             # (B, 1)
            nb_logit   = self.fc_new_block(out).squeeze(-1)  # (B,)

            all_act_logits.append(act_logits)
            all_ttne.append(ttne_pred)
            all_nb.append(nb_logit)

            if t + 1 < W:
                use_gt = torch.rand(B, device=out.device) < p_teacher  # (B,)

                pred_act  = act_logits.argmax(dim=-1).clamp(max=self.num_activities - 2)
                act_input = torch.where(use_gt, gt_acts[:, t + 1], pred_act)

                if has_norm:
                    ttne_secs = (ttne_pred[:, 0] * ttne_std + ttne_mean).clamp(min=0)
                    tss_secs  = (tss_curr * tss_std + tss_mean).clamp(min=0)
                    pred_tss  = (tss_secs + ttne_secs - tss_mean) / tss_std
                    # tsp feedback: concurrent at t → assume concurrent at t+1 → tsp = std(0)
                    pred_tsp  = torch.where(
                        nb_logit > 0,
                        (ttne_secs - tsp_mean) / tsp_std,
                        torch.full_like(tss_curr, -tsp_mean / tsp_std),
                    )
                    # GT path: suffix_num already encodes std(0) for concurrent events
                    tss_curr = torch.where(use_gt, suffix_num[:, t + 1, 0], pred_tss)
                    tsp_curr = torch.where(use_gt, suffix_num[:, t + 1, 1], pred_tsp)
                else:
                    tss_curr = suffix_num[:, t + 1, 0]
                    tsp_curr = suffix_num[:, t + 1, 1]

        act_out  = torch.stack(all_act_logits, dim=1)     # (B, W, C)
        ttne_out = torch.stack(all_ttne,       dim=1)     # (B, W, 1)
        nb_out   = torch.stack(all_nb,         dim=1)     # (B, W)
        return act_out, ttne_out, nb_out

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
