"""
GATv2 encoder + GRU decoder with cross-attention over prefix nodes and a
binary stop head.  Based on v2 (GATv2EncoderGRUDecoderStop) with one change:
instead of compressing the prefix graph into a single h0 vector, the decoder
attends over all encoded node embeddings at every step via cross-attention.

Architecture
------------
Encoder : same as v1/v2 — activity embedding → GATv2Conv + edge-attention
          residual → keeps per-node embeddings padded to (B, N_max, d_model)
          → global_mean_pool + last-node → BN → linear → GRU h0

Decoder : at each step, GRU output (query) attends over all encoded prefix
          nodes (keys/values) via nn.MultiheadAttention; the context vector is
          concatenated with the GRU output and projected to d_model before
          the prediction heads.

          fc_out_act  : activity logits
          fc_out_ttne : time-to-next-event regression
          fc_stop     : binary stop head (same as v2)

Training modes (selected at construction):
  use_scheduled_sampling=False : teacher forcing — all W steps in parallel;
                                  queries=(B,W,d), keys/values=(B,N,d).
  use_scheduled_sampling=True  : scheduled sampling — step-by-step with
                                  p_teacher annealing 1.0→0.0.
"""

import torch
import torch.nn as nn
import torch.nn.init as init
from torch_geometric.nn import GATv2Conv, global_mean_pool

from model_suffix_time_v1 import _EdgeAttnBias


class GATv2CrossAttnGRUDecoder(nn.Module):

    def __init__(self, num_activities, d_model=64, dropout=0.2,
                 n_layers=1, nhead=4, use_scheduled_sampling=False):
        """
        Parameters
        ----------
        num_activities : total classes incl. padding (0) and END (num_activities-1)
        d_model        : hidden size for GNN, GRU, and attention
        dropout        : applied after embeddings, GATv2, and in cross-attention
        n_layers       : GRU decoder layers
        nhead          : attention heads for GATv2Conv and cross-attention
        use_scheduled_sampling : if True, use scheduled sampling during training
        """
        super().__init__()
        self.num_activities         = num_activities
        self.d_model                = d_model
        self.n_layers               = n_layers
        self.use_scheduled_sampling = use_scheduled_sampling

        emb_size      = min(600, round(1.6 * (num_activities - 2) ** 0.56))
        self.emb_size = emb_size
        self.act_emb  = nn.Embedding(num_activities - 1, emb_size, padding_idx=0)
        self.dropout  = nn.Dropout(dropout)

        # GNN encoder
        self.gatv2     = GATv2Conv(emb_size, d_model, heads=nhead, concat=False, edge_dim=1)
        self.edge_bias = _EdgeAttnBias(d_model)
        self.bn_enc    = nn.BatchNorm1d(d_model * 2)
        self.enc_to_h  = nn.Linear(d_model * 2, d_model * n_layers)

        # GRU decoder
        dec_dropout  = dropout if n_layers > 1 else 0.0
        self.decoder = nn.GRU(
            input_size=emb_size + 2,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dec_dropout,
        )

        # Cross-attention: decoder queries attend over encoder node embeddings
        self.cross_attn   = nn.MultiheadAttention(d_model, nhead, batch_first=True,
                                                   dropout=dropout)
        self.context_proj = nn.Linear(d_model * 2, d_model)
        init.xavier_uniform_(self.context_proj.weight)

        # Output heads
        self.fc_out_act  = nn.Linear(d_model, num_activities)
        self.fc_out_ttne = nn.Linear(d_model, 1)
        self.fc_stop     = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self):
        for lin in (self.fc_out_act, self.fc_out_ttne, self.fc_stop):
            init.xavier_uniform_(lin.weight)
        for name, param in self.decoder.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                init.orthogonal_(param.data)

    # ── Encoder ───────────────────────────────────────────────────────────────

    def _encode(self, data):
        h = self.act_emb(data.cat_x[:, -1])
        h = self.dropout(h)
        h = self.gatv2(h, data.edge_index, data.edge_attr).relu()
        h = h + self.edge_bias(h, data.edge_index, data.edge_attr)
        h = self.dropout(h)

        # Build padded node tensor (B, N_max, d_model) for cross-attention.
        # node_mask is True at padding positions (key_padding_mask convention).
        sizes     = (data.ptr[1:] - data.ptr[:-1]).tolist()
        N_max     = max(sizes)
        B         = data.num_graphs
        node_h    = h.new_zeros(B, N_max, self.d_model)
        node_mask = torch.ones(B, N_max, dtype=torch.bool, device=h.device)
        for i, (start, sz) in enumerate(zip(data.ptr[:-1].tolist(), sizes)):
            node_h[i, :sz]    = h[start:start + sz]
            node_mask[i, :sz] = False

        h_global = global_mean_pool(h, data.batch)
        h_last   = h[data.ptr[1:] - 1]
        c        = self.bn_enc(torch.cat([h_global, h_last], dim=-1))
        return c, node_h, node_mask

    def _init_gru_state(self, c):
        B = c.shape[0]
        return (self.enc_to_h(c)
                .view(B, self.n_layers, self.d_model)
                .permute(1, 0, 2).contiguous())           # (n_layers, B, d_model)

    # ── Cross-attention helper ────────────────────────────────────────────────

    def _attend(self, out, node_h, node_mask):
        """
        out       : (B, T, d_model)  — GRU output; T=W (teacher) or T=1 (step)
        node_h    : (B, N_max, d_model)
        node_mask : (B, N_max) bool, True = padding
        Returns   : (B, T, d_model)  — projected([out; context])
        """
        ctx, _ = self.cross_attn(out, node_h, node_h, key_padding_mask=node_mask)
        return self.context_proj(torch.cat([out, ctx], dim=-1))

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, data, window_size=None, mean_std_ttne=None,
                mean_std_tss=None, mean_std_tsp=None, p_teacher=1.0):
        c, node_h, node_mask = self._encode(data)
        h0 = self._init_gru_state(c)
        if self.training:
            if self.use_scheduled_sampling:
                return self._scheduled_sampling(data, h0, node_h, node_mask, p_teacher,
                                                mean_std_ttne, mean_std_tss, mean_std_tsp)
            else:
                return self._teacher_forcing(data, h0, node_h, node_mask)
        else:
            return self._autoregressive(data, h0, node_h, node_mask, window_size,
                                        mean_std_ttne, mean_std_tss, mean_std_tsp)

    # ── Teacher forcing (all W steps in parallel) ─────────────────────────────

    def _teacher_forcing(self, data, h0, node_h, node_mask):
        B  = data.num_graphs
        W  = data.suffix_act.shape[0] // B
        suffix_act = data.suffix_act.view(B, W)
        suffix_num = data.suffix_num.view(B, W, 2)

        dec_start_act = data.cat_x[data.ptr[1:] - 1, -1]
        dec_acts = torch.cat(
            [dec_start_act.unsqueeze(-1), suffix_act[:, :-1]], dim=-1
        ).clamp(max=self.num_activities - 2)

        act_emb = self.act_emb(dec_acts)
        tss     = suffix_num[:, :, 0].unsqueeze(-1)
        tsp     = suffix_num[:, :, 1].unsqueeze(-1)
        dec_in  = torch.cat([act_emb, tss, tsp], dim=-1)          # (B, W, emb+2)

        output, _   = self.decoder(dec_in, h0)                     # (B, W, d_model)
        output      = self._attend(output, node_h, node_mask)      # (B, W, d_model)
        stop_logits = self.fc_stop(output).squeeze(-1)             # (B, W)
        return self.fc_out_act(output), self.fc_out_ttne(output), stop_logits

    # ── Scheduled sampling (step-by-step) ─────────────────────────────────────

    def _scheduled_sampling(self, data, h0, node_h, node_mask, p_teacher,
                             mean_std_ttne, mean_std_tss, mean_std_tsp):
        B  = data.num_graphs
        W  = data.suffix_act.shape[0] // B
        suffix_act = data.suffix_act.view(B, W)
        suffix_num = data.suffix_num.view(B, W, 2)

        dec_start_act = data.cat_x[data.ptr[1:] - 1, -1]
        gt_acts = torch.cat(
            [dec_start_act.unsqueeze(-1), suffix_act[:, :-1]], dim=-1
        ).clamp(max=self.num_activities - 2)

        act_input = gt_acts[:, 0]
        tss_curr  = suffix_num[:, 0, 0]
        tsp_curr  = suffix_num[:, 0, 1]
        h         = h0

        has_norm = mean_std_ttne is not None
        if has_norm:
            ttne_mean, ttne_std = mean_std_ttne
            tss_mean,  tss_std  = mean_std_tss
            tsp_mean,  tsp_std  = mean_std_tsp

        all_act_logits, all_ttne, all_stop = [], [], []

        for t in range(W):
            emb    = self.act_emb(act_input)
            dec_in = torch.cat([emb,
                                 tss_curr.unsqueeze(-1),
                                 tsp_curr.unsqueeze(-1)], dim=-1).unsqueeze(1)  # (B,1,emb+2)

            out, h = self.decoder(dec_in, h)                      # (B, 1, d_model)
            out    = self._attend(out, node_h, node_mask)         # (B, 1, d_model)
            out    = out.squeeze(1)                                # (B, d_model)

            act_logits = self.fc_out_act(out)
            ttne_pred  = self.fc_out_ttne(out)
            stop_logit = self.fc_stop(out).squeeze(-1)

            all_act_logits.append(act_logits)
            all_ttne.append(ttne_pred)
            all_stop.append(stop_logit)

            if t + 1 < W:
                use_gt    = torch.rand(B, device=out.device) < p_teacher
                pred_act  = act_logits.argmax(dim=-1).clamp(max=self.num_activities - 2)
                act_input = torch.where(use_gt, gt_acts[:, t + 1], pred_act)

                if has_norm:
                    ttne_secs = (ttne_pred[:, 0] * ttne_std + ttne_mean).clamp(min=0)
                    tss_secs  = (tss_curr  * tss_std  + tss_mean).clamp(min=0)
                    pred_tss  = (tss_secs + ttne_secs - tss_mean) / tss_std
                    pred_tsp  = (ttne_secs - tsp_mean) / tsp_std
                    tss_curr  = torch.where(use_gt, suffix_num[:, t + 1, 0], pred_tss)
                    tsp_curr  = torch.where(use_gt, suffix_num[:, t + 1, 1], pred_tsp)
                else:
                    tss_curr = suffix_num[:, t + 1, 0]
                    tsp_curr = suffix_num[:, t + 1, 1]

        act_out  = torch.stack(all_act_logits, dim=1)             # (B, W, C)
        ttne_out = torch.stack(all_ttne,       dim=1)             # (B, W, 1)
        stop_out = torch.stack(all_stop,       dim=1)             # (B, W)
        return act_out, ttne_out, stop_out

    # ── Autoregressive inference ───────────────────────────────────────────────

    def _autoregressive(self, data, h0, node_h, node_mask, window_size,
                        mean_std_ttne, mean_std_tss, mean_std_tsp):
        B      = h0.shape[1]
        device = h0.device
        ttne_mean, ttne_std = mean_std_ttne
        tss_mean,  tss_std  = mean_std_tss
        tsp_mean,  tsp_std  = mean_std_tsp

        W          = window_size if window_size is not None else (data.suffix_num.shape[0] // B)
        suffix_num = data.suffix_num.view(B, W, 2)

        suffix_acts = torch.zeros(B, W, dtype=torch.long,  device=device)
        suffix_ttne = torch.zeros(B, W, dtype=torch.float, device=device)

        act_input = data.cat_x[data.ptr[1:] - 1, -1].clamp(max=self.num_activities - 2)
        tss_curr  = suffix_num[:, 0, 0]
        tsp_curr  = suffix_num[:, 0, 1]
        h         = h0

        for t in range(W):
            emb    = self.act_emb(act_input)
            dec_in = torch.cat([emb,
                                 tss_curr.unsqueeze(-1),
                                 tsp_curr.unsqueeze(-1)], dim=-1).unsqueeze(1)

            out, h = self.decoder(dec_in, h)                      # (B, 1, d_model)
            out    = self._attend(out, node_h, node_mask)         # (B, 1, d_model)
            out    = out.squeeze(1)                                # (B, d_model)

            act_logits = self.fc_out_act(out)
            stop_logit = self.fc_stop(out).squeeze(-1)
            ttne_pred  = self.fc_out_ttne(out)

            act_logits[:, 0]  = -1e9
            act_logits[:, -1] = act_logits[:, -1] + stop_logit   # soft stop bias

            act_selected = act_logits.argmax(dim=-1)
            suffix_acts[:, t] = act_selected
            suffix_ttne[:, t] = ttne_pred[:, 0]

            ttne_secs = (ttne_pred[:, 0] * ttne_std + ttne_mean).clamp(min=0)
            tss_secs  = (tss_curr * tss_std + tss_mean).clamp(min=0)
            tss_curr  = (tss_secs + ttne_secs - tss_mean) / tss_std
            tsp_curr  = (ttne_secs - tsp_mean) / tsp_std
            act_input = act_selected.clamp(max=self.num_activities - 2)

        return suffix_acts, suffix_ttne
