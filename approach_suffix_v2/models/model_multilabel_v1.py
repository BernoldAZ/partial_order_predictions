"""GATv2 encoder + GRU decoder for multi-label next-activity prediction.

Architecture
------------
Encoder : same GATv2 + edge-bias + BN as v1/v2.

Decoder : at each step, the input is the average of act_emb embeddings for
          all activities active in the previous block (multi-hot average),
          concatenated with normalised [tss, tsp] time features.

          input per step: [avg_act_emb | tss | tsp]  (emb_size + 2)

Heads
-----
  fc_out_multilabel : nn.Linear(d_model, C)  — BCEWithLogitsLoss
      C = num_activities - 1  (regular activities 0-indexed + END at C-1)
  fc_out_tsp        : nn.Linear(d_model, 1)  — MAE for time-to-next-block

Multi-hot convention
--------------------
  Multi-hot index i  →  activity integer i+1 in the embedding table.
  Index C-1 = cardinality is the END_TOKEN slot.
  Only indices 0..C-2 are looked up in act_emb; END is excluded from input
  embeddings (it only appears at masked/padding decoder positions).

Training: teacher forcing (default) or scheduled sampling.
Inference: autoregressive greedy; stops when END (index C-1) is predicted
           or T_max steps are exhausted.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.utils import softmax as pyg_softmax, scatter


class _EdgeAttnBias(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.attn_lin = nn.Linear(1, 1)
        self.val_lin  = nn.Linear(d_model, d_model)

    def forward(self, h, edge_index, edge_attr):
        src, dst = edge_index
        attn = pyg_softmax(self.attn_lin(edge_attr), dst, num_nodes=h.size(0))
        msg  = attn * self.val_lin(h[src])
        return scatter(msg, dst, dim=0, dim_size=h.size(0), reduce='sum')


class GATv2GRUMultilabel(nn.Module):
    def __init__(self, num_activities: int, d_model: int = 64,
                 dropout: float = 0.2, n_layers: int = 1, nhead: int = 4,
                 use_scheduled_sampling: bool = False):
        """
        Parameters
        ----------
        num_activities : int
            Total classes incl. padding (0) and END (num_activities-1).
            C = num_activities - 1 (multi-hot dimension).
        """
        super().__init__()
        self.num_activities         = num_activities
        self.C                      = num_activities - 1   # multi-hot dim
        self.d_model                = d_model
        self.n_layers               = n_layers
        self.use_scheduled_sampling = use_scheduled_sampling

        emb_size = min(600, round(1.6 * (num_activities - 2) ** 0.56))
        self.emb_size = emb_size
        # Index 0 = padding (never in decoder input), 1..C-1 = regular activities.
        # act_emb has C = num_activities-1 entries (indices 0..C-1).
        self.act_emb  = nn.Embedding(self.C, emb_size, padding_idx=0)
        self.dropout  = nn.Dropout(dropout)

        self.gatv2     = GATv2Conv(emb_size, d_model, heads=nhead,
                                   concat=False, edge_dim=1)
        self.edge_bias = _EdgeAttnBias(d_model)
        self.bn_enc    = nn.BatchNorm1d(d_model * 2)

        self.enc_to_h  = nn.Linear(d_model * 2, d_model * n_layers)

        dec_dropout = dropout if n_layers > 1 else 0.0
        self.decoder = nn.GRU(
            input_size=emb_size + 2,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dec_dropout,
        )

        self.fc_out_multilabel = nn.Linear(d_model, self.C)
        self.fc_out_tsp        = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self):
        init.xavier_uniform_(self.fc_out_multilabel.weight)
        init.xavier_uniform_(self.fc_out_tsp.weight)
        for name, param in self.decoder.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                init.orthogonal_(param.data)

    # ── Encoder ───────────────────────────────────────────────────────────

    def _encode(self, data):
        h = self.act_emb(data.cat_x[:, -1])
        h = self.dropout(h)
        h = self.gatv2(h, data.edge_index, data.edge_attr).relu()
        h = h + self.edge_bias(h, data.edge_index, data.edge_attr)
        h = self.dropout(h)
        h_global = global_mean_pool(h, data.batch)
        h_last   = h[data.ptr[1:] - 1]
        h = torch.cat([h_global, h_last], dim=-1)
        return self.bn_enc(h)

    def _init_gru_state(self, c):
        B  = c.shape[0]
        h0 = (self.enc_to_h(c)
              .view(B, self.n_layers, self.d_model)
              .permute(1, 0, 2).contiguous())
        return h0

    # ── Multi-hot → averaged embedding ───────────────────────────────────

    def _embed_multihot(self, mh):
        """Average act_emb over active regular activities in multi-hot.

        mh : (*, C) float multi-hot
        Returns: (*, emb_size)
        Only indices 0..C-2 are looked up (index C-1 = END is excluded).
        """
        # act_emb.weight has shape (C, emb_size); index 0 = padding (zeros).
        # Multi-hot index i corresponds to act_emb weight at index i+1
        # (activity integer i+1), BUT our act_emb is sized C = num_activities-1
        # and the mapping is: multi-hot index 0..C-2 → act_emb index 1..C-1.
        # act_emb.weight[1:C] has shape (C-1, emb_size).
        mh_reg = mh[..., :self.C - 1].float()                     # exclude END dim
        # (*, C-1) @ (C-1, emb_size) → (*, emb_size)
        W = self.act_emb.weight[1:]                                # (C-1, emb_size)
        shape = mh_reg.shape[:-1]
        flat  = mh_reg.reshape(-1, self.C - 1)                    # (N, C-1)
        emb   = flat @ W                                           # (N, emb_size)
        count = flat.sum(dim=-1, keepdim=True).clamp(min=1)       # (N, 1)
        return (emb / count).reshape(*shape, self.emb_size)       # (*, emb_size)

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, data, T_max=None, mean_std_tsp=None, mean_std_tss=None,
                p_teacher=1.0):
        c  = self._encode(data)
        h0 = self._init_gru_state(c)
        if self.training:
            if self.use_scheduled_sampling:
                return self._scheduled_sampling(data, h0, p_teacher,
                                                mean_std_tsp, mean_std_tss)
            else:
                return self._teacher_forcing(data, h0)
        else:
            return self._autoregressive(data, h0, T_max, mean_std_tsp, mean_std_tss)

    # ── Teacher forcing ───────────────────────────────────────────────────

    def _teacher_forcing(self, data, h0):
        B  = data.num_graphs
        C  = self.C
        T  = data.suffix_multihot.shape[0] // B

        mh  = data.suffix_multihot.view(B, T, C)      # (B, T, C)
        tss = data.suffix_tss.view(B, T)               # (B, T)
        tsp = data.suffix_tsp.view(B, T)               # (B, T)

        # Step 0 input: one-hot of last prefix activity (as multi-hot)
        a_start = data.cat_x[data.ptr[1:] - 1, -1]   # (B,) activity integer 1..C-1
        # Convert to multi-hot of regular activities: index = a_start - 1
        mh0 = torch.zeros(B, C, device=mh.device)
        idx = (a_start - 1).clamp(min=0, max=C - 2)
        mh0.scatter_(1, idx.unsqueeze(1), 1.0)         # (B, C)

        # Decoder inputs: [mh0, mh[:, 0], ..., mh[:, T-2]]
        mh_in = torch.cat([mh0.unsqueeze(1), mh[:, :-1]], dim=1)  # (B, T, C)
        emb   = self._embed_multihot(mh_in)                        # (B, T, emb)

        dec_in = torch.cat([emb,
                             tss.unsqueeze(-1),
                             tsp.unsqueeze(-1)], dim=-1)            # (B, T, emb+2)
        output, _ = self.decoder(dec_in, h0)                        # (B, T, d_model)
        return self.fc_out_multilabel(output), self.fc_out_tsp(output)
        # shapes: (B, T, C), (B, T, 1)

    # ── Scheduled sampling ────────────────────────────────────────────────

    def _scheduled_sampling(self, data, h0, p_teacher, mean_std_tsp, mean_std_tss):
        B  = data.num_graphs
        C  = self.C
        T  = data.suffix_multihot.shape[0] // B

        mh  = data.suffix_multihot.view(B, T, C)      # (B, T, C) – GT
        tss = data.suffix_tss.view(B, T)
        tsp = data.suffix_tsp.view(B, T)

        a_start = data.cat_x[data.ptr[1:] - 1, -1]
        mh0 = torch.zeros(B, C, device=mh.device)
        mh0.scatter_(1, (a_start - 1).clamp(min=0, max=C - 2).unsqueeze(1), 1.0)

        cur_mh  = mh0
        tss_c   = tss[:, 0]
        tsp_c   = tsp[:, 0]
        h = h0

        has_norm = mean_std_tsp is not None
        if has_norm:
            tsp_mean, tsp_std = mean_std_tsp
            tss_mean, tss_std = mean_std_tss

        all_logits, all_tsp = [], []

        for t in range(T):
            emb    = self._embed_multihot(cur_mh)                   # (B, emb)
            dec_in = torch.cat([emb,
                                 tss_c.unsqueeze(-1),
                                 tsp_c.unsqueeze(-1)], dim=-1).unsqueeze(1)  # (B,1,emb+2)
            out, h = self.decoder(dec_in, h)
            out    = out.squeeze(1)

            logits   = self.fc_out_multilabel(out)                  # (B, C)
            tsp_pred = self.fc_out_tsp(out)                         # (B, 1)
            all_logits.append(logits)
            all_tsp.append(tsp_pred)

            if t + 1 < T:
                use_gt   = torch.rand(B, device=out.device) < p_teacher  # (B,)
                pred_mh  = (logits.sigmoid() > 0.5).float()               # (B, C)
                # Blend: select GT or predicted per instance
                cur_mh = torch.where(use_gt.unsqueeze(-1), mh[:, t], pred_mh)

                if has_norm:
                    tsp_secs = (tsp_pred[:, 0] * tsp_std + tsp_mean).clamp(min=0)
                    tss_secs = (tss_c        * tss_std + tss_mean).clamp(min=0)
                    pred_tss = (tss_secs + tsp_secs - tss_mean) / tss_std
                    pred_tp  = (tsp_secs             - tsp_mean) / tsp_std
                    tss_c = torch.where(use_gt, tss[:, t + 1], pred_tss)
                    tsp_c = torch.where(use_gt, tsp[:, t + 1], pred_tp)
                else:
                    tss_c = tss[:, t + 1]
                    tsp_c = tsp[:, t + 1]

        logits_out = torch.stack(all_logits, dim=1)    # (B, T, C)
        tsp_out    = torch.stack(all_tsp,    dim=1)    # (B, T, 1)
        return logits_out, tsp_out

    # ── Autoregressive inference ──────────────────────────────────────────

    def _autoregressive(self, data, h0, T_max, mean_std_tsp, mean_std_tss):
        B      = h0.shape[1]
        C      = self.C
        device = h0.device
        tsp_mean, tsp_std = mean_std_tsp
        tss_mean, tss_std = mean_std_tss

        T = T_max if T_max is not None else (data.suffix_multihot.shape[0] // B)
        suf_tss = data.suffix_tss.view(B, T)
        suf_tsp = data.suffix_tsp.view(B, T)

        a_start = data.cat_x[data.ptr[1:] - 1, -1]
        mh0 = torch.zeros(B, C, device=device)
        mh0.scatter_(1, (a_start - 1).clamp(min=0, max=C - 2).unsqueeze(1), 1.0)

        cur_mh = mh0
        tss_c  = suf_tss[:, 0]
        tsp_c  = suf_tsp[:, 0]
        h = h0

        pred_blocks = torch.zeros(B, T, C, dtype=torch.float, device=device)
        pred_tsp    = torch.zeros(B, T, dtype=torch.float, device=device)

        for t in range(T):
            emb    = self._embed_multihot(cur_mh)
            dec_in = torch.cat([emb,
                                 tss_c.unsqueeze(-1),
                                 tsp_c.unsqueeze(-1)], dim=-1).unsqueeze(1)
            out, h = self.decoder(dec_in, h)
            out    = out.squeeze(1)

            logits      = self.fc_out_multilabel(out)           # (B, C)
            tsp_pred    = self.fc_out_tsp(out)                  # (B, 1)
            pred_mh     = (logits.sigmoid() > 0.5).float()      # (B, C)

            pred_blocks[:, t] = pred_mh
            pred_tsp[:, t]    = tsp_pred[:, 0]

            # Update time for next step using predicted TSP (= TTNE to next block)
            tsp_secs = (tsp_pred[:, 0] * tsp_std + tsp_mean).clamp(min=0)
            tss_secs = (tss_c          * tss_std + tss_mean).clamp(min=0)
            tss_c = (tss_secs + tsp_secs - tss_mean) / tss_std
            tsp_c = (tsp_secs             - tsp_mean) / tsp_std

            cur_mh = pred_mh

        return pred_blocks, pred_tsp   # (B, T, C), (B, T)
