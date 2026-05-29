"""GRU encoder + GRU decoder v2.

Changes vs v1
-------------
1. Softplus time head       — guarantees positive dt predictions.
2. Detached time input      — prevents recursive gradient propagation through
                              autoregressive time in scheduled sampling.
3. EOS early stopping       — autoregressive decoding zeros out predictions
                              for sequences that already emitted EOS.
4. Consistency penalty      — training forward returns a 4th tensor of shape
                              (B, W): sigmoid(conc_logit) * time_pred**2,
                              penalising concurrent=True with large dt.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class GRUEncoderGRUDecoder(nn.Module):
    def __init__(
        self,
        num_classes,
        d_time=16,
        d_conc=8,
        d_model=64,
        hidden_dim=128,
        n_layers=1,
        dropout=0.0,
        use_scheduled_sampling=False,
    ):
        super().__init__()
        self.num_classes            = num_classes
        self.hidden_dim             = hidden_dim
        self.n_layers               = n_layers
        self.use_scheduled_sampling = use_scheduled_sampling

        emb_size      = min(600, round(1.6 * (num_classes - 2) ** 0.56))
        self.emb_size = emb_size

        self.act_emb  = nn.Embedding(num_classes, emb_size, padding_idx=0)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, d_time), nn.ReLU(), nn.Linear(d_time, d_time),
        )
        self.conc_emb   = nn.Embedding(2, d_conc)
        self.input_proj = nn.Linear(emb_size + d_time + d_conc, d_model)

        gru_kw = dict(
            input_size=d_model, hidden_size=hidden_dim,
            num_layers=n_layers, batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.encoder = nn.GRU(**gru_kw)
        self.decoder = nn.GRU(**gru_kw)

        self.fc_act  = nn.Linear(hidden_dim, num_classes)
        self.fc_time = nn.Linear(hidden_dim, 1)
        self.fc_conc = nn.Linear(hidden_dim, 1)

    # ── Shared embedding ──────────────────────────────────────────────────

    def _embed(self, act, dt, conc):
        seq = act.dim() == 2
        if not seq:
            act, dt, conc = act.unsqueeze(1), dt.unsqueeze(1), conc.unsqueeze(1)
        a = self.act_emb(act)
        t = self.time_mlp(dt.unsqueeze(-1))
        c = self.conc_emb(conc)
        x = self.input_proj(torch.cat([a, t, c], dim=-1))
        return x if seq else x.squeeze(1)

    # ── Encoder ───────────────────────────────────────────────────────────

    def _encode(self, data):
        x      = self._embed(data['prefix_act'], data['prefix_dt'], data['prefix_conc'])
        plen   = data['prefix_len'].cpu()
        packed = pack_padded_sequence(x, plen, batch_first=True, enforce_sorted=False)
        _, h0  = self.encoder(packed)
        return h0

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, data, p_teacher=1.0):
        """
        Training : returns (act_logits, time_pred, conc_logits, consistency)
                   all (B, W, *) — consistency is (B, W).
        Eval     : returns (pred_acts, pred_time, pred_conc) — all (B, W).
        """
        h0 = self._encode(data)
        if self.training:
            if self.use_scheduled_sampling:
                return self._scheduled_sampling(data, h0, p_teacher)
            return self._teacher_forcing(data, h0)
        return self._autoregressive(data, h0)

    # ── Teacher forcing (parallel over W) ────────────────────────────────

    def _teacher_forcing(self, data, h0):
        dec_act  = torch.cat([data['last_prefix_act'].unsqueeze(1),
                               data['suffix_act'][:, :-1]],  dim=1)
        dec_dt   = torch.cat([data['last_prefix_dt'].unsqueeze(1),
                               data['suffix_dt'][:, :-1]],   dim=1)
        dec_conc = torch.cat([data['last_prefix_conc'].unsqueeze(1),
                               data['suffix_conc'][:, :-1]], dim=1)

        x      = self._embed(dec_act, dec_dt, dec_conc)
        out, _ = self.decoder(x, h0)

        act_logits  = self.fc_act(out)                              # (B, W, num_classes)
        time_pred   = F.softplus(self.fc_time(out).squeeze(-1))     # (B, W) — positive
        conc_logits = self.fc_conc(out).squeeze(-1)                 # (B, W)
        consistency = torch.sigmoid(conc_logits) * time_pred ** 2  # (B, W)
        return act_logits, time_pred, conc_logits, consistency

    # ── Scheduled sampling (step-by-step) ────────────────────────────────

    def _scheduled_sampling(self, data, h0, p_teacher):
        B, W = data['suffix_act'].shape

        gt_in_act  = torch.cat([data['last_prefix_act'].unsqueeze(1),
                                 data['suffix_act'][:, :-1]],  dim=1)
        gt_in_dt   = torch.cat([data['last_prefix_dt'].unsqueeze(1),
                                 data['suffix_dt'][:, :-1]],   dim=1)
        gt_in_conc = torch.cat([data['last_prefix_conc'].unsqueeze(1),
                                 data['suffix_conc'][:, :-1]], dim=1)

        act_in  = gt_in_act[:, 0]
        dt_in   = gt_in_dt[:, 0]
        conc_in = gt_in_conc[:, 0]
        h = h0

        all_act, all_time, all_conc, all_cons = [], [], [], []

        for t in range(W):
            x       = self._embed(act_in, dt_in, conc_in)
            out, h  = self.decoder(x.unsqueeze(1), h)
            out     = out.squeeze(1)

            act_logits  = self.fc_act(out)
            time_pred   = F.softplus(self.fc_time(out).squeeze(-1))
            conc_logit  = self.fc_conc(out).squeeze(-1)
            consistency = torch.sigmoid(conc_logit) * time_pred ** 2

            all_act.append(act_logits)
            all_time.append(time_pred)
            all_conc.append(conc_logit)
            all_cons.append(consistency)

            if t + 1 < W:
                use_gt    = torch.rand(B, device=out.device) < p_teacher
                pred_act  = act_logits.argmax(dim=-1)
                pred_conc = (conc_logit > 0).long()
                act_in    = torch.where(use_gt, gt_in_act[:, t + 1],  pred_act)
                # detach: prevent recursive gradients through future time steps
                dt_in     = torch.where(use_gt, gt_in_dt[:, t + 1],   time_pred.detach())
                conc_in   = torch.where(use_gt, gt_in_conc[:, t + 1].long(), pred_conc)

        return (
            torch.stack(all_act,  dim=1),   # (B, W, num_classes)
            torch.stack(all_time, dim=1),   # (B, W)
            torch.stack(all_conc, dim=1),   # (B, W)
            torch.stack(all_cons, dim=1),   # (B, W)
        )

    # ── Autoregressive (eval, greedy) ─────────────────────────────────────

    @torch.no_grad()
    def _autoregressive(self, data, h0):
        B      = h0.shape[1]
        W      = data['suffix_act'].shape[1]
        device = h0.device
        EOS_ID = self.num_classes - 1

        pred_acts = torch.zeros(B, W, dtype=torch.long,  device=device)
        pred_time = torch.zeros(B, W, dtype=torch.float, device=device)
        pred_conc = torch.zeros(B, W, dtype=torch.long,  device=device)

        act_in   = data['last_prefix_act'].to(device)
        dt_in    = data['last_prefix_dt'].to(device)
        conc_in  = data['last_prefix_conc'].to(device)
        h        = h0
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for t in range(W):
            x      = self._embed(act_in, dt_in, conc_in)
            out, h = self.decoder(x.unsqueeze(1), h)
            out    = out.squeeze(1)

            act_logits = self.fc_act(out)
            act_logits[:, 0] = -1e9             # suppress PAD class
            pred_act = act_logits.argmax(dim=-1)
            pred_t   = F.softplus(self.fc_time(out).squeeze(-1))
            pred_c   = (self.fc_conc(out).squeeze(-1) > 0).long()

            # Zero out predictions for sequences that already emitted EOS
            pred_act = torch.where(finished, torch.zeros_like(pred_act), pred_act)
            pred_t   = torch.where(finished, torch.zeros_like(pred_t),   pred_t)
            pred_c   = torch.where(finished, torch.zeros_like(pred_c),   pred_c)

            pred_acts[:, t] = pred_act
            pred_time[:, t] = pred_t
            pred_conc[:, t] = pred_c

            finished |= (pred_act == EOS_ID)

            act_in  = pred_act
            dt_in   = pred_t
            conc_in = pred_c

        return pred_acts, pred_time, pred_conc
