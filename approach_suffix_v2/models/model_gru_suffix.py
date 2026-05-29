"""GRU encoder + GRU decoder for activity-suffix, time, and concurrency prediction.

Architecture
------------
Encoder : shared embedding (act_emb + time_mlp + conc_emb) → input_proj → GRU
          Pack-padded for variable-length prefix → final hidden state h0.

Decoder : step-by-step GRU with 3 output heads:
            fc_act  (hidden_dim → num_classes)  — CrossEntropyLoss
            fc_time (hidden_dim → 1)             — MSELoss on normalized log-dt
            fc_conc (hidden_dim → 1)             — BCEWithLogitsLoss

Embedding size
--------------
  emb_size = min(600, round(1.6 * (num_classes - 2) ** 0.56))
  (same formula as all other suffix approaches in this repo)

Training modes
--------------
  Teacher forcing     (use_scheduled_sampling=False):
    Decoder inputs are right-shifted ground-truth suffix events; all W
    steps run in a single GRU call (parallel, fast).

  Scheduled sampling  (use_scheduled_sampling=True):
    Each step uses GT with probability p_teacher, else the model's own
    previous prediction.  Anneal p_teacher 1.0 → 0.0 over training to
    close the train/inference gap.

  Autoregressive (eval):
    Greedy decoding; stops after window_size steps (EOS is the last
    activity class, index num_classes - 1).
"""

import torch
import torch.nn as nn
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
        """
        Parameters
        ----------
        num_classes : int
            Total vocab size from create_sequential_data:
            0=PAD, 1..N=activities, N+1=OOV, N+2=EOS.
        d_time : int
            Hidden size of the time MLP.
        d_conc : int
            Embedding size for the binary concurrency flag.
        d_model : int
            Projected input size fed into both encoder and decoder GRUs.
        hidden_dim : int
            GRU hidden size (encoder and decoder share the same size).
        n_layers : int
            Number of GRU layers.
        dropout : float
            Dropout between GRU layers (only active when n_layers > 1).
        use_scheduled_sampling : bool
            If True, _scheduled_sampling is used during training.
        """
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
        """Project (act, dt, conc) event(s) into d_model space.

        Accepts both sequences (B, T) and single steps (B,).
        Returns (B, T, d_model) or (B, d_model) accordingly.
        """
        seq = act.dim() == 2
        if not seq:
            act, dt, conc = act.unsqueeze(1), dt.unsqueeze(1), conc.unsqueeze(1)
        a = self.act_emb(act)                          # (B, T, emb_size)
        t = self.time_mlp(dt.unsqueeze(-1))             # (B, T, d_time)
        c = self.conc_emb(conc)                        # (B, T, d_conc)
        x = self.input_proj(torch.cat([a, t, c], dim=-1))  # (B, T, d_model)
        return x if seq else x.squeeze(1)

    # ── Encoder ───────────────────────────────────────────────────────────

    def _encode(self, data):
        """Embed prefix and run encoder GRU → h0 for the decoder."""
        x      = self._embed(data['prefix_act'], data['prefix_dt'], data['prefix_conc'])
        plen   = data['prefix_len'].cpu()
        packed = pack_padded_sequence(x, plen, batch_first=True, enforce_sorted=False)
        _, h0  = self.encoder(packed)   # (n_layers, B, hidden_dim)
        return h0

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, data, p_teacher=1.0):
        """
        Training : returns (act_logits, time_pred, conc_logits) — all (B, W, *).
        Eval     : returns (pred_acts, pred_time, pred_conc)     — all (B, W).
        """
        h0 = self._encode(data)
        if self.training:
            if self.use_scheduled_sampling:
                return self._scheduled_sampling(data, h0, p_teacher)
            return self._teacher_forcing(data, h0)
        return self._autoregressive(data, h0)

    # ── Teacher forcing (parallel over W) ────────────────────────────────

    def _teacher_forcing(self, data, h0):
        B, W = data['suffix_act'].shape

        # Decoder input at step t = ground-truth event at position t-1.
        # At t=0 the seed is the last prefix event.
        dec_act  = torch.cat([data['last_prefix_act'].unsqueeze(1),
                               data['suffix_act'][:, :-1]],  dim=1)   # (B, W)
        dec_dt   = torch.cat([data['last_prefix_dt'].unsqueeze(1),
                               data['suffix_dt'][:, :-1]],   dim=1)   # (B, W)
        dec_conc = torch.cat([data['last_prefix_conc'].unsqueeze(1),
                               data['suffix_conc'][:, :-1]], dim=1)   # (B, W)

        x      = self._embed(dec_act, dec_dt, dec_conc)   # (B, W, d_model)
        out, _ = self.decoder(x, h0)                       # (B, W, hidden_dim)

        return (
            self.fc_act(out),                 # (B, W, num_classes)
            self.fc_time(out).squeeze(-1),    # (B, W)
            self.fc_conc(out).squeeze(-1),    # (B, W)
        )

    # ── Scheduled sampling (step-by-step) ────────────────────────────────

    def _scheduled_sampling(self, data, h0, p_teacher):
        B, W = data['suffix_act'].shape

        # Ground-truth decoder inputs (shifted right with seed at t=0)
        gt_in_act  = torch.cat([data['last_prefix_act'].unsqueeze(1),
                                 data['suffix_act'][:, :-1]],  dim=1)
        gt_in_dt   = torch.cat([data['last_prefix_dt'].unsqueeze(1),
                                 data['suffix_dt'][:, :-1]],   dim=1)
        gt_in_conc = torch.cat([data['last_prefix_conc'].unsqueeze(1),
                                 data['suffix_conc'][:, :-1]], dim=1)

        act_in  = gt_in_act[:, 0]    # (B,)  seed
        dt_in   = gt_in_dt[:, 0]     # (B,)
        conc_in = gt_in_conc[:, 0]   # (B,)
        h = h0

        all_act, all_time, all_conc = [], [], []

        for t in range(W):
            x       = self._embed(act_in, dt_in, conc_in)   # (B, d_model)
            out, h  = self.decoder(x.unsqueeze(1), h)        # (B, 1, hidden_dim)
            out     = out.squeeze(1)                          # (B, hidden_dim)

            act_logits = self.fc_act(out)                    # (B, num_classes)
            time_pred  = self.fc_time(out).squeeze(-1)       # (B,)
            conc_logit = self.fc_conc(out).squeeze(-1)       # (B,)

            all_act.append(act_logits)
            all_time.append(time_pred)
            all_conc.append(conc_logit)

            if t + 1 < W:
                use_gt  = torch.rand(B, device=out.device) < p_teacher
                pred_act  = act_logits.argmax(dim=-1)
                pred_conc = (conc_logit > 0).long()
                act_in    = torch.where(use_gt, gt_in_act[:, t + 1],  pred_act)
                dt_in     = torch.where(use_gt, gt_in_dt[:, t + 1],   time_pred)
                conc_in   = torch.where(use_gt, gt_in_conc[:, t + 1].long(), pred_conc)

        return (
            torch.stack(all_act,  dim=1),    # (B, W, num_classes)
            torch.stack(all_time, dim=1),    # (B, W)
            torch.stack(all_conc, dim=1),    # (B, W)
        )

    # ── Autoregressive (eval, greedy) ─────────────────────────────────────

    @torch.no_grad()
    def _autoregressive(self, data, h0):
        B      = h0.shape[1]
        W      = data['suffix_act'].shape[1]
        device = h0.device

        pred_acts = torch.zeros(B, W, dtype=torch.long,  device=device)
        pred_time = torch.zeros(B, W, dtype=torch.float, device=device)
        pred_conc = torch.zeros(B, W, dtype=torch.long,  device=device)

        act_in  = data['last_prefix_act'].to(device)    # (B,)
        dt_in   = data['last_prefix_dt'].to(device)     # (B,)
        conc_in = data['last_prefix_conc'].to(device)   # (B,)
        h = h0

        for t in range(W):
            x      = self._embed(act_in, dt_in, conc_in)
            out, h = self.decoder(x.unsqueeze(1), h)
            out    = out.squeeze(1)

            act_logits = self.fc_act(out)
            act_logits[:, 0] = -1e9             # suppress PAD class
            pred_act  = act_logits.argmax(dim=-1)
            pred_t    = self.fc_time(out).squeeze(-1)
            pred_c    = (self.fc_conc(out).squeeze(-1) > 0).long()

            pred_acts[:, t] = pred_act
            pred_time[:, t] = pred_t
            pred_conc[:, t] = pred_c

            act_in  = pred_act
            dt_in   = pred_t
            conc_in = pred_c

        return pred_acts, pred_time, pred_conc
