"""
GATv2 encoder + GRU decoder with a binary stop head for activity suffix prediction
(no time prediction). Identical to GATv2EncoderGRUDecoderSuffix (v1), plus:
  fc_stop : nn.Linear(d_model, 1)
    - trained with BCE at every suffix position (target=1 at END, 0 elsewhere)
    - at inference its logit is added as a soft bias to the END class logit,
      so argmax still decides termination without a tunable threshold

Training mode is selected at construction via use_scheduled_sampling:
  False (default) : teacher forcing.
  True            : scheduled sampling — each step samples from GT with
                    probability p_teacher (passed to forward()).
"""

import torch
import torch.nn as nn
import torch.nn.init as init
from model_suffix_v1 import GATv2EncoderGRUDecoderSuffix


class GATv2EncoderGRUDecoderSuffixStop(GATv2EncoderGRUDecoderSuffix):
    """Same architecture as v1 plus a binary stop head trained over the full suffix."""

    def __init__(self, use_scheduled_sampling=False, **kwargs):
        super().__init__(**kwargs)
        self.fc_stop = nn.Linear(self.d_model, 1)
        init.xavier_uniform_(self.fc_stop.weight)
        self.use_scheduled_sampling = use_scheduled_sampling

    def forward(self, data, window_size=None, p_teacher=1.0):
        c  = self._encode(data)
        h0 = self._init_gru_state(c)
        if self.training:
            if self.use_scheduled_sampling:
                return self._scheduled_sampling(data, h0, p_teacher)
            else:
                return self._teacher_forcing(data, h0)
        else:
            return self._autoregressive(data, h0, window_size)

    def _teacher_forcing(self, data, h0):
        B  = data.num_graphs
        W  = data.suffix_act.shape[0] // B
        suffix_act = data.suffix_act.view(B, W)

        dec_start_act = data.cat_x[data.ptr[1:] - 1, -1]
        dec_acts = torch.cat(
            [dec_start_act.unsqueeze(-1), suffix_act[:, :-1]], dim=-1
        ).clamp(max=self.num_activities - 2)

        dec_in = self.act_emb(dec_acts)            # (B, W, emb)
        output, _ = self.decoder(dec_in, h0)       # (B, W, d_model)
        stop_logits = self.fc_stop(output).squeeze(-1)  # (B, W)
        return self.fc_out_act(output), stop_logits

    def _scheduled_sampling(self, data, h0, p_teacher):
        B  = data.num_graphs
        W  = data.suffix_act.shape[0] // B
        suffix_act = data.suffix_act.view(B, W)

        dec_start_act = data.cat_x[data.ptr[1:] - 1, -1]
        gt_acts = torch.cat(
            [dec_start_act.unsqueeze(-1), suffix_act[:, :-1]], dim=-1
        ).clamp(max=self.num_activities - 2)

        act_input = gt_acts[:, 0]
        h = h0

        all_act_logits = []
        all_stop       = []

        for t in range(W):
            emb    = self.act_emb(act_input)
            dec_in = emb.unsqueeze(1)              # (B, 1, emb)

            out, h = self.decoder(dec_in, h)
            out    = out.squeeze(1)                # (B, d_model)

            act_logits = self.fc_out_act(out)      # (B, C)
            stop_logit = self.fc_stop(out).squeeze(-1)  # (B,)

            all_act_logits.append(act_logits)
            all_stop.append(stop_logit)

            if t + 1 < W:
                use_gt    = torch.rand(B, device=out.device) < p_teacher
                pred_act  = act_logits.argmax(dim=-1).clamp(max=self.num_activities - 2)
                act_input = torch.where(use_gt, gt_acts[:, t + 1], pred_act)

        act_out  = torch.stack(all_act_logits, dim=1)  # (B, W, C)
        stop_out = torch.stack(all_stop,       dim=1)  # (B, W)
        return act_out, stop_out

    def _autoregressive(self, data, h0, window_size):
        B      = h0.shape[1]
        device = h0.device

        W = window_size if window_size is not None else (data.suffix_act.shape[0] // B)
        suffix_acts = torch.zeros(B, W, dtype=torch.long, device=device)

        act_input = data.cat_x[data.ptr[1:] - 1, -1].clamp(max=self.num_activities - 2)

        h = h0
        for t in range(W):
            emb    = self.act_emb(act_input)
            dec_in = emb.unsqueeze(1)              # (B, 1, emb)

            out, h = self.decoder(dec_in, h)
            out    = out.squeeze(1)                # (B, d_model)

            act_logits = self.fc_out_act(out)      # (B, C)
            stop_logit = self.fc_stop(out).squeeze(-1)  # (B,)

            act_logits[:, 0]  = -1e9               # mask padding
            act_logits[:, -1] = act_logits[:, -1] + stop_logit  # soft bias END

            act_selected = act_logits.argmax(dim=-1)
            suffix_acts[:, t] = act_selected
            act_input = act_selected.clamp(max=self.num_activities - 2)

        return suffix_acts
