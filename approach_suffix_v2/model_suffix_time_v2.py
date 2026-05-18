"""
GATv2 encoder + GRU decoder with a binary stop head, supporting both teacher
forcing and scheduled sampling during training.

Architecture is identical to GATv2EncoderGRUDecoder (v1), plus:
  fc_stop : nn.Linear(d_model, 1)
    - trained with BCE at every suffix position (target=1 at END, 0 elsewhere)
    - at inference its logit is added as a soft bias to the END class logit,
      so argmax still decides termination without a tunable threshold

Training mode is selected at construction via use_scheduled_sampling:
  False (default) : teacher forcing — all ground-truth inputs fed in parallel.
  True            : scheduled sampling — each step samples from GT with
                    probability p_teacher (passed to forward()), else uses
                    the model's own previous prediction.  Anneal p_teacher
                    1.0 → 0.0 over training to close the train/inference gap.
                    Pass mean_std_ttne/tss/tsp to forward() to also sample
                    time features; omit them to fall back to GT time features.
"""

import torch
import torch.nn as nn
import torch.nn.init as init
from model_suffix_time_v1 import GATv2EncoderGRUDecoder


class GATv2EncoderGRUDecoderStop(GATv2EncoderGRUDecoder):
    """Same architecture as v3 plus a binary stop head trained over the full suffix."""

    def __init__(self, use_scheduled_sampling=False, **kwargs):
        super().__init__(**kwargs)
        self.fc_stop = nn.Linear(self.d_model, 1)
        init.xavier_uniform_(self.fc_stop.weight)
        self.use_scheduled_sampling = use_scheduled_sampling

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
        tss     = suffix_num[:, :, 0].unsqueeze(-1)       # (B, W, 1)
        tsp     = suffix_num[:, :, 1].unsqueeze(-1)       # (B, W, 1)
        dec_in  = torch.cat([act_emb, tss, tsp], dim=-1)  # (B, W, emb+2)

        output, _ = self.decoder(dec_in, h0)               # (B, W, d_model)
        stop_logits = self.fc_stop(output).squeeze(-1)     # (B, W)
        return self.fc_out_act(output), self.fc_out_ttne(output), stop_logits

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
        tss_curr  = suffix_num[:, 0, 0]                   # (B,)
        tsp_curr  = suffix_num[:, 0, 1]                   # (B,)
        h = h0

        has_norm = mean_std_ttne is not None
        if has_norm:
            ttne_mean, ttne_std = mean_std_ttne
            tss_mean,  tss_std  = mean_std_tss
            tsp_mean,  tsp_std  = mean_std_tsp

        all_act_logits = []
        all_ttne       = []
        all_stop       = []

        for t in range(W):
            emb    = self.act_emb(act_input)
            dec_in = torch.cat([emb,
                                 tss_curr.unsqueeze(-1),
                                 tsp_curr.unsqueeze(-1)], dim=-1).unsqueeze(1)  # (B,1,emb+2)

            out, h = self.decoder(dec_in, h)
            out    = out.squeeze(1)                        # (B, d_model)

            act_logits = self.fc_out_act(out)              # (B, C)
            ttne_pred  = self.fc_out_ttne(out)             # (B, 1)
            stop_logit = self.fc_stop(out).squeeze(-1)     # (B,)

            all_act_logits.append(act_logits)
            all_ttne.append(ttne_pred)
            all_stop.append(stop_logit)

            if t + 1 < W:
                use_gt = torch.rand(B, device=out.device) < p_teacher  # (B,)

                pred_act  = act_logits.argmax(dim=-1).clamp(max=self.num_activities - 2)
                act_input = torch.where(use_gt, gt_acts[:, t + 1], pred_act)

                if has_norm:
                    ttne_secs = (ttne_pred[:, 0] * ttne_std + ttne_mean).clamp(min=0)
                    tss_secs  = (tss_curr * tss_std + tss_mean).clamp(min=0)
                    pred_tss  = (tss_secs + ttne_secs - tss_mean) / tss_std
                    pred_tsp  = (ttne_secs - tsp_mean) / tsp_std
                    tss_curr  = torch.where(use_gt, suffix_num[:, t + 1, 0], pred_tss)
                    tsp_curr  = torch.where(use_gt, suffix_num[:, t + 1, 1], pred_tsp)
                else:
                    tss_curr = suffix_num[:, t + 1, 0]
                    tsp_curr = suffix_num[:, t + 1, 1]

        act_out  = torch.stack(all_act_logits, dim=1)     # (B, W, C)
        ttne_out = torch.stack(all_ttne,       dim=1)     # (B, W, 1)
        stop_out = torch.stack(all_stop,       dim=1)     # (B, W)
        return act_out, ttne_out, stop_out

    def _autoregressive(self, data, h0, window_size, mean_std_ttne, mean_std_tss, mean_std_tsp):
        B      = h0.shape[1]
        device = h0.device
        ttne_mean, ttne_std = mean_std_ttne
        tss_mean,  tss_std  = mean_std_tss
        tsp_mean,  tsp_std  = mean_std_tsp

        W = window_size if window_size is not None else (data.suffix_num.shape[0] // B)
        suffix_num = data.suffix_num.view(B, W, 2)

        suffix_acts = torch.zeros(B, W, dtype=torch.long,  device=device)
        suffix_ttne = torch.zeros(B, W, dtype=torch.float, device=device)

        act_input = data.cat_x[data.ptr[1:] - 1, -1].clamp(max=self.num_activities - 2)
        tss_curr  = suffix_num[:, 0, 0]
        tsp_curr  = suffix_num[:, 0, 1]

        h = h0
        for t in range(W):
            emb    = self.act_emb(act_input)
            dec_in = torch.cat([emb,
                                 tss_curr.unsqueeze(-1),
                                 tsp_curr.unsqueeze(-1)], dim=-1).unsqueeze(1)

            out, h = self.decoder(dec_in, h)
            out    = out.squeeze(1)                                   # (B, d_model)

            act_logits = self.fc_out_act(out)                         # (B, C)
            stop_logit = self.fc_stop(out).squeeze(-1)                # (B,)
            ttne_pred  = self.fc_out_ttne(out)                        # (B, 1)

            act_logits[:, 0]  = -1e9                                  # mask padding
            act_logits[:, -1] = act_logits[:, -1] + stop_logit        # soft bias END

            act_selected = act_logits.argmax(dim=-1)                  # (B,)

            suffix_acts[:, t] = act_selected
            suffix_ttne[:, t] = ttne_pred[:, 0]

            ttne_secs = (ttne_pred[:, 0] * ttne_std + ttne_mean).clamp(min=0)
            tss_secs  = (tss_curr * tss_std + tss_mean).clamp(min=0)
            tss_curr  = (tss_secs + ttne_secs - tss_mean) / tss_std
            tsp_curr  = (ttne_secs - tsp_mean) / tsp_std

            act_input = act_selected.clamp(max=self.num_activities - 2)

        return suffix_acts, suffix_ttne
