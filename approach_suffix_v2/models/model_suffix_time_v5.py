"""
GATv2 encoder + GRU decoder with multi-label next-activity prediction.

Based on v2 (GATv2EncoderGRUDecoderStop). Key difference:
  Each decoder step corresponds to one partial-order level — a set of
  concurrent activities sharing the same timestamp.  The model predicts a
  binary vector of size C per step (BCEWithLogitsLoss), not a single class.

  Decoder input at step g : mean embedding of all activities in the GT group
                            g-1 (teacher forcing) or of the predicted set
                            (scheduled sampling / inference).
  Grouping                : derived from suffix_num[:,:,0] (tss) —
                            equal tss ⟹ concurrent events.
  Inference               : sigmoid > 0.5 threshold; fallback to argmax
                            when nothing crosses the threshold.  Activities
                            within each predicted group are sorted by class
                            index before being placed in the output (canonical
                            ordering that respects the partial order).
"""

import torch
import torch.nn as nn
import torch.nn.init as init
from model_suffix_time_v1 import GATv2EncoderGRUDecoder


class GATv2EncoderGRUDecoderMultiLabel(GATv2EncoderGRUDecoder):

    def __init__(self, use_scheduled_sampling=False, **kwargs):
        super().__init__(**kwargs)
        self.fc_stop = nn.Linear(self.d_model, 1)
        init.xavier_uniform_(self.fc_stop.weight)
        self.use_scheduled_sampling = use_scheduled_sampling

    # ── Grouping helper ───────────────────────────────────────────────────────

    def _build_groups(self, suffix_act, suffix_num, ttnext_flat):
        """
        Group consecutive suffix events with the same tss into partial-order steps.

        Args
        ----
        suffix_act   : (B, W) int   — activity indices; 0=padding, C-1=END
        suffix_num   : (B, W, 2)    — [:,:,0]=tss, [:,:,1]=tsp (normalised)
        ttnext_flat  : (B*W,) float — TTNE labels; -100 for padding/END

        Returns
        -------
        group_targets : (B, G, C) float — multi-hot targets per group
        group_tss     : (B, G)    float — tss of each group's first event
        group_tsp     : (B, G)    float — tsp of each group's first event
        group_mask    : (B, G)    bool  — True for valid (non-padded) groups
        group_ttne    : (B, G)    float — TTNE of each group's first event
        """
        B, W = suffix_act.shape
        C    = self.num_activities
        device = suffix_act.device
        tss = suffix_num[:, :, 0]   # (B, W)
        tsp = suffix_num[:, :, 1]   # (B, W)

        is_pad = (suffix_act == 0)
        valid  = ~is_pad

        # Boundary: first valid position, or where tss changes among valid events
        change = torch.zeros(B, W, dtype=torch.bool, device=device)
        change[:, 0] = valid[:, 0]
        change[:, 1:] = valid[:, 1:] & (tss[:, 1:] != tss[:, :-1])

        # 0-based group index per position; -1 for padding.
        # G is capped at W (no .item() GPU sync needed — W is always ≥ true G).
        group_id = change.long().cumsum(dim=1) - 1   # (B, W)
        group_id[is_pad] = -1
        G = W  # upper bound; empty trailing groups are masked out by group_mask

        b2d = torch.arange(B, device=device).unsqueeze(1).expand(B, W)
        g2d = group_id.clamp(min=0)
        a2d = suffix_act.clamp(min=0, max=C - 1)

        # Multi-hot targets (B, W, C) — trailing groups stay zero / masked
        group_targets = torch.zeros(B, G, C, device=device)
        group_targets[b2d[valid], g2d[valid], a2d[valid]] = 1.0

        # Group mask (B, W)
        group_mask = torch.zeros(B, G, dtype=torch.bool, device=device)
        group_mask[b2d[valid], g2d[valid]] = True

        # tss / tsp / ttne from the FIRST event of each group
        b_first = b2d[change]
        g_first = g2d[change]

        group_tss = torch.zeros(B, G, device=device)
        group_tsp = torch.zeros(B, G, device=device)
        group_ttne = torch.zeros(B, G, device=device)

        group_tss[b_first, g_first]  = tss[change]
        group_tsp[b_first, g_first]  = tsp[change]
        group_ttne[b_first, g_first] = ttnext_flat.reshape(B, W)[change]

        return group_targets, group_tss, group_tsp, group_mask, group_ttne

    # ── Mean embedding for a multi-hot group ─────────────────────────────────

    def _mean_group_emb(self, multi_hot):
        """
        multi_hot : (..., C) float — multi-hot over all C classes
        Returns   : (..., emb_size)

        act_emb covers indices 0..C-2 (END excluded from the embedding table).
        We take the mean of embeddings for all active non-END classes.
        """
        emb_part = multi_hot[..., :self.num_activities - 1]   # (..., C-1)
        n = emb_part.sum(dim=-1, keepdim=True).clamp(min=1)
        return (emb_part @ self.act_emb.weight) / n            # (..., emb_size)

    # ── forward ───────────────────────────────────────────────────────────────

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

    # ── Teacher forcing ───────────────────────────────────────────────────────

    def _teacher_forcing(self, data, h0):
        B  = data.num_graphs
        W  = data.suffix_act.shape[0] // B
        suffix_act = data.suffix_act.view(B, W)
        suffix_num = data.suffix_num.view(B, W, 2)

        group_targets, group_tss, group_tsp, group_mask, group_ttne = \
            self._build_groups(suffix_act, suffix_num, data.ttnext_label)
        G = group_targets.shape[1]

        # Input at group g: mean-emb of GT group g-1 (first step: last prefix act)
        dec_start_act = data.cat_x[data.ptr[1:] - 1, -1]
        start_emb = self.act_emb(
            dec_start_act.clamp(max=self.num_activities - 2))       # (B, emb)
        group_emb = self._mean_group_emb(group_targets)             # (B, G, emb)
        dec_emb   = torch.cat(
            [start_emb.unsqueeze(1), group_emb[:, :-1]], dim=1)    # (B, G, emb)

        dec_in = torch.cat(
            [dec_emb,
             group_tss.unsqueeze(-1),
             group_tsp.unsqueeze(-1)], dim=-1)                      # (B, G, emb+2)

        output, _   = self.decoder(dec_in, h0)                      # (B, G, d_model)
        act_logits  = self.fc_out_act(output)                       # (B, G, C)
        ttne_out    = self.fc_out_ttne(output)                      # (B, G, 1)
        stop_logits = self.fc_stop(output).squeeze(-1)              # (B, G)

        return act_logits, ttne_out, stop_logits, group_targets, group_mask, group_ttne

    # ── Scheduled sampling ────────────────────────────────────────────────────

    def _scheduled_sampling(self, data, h0, p_teacher,
                            mean_std_ttne, mean_std_tss, mean_std_tsp):
        B  = data.num_graphs
        W  = data.suffix_act.shape[0] // B
        suffix_act = data.suffix_act.view(B, W)
        suffix_num = data.suffix_num.view(B, W, 2)

        group_targets, group_tss, group_tsp, group_mask, group_ttne = \
            self._build_groups(suffix_act, suffix_num, data.ttnext_label)
        G = group_targets.shape[1]

        dec_start_act = data.cat_x[data.ptr[1:] - 1, -1]
        act_input_emb = self.act_emb(
            dec_start_act.clamp(max=self.num_activities - 2))       # (B, emb)
        tss_curr = group_tss[:, 0]   # (B,)
        tsp_curr = group_tsp[:, 0]   # (B,)
        h = h0

        has_norm = mean_std_ttne is not None
        if has_norm:
            ttne_mean, ttne_std = mean_std_ttne
            tss_mean,  tss_std  = mean_std_tss
            tsp_mean,  tsp_std  = mean_std_tsp

        all_act_logits, all_ttne, all_stop = [], [], []

        for g in range(G):
            dec_in = torch.cat(
                [act_input_emb,
                 tss_curr.unsqueeze(-1),
                 tsp_curr.unsqueeze(-1)], dim=-1).unsqueeze(1)      # (B, 1, emb+2)

            out, h      = self.decoder(dec_in, h)
            out         = out.squeeze(1)                            # (B, d_model)
            act_logits  = self.fc_out_act(out)                      # (B, C)
            ttne_pred   = self.fc_out_ttne(out)                     # (B, 1)
            stop_logit  = self.fc_stop(out).squeeze(-1)             # (B,)

            all_act_logits.append(act_logits)
            all_ttne.append(ttne_pred)
            all_stop.append(stop_logit)

            if g + 1 < G:
                use_gt = torch.rand(B, device=out.device) < p_teacher  # (B,)

                # GT embedding: mean of next GT group
                gt_emb = self._mean_group_emb(group_targets[:, g])     # (B, emb)

                # Predicted embedding: threshold sigmoid, zero out padding class
                pred_hot = (act_logits.detach().sigmoid() > 0.5).float()
                pred_hot[:, 0] = 0.0
                pred_emb = self._mean_group_emb(pred_hot)              # (B, emb)

                act_input_emb = torch.where(
                    use_gt.unsqueeze(-1), gt_emb, pred_emb)

                if has_norm:
                    ttne_secs = (ttne_pred[:, 0] * ttne_std + ttne_mean).clamp(min=0)
                    tss_secs  = (tss_curr * tss_std + tss_mean).clamp(min=0)
                    pred_tss  = (tss_secs + ttne_secs - tss_mean) / tss_std
                    pred_tsp  = (ttne_secs - tsp_mean) / tsp_std
                    tss_curr  = torch.where(use_gt, group_tss[:, g + 1], pred_tss)
                    tsp_curr  = torch.where(use_gt, group_tsp[:, g + 1], pred_tsp)
                else:
                    tss_curr = group_tss[:, g + 1]
                    tsp_curr = group_tsp[:, g + 1]

        act_out  = torch.stack(all_act_logits, dim=1)   # (B, G, C)
        ttne_out = torch.stack(all_ttne,       dim=1)   # (B, G, 1)
        stop_out = torch.stack(all_stop,       dim=1)   # (B, G)
        return act_out, ttne_out, stop_out, group_targets, group_mask, group_ttne

    # ── Autoregressive inference ───────────────────────────────────────────────

    def _autoregressive(self, data, h0, window_size,
                        mean_std_ttne, mean_std_tss, mean_std_tsp):
        B      = h0.shape[1]
        device = h0.device
        ttne_mean, ttne_std = mean_std_ttne
        tss_mean,  tss_std  = mean_std_tss
        tsp_mean,  tsp_std  = mean_std_tsp

        W = window_size if window_size is not None else (data.suffix_num.shape[0] // B)
        suffix_num = data.suffix_num.view(B, W, 2)
        end_tok = self.num_activities - 1
        C = self.num_activities

        act_input_emb = self.act_emb(
            data.cat_x[data.ptr[1:] - 1, -1].clamp(max=end_tok - 1))  # (B, emb)
        tss_curr = suffix_num[:, 0, 0]
        tsp_curr = suffix_num[:, 0, 1]

        # Accumulate per-step multi-hot predictions and TTNEs — no Python
        # loops or .item() calls, so no GPU synchronisation inside the loop.
        pred_matrix = torch.zeros(B, W, C, dtype=torch.bool,  device=device)
        ttne_matrix = torch.zeros(B, W,    dtype=torch.float, device=device)

        h = h0
        for t in range(W):
            dec_in = torch.cat(
                [act_input_emb,
                 tss_curr.unsqueeze(-1),
                 tsp_curr.unsqueeze(-1)], dim=-1).unsqueeze(1)

            out, h = self.decoder(dec_in, h)
            out    = out.squeeze(1)                                 # (B, d_model)

            act_logits = self.fc_out_act(out)                       # (B, C)
            stop_logit = self.fc_stop(out).squeeze(-1)              # (B,)
            ttne_pred  = self.fc_out_ttne(out)                      # (B, 1)

            act_logits[:, 0]       = -1e9
            act_logits[:, end_tok] = act_logits[:, end_tok] + stop_logit

            pred_set = (act_logits.sigmoid() > 0.5)                 # (B, C) bool
            pred_set[:, 0] = False

            # Vectorised fallback: samples with no prediction above threshold
            # get the argmax class — no .item() or Python loop needed.
            empty    = ~pred_set.any(dim=-1)                        # (B,) bool
            fallback = act_logits.argmax(dim=-1)                    # (B,) long
            pred_set[empty] = False
            pred_set[empty, fallback[empty]] = True

            pred_matrix[:, t, :] = pred_set
            ttne_matrix[:, t]    = ttne_pred[:, 0]

            # Update time and embedding for next step
            ttne_secs = (ttne_pred[:, 0] * ttne_std + ttne_mean).clamp(min=0)
            tss_secs  = (tss_curr * tss_std + tss_mean).clamp(min=0)
            tss_curr  = (tss_secs + ttne_secs - tss_mean) / tss_std
            tsp_curr  = (ttne_secs - tsp_mean) / tsp_std

            pred_hot = pred_set.float()
            pred_hot[:, end_tok] = 0.0
            act_input_emb = self._mean_group_emb(pred_hot)

        # Vectorised canonical flattening — no Python loops, no GPU syncs.
        #
        # For each (b, t, c): if pred_matrix[b,t,c] is True, activity c
        # is predicted at group t.  Scanning (t, c) in row-major order
        # is already the canonical order (group first, then ascending index).
        #
        # write_pos[b, i] = sequential output position for flat slot i:
        #   cumsum of True values up to i, minus 1 (0-indexed).
        # Non-predicted slots are redirected to the dummy column W.

        ttne_expanded = ttne_matrix.unsqueeze(-1).expand(B, W, C)           # (B, W, C)
        act_idx       = torch.arange(C, device=device).view(1, 1, C) \
                               .expand(B, W, C)                             # (B, W, C)

        mask_flat = pred_matrix.reshape(B, -1)                              # (B, W*C)
        acts_flat = act_idx.reshape(B, -1)                                  # (B, W*C)
        ttne_flat = ttne_expanded.reshape(B, -1)                            # (B, W*C)

        write_pos = mask_flat.long().cumsum(dim=1) - 1                      # (B, W*C)
        write_pos = torch.where(mask_flat, write_pos,
                                write_pos.new_full((), W))                  # redirect non-predicted to dummy

        suffix_acts = acts_flat.new_zeros(B, W + 1)
        suffix_ttne = ttne_flat.new_zeros(B, W + 1)
        suffix_acts.scatter_(1, write_pos, acts_flat)
        suffix_ttne.scatter_(1, write_pos, ttne_flat)

        return suffix_acts[:, :W], suffix_ttne[:, :W]
