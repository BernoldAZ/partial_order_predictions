"""
GATv2 encoder (pre-norm residual, prefix block-index embedding) + Transformer
decoder with explicit new-block feedback. Activities and partial-order
structure only — no time outputs. Standalone implementation — no inheritance
from v1..v5.

Predicts:
  - activity suffix      (fc_out_act)
  - new-block label      (fc_new_block) BCE-trained
      target=1  event starts a new concurrent block
      target=0  event is concurrent with the previous one

Time is still used as encoder INPUT (node tss, edge tsp), never predicted.

Encoder
    h = input_proj([act_emb, tss]) + block_emb(block index counted from the
    END of the prefix, last block = 0, clamped to max_blocks-1).
    The block index is derived from the edges: only intra-block edges point
    backwards, so node i shares a block with node i-1 iff edge (i -> i-1)
    exists.
    n_gnn_layers pre-norm residual GATv2 layers:
        h = h + dropout(relu(gnn(LN(h))))
    followed by a final LayerNorm. h_global (attention pooling + BatchNorm) is
    prepended to the per-node embeddings as an extra memory token.

Decoder (nn.TransformerDecoder, pre-norm, causal)
    token_t = dec_act_proj(act_emb(a_{t-1})) + nb_emb(nb_{t-1}) + dec_block_emb(blk_{t-1})
    step 0: mean act embedding over the prefix's last block (order invariant),
            nb=1, blk=0
    blk_{t-1} = number of new-block bits among suffix events 0..t-1, i.e. the
                block position inside the suffix.

Training uses teacher forcing (one parallel pass). Inference is greedy
autoregressive: predicted activity and new-block bit are fed back.
"""

import torch
import torch.nn as nn
import torch.nn.init as init
from torch_geometric.nn import GATv2Conv, global_mean_pool, AttentionalAggregation
from torch_geometric.utils import to_dense_batch


class GATv2EncoderTransformerDecoderNewBlockV6(nn.Module):
    """
    Parameters
    ----------
    num_activities : int
        Total classes incl. padding (0) and END (num_activities-1).
    d_model : int
        Hidden size for GNN and Transformer decoder.
    dropout : float
    n_gnn_layers : int
        Number of residual GATv2Conv layers in the encoder.
    n_dec_layers : int
        Number of Transformer decoder layers.
    nhead : int
        Number of GATv2Conv / decoder attention heads.
    max_blocks : int
        Size of the prefix / suffix block-index embeddings (indices are clamped).
    """

    def __init__(self, num_activities: int, d_model: int = 64,
                 dropout: float = 0.2, n_gnn_layers: int = 2, n_dec_layers: int = 2,
                 nhead: int = 4, max_blocks: int = 16):
        super().__init__()
        self.num_activities = num_activities
        self.d_model        = d_model
        self.max_blocks     = max_blocks

        emb_size      = min(600, round(1.6 * (num_activities - 2) ** 0.56))
        self.emb_size = emb_size
        self.act_emb  = nn.Embedding(num_activities - 1, emb_size, padding_idx=0)
        self.dropout  = nn.Dropout(dropout)

        # ── Encoder ──
        self.input_proj = nn.Linear(emb_size + 1, d_model)   # + 1 because of tss
        self.block_emb  = nn.Embedding(max_blocks, d_model)
        self.gnn_norms  = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_gnn_layers)])
        self.gnn_layers = nn.ModuleList(
            [GATv2Conv(d_model, d_model, heads=nhead, concat=False, edge_dim=1)
             for _ in range(n_gnn_layers)])
        self.ln_enc   = nn.LayerNorm(d_model)
        self.att_pool = AttentionalAggregation(gate_nn=nn.Linear(d_model, 1))
        self.bn_enc   = nn.BatchNorm1d(d_model)

        # ── Decoder ──
        self.dec_act_proj  = nn.Linear(emb_size, d_model)
        self.nb_emb        = nn.Embedding(2, d_model)
        self.dec_block_emb = nn.Embedding(max_blocks, d_model)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(
            dec_layer, num_layers=n_dec_layers, norm=nn.LayerNorm(d_model))

        self.fc_out_act   = nn.Linear(d_model, num_activities)
        self.fc_new_block = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self):
        init.xavier_uniform_(self.fc_out_act.weight)
        init.xavier_uniform_(self.fc_new_block.weight)

    # ── Encoder ──────────────────────────────────────────────────────────────

    def _prefix_block_from_end(self, data):
        """Block index of every node counted from the end of its prefix (last block = 0)."""
        src, dst = data.edge_index
        same_prev = torch.zeros(data.num_nodes, dtype=torch.bool, device=src.device)
        back = dst == src - 1                       # only intra-block edges point backwards
        same_prev[src[back]] = True
        cum = (~same_prev).long().cumsum(0)         # global block counter
        last_cum = cum[data.ptr[1:] - 1]            # counter at last node of each graph
        return last_cum[data.batch] - cum

    def _encode(self, data):
        B = data.num_graphs
        h = torch.cat([self.act_emb(data.cat_x[:, -1]), data.x[:, [0]]], dim=-1)
        blk = self._prefix_block_from_end(data).clamp(max=self.max_blocks - 1)
        h = self.input_proj(h) + self.block_emb(blk)
        h = self.dropout(h)
        for norm, gnn in zip(self.gnn_norms, self.gnn_layers):
            h = h + self.dropout(gnn(norm(h), data.edge_index, data.edge_attr).relu())
        h = self.ln_enc(h)
        h_global = self.bn_enc(self.att_pool(h, data.batch))                # (B, d_model)
        mem, mem_mask = to_dense_batch(h, data.batch)                       # (B, N_max, d), (B, N_max)
        mem      = torch.cat([h_global.unsqueeze(1), mem], dim=1)           # (B, 1+N_max, d)
        mem_mask = torch.cat([mem_mask.new_ones(B, 1), mem_mask], dim=1)
        return mem, mem_mask

    def _start_emb(self, data):
        """Mean activity embedding over the prefix's last block (order invariant)."""
        lb = data.last_block_mask
        act_emb = self.act_emb(data.cat_x[:, -1])
        return global_mean_pool(act_emb[lb], data.batch[lb], size=data.num_graphs)  # (B, emb)

    # ── Decoder ──────────────────────────────────────────────────────────────

    def _tokens(self, act_e, nb, blk):
        # act_e: (B, T, emb); nb, blk: (B, T) long
        tok = (self.dec_act_proj(act_e) + self.nb_emb(nb)
               + self.dec_block_emb(blk.clamp(max=self.max_blocks - 1)))
        return self.dropout(tok)

    def _decode(self, tok, mem, mem_mask):
        T = tok.shape[1]
        causal = nn.Transformer.generate_square_subsequent_mask(T, device=tok.device)
        out = self.decoder(tok, mem, tgt_mask=causal,
                           memory_key_padding_mask=~mem_mask)
        return self.fc_out_act(out), self.fc_new_block(out).squeeze(-1)

    def forward(self, data, window_size=None):
        mem, mem_mask = self._encode(data)
        start_emb = self._start_emb(data)                                   # (B, emb)
        B = data.num_graphs
        W = window_size if window_size is not None else (data.suffix_act.shape[0] // B)

        if self.training:
            return self._teacher_forced(data, start_emb, mem, mem_mask, W)
        return self._greedy(start_emb, mem, mem_mask, W)

    def _teacher_forced(self, data, start_emb, mem, mem_mask, W):
        B = start_emb.shape[0]
        suffix_act = data.suffix_act.view(B, W)
        nb_label   = data.new_block_label.view(B, W).long()

        acts  = suffix_act[:, :-1].clamp(max=self.num_activities - 2)       # inputs of steps 1..W-1
        act_e = torch.cat([start_emb.unsqueeze(1), self.act_emb(acts)], dim=1)
        ones  = nb_label.new_ones(B, 1)
        nb    = torch.cat([ones, nb_label[:, :-1]], dim=1)
        blk   = torch.cat([ones - 1, nb_label.cumsum(dim=1)[:, :-1]], dim=1)

        return self._decode(self._tokens(act_e, nb, blk), mem, mem_mask)    # (B,W,C), (B,W)

    @torch.no_grad()
    def _greedy(self, start_emb, mem, mem_mask, W):
        B      = start_emb.shape[0]
        device = start_emb.device
        act_e  = start_emb.unsqueeze(1)                                     # (B, 1, emb)
        nb     = torch.ones(B, 1, dtype=torch.long, device=device)
        blk    = torch.zeros(B, 1, dtype=torch.long, device=device)

        all_act, all_nb = [], []
        for t in range(W):
            act_logits, nb_logits = self._decode(self._tokens(act_e, nb, blk), mem, mem_mask)
            logits = act_logits[:, -1].clone()
            logits[:, 0] = -1e9
            pred_act = logits.argmax(dim=-1)                                # (B,)
            pred_nb  = (nb_logits[:, -1] > 0).long()                        # (B,)
            all_act.append(pred_act)
            all_nb.append(pred_nb)

            if t == W - 1:
                break
            next_e = self.act_emb(pred_act.clamp(max=self.num_activities - 2))
            act_e  = torch.cat([act_e, next_e.unsqueeze(1)], dim=1)
            nb     = torch.cat([nb, pred_nb.unsqueeze(1)], dim=1)
            blk    = torch.cat([blk, (blk[:, -1] + pred_nb).unsqueeze(1)], dim=1)

        return torch.stack(all_act, dim=1), torch.stack(all_nb, dim=1).float()
