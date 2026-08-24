"""
GATv2 encoder with a single remaining-time regression head.

Predicts:
  - remaining time (time from end of prefix to end of trace)  (fc_out_rtime)

No decoder. The encoder context vector is used directly for regression.
"""

import torch
import torch.nn as nn
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


class GATv2RemainingTime(nn.Module):
    """
    GATv2 encoder with a single remaining-time regression head.

    Parameters
    ----------
    num_activities : int
        Total classes incl. padding (0) and END (num_activities-1).
    d_model : int
        Hidden size for GNN layers.
    dropout : float
    nhead : int
        Number of GATv2Conv attention heads (concat=False → output is d_model).
    """

    def __init__(self, num_activities: int, d_model: int = 64,
                 dropout: float = 0.2, nhead: int = 4):
        super().__init__()
        self.num_activities = num_activities
        self.d_model        = d_model

        emb_size      = min(600, round(1.6 * (num_activities - 2) ** 0.56))
        self.emb_size = emb_size
        self.act_emb  = nn.Embedding(num_activities - 1, emb_size, padding_idx=0)
        self.dropout  = nn.Dropout(dropout)

        self.gatv2     = GATv2Conv(emb_size + 1, d_model, heads=nhead, concat=False, edge_dim=1)
        self.edge_bias = _EdgeAttnBias(d_model)
        self.bn_enc    = nn.BatchNorm1d(d_model * 2)

        self.fc_out_rtime = nn.Linear(d_model * 2, 1)
        init.xavier_uniform_(self.fc_out_rtime.weight)

    def _encode(self, data):
        h = torch.cat([self.act_emb(data.cat_x[:, -1]), data.x[:, [0]]], dim=-1)
        h = self.dropout(h)
        h = self.gatv2(h, data.edge_index, data.edge_attr).relu()
        h = h + self.edge_bias(h, data.edge_index, data.edge_attr)
        h = self.dropout(h)
        h_global = global_mean_pool(h, data.batch)
        h_last   = global_mean_pool(h[data.last_block_mask], data.batch[data.last_block_mask])
        h = torch.cat([h_global, h_last], dim=-1)    # (B, 2*d_model)
        return self.bn_enc(h)

    def forward(self, data):
        c = self._encode(data)          # (B, 2*d_model)
        return self.fc_out_rtime(c)     # (B, 1)
