"""GATv2 encoder (data-aware) + GRU decoder with binary stop head.

Extends GATv2EncoderGRUDecoderStop (v2) by incorporating all prefix node
features into the encoder, not just the activity label:

  - cat_x[:, -1]   : activity label          (existing act_emb)
  - cat_x[:, :-1]  : other categorical fts   (one Embedding per feature)
  - data.x         : numeric node features   (passed directly, already standardised)

All are concatenated and projected to emb_size before GATv2Conv, so the
rest of the encoder and the entire decoder are unchanged from v2.
"""

import torch
import torch.nn as nn
import torch.nn.init as init
from torch_geometric.nn import global_mean_pool

from model_suffix_time_v2 import GATv2EncoderGRUDecoderStop


class GATv2DataAwareEncoderGRUDecoderStop(GATv2EncoderGRUDecoderStop):

    def __init__(self, cat_cardinalities, n_numeric, **kwargs):
        """
        Parameters
        ----------
        cat_cardinalities : list[int]
            Cardinality of each non-activity categorical prefix feature
            (i.e. pref_cat_cars[:-1]).  Empty list if none exist.
        n_numeric : int
            Number of standardised numeric node features (data.x.shape[1]).
            0 if none exist.
        **kwargs : passed to GATv2EncoderGRUDecoderStop
        """
        super().__init__(**kwargs)

        cat_emb_size = 16
        self.cat_embs = nn.ModuleList([
            nn.Embedding(c + 1, cat_emb_size, padding_idx=0)
            for c in cat_cardinalities
        ])

        total_in = self.emb_size + len(cat_cardinalities) * cat_emb_size + n_numeric
        self.input_proj = nn.Linear(total_in, self.emb_size)
        init.xavier_uniform_(self.input_proj.weight)

    def _encode(self, data):
        parts = [self.act_emb(data.cat_x[:, -1])]
        for i, emb in enumerate(self.cat_embs):
            parts.append(emb(data.cat_x[:, i]))
        if data.x.shape[1] > 0:
            parts.append(data.x)
        h = self.input_proj(torch.cat(parts, dim=-1)).relu()
        h = self.dropout(h)
        h = self.gatv2(h, data.edge_index, data.edge_attr).relu()
        h = h + self.edge_bias(h, data.edge_index, data.edge_attr)
        h = self.dropout(h)
        h_global = global_mean_pool(h, data.batch)
        last_idx  = data.ptr[1:] - 1
        h_last    = h[last_idx]
        h = torch.cat([h_global, h_last], dim=-1)
        h = self.bn_enc(h)
        return h
