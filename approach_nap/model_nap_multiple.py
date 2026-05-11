"""
GNN model for Multiple-Activity Prediction (MAP).

Same architecture as model_nap.py (GNNNextActivity) but trained with
BCEWithLogitsLoss for multi-label output: the model predicts the SET of
activities that occur at the next concurrent timestamp block.

Metrics
-------
  exact_match : fraction of samples where the predicted set exactly equals
                the ground-truth set (no missing, no extra activities)
  f1_set      : per-sample F1 between predicted and true sets, averaged
                over all samples  (sklearn average='samples')
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch_geometric.nn import SAGEConv, global_mean_pool, global_max_pool


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────

class GNNMultipleActivity(nn.Module):
    """
    GraphSAGE encoder + MLP head for multi-label next-block prediction.

    Architecture is identical to GNNNextActivity in model_nap.py:
      embedding → 2-layer GraphSAGE with BatchNorm + residuals
      → global mean+max pool → 2-layer MLP
    Output is one logit per activity (interpreted with BCEWithLogitsLoss).

    Parameters
    ----------
    num_activities : int
    emb_dim : int
    hidden_channels : int
    out_channels : int  — equals num_activities
    dropout : float
    """

    def __init__(self, num_activities: int, emb_dim: int,
                 hidden_channels: int, out_channels: int, dropout: float = 0.3):
        super().__init__()

        # +1 for UNK
        self.act_emb = nn.Embedding(num_activities + 1, emb_dim)

        gnn_in = emb_dim + 1   # +1 for block-index position encoding

        self.conv1    = SAGEConv(gnn_in, hidden_channels)
        self.bn1      = nn.BatchNorm1d(hidden_channels)
        self.res_proj = nn.Linear(gnn_in, hidden_channels, bias=False)

        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2   = nn.BatchNorm1d(hidden_channels)

        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x, edge_index, batch, node_pos):
        x = self.act_emb(x)                          # (N, emb_dim)
        x = torch.cat([x, node_pos], dim=-1)         # (N, emb_dim + 1)

        x1 = self.bn1(self.conv1(x, edge_index)).relu()
        x1 = x1 + self.res_proj(x)
        x1 = self.dropout(x1)

        x2 = self.bn2(self.conv2(x1, edge_index)).relu()
        x2 = x2 + x1
        x2 = self.dropout(x2)

        graph_emb = torch.cat([
            global_mean_pool(x2, batch),
            global_max_pool(x2, batch),
        ], dim=-1)                                    # (B, 2 * hidden_channels)

        return self.mlp(graph_emb)                   # (B, out_channels) — raw logits


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    """One training epoch; returns mean loss."""
    model.train()
    total_loss = 0.0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        logits  = model(data.x, data.edge_index, data.batch, data.node_pos)
        targets = data.y                              # (B, num_activities), float
        loss    = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """
    Evaluate on a DataLoader.

    Returns
    -------
    mean_loss : float
    metrics   : dict
        exact_match — fraction of samples with perfectly predicted set
        f1_set      — mean per-sample F1 between predicted and true sets
    """
    model.eval()
    total_loss   = 0.0
    preds_list   = []
    targets_list = []

    for data in loader:
        data    = data.to(device)
        logits  = model(data.x, data.edge_index, data.batch, data.node_pos)
        targets = data.y
        loss    = criterion(logits, targets)
        total_loss += loss.item()
        preds = (torch.sigmoid(logits) > 0.5).float()
        preds_list.append(preds.cpu())
        targets_list.append(targets.cpu())

    y_pred = torch.cat(preds_list).numpy()
    y_true = torch.cat(targets_list).numpy()

    return total_loss / len(loader), _compute_metrics(y_pred, y_true)


def _compute_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> dict:
    def _scores(yp, yt):
        if len(yt) == 0:
            return 0.0, 0.0
        exact = float(np.all(yp == yt, axis=1).mean())
        f1    = float(f1_score(yt, yp, average='samples', zero_division=0))
        return exact, f1

    exact,        f1        = _scores(y_pred, y_true)

    single_mask = y_true.sum(axis=1) == 1
    multi_mask  = y_true.sum(axis=1) >  1

    exact_single, f1_single = _scores(y_pred[single_mask], y_true[single_mask])
    exact_multi,  f1_multi  = _scores(y_pred[multi_mask],  y_true[multi_mask])

    return {
        'exact_match':        exact,
        'f1_set':             f1,
        'exact_match_single': exact_single,
        'f1_set_single':      f1_single,
        'exact_match_multi':  exact_multi,
        'f1_set_multi':       f1_multi,
        'n_single':           int(single_mask.sum()),
        'n_multi':            int(multi_mask.sum()),
    }
