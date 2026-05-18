"""
GNN model for Multiple-Activity Prediction (MAP).

Architecture: embed → SAGEConv+BN+ReLU+Dropout → SAGEConv+BN+ReLU+Dropout → mean_pool → Linear head

Loss    : BCEWithLogitsLoss (multi-label)
Metrics : exact_match, f1_set (overall, single-event, multi-event)
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch_geometric.nn import SAGEConv, global_mean_pool


class GNNMultipleActivity(nn.Module):
    def __init__(self, num_activities: int, emb_dim: int,
                 hidden_channels: int, out_channels: int, dropout: float = 0.3):
        super().__init__()
        self.act_emb = nn.Embedding(num_activities + 1, emb_dim)
        self.conv1   = SAGEConv(emb_dim, hidden_channels)
        self.bn1     = nn.BatchNorm1d(hidden_channels)
        self.conv2   = SAGEConv(hidden_channels, hidden_channels)
        self.bn2     = nn.BatchNorm1d(hidden_channels)
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.act_emb(x)
        x = self.dropout(self.bn1(self.conv1(x, edge_index)).relu())
        x = self.dropout(self.bn2(self.conv2(x, edge_index)).relu())
        return self.head(global_mean_pool(x, batch))


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        logits  = model(data.x, data.edge_index, data.batch)
        targets = data.y
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
    model.eval()
    total_loss   = 0.0
    preds_list   = []
    targets_list = []

    for data in loader:
        data    = data.to(device)
        logits  = model(data.x, data.edge_index, data.batch)
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
