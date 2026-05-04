"""
GNN model for Next Activity Prediction (NAP).

Architecture:
  - 2-layer GraphSAGE encoder over the prefix DAG
  - Global mean pooling → graph-level embedding
  - 2-layer MLP classification head (one logit per activity)

Loss     : CrossEntropyLoss  (single-label; multi-hot target converted via argmax)
Metrics  : accuracy and weighted F1  — same as the LSTM baselines in
           baselines/next_activity_prediction/next_activity_prediction.py
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.nn import SAGEConv, global_mean_pool


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────

class GNNNextActivity(nn.Module):
    """
    GraphSAGE encoder + MLP head for next-activity classification.

    Parameters
    ----------
    in_channels : int
        Node feature dimension (vocabulary size for one-hot encoding).
    hidden_channels : int
        Hidden size of GNN and MLP layers.
    out_channels : int
        Number of activity classes (vocabulary size).
    dropout : float
    """

    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, dropout: float = 0.3):
        super().__init__()
        self.conv1   = SAGEConv(in_channels, hidden_channels)
        self.conv2   = SAGEConv(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(dropout)
        self.mlp     = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x, edge_index, batch):
        """
        Parameters
        ----------
        x          : (num_nodes, in_channels)
        edge_index : (2, num_edges)
        batch      : (num_nodes,)

        Returns
        -------
        logits : (batch_size, out_channels)
        """
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index).relu()
        x = self.dropout(x)
        x = global_mean_pool(x, batch)
        return self.mlp(x)


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
        logits  = model(data.x, data.edge_index, data.batch)
        # data.y shape: (B, num_activities)  — multi-hot from pipeline
        # Convert to integer class label via argmax for CrossEntropyLoss
        targets = data.y.argmax(dim=-1)       # (B,)
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
        accuracy  — argmax accuracy (matches baseline)
        f1        — weighted F1     (matches baseline)
    """
    model.eval()
    total_loss   = 0.0
    preds_list   = []
    targets_list = []

    for data in loader:
        data    = data.to(device)
        logits  = model(data.x, data.edge_index, data.batch)
        targets = data.y.argmax(dim=-1)
        loss    = criterion(logits, targets)
        total_loss += loss.item()
        preds_list.append(logits.argmax(dim=-1).cpu())
        targets_list.append(targets.cpu())

    y_pred = torch.cat(preds_list).numpy()
    y_true = torch.cat(targets_list).numpy()

    return total_loss / len(loader), _compute_metrics(y_pred, y_true)


def _compute_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> dict:
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1':       float(f1_score(y_true, y_pred, average='weighted',
                                   zero_division=0)),
    }
