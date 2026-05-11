"""
GNN model for Next Activity Prediction (NAP).

Architecture:
  - Learned activity embedding (emb_dim set dynamically per log)
  - Block-index position encoding appended to embedding (same value for
    all concurrent nodes, preserving their structural symmetry)
  - 2-layer GraphSAGE encoder with BatchNorm and residual connections
  - Global mean + max pooling concatenated → graph-level embedding
  - 2-layer MLP classification head (one logit per activity)

Loss     : CrossEntropyLoss  (single-label; multi-hot target converted via argmax)
Metrics  : accuracy and weighted F1  — same as the LSTM baselines in
           baselines/next_activity_prediction/next_activity_prediction.py
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.nn import SAGEConv, global_mean_pool, global_max_pool


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────

class GNNNextActivity(nn.Module):
    """
    GraphSAGE encoder + MLP head for next-activity classification.

    Parameters
    ----------
    num_activities : int
        Vocabulary size (number of unique activities in training data).
    emb_dim : int
        Activity embedding dimension. Set dynamically per log.
    hidden_channels : int
        Hidden size of GNN and MLP layers.
    out_channels : int
        Number of activity classes (vocabulary size).
    dropout : float
    """

    def __init__(self, num_activities: int, emb_dim: int,
                 hidden_channels: int, out_channels: int, dropout: float = 0.3):
        super().__init__()

        # +1 for UNK (activities seen in val/test but not in vocab)
        self.act_emb = nn.Embedding(num_activities + 1, emb_dim)

        # Position encoding: block index appended → input width = emb_dim + 1
        gnn_in = emb_dim + 1

        # Layer 1: project to hidden_channels
        self.conv1    = SAGEConv(gnn_in, hidden_channels)
        self.bn1      = nn.BatchNorm1d(hidden_channels)
        self.res_proj = nn.Linear(gnn_in, hidden_channels, bias=False)

        # Layer 2: hidden_channels → hidden_channels (residual without projection)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2   = nn.BatchNorm1d(hidden_channels)

        self.dropout = nn.Dropout(dropout)

        # Mean + max pooling → 2 * hidden_channels
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x, edge_index, batch, node_pos):
        """
        Parameters
        ----------
        x        : (num_nodes,)      — integer activity IDs
        edge_index : (2, num_edges)
        batch    : (num_nodes,)
        node_pos : (num_nodes, 1)    — normalized block index ∈ [0, 1]

        Returns
        -------
        logits : (batch_size, out_channels)
        """
        x = self.act_emb(x)                              # (N, emb_dim)
        x = torch.cat([x, node_pos], dim=-1)             # (N, emb_dim + 1)

        # Layer 1 with residual
        x1 = self.bn1(self.conv1(x, edge_index)).relu()
        x1 = x1 + self.res_proj(x)                      # skip from input
        x1 = self.dropout(x1)

        # Layer 2 with residual
        x2 = self.bn2(self.conv2(x1, edge_index)).relu()
        x2 = x2 + x1                                     # skip from layer 1
        x2 = self.dropout(x2)

        # Mean + max pooling
        graph_emb = torch.cat([
            global_mean_pool(x2, batch),
            global_max_pool(x2, batch),
        ], dim=-1)                                        # (B, 2 * hidden_channels)

        return self.mlp(graph_emb)


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
        logits  = model(data.x, data.edge_index, data.batch, data.node_pos)
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
