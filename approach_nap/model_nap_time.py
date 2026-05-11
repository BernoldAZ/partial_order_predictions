"""
GNN model for joint Next Activity + Next Timestamp Prediction (NAP-Time).

Extends the NAP approach with the multi-task idea from:
  N. Tax, I. Verenich, M. La Rosa, M. Dumas,
  "Predictive business process monitoring with LSTM neural networks", 2017.

Architecture:
  - Learned activity embedding (emb_dim set dynamically per log)
  - Block-index position encoding on nodes
  - 4 Tax et al. time features stored on edges (same for all edges arriving
    at the same concurrent block):
      dt_prev_norm   — time since previous block  / mean_dt_prev
      dt_start_norm  — time since case start      / mean_dt_start
      time_of_day    — seconds since midnight     / 86400
      day_of_week    — weekday (0-6)              / 7
  - 2-layer GATv2Conv encoder (edge_dim=4) with BatchNorm and residuals
  - Global mean + max pooling concatenated
  - Two separate MLP heads sharing the same graph embedding:
      act_head  — softmax classification (next activity)
      time_head — regression (time to next event, normalized)

Loss     : CrossEntropyLoss (activity) + λ * L1Loss (time)
           λ = 1.0 by default, matching Tax et al. equal-weight approach
Metrics  : accuracy, weighted F1 (activity) + MAE in normalized units (time)
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────

class GNNNextActivityTime(nn.Module):
    """
    GATv2Conv encoder + dual MLP head for joint next-activity and
    next-timestamp prediction.

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

        # Node input: embedding + block position scalar
        gnn_in = emb_dim + 1

        # Layer 1
        self.conv1 = GATv2Conv(
            in_channels=gnn_in,
            out_channels=hidden_channels,
            heads=4,
            concat=False,
            edge_dim=4,
            dropout=dropout,
        )
        self.bn1      = nn.BatchNorm1d(hidden_channels)
        self.res_proj = nn.Linear(gnn_in, hidden_channels, bias=False)

        # Layer 2 (residual without projection — same width)
        self.conv2 = GATv2Conv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            heads=4,
            concat=False,
            edge_dim=4,
            dropout=dropout,
        )
        self.bn2 = nn.BatchNorm1d(hidden_channels)

        self.dropout = nn.Dropout(dropout)

        # Mean + max pooling → 2 * hidden_channels into both heads
        pool_dim = 2 * hidden_channels

        self.act_head = nn.Sequential(
            nn.Linear(pool_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )

        self.time_head = nn.Sequential(
            nn.Linear(pool_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, x, edge_index, edge_attr, batch, node_pos):
        """
        Parameters
        ----------
        x          : (num_nodes,)    — integer activity IDs
        edge_index : (2, num_edges)
        edge_attr  : (num_edges, 4)  — Tax et al. time features per edge
        batch      : (num_nodes,)
        node_pos   : (num_nodes, 1)  — normalized block index ∈ [0, 1]

        Returns
        -------
        act_logits : (batch_size, out_channels)
        time_pred  : (batch_size, 1)   — normalized time to next event
        """
        x = self.act_emb(x)                            # (N, emb_dim)
        x = torch.cat([x, node_pos], dim=-1)           # (N, gnn_in)

        # Layer 1 with residual
        x1 = self.bn1(self.conv1(x, edge_index, edge_attr)).relu()
        x1 = x1 + self.res_proj(x)
        x1 = self.dropout(x1)

        # Layer 2 with residual
        x2 = self.bn2(self.conv2(x1, edge_index, edge_attr)).relu()
        x2 = x2 + x1
        x2 = self.dropout(x2)

        graph_emb = torch.cat([
            global_mean_pool(x2, batch),
            global_max_pool(x2, batch),
        ], dim=-1)                                      # (B, 2*hidden_channels)

        return self.act_head(graph_emb), self.time_head(graph_emb)


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion_act, criterion_time,
                device, lambda_time=1.0):
    """
    One training epoch with multi-task loss.

    loss = CrossEntropy(activity) + lambda_time * L1(time)

    Returns mean combined loss.
    """
    model.train()
    total_loss = 0.0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        act_logits, time_pred = model(
            data.x, data.edge_index, data.edge_attr,
            data.batch, data.node_pos)

        targets = data.y.argmax(dim=-1)               # (B,)
        y_time  = data.y_time
        if y_time.dim() == 1:
            y_time = y_time.unsqueeze(1)

        loss = (criterion_act(act_logits, targets)
                + lambda_time * criterion_time(time_pred, y_time))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, criterion_act, criterion_time, device,
             lambda_time=1.0):
    """
    Evaluate on a DataLoader.

    Returns
    -------
    mean_loss : float  — combined loss
    metrics   : dict
        accuracy  — argmax accuracy
        f1        — weighted F1
        time_mae  — mean absolute error on normalized time prediction
    """
    model.eval()
    total_loss   = 0.0
    preds_list   = []
    targets_list = []
    time_abs_err = []

    for data in loader:
        data = data.to(device)

        act_logits, time_pred = model(
            data.x, data.edge_index, data.edge_attr,
            data.batch, data.node_pos)

        targets = data.y.argmax(dim=-1)
        y_time  = data.y_time
        if y_time.dim() == 1:
            y_time = y_time.unsqueeze(1)

        loss = (criterion_act(act_logits, targets)
                + lambda_time * criterion_time(time_pred, y_time))
        total_loss += loss.item()

        preds_list.append(act_logits.argmax(dim=-1).cpu())
        targets_list.append(targets.cpu())
        time_abs_err.append((time_pred - y_time).abs().cpu())

    y_pred = torch.cat(preds_list).numpy()
    y_true = torch.cat(targets_list).numpy()
    time_mae = float(torch.cat(time_abs_err).mean())

    metrics = _compute_metrics(y_pred, y_true)
    metrics['time_mae'] = time_mae

    return total_loss / len(loader), metrics


def _compute_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> dict:
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1':       float(f1_score(y_true, y_pred, average='weighted',
                                   zero_division=0)),
    }
