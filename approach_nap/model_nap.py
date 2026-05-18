"""
GNN model for Next Activity Prediction (NAP).

Architecture: embed → SAGEConv+BN+ReLU+Dropout → SAGEConv+BN+ReLU+Dropout → mean_pool → Linear head

Loss    : CrossEntropyLoss
Metrics : accuracy, weighted F1
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.nn import SAGEConv, global_mean_pool


class GNNNextActivity(nn.Module):
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
        targets = data.y.argmax(dim=-1)
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
        targets = data.y.argmax(dim=-1)
        loss    = criterion(logits, targets)
        total_loss += loss.item()
        preds_list.append(logits.argmax(dim=-1).cpu())
        targets_list.append(targets.cpu())

    y_pred = torch.cat(preds_list).numpy()
    y_true = torch.cat(targets_list).numpy()

    return total_loss / len(loader), {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1':       float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
    }
