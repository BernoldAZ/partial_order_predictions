"""
GNN encoder + MLP heads for Next Activity + Timestamp Prediction.

Architecture: embed → SAGEConv+BN+ReLU+Dropout → SAGEConv+BN+ReLU+Dropout → mean_pool → act_head / time_head

Loss    : CrossEntropyLoss (activity) + λ * L1Loss (time)
Metrics : accuracy, weighted F1, time MAE
"""

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.nn import SAGEConv, global_mean_pool


class GNNNAPTimeMLPModel(nn.Module):
    def __init__(self, num_activities: int, emb_dim: int,
                 hidden_channels: int, out_channels: int, dropout: float = 0.3):
        super().__init__()
        self.node_emb  = nn.Embedding(num_activities + 1, emb_dim)
        self.conv1     = SAGEConv(emb_dim, hidden_channels)
        self.bn1       = nn.BatchNorm1d(hidden_channels)
        self.conv2     = SAGEConv(hidden_channels, hidden_channels)
        self.bn2       = nn.BatchNorm1d(hidden_channels)
        self.dropout   = nn.Dropout(dropout)
        self.act_head  = nn.Linear(hidden_channels, out_channels)
        self.time_head = nn.Sequential(nn.Linear(hidden_channels, 1), nn.Softplus())

    def forward(self, x, edge_index, batch):
        x = self.node_emb(x)
        x = self.dropout(self.bn1(self.conv1(x, edge_index)).relu())
        x = self.dropout(self.bn2(self.conv2(x, edge_index)).relu())
        g = global_mean_pool(x, batch)
        return self.act_head(g), self.time_head(g)


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion_act, criterion_time,
                device, lambda_time=1.0):
    model.train()
    total_loss = 0.0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        act_logits, time_pred = model(data.x, data.edge_index, data.batch)

        targets = data.y.argmax(dim=-1)
        y_time  = data.y_time
        if y_time.dim() == 1:
            y_time = y_time.unsqueeze(1)

        loss = criterion_act(act_logits, targets) + lambda_time * criterion_time(time_pred, y_time)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, criterion_act, criterion_time, device, lambda_time=1.0):
    model.eval()
    total_loss   = 0.0
    preds_list   = []
    targets_list = []
    time_abs_err = []

    for data in loader:
        data = data.to(device)

        act_logits, time_pred = model(data.x, data.edge_index, data.batch)

        targets = data.y.argmax(dim=-1)
        y_time  = data.y_time
        if y_time.dim() == 1:
            y_time = y_time.unsqueeze(1)

        loss = criterion_act(act_logits, targets) + lambda_time * criterion_time(time_pred, y_time)
        total_loss += loss.item()

        preds_list.append(act_logits.argmax(dim=-1).cpu())
        targets_list.append(targets.cpu())
        time_abs_err.append((time_pred - y_time).abs().cpu())

    y_pred   = torch.cat(preds_list).numpy()
    y_true   = torch.cat(targets_list).numpy()
    time_mae = float(torch.cat(time_abs_err).mean())

    return total_loss / len(loader), {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1':       float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'time_mae': time_mae,
    }
