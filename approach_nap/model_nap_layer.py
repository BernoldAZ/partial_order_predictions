"""
GNN model for Next Activity Prediction with Layer Prediction (NAP + Layer).

Architecture: embed → SAGEConv+BN+ReLU+Dropout → SAGEConv+BN+ReLU+Dropout
              → global_mean_pool(all) + global_mean_pool(last_layer)
              → cat → Linear head (activity) + Linear head (layer)

Loss    : alpha * CrossEntropyLoss(activity) + (1-alpha) * CrossEntropyLoss(layer)
Metrics : activity accuracy, activity weighted F1, layer recall per class
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch_geometric.nn import SAGEConv, global_mean_pool


class GNNNextActivityLayer(nn.Module):
    def __init__(self, num_activities: int, emb_dim: int,
                 hidden_channels: int, out_channels: int, dropout: float = 0.3):
        super().__init__()
        self.act_emb  = nn.Embedding(num_activities + 1, emb_dim)
        self.conv1    = SAGEConv(emb_dim, hidden_channels)
        self.bn1      = nn.BatchNorm1d(hidden_channels)
        self.conv2    = SAGEConv(hidden_channels, hidden_channels)
        self.bn2      = nn.BatchNorm1d(hidden_channels)
        self.dropout  = nn.Dropout(dropout)
        # Both heads receive [g_global || g_last] → 2 * hidden_channels
        self.head_act   = nn.Linear(hidden_channels * 2, out_channels)
        self.head_layer = nn.Linear(hidden_channels * 2, 2)

    def forward(self, x, edge_index, batch, last_layer_mask):
        h = self.act_emb(x)
        h = self.dropout(self.bn1(self.conv1(h, edge_index)).relu())
        h = self.dropout(self.bn2(self.conv2(h, edge_index)).relu())

        batch_size = int(batch.max()) + 1
        g_global = global_mean_pool(h, batch)                                      # (B, H)
        g_last   = global_mean_pool(h[last_layer_mask], batch[last_layer_mask],
                                    size=batch_size)                               # (B, H)
        g = torch.cat([g_global, g_last], dim=-1)                                  # (B, 2H)

        return self.head_act(g), self.head_layer(g)


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion_act, criterion_layer,
                device, alpha: float = 0.7):
    model.train()
    total_loss = 0.0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        act_logits, layer_logits = model(
            data.x, data.edge_index, data.batch, data.last_layer_mask)
        act_targets   = data.y.argmax(dim=-1)
        layer_targets = data.y_layer.squeeze(-1)
        loss = (alpha * criterion_act(act_logits, act_targets)
                + (1 - alpha) * criterion_layer(layer_logits, layer_targets))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, criterion_act, criterion_layer, device,
             alpha: float = 0.8):
    model.eval()
    total_loss        = 0.0
    act_preds_list    = []
    act_targets_list  = []
    layer_preds_list  = []
    layer_targets_list = []

    for data in loader:
        data = data.to(device)
        act_logits, layer_logits = model(
            data.x, data.edge_index, data.batch, data.last_layer_mask)
        act_targets   = data.y.argmax(dim=-1)
        layer_targets = data.y_layer.squeeze(-1)

        loss = (alpha * criterion_act(act_logits, act_targets)
                + (1 - alpha) * criterion_layer(layer_logits, layer_targets))
        total_loss += loss.item()

        act_preds_list.append(act_logits.argmax(dim=-1).cpu())
        act_targets_list.append(act_targets.cpu())
        layer_preds_list.append(layer_logits.argmax(dim=-1).cpu())
        layer_targets_list.append(layer_targets.cpu())

    y_act_pred   = torch.cat(act_preds_list).numpy()
    y_act_true   = torch.cat(act_targets_list).numpy()
    y_layer_pred = torch.cat(layer_preds_list).numpy()
    y_layer_true = torch.cat(layer_targets_list).numpy()

    layer_recalls = recall_score(y_layer_true, y_layer_pred,
                                 average=None, zero_division=0, labels=[0, 1])
    return total_loss / len(loader), {
        'accuracy':       float(accuracy_score(y_act_true, y_act_pred)),
        'f1':             float(f1_score(y_act_true, y_act_pred,
                                         average='weighted', zero_division=0)),
        'layer_recall_0': float(layer_recalls[0]),
        'layer_recall_1': float(layer_recalls[1]),
    }
