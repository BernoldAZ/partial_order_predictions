from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, global_mean_pool
from sklearn.metrics import f1_score
import numpy as np

# ---------------------------
# Simple GNN
# ---------------------------
class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        return self.lin(x)  # logits

# ---------------------------
# Training
# ---------------------------
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)

# ---------------------------
# Validation / evaluation
# ---------------------------
def validation(model, loader, criterion, device):
    model.eval()
    total_loss = 0

    preds_all = []
    targets_all = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            total_loss += loss.item()

            preds = (torch.sigmoid(out) > 0.5).float()
            preds_all.append(preds.cpu())
            targets_all.append(data.y.cpu())

    preds_all = torch.cat(preds_all, dim=0).numpy()
    targets_all = torch.cat(targets_all, dim=0).numpy()

    # Normalize loss if provided
    normalized_loss = (total_loss / len(loader))
    overall_exact, overall_f1, single_exact, single_f1, multi_exact, multi_f1 = compute_occurrence_metrics(preds_all, targets_all)
    
    return normalized_loss, overall_exact, overall_f1, single_exact, single_f1, multi_exact, multi_f1

# ---------------------------
# Prediction function
# ---------------------------
def predict(model, loader, device):
    """
    Generate predictions (0/1) for all samples in a DataLoader.
    """
    model.eval()
    preds_all = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            preds = (torch.sigmoid(out) > 0.5).float()
            preds_all.append(preds.cpu())

    preds_all = torch.cat(preds_all, dim=0).numpy()
    return preds_all

# ---------------------------
#  functions for metrics and evaluation
# ---------------------------
def compute_metrics(preds, targets):
    if len(targets) == 0:
        return 0.0, 0.0
    exact = np.all(preds == targets, axis=1).mean()
    f1 = f1_score(targets, preds, average="micro")
    return exact, f1

def compute_occurrence_metrics(preds_all, targets_all):
    """
    Compute metrics for predictions split by single-activity and multi-activity samples.

    Parameters:
        preds_all (np.ndarray or torch.Tensor): Binary predictions (N x num_labels)
        targets_all (np.ndarray or torch.Tensor): Ground truth labels (N x num_labels)
        total_loss (float, optional): Total loss to normalize
        loader_len (int, optional): Number of batches for loss normalization

    Returns:
        tuple:
            - normalized_loss (float or None)
            - overall_exact, overall_f1
            - single_exact, single_f1
            - multi_exact, multi_f1
    """

    # Convert to numpy if torch tensors
    if isinstance(preds_all, torch.Tensor):
        preds_all = preds_all.cpu().numpy()
    if isinstance(targets_all, torch.Tensor):
        targets_all = targets_all.cpu().numpy()

    # Split by occurrence
    target_sums = targets_all.sum(axis=1)
    single_mask = target_sums == 1
    multi_mask = target_sums > 1

    # Single-activity
    single_preds = preds_all[single_mask]
    single_targets = targets_all[single_mask]
    single_exact, single_f1 = compute_metrics(single_preds, single_targets)

    # Multi-activity
    multi_preds = preds_all[multi_mask]
    multi_targets = targets_all[multi_mask]
    multi_exact, multi_f1 = compute_metrics(multi_preds, multi_targets)

    # Overall metrics
    overall_exact, overall_f1 = compute_metrics(preds_all, targets_all)

    return overall_exact, overall_f1, single_exact, single_f1, multi_exact, multi_f1