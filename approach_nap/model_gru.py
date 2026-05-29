"""
GRU model for Next Event Prediction (activity + delta time + concurrency).

Architecture:
  (activity, dt, conc)
        ↓
  embeddings (act_emb + time_mlp + conc_emb)
        ↓
  input_proj → GRU encoder
        ↓
  hidden state h_t
        ↓
   ├── activity head   (CrossEntropyLoss)
   ├── time head       (MSELoss on normalized log-dt)
   └── concurrency head (BCEWithLogitsLoss)

Multi-task loss: L = L_act + λ1*L_time + λ2*L_conc
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score


class GRUNextEvent(nn.Module):
    def __init__(self, num_activities, d_act, d_time, d_conc, d_model, hidden_dim):
        super().__init__()
        self.act_emb  = nn.Embedding(num_activities + 1, d_act)  # +1 for unk
        self.time_mlp = nn.Sequential(
            nn.Linear(1, d_time),
            nn.ReLU(),
            nn.Linear(d_time, d_time),
        )
        self.conc_emb   = nn.Embedding(2, d_conc)
        self.input_proj = nn.Linear(d_act + d_time + d_conc, d_model)
        self.gru = nn.GRU(input_size=d_model, hidden_size=hidden_dim,
                          batch_first=True)
        self.act_head  = nn.Linear(hidden_dim, num_activities)
        self.time_head = nn.Linear(hidden_dim, 1)
        self.conc_head = nn.Linear(hidden_dim, 1)

    def forward(self, activity, dt, conc):
        # activity: (B, T)  dt: (B, T)  conc: (B, T)
        a = self.act_emb(activity)           # (B, T, d_act)
        t = self.time_mlp(dt.unsqueeze(-1))  # (B, T, d_time)
        c = self.conc_emb(conc)              # (B, T, d_conc)
        x = self.input_proj(torch.cat([a, t, c], dim=-1))  # (B, T, d_model)
        h, _ = self.gru(x)                   # (B, T, hidden_dim)
        return (
            self.act_head(h),                # (B, T, num_activities)
            self.time_head(h).squeeze(-1),   # (B, T)
            self.conc_head(h).squeeze(-1),   # (B, T)
        )


# ─────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────

def compute_loss(act_logits, time_pred, conc_logits, batch,
                 lambda1=0.3, lambda2=0.1, unk_idx=None):
    mask        = batch['mask']                    # (B, T) bool
    act_target  = batch['act_target']              # (B, T)
    dt_target   = batch['dt_target']               # (B, T)
    conc_target = batch['conc_target'].float()     # (B, T)

    ignore = unk_idx if unk_idx is not None else -100
    L_act  = F.cross_entropy(act_logits[mask], act_target[mask],
                              ignore_index=ignore)
    L_time = F.mse_loss(time_pred[mask], dt_target[mask])
    L_conc = F.binary_cross_entropy_with_logits(conc_logits[mask],
                                                 conc_target[mask])
    return L_act + lambda1 * L_time + lambda2 * L_conc


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device,
                lambda1=0.3, lambda2=0.1, unk_idx=None):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        act_logits, time_pred, conc_logits = model(
            batch['act'], batch['dt'], batch['conc'])
        loss = compute_loss(act_logits, time_pred, conc_logits, batch,
                            lambda1, lambda2, unk_idx)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, dt_std=1.0,
             lambda1=0.3, lambda2=0.1, unk_idx=None):
    model.eval()
    total_loss = 0.0
    act_preds_list    = []
    act_targets_list  = []
    time_preds_list   = []
    time_targets_list = []
    conc_preds_list   = []
    conc_targets_list = []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        act_logits, time_pred, conc_logits = model(
            batch['act'], batch['dt'], batch['conc'])

        total_loss += compute_loss(act_logits, time_pred, conc_logits, batch,
                                   lambda1, lambda2, unk_idx).item()

        mask = batch['mask']
        act_preds_list.append(act_logits[mask].argmax(dim=-1).cpu())
        act_targets_list.append(batch['act_target'][mask].cpu())
        time_preds_list.append(time_pred[mask].cpu())
        time_targets_list.append(batch['dt_target'][mask].cpu())
        conc_preds_list.append((conc_logits[mask] > 0).long().cpu())
        conc_targets_list.append(batch['conc_target'][mask].cpu())

    y_act_pred  = torch.cat(act_preds_list).numpy()
    y_act_true  = torch.cat(act_targets_list).numpy()
    y_time_pred = torch.cat(time_preds_list).numpy()
    y_time_true = torch.cat(time_targets_list).numpy()
    y_conc_pred = torch.cat(conc_preds_list).numpy()
    y_conc_true = torch.cat(conc_targets_list).numpy()

    # Exclude unk targets from activity metrics
    if unk_idx is not None:
        valid = y_act_true != unk_idx
        y_act_pred = y_act_pred[valid]
        y_act_true = y_act_true[valid]

    # MAE in log(1+dt) space: multiply normalized MAE by dt_std
    mae_log = float(np.mean(np.abs(y_time_pred - y_time_true)) * dt_std)

    return total_loss / len(loader), {
        'accuracy': float(accuracy_score(y_act_true, y_act_pred)),
        'f1':       float(f1_score(y_act_true, y_act_pred,
                                   average='weighted', zero_division=0)),
        'mae_time': mae_log,
        'acc_conc': float(accuracy_score(y_conc_true, y_conc_pred)),
    }
