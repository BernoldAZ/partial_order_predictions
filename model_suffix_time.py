"""
GNN encoder + LSTM decoder for Activity Suffix + Time Prediction.

Same graph encoder and LSTM decoder as model_suffix.py, extended with two
scalar regression heads applied directly to the graph embedding:
  - TTNE head : time till next event  (seconds, normalised by training mean)
  - RRT head  : remaining runtime     (seconds, normalised by training mean)

Training loss  : CrossEntropy (activity) + L1 (TTNE) + L1 (RRT)
Time metrics   : MAE TTNE (minutes), MAE RRT (minutes)
Activity metrics: per-position accuracy, Damerau-Levenshtein Similarity
"""

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, global_mean_pool


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────

class GNNSuffixTimeModel(nn.Module):
    """
    GNN encoder + LSTM decoder with TTNE and RRT regression heads.

    Parameters
    ----------
    in_channels : int
    hidden_channels : int
    num_activities : int
    emb_dim : int
    lstm_hidden : int
    dropout : float
    """

    def __init__(self, in_channels: int, hidden_channels: int,
                 num_activities: int, emb_dim: int = 64,
                 lstm_hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        vocab_size = num_activities + 2          # 0=PAD, 1..N=acts, N+1=END

        # GNN encoder
        self.conv1   = SAGEConv(in_channels, hidden_channels)
        self.conv2   = SAGEConv(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(dropout)

        # LSTM decoder (activity suffix)
        self.enc_to_h  = nn.Linear(hidden_channels, lstm_hidden)
        self.enc_to_c  = nn.Linear(hidden_channels, lstm_hidden)
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm      = nn.LSTM(emb_dim, lstm_hidden, batch_first=True)
        self.out_proj  = nn.Linear(lstm_hidden, vocab_size)

        # Time regression heads
        self.ttne_head = nn.Linear(hidden_channels, 1)
        self.rrt_head  = nn.Linear(hidden_channels, 1)

    def encode(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index).relu()
        x = self.dropout(x)
        return global_mean_pool(x, batch)

    def forward(self, x, edge_index, batch, y_in):
        """
        Parameters
        ----------
        y_in : (B, max_suffix_len)  teacher-forced decoder input

        Returns
        -------
        logits    : (B, max_suffix_len, vocab_size)
        ttne_pred : (B,)  normalised TTNE prediction
        rrt_pred  : (B,)  normalised RRT prediction
        """
        graph_emb = self.encode(x, edge_index, batch)

        ttne_pred = self.ttne_head(graph_emb).squeeze(-1)   # (B,)
        rrt_pred  = self.rrt_head(graph_emb).squeeze(-1)    # (B,)

        h0  = self.enc_to_h(graph_emb).unsqueeze(0)
        c0  = self.enc_to_c(graph_emb).unsqueeze(0)
        emb = self.embedding(y_in)
        out, _ = self.lstm(emb, (h0, c0))
        logits  = self.out_proj(out)                         # (B, L, vocab)

        return logits, ttne_pred, rrt_pred

    @torch.no_grad()
    def greedy_decode(self, x, edge_index, batch, end_token_idx: int, max_len: int):
        """Autoregressive greedy decoding; returns (batch_size, max_len) token tensor."""
        self.eval()
        batch_size = int(batch.max().item()) + 1
        device     = x.device

        graph_emb = self.encode(x, edge_index, batch)
        h = self.enc_to_h(graph_emb).unsqueeze(0)
        c = self.enc_to_c(graph_emb).unsqueeze(0)

        token    = torch.zeros(batch_size, dtype=torch.long, device=device)
        outputs  = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len):
            emb              = self.embedding(token.unsqueeze(1))
            lstm_out, (h, c) = self.lstm(emb, (h, c))
            logits           = self.out_proj(lstm_out.squeeze(1))
            pred             = logits.argmax(dim=-1)
            pred             = pred.masked_fill(finished, 0)
            outputs.append(pred)
            finished         = finished | (pred == end_token_idx)
            if finished.all():
                break
            token = pred

        while len(outputs) < max_len:
            outputs.append(torch.zeros(batch_size, dtype=torch.long, device=device))

        return torch.stack(outputs, dim=1)


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion_ce, device,
                mean_ttne: float, mean_rrt: float,
                lambda_ttne: float = 1.0, lambda_rrt: float = 1.0):
    """
    One training epoch.  Time targets are normalised by their training means
    so all loss terms are on a comparable scale.

    Parameters
    ----------
    mean_ttne, mean_rrt : float
        Mean TTNE / RRT in seconds over the training set.
    lambda_ttne, lambda_rrt : float
        Loss weights for the two time heads.
    """
    model.train()
    total_loss = 0.0
    mae_loss   = nn.L1Loss()

    for data in loader:
        data = data.to(device)
        y    = data.y
        bos  = torch.zeros(y.size(0), 1, dtype=torch.long, device=device)
        y_in = torch.cat([bos, y[:, :-1]], dim=1)

        ttne_norm = data.ttne.squeeze(-1) / mean_ttne   # normalised targets
        rrt_norm  = data.rrt.squeeze(-1)  / mean_rrt

        optimizer.zero_grad()
        logits, ttne_pred, rrt_pred = model(data.x, data.edge_index, data.batch, y_in)

        loss = (criterion_ce(logits.view(-1, logits.size(-1)), y.view(-1))
                + lambda_ttne * mae_loss(ttne_pred, ttne_norm)
                + lambda_rrt  * mae_loss(rrt_pred,  rrt_norm))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, criterion_ce, device,
             end_token_idx: int, max_len: int,
             mean_ttne: float, mean_rrt: float):
    """
    Returns
    -------
    mean_loss : float   (CE loss only)
    metrics   : dict
        activity_accuracy, mean_dls, mae_ttne_minutes, mae_rrt_minutes
    """
    model.eval()
    total_loss     = 0.0
    all_dls        = []
    correct        = 0
    total_pos      = 0
    ttne_err_s_sum = 0.0
    rrt_err_s_sum  = 0.0
    n_instances    = 0

    for data in loader:
        data  = data.to(device)
        y     = data.y
        bos   = torch.zeros(y.size(0), 1, dtype=torch.long, device=device)
        y_in  = torch.cat([bos, y[:, :-1]], dim=1)

        ttne_true = data.ttne.squeeze(-1)   # seconds
        rrt_true  = data.rrt.squeeze(-1)    # seconds

        logits, ttne_pred, rrt_pred = model(data.x, data.edge_index, data.batch, y_in)
        total_loss += criterion_ce(logits.view(-1, logits.size(-1)), y.view(-1)).item()

        # Denormalise predictions → seconds, accumulate absolute error
        ttne_err_s_sum += ((ttne_pred * mean_ttne - ttne_true).abs()).sum().item()
        rrt_err_s_sum  += ((rrt_pred  * mean_rrt  - rrt_true).abs()).sum().item()
        n_instances    += int(ttne_true.size(0))

        # Activity sequence metrics (greedy decode)
        preds   = model.greedy_decode(data.x, data.edge_index, data.batch,
                                      end_token_idx, max_len)
        y_np    = y.cpu().numpy()
        pred_np = preds.cpu().numpy()

        for i in range(y_np.shape[0]):
            true_seq = _strip_seq(y_np[i],    end_token_idx)
            pred_seq = _strip_seq(pred_np[i], end_token_idx)
            all_dls.append(_dls(pred_seq, true_seq))

            mask = (y_np[i] != 0) & (y_np[i] != end_token_idx)
            if mask.any():
                correct   += int((pred_np[i][mask] == y_np[i][mask]).sum())
                total_pos += int(mask.sum())

    act_acc  = correct / total_pos if total_pos > 0 else 0.0
    mean_dls = float(np.mean(all_dls)) if all_dls else 0.0
    # Convert accumulated absolute errors from seconds to minutes
    mae_ttne = (ttne_err_s_sum / n_instances) / 60.0 if n_instances else 0.0
    mae_rrt  = (rrt_err_s_sum  / n_instances) / 60.0 if n_instances else 0.0

    return total_loss / len(loader), {
        'activity_accuracy': act_acc,
        'mean_dls':          mean_dls,
        'mae_ttne_minutes':  mae_ttne,
        'mae_rrt_minutes':   mae_rrt,
    }


# ─────────────────────────────────────────────
# Metric helpers  (identical to model_suffix.py)
# ─────────────────────────────────────────────

def _strip_seq(seq: np.ndarray, end_token_idx: int) -> list:
    result = []
    for tok in seq:
        if tok == 0 or tok == end_token_idx:
            break
        result.append(int(tok))
    return result


def _dl_distance(s1: list, s2: list) -> int:
    n, m = len(s1), len(s2)
    if n == 0:
        return m
    if m == 0:
        return n
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost    = 0 if s1[i - 1] == s2[j - 1] else 1
            d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + cost)
            if i > 1 and j > 1 and s1[i-1] == s2[j-2] and s1[i-2] == s2[j-1]:
                d[i][j] = min(d[i][j], d[i-2][j-2] + cost)
    return d[n][m]


def _dls(pred: list, true: list) -> float:
    denom = max(len(pred), len(true), 1)
    return 1.0 - _dl_distance(pred, true) / denom
