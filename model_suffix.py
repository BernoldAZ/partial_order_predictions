"""
GNN encoder + LSTM decoder model for Activity Suffix Prediction.

Architecture:
  GNN Encoder:
    - 2-layer GraphSAGE over the prefix DAG
    - Global mean pooling → graph embedding
    - Two linear projections to seed the LSTM's (h0, c0)

  LSTM Decoder:
    - Activity embedding layer  (vocab_size = num_activities + 2;
      0 = PAD, 1…N = activities, N+1 = END_TOKEN)
    - Single-layer LSTM
    - Linear projection to vocabulary logits

Training  : teacher forcing with CrossEntropyLoss (ignore_index=0 for PAD)
Inference : greedy autoregressive decoding, stops at END_TOKEN or max_len

Metrics (same convention as SuffixTransformerNetwork baseline):
  - Activity accuracy  (per position, ignoring PAD / END tokens)
  - Damerau-Levenshtein Similarity (DLS) averaged over the test set
"""

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, global_mean_pool


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────

class GNNSuffixModel(nn.Module):
    """
    GNN encoder + LSTM decoder for activity suffix prediction.

    Parameters
    ----------
    in_channels : int
        Node feature dimension (vocabulary size for one-hot encoding).
    hidden_channels : int
        Hidden dimension of the GNN layers.
    num_activities : int
        Number of unique activities (from training vocabulary).
        Decoder vocab_size = num_activities + 2  (0=PAD, 1..N=acts, N+1=END).
    emb_dim : int
        Activity embedding dimension for the LSTM decoder input.
    lstm_hidden : int
        LSTM hidden state size.
    dropout : float
    """

    def __init__(self, in_channels: int, hidden_channels: int,
                 num_activities: int, emb_dim: int = 64,
                 lstm_hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        vocab_size = num_activities + 2          # 0=PAD, 1..N=acts, N+1=END

        # ── GNN encoder ───────────────────────────────────────────────────
        self.conv1   = SAGEConv(in_channels, hidden_channels)
        self.conv2   = SAGEConv(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(dropout)

        # Project graph embedding → LSTM initial state
        self.enc_to_h = nn.Linear(hidden_channels, lstm_hidden)
        self.enc_to_c = nn.Linear(hidden_channels, lstm_hidden)

        # ── LSTM decoder ──────────────────────────────────────────────────
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm      = nn.LSTM(emb_dim, lstm_hidden, batch_first=True)
        self.out_proj  = nn.Linear(lstm_hidden, vocab_size)

    # ── Shared encoder ────────────────────────────────────────────────────

    def encode(self, x, edge_index, batch):
        """Return graph-level embedding: (batch_size, hidden_channels)."""
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index).relu()
        x = self.dropout(x)
        return global_mean_pool(x, batch)

    # ── Training forward (teacher forcing) ────────────────────────────────

    def forward(self, x, edge_index, batch, y_in):
        """
        Teacher-forcing forward pass.

        Parameters
        ----------
        x, edge_index, batch : standard PyG inputs
        y_in : (B, max_suffix_len)  — decoder input (ground-truth shifted right;
               column 0 is the BOS/zero token, last column dropped vs target)

        Returns
        -------
        logits : (B, max_suffix_len, vocab_size)
        """
        graph_emb = self.encode(x, edge_index, batch)
        h0 = self.enc_to_h(graph_emb).unsqueeze(0)   # (1, B, lstm_hidden)
        c0 = self.enc_to_c(graph_emb).unsqueeze(0)

        emb      = self.embedding(y_in)               # (B, L, emb_dim)
        out, _   = self.lstm(emb, (h0, c0))           # (B, L, lstm_hidden)
        return self.out_proj(out)                      # (B, L, vocab_size)

    # ── Greedy inference ──────────────────────────────────────────────────

    @torch.no_grad()
    def greedy_decode(self, x, edge_index, batch, end_token_idx: int, max_len: int):
        """
        Autoregressive greedy decoding.

        Returns
        -------
        preds : (batch_size, max_len)  — predicted token indices (0-padded)
        """
        self.eval()
        batch_size = int(batch.max().item()) + 1
        device     = x.device

        graph_emb = self.encode(x, edge_index, batch)
        h = self.enc_to_h(graph_emb).unsqueeze(0)
        c = self.enc_to_c(graph_emb).unsqueeze(0)

        # BOS: zero token (PAD index used as start-of-sequence)
        token    = torch.zeros(batch_size, dtype=torch.long, device=device)
        outputs  = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len):
            emb          = self.embedding(token.unsqueeze(1))    # (B, 1, emb_dim)
            lstm_out, (h, c) = self.lstm(emb, (h, c))            # (B, 1, hidden)
            logits       = self.out_proj(lstm_out.squeeze(1))    # (B, vocab)
            pred         = logits.argmax(dim=-1)                  # (B,)
            pred         = pred.masked_fill(finished, 0)          # pad finished
            outputs.append(pred)
            finished     = finished | (pred == end_token_idx)
            if finished.all():
                break
            token = pred

        # Pad any remaining steps to max_len
        while len(outputs) < max_len:
            outputs.append(torch.zeros(batch_size, dtype=torch.long, device=device))

        return torch.stack(outputs, dim=1)  # (B, max_len)


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    """One training epoch with teacher forcing; returns mean loss."""
    model.train()
    total_loss = 0.0

    for data in loader:
        data = data.to(device)
        y    = data.y                              # (B, max_suffix_len) – 1-indexed
        # Decoder input: shift right (prepend zero BOS, drop last token)
        bos  = torch.zeros(y.size(0), 1, dtype=torch.long, device=device)
        y_in = torch.cat([bos, y[:, :-1]], dim=1) # (B, max_suffix_len)

        optimizer.zero_grad()
        logits = model(data.x, data.edge_index, data.batch, y_in)
        # logits: (B, L, vocab) → flatten; y: (B, L) → flatten
        loss   = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, criterion, device, end_token_idx: int, max_len: int):
    """
    Evaluate suffix predictions.

    Returns
    -------
    mean_loss  : float
    metrics    : dict
        activity_accuracy — per-position accuracy ignoring PAD/END tokens
        mean_dls          — mean Damerau-Levenshtein Similarity over all samples
    """
    model.eval()
    total_loss = 0.0
    all_dls    = []
    correct    = 0
    total_pos  = 0

    for data in loader:
        data  = data.to(device)
        y     = data.y                             # (B, max_suffix_len)
        bos   = torch.zeros(y.size(0), 1, dtype=torch.long, device=device)
        y_in  = torch.cat([bos, y[:, :-1]], dim=1)

        logits = model(data.x, data.edge_index, data.batch, y_in)
        loss   = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss.item()

        # Greedy predictions for DLS / accuracy
        preds = model.greedy_decode(
            data.x, data.edge_index, data.batch, end_token_idx, max_len)

        y_np    = y.cpu().numpy()
        pred_np = preds.cpu().numpy()

        for i in range(y_np.shape[0]):
            true_seq = _strip_seq(y_np[i],    end_token_idx)
            pred_seq = _strip_seq(pred_np[i], end_token_idx)
            all_dls.append(_dls(pred_seq, true_seq))

            # Per-position accuracy on the raw non-padded, non-END positions
            mask = (y_np[i] != 0) & (y_np[i] != end_token_idx)
            if mask.any():
                correct   += int((pred_np[i][mask] == y_np[i][mask]).sum())
                total_pos += int(mask.sum())

    act_acc  = correct / total_pos if total_pos > 0 else 0.0
    mean_dls = float(np.mean(all_dls)) if all_dls else 0.0

    return total_loss / len(loader), {
        'activity_accuracy': act_acc,
        'mean_dls':          mean_dls,
    }


# ─────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────

def _strip_seq(seq: np.ndarray, end_token_idx: int) -> list:
    """Remove PAD (0) and END_TOKEN, return the content as a list."""
    result = []
    for tok in seq:
        if tok == 0 or tok == end_token_idx:
            break
        result.append(int(tok))
    return result


def _dl_distance(s1: list, s2: list) -> int:
    """Optimal-string-alignment (restricted) Damerau-Levenshtein distance."""
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
            d[i][j] = min(d[i-1][j] + 1,
                          d[i][j-1] + 1,
                          d[i-1][j-1] + cost)
            if i > 1 and j > 1 and s1[i-1] == s2[j-2] and s1[i-2] == s2[j-1]:
                d[i][j] = min(d[i][j], d[i-2][j-2] + cost)
    return d[n][m]


def _dls(pred: list, true: list) -> float:
    """Damerau-Levenshtein Similarity ∈ [0, 1]."""
    denom = max(len(pred), len(true), 1)
    return 1.0 - _dl_distance(pred, true) / denom
