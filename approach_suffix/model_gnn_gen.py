"""
Simple graph generation model: GNN encoder + LSTM decoder.

Predicts the suffix as a sequence of (activity, Δt) pairs.
These pairs define the suffix graph: nodes are predicted events,
edges carry the time deltas following the partial-order DAG rule:
  Δt_seconds >= threshold → new layer  (connect from full frontier)
  Δt_seconds <  threshold → concurrent (connect from prev_frontier)

The edge connecting the last prefix node to the first suffix node
carries Δt[0], which equals TTNE.
RRT = sum of all predicted Δt values.

Δt values predicted by the model are normalized (divided by mean_ttne
at training time). Multiply by mean_ttne to recover seconds.
"""

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, global_mean_pool


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────

class GraphGenModel(nn.Module):
    """
    Parameters
    ----------
    in_channels     : one-hot activity feature size (= num_activities)
    hidden_channels : GNN hidden dimension
    num_activities  : number of distinct activities
    emb_dim         : activity embedding dimension for LSTM input
    lstm_hidden     : LSTM hidden size
    dropout         : dropout probability
    """

    def __init__(self, in_channels, hidden_channels, num_activities,
                 emb_dim=64, lstm_hidden=128, dropout=0.3):
        super().__init__()
        self.num_activities = num_activities
        vocab = num_activities + 2  # 0=PAD/BOS, 1..N=acts, N+1=END

        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.drop  = nn.Dropout(dropout)

        self.h_init  = nn.Linear(hidden_channels, lstm_hidden)
        self.act_emb = nn.Embedding(vocab, emb_dim, padding_idx=0)
        self.lstm    = nn.LSTM(emb_dim, lstm_hidden, batch_first=True)

        self.act_head  = nn.Linear(lstm_hidden, vocab)
        # Softplus ensures Δt > 0; the model learns near-zero output for
        # concurrent events (ground-truth Δt = 0) and larger values otherwise.
        self.time_head = nn.Sequential(nn.Linear(lstm_hidden, 1), nn.Softplus())

    def _encode(self, data):
        h = self.conv1(data.x, data.edge_index).relu()
        h = self.drop(h)
        h = self.conv2(h, data.edge_index).relu()
        return global_mean_pool(h, data.batch)  # (B, hidden_channels)

    def forward(self, data, end_token_idx):
        """
        Teacher-forced forward pass.

        Returns
        -------
        act_logits : (B, T, vocab)
        time_preds : (B, T)  — normalized Δt values (divide targets by mean_ttne for loss)
        """
        B      = data.num_graphs
        device = data.x.device
        y      = data.y  # (B, T)  1-based activity indices + END + PAD

        graph_emb = self._encode(data)                               # (B, H)
        h0 = self.h_init(graph_emb).tanh().unsqueeze(0)             # (1, B, lstm_hidden)
        c0 = torch.zeros_like(h0)

        # Shift targets right: step k receives y[k-1] as input (BOS at step 0)
        bos      = torch.zeros(B, 1, dtype=torch.long, device=device)
        inp_acts = torch.cat([bos, y[:, :-1]], dim=1)               # (B, T)
        lstm_in  = self.act_emb(inp_acts)                           # (B, T, emb_dim)

        out, _     = self.lstm(lstm_in, (h0, c0))                   # (B, T, lstm_hidden)
        act_logits = self.act_head(out)                             # (B, T, vocab)
        time_preds = self.time_head(out).squeeze(-1)                # (B, T)

        return act_logits, time_preds

    @torch.no_grad()
    def greedy_decode(self, data, end_token_idx, max_len):
        """
        Autoregressive greedy decoding.

        Returns
        -------
        list of (act_tokens, time_deltas_normalized) per sample.
        Multiply time_deltas by mean_ttne to get seconds.
        """
        B      = data.num_graphs
        device = data.x.device

        graph_emb = self._encode(data)
        h = self.h_init(graph_emb).tanh().unsqueeze(0)
        c = torch.zeros_like(h)

        act_in  = torch.zeros(B, dtype=torch.long, device=device)  # BOS
        results = [{'acts': [], 'dts': []} for _ in range(B)]
        done    = [False] * B

        for _ in range(max_len):
            emb         = self.act_emb(act_in).unsqueeze(1)           # (B, 1, emb_dim)
            out, (h, c) = self.lstm(emb, (h, c))
            out         = out.squeeze(1)                               # (B, lstm_hidden)

            act_pred  = self.act_head(out).argmax(dim=-1)             # (B,)
            time_pred = self.time_head(out).squeeze(-1)               # (B,)

            for b in range(B):
                if done[b]:
                    continue
                a = act_pred[b].item()
                if a == 0 or a == end_token_idx:
                    done[b] = True
                    continue
                results[b]['acts'].append(a)
                results[b]['dts'].append(time_pred[b].item())

            if all(done):
                break
            act_in = act_pred

        return [(r['acts'], r['dts']) for r in results]


# ─────────────────────────────────────────────
# Graph construction utility
# ─────────────────────────────────────────────

def build_output_graph(x_prefix, ei_prefix, acts, dts_normalized,
                        num_activities, mean_ttne, threshold_sec=1.0):
    """
    Extend a prefix graph with the predicted suffix nodes and edges.

    Δt_seconds = dt_normalized * mean_ttne.
    If Δt_seconds < threshold_sec the new node is concurrent with the
    current frontier (same timestamp layer); otherwise it starts a new layer.

    Parameters
    ----------
    x_prefix        : (N, num_activities) node features of the prefix
    ei_prefix       : (2, E) edge index of the prefix
    acts            : list of predicted activity indices (1-based)
    dts_normalized  : list of normalized Δt predictions (same length as acts)
    num_activities  : vocabulary size for one-hot features
    mean_ttne       : training mean TTNE in seconds (used to de-normalize Δt)
    threshold_sec   : Δt_seconds below this value is treated as concurrent (default 1 s)

    Returns
    -------
    x          : (N + len(acts), num_activities)
    edge_index : (2, E') — full trace graph edges
    edge_attr  : (E', 1) — time delta in seconds for each edge
    """
    device = x_prefix.device
    x      = x_prefix.clone()
    ei     = ei_prefix.clone()

    n = x.shape[0]
    if ei.shape[1] > 0:
        is_leaf = torch.ones(n, dtype=torch.bool, device=device)
        is_leaf[ei[0]] = False
        frontier = is_leaf.nonzero(as_tuple=True)[0].tolist()
    else:
        frontier = list(range(n))

    prev_frontier = []
    new_edges     = []
    new_attrs     = []

    for act, dt_norm in zip(acts, dts_normalized):
        dt_sec = dt_norm * mean_ttne

        feat = torch.zeros(num_activities, device=device)
        if 0 < act <= num_activities:
            feat[act - 1] = 1.0
        x = torch.cat([x, feat.unsqueeze(0)], dim=0)
        new_idx = x.shape[0] - 1

        if dt_sec >= threshold_sec:
            connect_from  = frontier
            prev_frontier = frontier
            frontier      = [new_idx]
        else:
            connect_from  = prev_frontier if prev_frontier else frontier
            frontier      = frontier + [new_idx]

        for src in connect_from:
            new_edges.append([src, new_idx])
            new_attrs.append(dt_sec)

    if new_edges:
        new_ei   = torch.tensor(new_edges, dtype=torch.long, device=device).t().contiguous()
        new_attr = torch.tensor(new_attrs, dtype=torch.float, device=device).unsqueeze(1)
        ei       = torch.cat([ei, new_ei], dim=1)
        if ei_prefix.shape[1] > 0:
            old_attr = torch.zeros(ei_prefix.shape[1], 1, device=device)
        else:
            old_attr = torch.empty(0, 1, device=device)
        edge_attr = torch.cat([old_attr, new_attr], dim=0)
    else:
        edge_attr = torch.zeros(ei.shape[1], 1, device=device)

    return x, ei, edge_attr


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion_ce, device,
                end_token_idx, mean_ttne, lambda_time=1.0):
    model.train()
    l1         = nn.L1Loss()
    total_loss = 0.0
    n_batches  = len(loader)

    for batch_idx, data in enumerate(loader):
        data = data.to(device)
        y    = data.y        # (B, T)
        yt   = data.y_times  # (B, T) raw seconds

        optimizer.zero_grad()
        act_logits, time_preds = model(data, end_token_idx)

        loss_ce = criterion_ce(act_logits.view(-1, act_logits.size(-1)), y.view(-1))

        # Time loss only on real activity positions (exclude PAD and END)
        mask   = (y != 0) & (y != end_token_idx)
        loss_t = (l1(time_preds[mask], yt[mask] / mean_ttne)
                  if mask.any() else loss_ce.new_zeros(1).squeeze())

        loss = loss_ce + lambda_time * loss_t
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / n_batches


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, criterion_ce, device,
             end_token_idx, max_len, mean_ttne):
    model.eval()
    total_loss = 0.0
    all_dls    = []
    correct    = 0
    total_pos  = 0
    ttne_err_s = 0.0
    rrt_err_s  = 0.0
    n_inst     = 0

    for data in loader:
        data  = data.to(device)
        y     = data.y
        y_np  = y.cpu().numpy()
        T     = y_np.shape[1]

        act_logits, _ = model(data, end_token_idx)
        total_loss   += criterion_ce(
            act_logits.view(-1, act_logits.size(-1)), y.view(-1)).item()

        results = model.greedy_decode(data, end_token_idx, max_len)

        for b, (acts, dts) in enumerate(results):
            pred_arr = np.zeros(T, dtype=np.int64)
            for j, a in enumerate(acts):
                if j >= T:
                    break
                pred_arr[j] = a

            true_seq = _strip_seq(y_np[b], end_token_idx)
            pred_seq = _strip_seq(pred_arr, end_token_idx)
            all_dls.append(_dls(pred_seq, true_seq))

            mask = (y_np[b] != 0) & (y_np[b] != end_token_idx)
            if mask.any():
                correct   += int((pred_arr[mask] == y_np[b][mask]).sum())
                total_pos += int(mask.sum())

            # TTNE = first predicted Δt; RRT = sum of all Δt (both × mean_ttne)
            ttne_pred_s = dts[0] * mean_ttne if dts else 0.0
            rrt_pred_s  = sum(dts) * mean_ttne
            ttne_err_s += abs(ttne_pred_s - data.ttne[b].item())
            rrt_err_s  += abs(rrt_pred_s  - data.rrt[b].item())
            n_inst     += 1

    act_acc  = correct / total_pos if total_pos > 0 else 0.0
    mean_dls = float(np.mean(all_dls)) if all_dls else 0.0
    mae_ttne = (ttne_err_s / n_inst) / 60.0 if n_inst else 0.0
    mae_rrt  = (rrt_err_s  / n_inst) / 60.0 if n_inst else 0.0

    return total_loss / len(loader), {
        'activity_accuracy': act_acc,
        'mean_dls':          mean_dls,
        'mae_ttne_minutes':  mae_ttne,
        'mae_rrt_minutes':   mae_rrt,
    }


# ─────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────

def _strip_seq(seq, end_token_idx):
    result = []
    for tok in seq:
        if tok == 0 or tok == end_token_idx:
            break
        result.append(int(tok))
    return result


def _dl_distance(s1, s2):
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


def _dls(pred, true):
    denom = max(len(pred), len(true), 1)
    return 1.0 - _dl_distance(pred, true) / denom
