"""
Scaffold-based suffix graph generation following Lim et al. 2020.

At each decoder step the model:
  1. addNode    — predict next activity (or END)
  2. predictTime — predict time delta to this event (becomes the edge attribute)

The GNN is re-run on the growing transient graph after every node addition.
Connectivity follows the partial-order DAG rule (matching the prefix graph):
  - dt > 0  -> new layer: connect new node from the full current frontier
  - dt == 0 -> concurrent: connect new node from prev_frontier (same layer)

VAE: posterior q(z | full trace), conditional prior p(z | prefix).
TTNE = first predicted time delta * mean_ttne.
RRT  = sum of all predicted time deltas * mean_ttne.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _kl_divergence(mu, logvar, prior_mu, prior_logvar):
    pvar = prior_logvar.exp()
    qvar = logvar.exp()
    return 0.5 * (
        prior_logvar - logvar
        + (qvar + (mu - prior_mu).pow(2)) / pvar
        - 1
    ).sum(-1).mean()


def _extend_with_suffix(x, edge_index, y, y_times, num_activities, end_token_idx):
    """
    Append ground-truth suffix nodes + edges to a prefix graph.
    Used to build the full trace graph for VAE posterior encoding.
    Connectivity rule matches _build_prefix_graph in data_pipeline_suffix.py.
    """
    device = x.device
    n = x.shape[0]

    if edge_index.shape[1] > 0:
        leaf = torch.ones(n, dtype=torch.bool, device=device)
        leaf[edge_index[0]] = False
        frontier = leaf.nonzero(as_tuple=True)[0].tolist()
    else:
        frontier = list(range(n))

    prev_frontier = []

    for k in range(len(y)):
        act_k = y[k].item()
        dt_k  = y_times[k].item()
        if act_k == 0:
            break

        feat = torch.zeros(num_activities, device=device)
        if 0 < act_k <= num_activities:
            feat[act_k - 1] = 1.0
        x = torch.cat([x, feat.unsqueeze(0)], dim=0)
        new_idx = x.shape[0] - 1

        if dt_k > 0:
            connect_from  = frontier
            prev_frontier = frontier
            frontier      = [new_idx]
        else:
            connect_from  = prev_frontier if prev_frontier else frontier
            frontier      = frontier + [new_idx]

        for src in connect_from:
            edge_index = torch.cat(
                [edge_index,
                 torch.tensor([[src], [new_idx]], dtype=torch.long, device=device)],
                dim=1,
            )

        if act_k == end_token_idx:
            break

    return x, edge_index


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────

class GraphScaffoldModel(nn.Module):
    """
    Scaffold-based graph VAE for activity suffix + time-delta prediction.

    Parameters
    ----------
    in_channels     : one-hot activity feature size
    hidden_channels : GNN / MLP hidden dimension
    num_activities  : number of distinct activities
    latent_dim      : VAE latent size
    dropout         : dropout probability
    """

    def __init__(self, in_channels, hidden_channels, num_activities,
                 latent_dim=64, dropout=0.3):
        super().__init__()
        self.in_channels    = in_channels
        self.num_activities = num_activities
        self.latent_dim     = latent_dim
        self.dropout_p      = dropout
        H, L  = hidden_channels, latent_dim
        vocab = num_activities + 2          # 0=PAD  1..N=acts  N+1=END

        # Shared GNN backbone (re-run at every scaffold step)
        self.conv1 = SAGEConv(in_channels, H)
        self.conv2 = SAGEConv(H, H)
        self.drop  = nn.Dropout(dropout)

        # VAE heads
        self.post_mu     = nn.Linear(H, L)
        self.post_logvar = nn.Linear(H, L)
        self.prior_mu_h  = nn.Linear(H, L)
        self.prior_lv_h  = nn.Linear(H, L)

        # Activity embedding used by predictTime
        self.act_emb = nn.Embedding(vocab, H, padding_idx=0)

        # addNode: [graph_emb || z] -> activity logits (full vocab)
        self.add_node = nn.Sequential(
            nn.Linear(H + L, H), nn.ReLU(), nn.Linear(H, vocab))

        # predictTime: [graph_emb || act_emb || z] -> softplus scalar
        self.time_mlp = nn.Sequential(
            nn.Linear(H + H + L, H), nn.ReLU(), nn.Linear(H, 1))

    # ── private helpers ─────────────────────────────────────────────────

    def _encode(self, x, ei):
        h = self.conv1(x, ei).relu()
        h = self.drop(h)
        h = self.conv2(h, ei).relu()
        return h

    def _gemb(self, x, ei):
        return self._encode(x, ei).mean(0)

    def _reparam(self, mu, logvar):
        if self.training:
            return mu + (0.5 * logvar).exp() * torch.randn_like(mu)
        return mu

    def _extract(self, data, b):
        """Return (x_b, ei_b) for sample b from a batched Data object."""
        mask = data.batch == b
        x_b  = data.x[mask]
        off  = mask.nonzero(as_tuple=True)[0][0].item()
        em   = data.batch[data.edge_index[0]] == b
        ei_b = data.edge_index[:, em] - off
        return x_b, ei_b

    def _frontier(self, x, ei):
        """Indices of last-layer nodes (no outgoing edges)."""
        n = x.shape[0]
        if ei.shape[1] > 0:
            leaf = torch.ones(n, dtype=torch.bool, device=x.device)
            leaf[ei[0]] = False
            return leaf.nonzero(as_tuple=True)[0].tolist()
        return list(range(n))

    def _decode_one(self, x0, ei0, y_b, yt_b, z, end_token_idx, device):
        """Teacher-forced scaffold decode for a single sample."""
        x, ei          = x0.clone(), ei0.clone()
        frontier       = self._frontier(x, ei)
        prev_frontier  = []
        act_logits, time_preds = [], []

        for k in range(len(y_b)):
            act_k = y_b[k].item()
            dt_k  = yt_b[k].item()

            g     = self._gemb(x, ei)
            logit = self.add_node(torch.cat([g, z]))
            act_logits.append(logit)

            if act_k == 0:                              # PAD
                time_preds.append(torch.zeros(1, device=device))
                continue

            ae = self.act_emb(torch.tensor(act_k, device=device))
            dt = F.softplus(self.time_mlp(torch.cat([g, ae, z]))).reshape(1)
            time_preds.append(dt)

            if act_k == end_token_idx:                  # END — no node added
                continue

            # Teacher-force: add ground-truth node to the transient graph
            feat = torch.zeros(self.in_channels, device=device)
            if 0 < act_k <= self.num_activities:
                feat[act_k - 1] = 1.0
            x = torch.cat([x, feat.unsqueeze(0)], dim=0)
            new_idx = x.shape[0] - 1

            if dt_k > 0:
                connect_from  = frontier
                prev_frontier = frontier
                frontier      = [new_idx]
            else:
                connect_from  = prev_frontier if prev_frontier else frontier
                frontier      = frontier + [new_idx]

            for src in connect_from:
                ei = torch.cat(
                    [ei, torch.tensor([[src], [new_idx]], dtype=torch.long, device=device)],
                    dim=1,
                )

        return torch.stack(act_logits), torch.cat(time_preds)

    # ── forward ─────────────────────────────────────────────────────────

    def forward(self, data, end_token_idx):
        device = data.x.device
        B      = data.num_graphs
        y      = data.y                     # (B, T)
        yt     = data.y_times               # (B, T)

        all_logits, all_times          = [], []
        mus, lvars, pmus, plvs         = [], [], [], []

        for b in range(B):
            x_b, ei_b = self._extract(data, b)

            # Conditional prior p(z | prefix)
            g_pre = self._gemb(x_b, ei_b)
            pm    = self.prior_mu_h(g_pre)
            plv   = self.prior_lv_h(g_pre)
            pmus.append(pm)
            plvs.append(plv)

            # Posterior q(z | full trace)
            xf, eif = _extend_with_suffix(
                x_b, ei_b, y[b], yt[b], self.num_activities, end_token_idx)
            g_full = self._gemb(xf, eif)
            qm     = self.post_mu(g_full)
            qlv    = self.post_logvar(g_full)
            mus.append(qm)
            lvars.append(qlv)

            z_b = self._reparam(qm, qlv)

            # Scaffold decode (teacher-forced)
            lg, tp = self._decode_one(x_b, ei_b, y[b], yt[b], z_b, end_token_idx, device)
            all_logits.append(lg)
            all_times.append(tp)

        return (
            torch.stack(all_logits),        # (B, T, vocab)
            torch.stack(all_times),         # (B, T)
            torch.stack(mus),
            torch.stack(lvars),
            torch.stack(pmus),
            torch.stack(plvs),
        )

    # ── greedy decode ────────────────────────────────────────────────────

    @torch.no_grad()
    def greedy_decode(self, data, end_token_idx, max_len):
        """
        Autoregressive inference. Returns list of (act_tokens, time_deltas) per sample.
        time_deltas are in normalised units; multiply by mean_ttne to get seconds.
        """
        device  = data.x.device
        B       = data.num_graphs
        results = []

        for b in range(B):
            x_b, ei_b = self._extract(data, b)
            g_pre     = self._gemb(x_b, ei_b)
            z_b       = self.prior_mu_h(g_pre)      # MAP estimate: z = prior mean

            x, ei    = x_b.clone(), ei_b.clone()
            frontier = self._frontier(x, ei)
            acts, dts = [], []

            for _ in range(max_len):
                g_cur = self._gemb(x, ei)
                raw   = self.add_node(torch.cat([g_cur, z_b])).argmax().item()

                if raw == 0 or raw == end_token_idx:
                    break

                ae  = self.act_emb(torch.tensor(raw, device=device))
                dt  = F.softplus(self.time_mlp(torch.cat([g_cur, ae, z_b]))).item()
                acts.append(raw)
                dts.append(dt)

                feat = torch.zeros(self.in_channels, device=device)
                if 0 < raw <= self.num_activities:
                    feat[raw - 1] = 1.0
                x = torch.cat([x, feat.unsqueeze(0)], dim=0)
                new_idx  = x.shape[0] - 1
                prev_fr  = frontier
                frontier = [new_idx]
                for src in prev_fr:
                    ei = torch.cat(
                        [ei, torch.tensor([[src], [new_idx]], dtype=torch.long, device=device)],
                        dim=1,
                    )

            results.append((acts, dts))

        return results


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion_ce, device,
                end_token_idx, mean_ttne, beta=1.0, lambda_time=1.0,
                print_every=10):
    model.train()
    total_loss = 0.0
    l1 = nn.L1Loss()
    n_batches = len(loader)

    for batch_idx, data in enumerate(loader):
        data = data.to(device)
        y    = data.y           # (B, T)
        yt   = data.y_times     # (B, T)

        optimizer.zero_grad()
        act_logits, time_preds, mu, logvar, p_mu, p_lv = model(data, end_token_idx)

        loss_ce = criterion_ce(act_logits.view(-1, act_logits.size(-1)), y.view(-1))

        # Time L1 only on real-activity positions (not PAD, not END)
        mask   = (y != 0) & (y != end_token_idx)
        loss_t = l1(time_preds[mask], yt[mask] / mean_ttne) if mask.any() else loss_ce.new_zeros(1).squeeze()

        loss_kl = _kl_divergence(mu, logvar, p_mu, p_lv)

        loss = loss_ce + lambda_time * loss_t + beta * loss_kl
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item()

        if (batch_idx + 1) % print_every == 0 or (batch_idx + 1) == n_batches:
            print(f"  batch {batch_idx+1}/{n_batches}  loss={total_loss/(batch_idx+1):.4f}",
                  flush=True)

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
        y     = data.y          # (B, T)
        y_np  = y.cpu().numpy()
        T     = y_np.shape[1]

        act_logits, _, mu, logvar, p_mu, p_lv = model(data, end_token_idx)
        total_loss += criterion_ce(act_logits.view(-1, act_logits.size(-1)), y.view(-1)).item()

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
