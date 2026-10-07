"""Partial-order suffix metrics shared by the run_suffix_time_v* scripts.

  - GES: graph edit similarity between block graphs built from (acts, nb bits).
  - Bucket-order similarity: Kendall distance with penalty p = 1/2 (K_prof)
    from Fagin et al., "Comparing Partial Rankings" (SIAM J. Discrete Math,
    2006), computed on the activity occurrences shared by prediction and
    ground truth, plus a content F1 over the occurrence multisets.
"""
import networkx as nx


# ─── GES helpers ──────────────────────────────────────────────────────────────

def _build_suffix_graph(acts, nb_bits):
    n = len(acts)
    G = nx.DiGraph()
    if n == 0:
        return G
    for i in range(n):
        G.add_node(i, act=int(acts[i]))
    blocks = [[0]]
    for i in range(1, n):
        if nb_bits[i]:
            blocks.append([i])
        else:
            blocks[-1].append(i)
    for bi in range(len(blocks) - 1):
        for u in blocks[bi]:
            for v in blocks[bi + 1]:
                G.add_edge(u, v)
    return G


def _node_match(n1, n2):
    return n1['act'] == n2['act']


def _graph_edit_similarity(G_pred, G_true):
    np_n, np_e = G_pred.number_of_nodes(), G_pred.number_of_edges()
    nt_n, nt_e = G_true.number_of_nodes(), G_true.number_of_edges()
    if np_n == 0 and nt_n == 0:
        return 1.0
    if np_n == 0 or nt_n == 0:
        return 0.0
    denom = (np_n + np_e) + (nt_n + nt_e)
    ged   = next(nx.optimize_graph_edit_distance(G_pred, G_true, node_match=_node_match))
    return 1.0 - ged / denom


def ges_compute_sample(args):
    sa_list, nbp_list, la_list, nbl_list = args
    return _graph_edit_similarity(
        _build_suffix_graph(sa_list, nbp_list),
        _build_suffix_graph(la_list, nbl_list),
    )


# ─── Bucket-order helpers ─────────────────────────────────────────────────────

def _bucket_of_occurrence(acts, nb_bits):
    """Map (activity, k-th occurrence) → bucket index. Position 0 opens bucket 0;
    nb_bits[i] (i >= 1) opens a new bucket, as in _build_suffix_graph."""
    buckets, counts = {}, {}
    b = 0
    for i, a in enumerate(acts):
        if i > 0 and nb_bits[i]:
            b += 1
        a = int(a)
        counts[a] = counts.get(a, 0) + 1
        buckets[(a, counts[a])] = b
    return buckets


def bucket_order_similarity(acts_pred, nb_pred, acts_true, nb_true):
    """Returns (order_sim, content_f1, combined).

    order_sim  : 1 - K^(1/2) / C(m, 2) over the m occurrences present on both
                 sides (1.0 when m < 2).
    content_f1 : 2 * m / (len_pred + len_true) (1.0 when both are empty).
    combined   : order_sim * content_f1.
    """
    bp = _bucket_of_occurrence(acts_pred, nb_pred)
    bt = _bucket_of_occurrence(acts_true, nb_true)
    common = [k for k in bp if k in bt]
    m = len(common)

    n_total = len(acts_pred) + len(acts_true)
    content_f1 = 1.0 if n_total == 0 else 2 * m / n_total

    if m < 2:
        order_sim = 1.0
    else:
        k_dist = 0.0
        for i in range(m):
            for j in range(i + 1, m):
                x, y = common[i], common[j]
                dp = (bp[x] > bp[y]) - (bp[x] < bp[y])
                dt = (bt[x] > bt[y]) - (bt[x] < bt[y])
                if dp == dt:
                    continue
                k_dist += 0.5 if (dp == 0 or dt == 0) else 1.0
        order_sim = 1.0 - k_dist / (m * (m - 1) / 2)

    return order_sim, content_f1, order_sim * content_f1
