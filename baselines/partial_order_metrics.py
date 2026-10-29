"""Partial-order suffix metrics shared by the run_suffix_time_v* scripts.

  - GES: graph edit similarity between block graphs built from (acts, nb bits).
  - Exact GES: same block graphs, exact GED from the F2 binary program of
    Lerouge et al., "New binary linear programming formulation to compute the
    graph edit distance" (Pattern Recognition, 2017), solved with PuLP/CBC,
    normalised by max(n_pred, n_true) + e_pred + e_true.
  - Bucket-order similarity: Kendall distance with penalty p = 1/2 (K_prof)
    from Fagin et al., "Comparing Partial Rankings" (SIAM J. Discrete Math,
    2006), computed on the activity occurrences shared by prediction and
    ground truth, plus a content F1 over the occurrence multisets.
"""
import networkx as nx
import pulp


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


# ─── Exact GES helpers ────────────────────────────────────────────────────────

def _exact_ged(G1, G2):
    """Unit-cost GED (node subst 0/1 by 'act', node/edge del/ins 1, edge subst 0)
    via F2: x maps nodes, y maps edges; the objective is the cost of deleting and
    inserting everything minus the savings of each substitution."""
    prob = pulp.LpProblem("ged", pulp.LpMinimize)
    x = {(i, k): pulp.LpVariable(f"x_{i}_{k}", cat="Binary")
         for i in G1.nodes for k in G2.nodes}
    y = {(e, f): pulp.LpVariable(f"y_{e[0]}_{e[1]}_{f[0]}_{f[1]}", cat="Binary")
         for e in G1.edges for f in G2.edges}

    const = (G1.number_of_nodes() + G2.number_of_nodes()
             + G1.number_of_edges() + G2.number_of_edges())
    node_cost = pulp.lpSum(
        x[i, k] * ((0 if G1.nodes[i]['act'] == G2.nodes[k]['act'] else 1) - 2)
        for (i, k) in x)
    prob += const + node_cost - 2 * pulp.lpSum(y.values())

    for i in G1.nodes:
        prob += pulp.lpSum(x[i, k] for k in G2.nodes) <= 1
    for k in G2.nodes:
        prob += pulp.lpSum(x[i, k] for i in G1.nodes) <= 1
    for (i, j) in G1.edges:
        for k in G2.nodes:   # edge tail i must map to the tail k of the matched edge
            prob += pulp.lpSum(y[(i, j), (k, l)] for l in G2.successors(k)) <= x[i, k]
        for l in G2.nodes:   # edge head j must map to the head l of the matched edge
            prob += pulp.lpSum(y[(i, j), (k, l)] for k in G2.predecessors(l)) <= x[j, l]

    prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=60))  # 1 min cap: best solution found so far
    return round(pulp.value(prob.objective))


def _graph_edit_similarity_exact(G_pred, G_true):
    np_n, np_e = G_pred.number_of_nodes(), G_pred.number_of_edges()
    nt_n, nt_e = G_true.number_of_nodes(), G_true.number_of_edges()
    if np_n == 0 and nt_n == 0:
        return 1.0
    if np_n == 0 or nt_n == 0:
        return 0.0
    # Upper bound on the unit-cost GED: substitute min(n) node pairs, del/ins the
    # rest (<= max(n)), delete and insert every edge (e_pred + e_true).
    denom = max(np_n, nt_n) + np_e + nt_e
    return 1.0 - _exact_ged(G_pred, G_true) / denom


def ges_exact_compute_sample(args):
    sa_list, nbp_list, la_list, nbl_list = args
    return _graph_edit_similarity_exact(
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
