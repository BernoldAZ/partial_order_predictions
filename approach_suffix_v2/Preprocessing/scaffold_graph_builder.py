"""Scaffold graph builder utilities shared by create_scaffold_data.py
and run_suffix_scaffold_v1.py.

_init_block_state and _update_graph are identical to the versions in
run_suffix_nap_v1.py.  build_scaffold_dataset is new: it pre-computes
the W growing scaffold states for every sample in a graph dataset.
"""
import torch
from torch_geometric.data import Data


def _init_block_state(graph):
    """Extract last-block and second-to-last-block node indices from a prefix graph.

    Returns (curr_block, prev_block, prev_to_curr_attr, curr_block_tss).
    """
    n       = graph.num_nodes
    last_ts = graph.x[-1, 0].item()

    curr_block = [i for i in range(n) if graph.x[i, 0].item() == last_ts]
    remaining  = [i for i in range(n) if i not in set(curr_block)]

    if remaining:
        prev_ts    = max(graph.x[i, 0].item() for i in remaining)
        prev_block = [i for i in remaining if graph.x[i, 0].item() == prev_ts]
        prev_set   = set(prev_block)
        curr_set   = set(curr_block)
        p2c_attr   = None
        for e in range(graph.edge_index.shape[1]):
            if (graph.edge_index[0, e].item() in prev_set and
                    graph.edge_index[1, e].item() in curr_set):
                p2c_attr = graph.edge_attr[e, 0].item()
                break
        if p2c_attr is None:
            p2c_attr = 0.0
    else:
        prev_block = []
        p2c_attr   = None

    return curr_block, prev_block, p2c_attr, last_ts


def _update_graph(graph, act_pred, new_block, ttne_raw,
                  curr_block, prev_block, p2c_attr, curr_tss,
                  tss_mean, tss_std, tsp_mean, tsp_std):
    """Append a predicted event node to the graph and wire its edges.

    new_block=True  : starts a new block (different timestamp).
                      Adds directed edges from curr_block -> new node.
    new_block=False : concurrent with previous event (same block).
                      Adds bidirectional edges to all curr_block nodes
                      and directed edges from prev_block -> new node.

    Returns updated (graph, curr_block, prev_block, p2c_attr, curr_tss).
    """
    n         = graph.num_nodes
    num_cat   = graph.cat_x.shape[1]
    num_num   = graph.x.shape[1]
    intra_val = -tsp_mean / tsp_std

    cat_new        = torch.zeros(1, num_cat, dtype=torch.long)
    cat_new[0, -1] = int(act_pred) + 1
    x_new          = torch.zeros(1, num_num, dtype=torch.float32)

    new_src, new_dst, new_attr = [], [], []

    if new_block:
        tsp_new      = (ttne_raw - tsp_mean) / tsp_std
        curr_raw     = curr_tss * tss_std + tss_mean
        new_tss      = (curr_raw + ttne_raw - tss_mean) / tss_std
        x_new[0, 0]  = new_tss
        for u in curr_block:
            new_src.append(u); new_dst.append(n); new_attr.append(tsp_new)
        out_curr = [n]
        out_prev = list(curr_block)
        out_p2c  = tsp_new
        out_tss  = new_tss
    else:
        x_new[0, 0] = curr_tss
        for u in curr_block:
            new_src += [u, n]; new_dst += [n, u]; new_attr += [intra_val, intra_val]
        if prev_block and p2c_attr is not None:
            for u in prev_block:
                new_src.append(u); new_dst.append(n); new_attr.append(p2c_attr)
        out_curr = curr_block + [n]
        out_prev = prev_block
        out_p2c  = p2c_attr
        out_tss  = curr_tss

    new_cat_x = torch.cat([graph.cat_x, cat_new], dim=0)
    new_x     = torch.cat([graph.x,     x_new],   dim=0)

    if new_src:
        add_ei = torch.tensor([new_src, new_dst], dtype=torch.long)
        add_ea = torch.tensor(new_attr, dtype=torch.float32).unsqueeze(1)
        new_ei = torch.cat([graph.edge_index, add_ei], dim=1)
        new_ea = torch.cat([graph.edge_attr,  add_ea], dim=0)
    else:
        new_ei = graph.edge_index
        new_ea = graph.edge_attr

    new_graph = Data(x=new_x, cat_x=new_cat_x, edge_index=new_ei, edge_attr=new_ea)
    return new_graph, out_curr, out_prev, out_p2c, out_tss


def build_scaffold_dataset(graph_dataset, end_tok, tss_mean, tss_std, tsp_mean, tsp_std):
    """Pre-compute W scaffold states for every sample in graph_dataset.

    Returns a list of N items.  Each item is a list of W Data objects:
      item[t] = prefix graph + first t ground-truth suffix events
                (graph-only fields: x, cat_x, edge_index, edge_attr).

    scaffold[t] is used as encoder input when predicting suffix event t,
    so scaffold[0] = prefix only, scaffold[1] = prefix + GT event 0, etc.

    Parameters
    ----------
    graph_dataset : list of torch_geometric.data.Data
    end_tok : int
        Activity index for the END token (= num_activities - 1).
    tss_mean, tss_std, tsp_mean, tsp_std : float
        Normalization statistics for ts_start and ts_prev (from suffix_df).
    """
    result = []
    for sample in graph_dataset:
        result.append(_build_sample_scaffolds(
            sample, end_tok, tss_mean, tss_std, tsp_mean, tsp_std))
    return result


def _build_sample_scaffolds(sample, end_tok, tss_mean, tss_std, tsp_mean, tsp_std):
    W = sample.suffix_act.shape[0]

    graph = Data(
        x=sample.x,
        cat_x=sample.cat_x,
        edge_index=sample.edge_index,
        edge_attr=sample.edge_attr,
    )
    cb, pb, p2c, ctss = _init_block_state(graph)

    steps   = []
    stopped = False

    for t in range(W):
        # Snapshot before adding event t: used to predict event t.
        # _update_graph always creates new tensors via torch.cat, so
        # the tensors referenced here are never modified in-place.
        steps.append(Data(
            x=graph.x,
            cat_x=graph.cat_x,
            edge_index=graph.edge_index,
            edge_attr=graph.edge_attr,
        ))

        if stopped:
            continue

        act_t = sample.suffix_act[t].item()      # 0 = padding
        lbl_t = sample.act_label_seq[t].item()   # end_tok = END token

        if act_t == 0 or lbl_t == end_tok:
            stopped = True
            continue

        nb       = sample.new_block_label[t].item() > 0.5
        # suffix_num[:, 1] = ts_prev (normalized); denormalize to get seconds
        ttne_raw = max(0.0, sample.suffix_num[t, 1].item() * tsp_std + tsp_mean)

        graph, cb, pb, p2c, ctss = _update_graph(
            graph, act_t - 1, nb, ttne_raw,
            cb, pb, p2c, ctss,
            tss_mean, tss_std, tsp_mean, tsp_std)

    return steps
