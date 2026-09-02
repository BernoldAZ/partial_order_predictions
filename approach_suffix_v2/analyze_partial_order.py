"""Partial-order (same-timestamp) statistics for a test set.

"Partial order" = two or more events sharing an identical timestamp (the repo
also calls these concurrent events / parallel blocks).

Consumes a pre-built graph dataset (one PyG Data per prefix-suffix pair):
    results_per_log/<log_name>/test_graphdataset.pt

For prefixes and (separately) for suffixes it reports:
  - how many have a partial order (>= 2 events share a timestamp)
  - how many events share a timestamp with >= 1 other event
        * summed over every prefix / suffix window
        * de-duplicated once per original case (case-level block)
  - how many have no partial order, split into:
        * length < 2  (trivially impossible)
        * length >= 2 but all timestamps distinct

Usage
-----
    python analyze_partial_order.py <path-to-test_graphdataset.pt> [--tss-index N] [--csv OUT.csv]

    python analyze_partial_order.py results_per_log/Sepsis/test_graphdataset.pt --csv results_per_log/Sepsis/sepsis_po.csv
    python analyze_partial_order.py results_per_log/BPI_Challenge_2012_O/test_graphdataset.pt --csv results_per_log/BPI_Challenge_2012_O/BPI_Challenge_2012_O_po.csv
    python analyze_partial_order.py results_per_log/BPIC15_4/test_graphdataset.pt --csv results_per_log/BPIC15_4/BPIC15_4_po.csv
"""

import argparse
import csv
import os

import torch

try:  # needed so torch.load can resolve the pickled PyG classes
    import torch_geometric  # noqa: F401
except ImportError:
    pass


def block_sizes(equal_to_prev):
    """Given a list of bools (element i shares its key with element i-1),
    return the list of block sizes (maximal runs of consecutive shared keys)."""
    sizes, cur = [], 1
    for shared in equal_to_prev:
        if shared:
            cur += 1
        else:
            sizes.append(cur)
            cur = 1
    sizes.append(cur)
    return sizes


def shared_count(sizes):
    """Number of elements that live in a block of size >= 2."""
    return sum(s for s in sizes if s >= 2)


def prefix_stats(data, tss_index):
    """(length k, shared_events, has_po) for the prefix of one sample."""
    ts = data.x[:, tss_index]
    k = ts.size(0)
    if k < 2:
        return k, 0, False
    equal_to_prev = [bool(ts[i].item() == ts[i - 1].item()) for i in range(1, k)]
    sizes = block_sizes(equal_to_prev)
    sc = shared_count(sizes)
    return k, sc, sc > 0


def suffix_stats(data):
    """(length m, shared_events, has_po) for the predicted suffix (events p+1..L).

    new_block_label is aligned to the decoder-input suffix (events p..L):
      nb[0] <-> event p (always 0), nb[j] <-> event p+j for j = 1..m,
      1.0 = new timestamp block vs previous event, 0.0 = same timestamp.
    m (number of predicted suffix events) = (# non-zero act_label_seq) - 1,
    because act_label_seq holds events p+1..L plus the END token.
    """
    m = int((data.act_label_seq != 0).sum().item()) - 1
    if m < 2:
        return m, 0, False
    nb = data.new_block_label[:m + 1]
    # blocks over events p+1..L (nb indices 1..m); the p / p+1 boundary is ignored
    equal_to_prev = [nb[j].item() == 0.0 for j in range(2, m + 1)]
    sizes = block_sizes(equal_to_prev)
    sc = shared_count(sizes)
    return m, sc, sc > 0


def pct(n, total):
    return f"{100.0 * n / total:.1f}%" if total else "n/a"


def analyze(dataset, tss_index):
    n = len(dataset)
    r = dict(
        n_pairs=n,
        pref_po=0, pref_nopo_len_lt2=0, pref_nopo_distinct=0, pref_shared_events=0,
        pref_total_events=0,
        suf_po=0, suf_nopo_len_lt2=0, suf_nopo_distinct=0, suf_shared_events=0,
        suf_total_events=0,
        n_cases=0, case_po=0, case_nopo_len_lt2=0, case_nopo_distinct=0,
        case_shared_events=0, case_total_events=0,
    )

    for data in dataset:
        k, p_sc, p_po = prefix_stats(data, tss_index)
        r["pref_shared_events"] += p_sc
        r["pref_total_events"] += k
        if p_po:
            r["pref_po"] += 1
        elif k < 2:
            r["pref_nopo_len_lt2"] += 1
        else:
            r["pref_nopo_distinct"] += 1

        m, s_sc, s_po = suffix_stats(data)
        r["suf_shared_events"] += s_sc
        r["suf_total_events"] += m
        if s_po:
            r["suf_po"] += 1
        elif m < 2:
            r["suf_nopo_len_lt2"] += 1
        else:
            r["suf_nopo_distinct"] += 1

        # full-case sample: suffix is only the END token -> prefix spans the whole case
        if m == 0:
            r["n_cases"] += 1
            r["case_shared_events"] += p_sc
            r["case_total_events"] += k
            if p_po:
                r["case_po"] += 1
            elif k < 2:
                r["case_nopo_len_lt2"] += 1
            else:
                r["case_nopo_distinct"] += 1

    assert r["pref_po"] + r["pref_nopo_len_lt2"] + r["pref_nopo_distinct"] == n
    assert r["suf_po"] + r["suf_nopo_len_lt2"] + r["suf_nopo_distinct"] == n
    assert r["n_cases"] >= 1
    return r


def print_report(log_name, r):
    n, nc = r["n_pairs"], r["n_cases"]
    print(f"\nLog: {log_name}")
    print(f"Prefix-suffix pairs: {n}    Test cases: {nc}\n")

    print("PREFIXES (one per pair)")
    print(f"  with partial order          : {r['pref_po']:>8}  ({pct(r['pref_po'], n)})")
    print(f"  no partial order            : {r['pref_nopo_len_lt2'] + r['pref_nopo_distinct']:>8}")
    print(f"    - length < 2              : {r['pref_nopo_len_lt2']:>8}  ({pct(r['pref_nopo_len_lt2'], n)})")
    print(f"    - length >= 2, distinct ts: {r['pref_nopo_distinct']:>8}  ({pct(r['pref_nopo_distinct'], n)})")
    print(f"  events sharing a timestamp  : {r['pref_shared_events']:>8}  ({pct(r['pref_shared_events'], r['pref_total_events'])} of all prefix events)\n")

    print("SUFFIXES (one per pair, events after the prefix)")
    print(f"  with partial order          : {r['suf_po']:>8}  ({pct(r['suf_po'], n)})")
    print(f"  no partial order            : {r['suf_nopo_len_lt2'] + r['suf_nopo_distinct']:>8}")
    print(f"    - length < 2              : {r['suf_nopo_len_lt2']:>8}  ({pct(r['suf_nopo_len_lt2'], n)})")
    print(f"    - length >= 2, distinct ts: {r['suf_nopo_distinct']:>8}  ({pct(r['suf_nopo_distinct'], n)})")
    print(f"  events sharing a timestamp  : {r['suf_shared_events']:>8}  ({pct(r['suf_shared_events'], r['suf_total_events'])} of all suffix events)\n")

    print("CASE-LEVEL (de-duplicated, full trace counted once)")
    print(f"  cases with >= 1 partial order: {r['case_po']:>8}  ({pct(r['case_po'], nc)})")
    print(f"  cases with no partial order  : {r['case_nopo_len_lt2'] + r['case_nopo_distinct']:>8}")
    print(f"    - length < 2              : {r['case_nopo_len_lt2']:>8}  ({pct(r['case_nopo_len_lt2'], nc)})")
    print(f"    - length >= 2, distinct ts: {r['case_nopo_distinct']:>8}  ({pct(r['case_nopo_distinct'], nc)})")
    print(f"  events sharing a timestamp  : {r['case_shared_events']:>8}  ({pct(r['case_shared_events'], r['case_total_events'])} of all case events)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_path", help="path to test_graphdataset.pt")
    ap.add_argument("--tss-index", type=int, default=None,
                    help="column index of ts_start in data.x "
                         "(default: read tss_index.txt next to the dataset)")
    ap.add_argument("--csv", default=None, help="write a one-row summary CSV here")
    args = ap.parse_args()

    dataset_dir = os.path.dirname(os.path.abspath(args.dataset_path))
    log_name = os.path.basename(dataset_dir)

    tss_index = args.tss_index
    if tss_index is None:
        tss_path = os.path.join(dataset_dir, "tss_index.txt")
        if not os.path.exists(tss_path):
            ap.error(f"tss_index.txt not found in {dataset_dir}; pass --tss-index N")
        with open(tss_path) as f:
            tss_index = int(f.read().strip())

    dataset = torch.load(args.dataset_path, weights_only=False)
    r = analyze(dataset, tss_index)
    print_report(log_name, r)

    if args.csv:
        r_row = {"log_name": log_name, **r}
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(r_row.keys()))
            w.writeheader()
            w.writerow(r_row)
        print(f"\nCSV written to {args.csv}")


if __name__ == "__main__":
    main()
