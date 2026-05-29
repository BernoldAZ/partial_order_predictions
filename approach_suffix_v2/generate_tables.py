import re

# ── parsing ───────────────────────────────────────────────────────────────────

def parse_float(s):
    s = s.strip()
    if s in ('NaN', ''):
        return None
    try:
        return float(s)
    except:
        return None

GNN_MODELS = {'GNN_Suffix', 'GNN_Suffix_Stop', 'GNN_Suffix_time', 'GNN_Suffix_time_Stop'}
GNN_LIST   = ['GNN_Suffix', 'GNN_Suffix_Stop', 'GNN_Suffix_time', 'GNN_Suffix_time_Stop']

def parse_results(filename):
    logs, log_order, current_log = {}, [], None
    with open(filename, 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            m = re.match(r'=== Log: (.+) ===', line.strip())
            if m:
                current_log = m.group(1)
                logs[current_log] = {}
                log_order.append(current_log)
                continue
            if not current_log or not line.strip() or 'DL similarity' in line:
                continue
            parts = re.split(r'\s{2,}', line.strip())
            if len(parts) >= 7:
                logs[current_log][parts[0]] = {
                    'dl':   parse_float(parts[1]),
                    'ttne': parse_float(parts[3]),
                    'rrt':  parse_float(parts[5]),
                }
    return logs, log_order

# ── format helpers ────────────────────────────────────────────────────────────

def non_gnn(log_data):
    return {k: v for k, v in log_data.items() if k not in GNN_MODELS}

def pct(val, ref):
    if val is None or ref is None:
        return None
    return (val - ref) / abs(ref) * 100

def fmt_dl(v):  return f"{v:.4f}" if v is not None else "N/A"
def fmt_int(v): return f"{int(round(v))}" if v is not None else "N/A"
def fmt_pct(v):
    if v is None: return "N/A"
    return f"{'+'if v>0 else ''}{v:.1f}%"
def avg(lst):   return sum(lst)/len(lst) if lst else None

SEP = "─"

# ── auto-sizing table printer ─────────────────────────────────────────────────

def auto_widths(headers, rows, min_w=3):
    """Return column widths = max(header len, max cell len, min_w) per column."""
    n = len(headers)
    widths = [max(min_w, len(str(h))) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < n:
                widths[i] = max(widths[i], len(str(cell)))
    return widths

def fmt_cell(val, w, align):
    s = str(val)
    if align == 'right':  return s.rjust(w)
    if align == 'center': return s.center(w)
    return s.ljust(w)

def print_sub_table(title, headers, aligns, rows):
    widths = auto_widths(headers, rows)
    sep = " │ "
    header_cells = [fmt_cell(h, w, a) for h, w, a in zip(headers, widths, aligns)]
    header_line  = sep.join(header_cells)
    total_w = len(header_line)

    print(f"  {title}")
    print("  " + SEP * total_w)
    print("  " + header_line)
    print("  " + SEP * total_w)
    for row in rows:
        cells = [fmt_cell(row[i] if i < len(row) else "", widths[i], aligns[i])
                 for i in range(len(headers))]
        print("  " + sep.join(cells))
    print("  " + SEP * total_w)
    print()

# ── TABLE 1 per GNN: per-log detail split by metric ──────────────────────────

def make_detail_rows(logs, log_order, gnn_model):
    rows_dl, rows_ttne, rows_rrt = [], [], []

    for log_name in log_order:
        ld = logs[log_name]
        if gnn_model not in ld:
            continue
        gnn = ld[gnn_model]
        ng  = non_gnn(ld)

        # DL ─────────────────────────────────────────────────────────────────
        dl_pairs = [(k, v['dl']) for k, v in ng.items() if v['dl'] is not None]
        dl_only  = [v for _, v in dl_pairs]
        # include GNN itself so it can be the best (highest DL)
        all_dl_pairs = dl_pairs + ([(gnn_model, gnn['dl'])] if gnn['dl'] is not None else [])
        best_dl_model, best_dl = max(all_dl_pairs, key=lambda x: x[1]) if all_dl_pairs else (None, None)
        mean_dl  = avg(dl_only)  # mean stays non-GNN only

        if gnn['dl'] is not None and dl_only:
            all_dl   = dl_only + [gnn['dl']]
            rank     = sorted(all_dl, reverse=True).index(gnn['dl']) + 1
            rank_str = f"{rank}/{len(all_dl)}"
        else:
            rank_str = "N/A"

        rows_dl.append([
            log_name, rank_str,
            fmt_dl(gnn['dl']), fmt_dl(best_dl), best_dl_model or "N/A",
            fmt_pct(pct(gnn['dl'], best_dl)),
            fmt_dl(mean_dl),
            fmt_pct(pct(gnn['dl'], mean_dl)),
        ])

        # TTNE ────────────────────────────────────────────────────────────────
        ttne_pairs = [(k, v['ttne']) for k, v in ng.items() if v['ttne'] is not None]
        ttne_only  = [v for _, v in ttne_pairs]
        # include GNN itself so it can be the best (lowest TTNE)
        all_ttne_pairs = ttne_pairs + ([(gnn_model, gnn['ttne'])] if gnn['ttne'] is not None else [])
        best_ttne_model, best_ttne = min(all_ttne_pairs, key=lambda x: x[1]) if all_ttne_pairs else (None, None)
        mean_ttne = avg(ttne_only)  # mean stays non-GNN only

        if gnn['ttne'] is not None and ttne_only:
            all_ttne   = ttne_only + [gnn['ttne']]
            ttne_rank  = sorted(all_ttne).index(gnn['ttne']) + 1
            ttne_rank_str = f"{ttne_rank}/{len(all_ttne)}"
        else:
            ttne_rank_str = "N/A"

        rows_ttne.append([
            log_name, ttne_rank_str,
            fmt_int(gnn['ttne']), fmt_int(best_ttne), best_ttne_model or "N/A",
            fmt_pct(pct(gnn['ttne'], best_ttne)),
            fmt_int(mean_ttne),
            fmt_pct(pct(gnn['ttne'], mean_ttne)),
        ])

        # RRT ─────────────────────────────────────────────────────────────────
        rrt_pairs = [(k, v['rrt']) for k, v in ng.items() if v['rrt'] is not None]
        rrt_only  = [v for _, v in rrt_pairs]
        # include GNN itself so it can be the best (lowest RRT)
        all_rrt_pairs = rrt_pairs + ([(gnn_model, gnn['rrt'])] if gnn['rrt'] is not None else [])
        best_rrt_model, best_rrt = min(all_rrt_pairs, key=lambda x: x[1]) if all_rrt_pairs else (None, None)
        mean_rrt  = avg(rrt_only)  # mean stays non-GNN only

        if gnn['rrt'] is not None and rrt_only:
            all_rrt   = rrt_only + [gnn['rrt']]
            rrt_rank  = sorted(all_rrt).index(gnn['rrt']) + 1
            rrt_rank_str = f"{rrt_rank}/{len(all_rrt)}"
        else:
            rrt_rank_str = "N/A"

        rows_rrt.append([
            log_name, rrt_rank_str,
            fmt_int(gnn['rrt']), fmt_int(best_rrt), best_rrt_model or "N/A",
            fmt_pct(pct(gnn['rrt'], best_rrt)),
            fmt_int(mean_rrt),
            fmt_pct(pct(gnn['rrt'], mean_rrt)),
        ])

    return rows_dl, rows_ttne, rows_rrt


# ── TABLE 2 per GNN: performance vs baselines ─────────────────────────────────

def make_vs_baselines(logs, log_order, gnn_model):
    baseline_names = sorted({k for ld in logs.values() for k in ld if k not in GNN_MODELS})
    all_models = [gnn_model] + baseline_names

    dl_vals_a    = {m: [] for m in all_models}
    ttne_vals_a  = {m: [] for m in all_models}
    rrt_vals_a   = {m: [] for m in all_models}
    dl_ranks_a   = {m: [] for m in all_models}
    ttne_ranks_a = {m: [] for m in all_models}
    rrt_ranks_a  = {m: [] for m in all_models}
    dl_best_a    = {m: [] for m in all_models}
    dl_mean_a    = {m: [] for m in all_models}
    ttne_best_a  = {m: [] for m in all_models}
    ttne_mean_a  = {m: [] for m in all_models}
    rrt_best_a   = {m: [] for m in all_models}
    rrt_mean_a   = {m: [] for m in all_models}

    for log_name in log_order:
        ld = logs[log_name]
        present = [m for m in all_models if m in ld]

        # DL
        dl_p = [(m, ld[m]['dl']) for m in present if ld[m]['dl'] is not None]
        if dl_p:
            best_v = max(v for _, v in dl_p)
            mean_v = avg([v for _, v in dl_p])
            for rank, (m, v) in enumerate(sorted(dl_p, key=lambda x: x[1], reverse=True), 1):
                dl_vals_a[m].append(v);  dl_ranks_a[m].append(rank)
                dl_best_a[m].append(pct(v, best_v)); dl_mean_a[m].append(pct(v, mean_v))

        # TTNE
        ttne_p = [(m, ld[m]['ttne']) for m in present if ld[m].get('ttne') is not None]
        if ttne_p:
            best_v = min(v for _, v in ttne_p)
            mean_v = avg([v for _, v in ttne_p])
            for rank, (m, v) in enumerate(sorted(ttne_p, key=lambda x: x[1]), 1):
                ttne_vals_a[m].append(v); ttne_ranks_a[m].append(rank)
                ttne_best_a[m].append(pct(v, best_v)); ttne_mean_a[m].append(pct(v, mean_v))

        # RRT
        rrt_p = [(m, ld[m]['rrt']) for m in present if ld[m].get('rrt') is not None]
        if rrt_p:
            best_v = min(v for _, v in rrt_p)
            mean_v = avg([v for _, v in rrt_p])
            for rank, (m, v) in enumerate(sorted(rrt_p, key=lambda x: x[1]), 1):
                rrt_vals_a[m].append(v);  rrt_ranks_a[m].append(rank)
                rrt_best_a[m].append(pct(v, best_v)); rrt_mean_a[m].append(pct(v, mean_v))

    def wins(lst): return sum(1 for r in lst if r == 1)

    rows = []
    for m in all_models:
        marker = " ◄" if m == gnn_model else ""
        avg_dl_r   = avg(dl_ranks_a[m])
        avg_ttne_r = avg(ttne_ranks_a[m])
        avg_rrt_r  = avg(rrt_ranks_a[m])
        rows.append([
            m + marker,
            str(len(dl_vals_a[m])),
            fmt_dl(avg(dl_vals_a[m])),
            f"{avg_dl_r:.2f}"   if avg_dl_r   is not None else "N/A",
            str(wins(dl_ranks_a[m])),
            fmt_pct(avg(dl_best_a[m])),
            fmt_pct(avg(dl_mean_a[m])),
            fmt_int(avg(ttne_vals_a[m])),
            f"{avg_ttne_r:.2f}" if avg_ttne_r is not None else "N/A",
            str(wins(ttne_ranks_a[m])),
            fmt_pct(avg(ttne_best_a[m])),
            fmt_pct(avg(ttne_mean_a[m])),
            fmt_int(avg(rrt_vals_a[m])),
            f"{avg_rrt_r:.2f}"  if avg_rrt_r  is not None else "N/A",
            str(wins(rrt_ranks_a[m])),
            fmt_pct(avg(rrt_best_a[m])),
            fmt_pct(avg(rrt_mean_a[m])),
        ])
    return rows


# ── print one full GNN model block ────────────────────────────────────────────

def print_gnn_block(logs, log_order, gnn_model, idx):
    print()
    title = f"  TABLE {idx}: {gnn_model}"
    bar = "═" * 160
    print(f"╔{bar}╗")
    print(f"║{title:<160}║")
    print(f"╚{bar}╝")
    print()

    rows_dl, rows_ttne, rows_rrt = make_detail_rows(logs, log_order, gnn_model)

    dl_headers = ["Log","Rank","DL","Best DL","Best Model (DL)","Δ vs Best","Mean DL","Δ vs Mean"]
    dl_aligns  = ['left','center','right','right','left','right','right','right']
    print_sub_table("DL Similarity ↑  (higher is better)", dl_headers, dl_aligns, rows_dl)

    ttne_headers = ["Log","Rank","TTNE (min)","Best TTNE","Best Model (TTNE)","Δ vs Best","Mean TTNE","Δ vs Mean"]
    ttne_aligns  = ['left','center','right','right','left','right','right','right']
    print_sub_table("MAE TTNE ↓  (lower is better)", ttne_headers, ttne_aligns, rows_ttne)

    rrt_headers = ["Log","Rank","RRT (min)","Best RRT","Best Model (RRT)","Δ vs Best","Mean RRT","Δ vs Mean"]
    rrt_aligns  = ['left','center','right','right','left','right','right','right']
    print_sub_table("MAE RRT ↓  (lower is better)", rrt_headers, rrt_aligns, rows_rrt)

    vs_headers = [
        "Model","N",
        "Avg DL","DL Rank","DL↑ Wins","Avg Δ Best","Avg Δ Mean",
        "Avg TTNE","TTNE Rank","TTNE↓ Wins","Avg Δ Best","Avg Δ Mean",
        "Avg RRT","RRT Rank","RRT↓ Wins","Avg Δ Best","Avg Δ Mean",
    ]
    vs_aligns = [
        'left','right',
        'right','right','right','right','right',
        'right','right','right','right','right',
        'right','right','right','right','right',
    ]
    vs_rows = make_vs_baselines(logs, log_order, gnn_model)
    print_sub_table(
        "Performance vs Baselines  (◄ = this GNN model | rank among GNN+baselines | Δ% vs best/mean per log, averaged)",
        vs_headers, vs_aligns, vs_rows
    )


# ── Final summary: GNN variants head-to-head ─────────────────────────────────

def print_gnn_head_to_head(logs, log_order):
    print()
    bar = "═" * 120
    print(f"╔{bar}╗")
    print(f"║  {'SUMMARY: GNN_Suffix Variants Head-to-Head':<118}║")
    print(f"╚{bar}╝")
    print()

    dl_ranks_a   = {m: [] for m in GNN_LIST}
    ttne_ranks_a = {m: [] for m in GNN_LIST}
    rrt_ranks_a  = {m: [] for m in GNN_LIST}
    dl_wins      = {m: 0  for m in GNN_LIST}
    ttne_wins    = {m: 0  for m in GNN_LIST}
    rrt_wins     = {m: 0  for m in GNN_LIST}
    dl_best_a    = {m: [] for m in GNN_LIST}
    dl_mean_a    = {m: [] for m in GNN_LIST}
    ttne_best_a  = {m: [] for m in GNN_LIST}
    ttne_mean_a  = {m: [] for m in GNN_LIST}
    rrt_best_a   = {m: [] for m in GNN_LIST}
    rrt_mean_a   = {m: [] for m in GNN_LIST}

    for log_name in log_order:
        ld = logs[log_name]
        ng = non_gnn(ld)
        present = [m for m in GNN_LIST if m in ld]
        if not present:
            continue

        # DL (higher is better)
        dl_p = [(m, ld[m]['dl']) for m in present if ld[m]['dl'] is not None]
        if dl_p:
            dl_ng = [v['dl'] for v in ng.values() if v['dl'] is not None]
            best_ng = max(dl_ng) if dl_ng else None
            mean_ng = avg(dl_ng)
            for rank, (m, v) in enumerate(sorted(dl_p, key=lambda x: x[1], reverse=True), 1):
                dl_ranks_a[m].append(rank)
                dl_best_a[m].append(pct(v, best_ng))
                dl_mean_a[m].append(pct(v, mean_ng))
            dl_wins[sorted(dl_p, key=lambda x: x[1], reverse=True)[0][0]] += 1

        # TTNE (lower is better)
        ttne_p = [(m, ld[m]['ttne']) for m in present if ld[m]['ttne'] is not None]
        if ttne_p:
            ttne_ng = [v['ttne'] for v in ng.values() if v['ttne'] is not None]
            best_ng = min(ttne_ng) if ttne_ng else None
            mean_ng = avg(ttne_ng)
            for rank, (m, v) in enumerate(sorted(ttne_p, key=lambda x: x[1]), 1):
                ttne_ranks_a[m].append(rank)
                ttne_best_a[m].append(pct(v, best_ng))
                ttne_mean_a[m].append(pct(v, mean_ng))
            ttne_wins[sorted(ttne_p, key=lambda x: x[1])[0][0]] += 1

        # RRT (lower is better)
        rrt_p = [(m, ld[m]['rrt']) for m in present if ld[m]['rrt'] is not None]
        if rrt_p:
            rrt_ng = [v['rrt'] for v in ng.values() if v['rrt'] is not None]
            best_ng = min(rrt_ng) if rrt_ng else None
            mean_ng = avg(rrt_ng)
            for rank, (m, v) in enumerate(sorted(rrt_p, key=lambda x: x[1]), 1):
                rrt_ranks_a[m].append(rank)
                rrt_best_a[m].append(pct(v, best_ng))
                rrt_mean_a[m].append(pct(v, mean_ng))
            rrt_wins[sorted(rrt_p, key=lambda x: x[1])[0][0]] += 1

    headers = [
        "Model",
        "DL Rank","DL↑ Wins","Avg Δ Best DL","Avg Δ Mean DL",
        "TTNE Rank","TTNE↓ Wins","Avg Δ Best TTNE","Avg Δ Mean TTNE",
        "RRT Rank","RRT↓ Wins","Avg Δ Best RRT","Avg Δ Mean RRT",
    ]
    aligns = ['left'] + ['right']*12

    scores = {}
    rows   = []
    for m in GNN_LIST:
        avg_dl_r   = avg(dl_ranks_a[m])
        avg_ttne_r = avg(ttne_ranks_a[m])
        avg_rrt_r  = avg(rrt_ranks_a[m])
        rows.append([
            m,
            f"{avg_dl_r:.2f}"   if avg_dl_r   is not None else "N/A",
            str(dl_wins[m]),
            fmt_pct(avg(dl_best_a[m])),
            fmt_pct(avg(dl_mean_a[m])),
            f"{avg_ttne_r:.2f}" if avg_ttne_r is not None else "N/A",
            str(ttne_wins[m]),
            fmt_pct(avg(ttne_best_a[m])),
            fmt_pct(avg(ttne_mean_a[m])),
            f"{avg_rrt_r:.2f}"  if avg_rrt_r  is not None else "N/A",
            str(rrt_wins[m]),
            fmt_pct(avg(rrt_best_a[m])),
            fmt_pct(avg(rrt_mean_a[m])),
        ])
        scores[m] = (
            -3 * (avg_dl_r   or 99)
            -2 * (avg_ttne_r or 99)
            -1 * (avg_rrt_r  or 99)
        )

    print_sub_table(
        "Rank among GNN variants only (1=best).  Avg Δ vs best/mean of non-GNN baselines per log, then averaged.",
        headers, aligns, rows
    )

    best = max(scores, key=scores.get)
    print(f"  ★  Best overall GNN_Suffix variant: {best}")
    print()
    print("  Notes:")
    print("  - DL ↑ higher is better;  TTNE ↓ and RRT ↓ lower is better.")
    print("  - GNN_Suffix and GNN_Suffix_Stop produce no TTNE/RRT output → N/A.")
    print("  - Winner weighted score: 3×DL_rank + 2×TTNE_rank + 1×RRT_rank (lower avg rank = higher score).")
    print()


# ── entry point ───────────────────────────────────────────────────────────────

logs, log_order = parse_results('results.txt')

for idx, gnn_model in enumerate(GNN_LIST, 1):
    print_gnn_block(logs, log_order, gnn_model, idx)

print_gnn_head_to_head(logs, log_order)
