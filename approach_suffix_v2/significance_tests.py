"""Statistical significance tests for the suffix-prediction results.

Every comparison runs a Friedman test across the 8 logs (block = log, one
seed-averaged mean score per (log, model)) and, if it is significant, a Nemenyi
post-hoc with a critical-difference diagram. With only 2 models (e.g.
ablation-decoder) Friedman is undefined in scipy and reduces to the two-sided
sign test across the logs, which is run instead (no Nemenyi / CD diagram).

  main              POG vs the baselines SEP-LSTM, ED-LSTM, SuTraN (NDA) on
                    GES, BOS and DLS. The baselines' GES / BOS are those of the
                    TTNE-derived partial order (po_ttne_results.pkl, see
                    baselines/eval_ttne_partial_order.py).
  ablation-input    input-format ablation on GES, BOS and DLS. Variants:
                    Graph, Seq, GraphFlip, GraphRandom, SeqFlip, SeqRandom;
                    pick a subset with --input-variants (default: all six).
  ablation-decoder  event-wise new-block decoder (NB) vs block decoder (Block),
                    graph input, on GES and BOS (the block decoder has no DLS).
  all               main + ablation-input + ablation-decoder.

Run from the repo root, e.g.

    python significance_tests.py all
    python approach_suffix_v2/significance_tests.py ablation-input --input-variants Graph,Seq,GraphFlip

Data sources (per log, mean over the runs found on disk):
    own variants   <subdir>/run_N/results_suffix_time_gnn[_<tag>].csv
                   columns dl_similarity / ges_exact / bo_combined
    baselines      <baselines_root>/<log>/<DIR>_runN/TEST_SET_RESULTS/
                   averaged_results.pkl ('DL sim') and
                   po_ttne_results.pkl ('ges_exact', 'bo_combined')

Outputs (into --out-dir, default approach_suffix_v2/significance_tests/):
    significance_tests.txt          box tables for every section that ran
    significance_tests.csv          tidy long format, one row per test
    cd_<table>_<metric>.png         critical-difference diagram (if Friedman p < alpha)
"""
import argparse
import csv
import math
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)

# ── Config ───────────────────────────────────────────────────────────────────

LOGS = [
    "Sepsis", "BPI_Challenge_2012_A", "BPI_Challenge_2012_O",
    "BPIC15_1", "BPIC15_2", "BPIC15_3", "BPIC15_4", "BPIC15_5",
]

POG_SUB = "approach_suffix_v2/results_time_gatv2_gru_nb_v2_actnb"
SEQ_SUB = "approach_suffix_v2/results_time_gatv2_seq_gru_nb_v2_actnb"
BLOCK_SUB = "approach_suffix_v2/results_time_gatv2_gru_block_v2"
OWN_CSV = "results_suffix_time_gnn.csv"
BASELINES_ROOT_DEFAULT = "baselines/results_per_log"

POG_NAME = "POG"

# display name -> result-dir base under <baselines_root>/<log>/
BASELINES = [
    ("SEP-LSTM", "SEP_LSTM_results"),
    ("ED-LSTM", "ED_LSTM_results"),
    ("SuTraN", "SUTRAN_NDA_results"),
]

# name -> (results subdir, csv file)
INPUT_VARIANTS = {
    "Graph": (POG_SUB, OWN_CSV),
    "Seq": (SEQ_SUB, OWN_CSV),
    #"GraphFlip": (POG_SUB, "results_suffix_time_gnn_prefixflip.csv"),
    #"GraphRandom": (POG_SUB, "results_suffix_time_gnn_prefixrandom.csv"),
    "SeqFlip": (SEQ_SUB, "results_suffix_time_gnn_prefixflip.csv"),
    #"SeqRandom": (SEQ_SUB, "results_suffix_time_gnn_prefixrandom.csv"),
}
DECODER_VARIANTS = {
    "NB": (POG_SUB, OWN_CSV),
    "Block": (BLOCK_SUB, OWN_CSV),
}

# metric key -> label; all higher-is-better
METRICS = {"ges_exact": "GES", "bo_combined": "BOS", "dl_similarity": "DLS"}

# metric key -> (baseline result file, key in it)
_BASELINE_SRC = {
    "dl_similarity": ("averaged_results.pkl", "DL sim"),
    "ges_exact": ("po_ttne_results.pkl", "ges_exact"),
    "bo_combined": ("po_ttne_results.pkl", "bo_combined"),
}

CSV_FIELDS = [
    "table", "metric", "comparison", "test", "model_a", "model_b",
    "statistic", "p_value", "n", "note",
]


# ── Small helpers ────────────────────────────────────────────────────────────

def _resolve(p):
    return p if os.path.isabs(p) else os.path.join(_PROJECT_ROOT, p)


def _parse_runs(spec):
    return [int(x) for x in str(spec).split(",") if x.strip() != ""]


class Tee:
    """Collect every emitted line and mirror it to stdout."""

    def __init__(self):
        self.lines = []

    def __call__(self, line=""):
        print(line)
        self.lines.append(line)

    def dump(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines) + "\n")


def print_table(emit, title, headers, aligns, rows):
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    sep = " | "

    def fmt(val, w, a):
        s = str(val)
        return s.rjust(w) if a == "right" else (s.center(w) if a == "center" else s.ljust(w))

    header_line = sep.join(fmt(h, w, a) for h, w, a in zip(headers, widths, aligns))
    total_w = len(header_line)
    emit("  " + title)
    emit("  " + "-" * total_w)
    emit("  " + header_line)
    emit("  " + "-" * total_w)
    for row in rows:
        emit("  " + sep.join(fmt(row[i] if i < len(row) else "", widths[i], aligns[i])
                             for i in range(len(headers))))
    emit("  " + "-" * total_w)
    emit("")


def _f(v, nd=4):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    return f"{v:.{nd}f}"


def _p(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    return "<1e-4" if 0 <= v < 1e-4 else f"{v:.4g}"


def _add_row(ctx, **kw):
    row = {k: "" for k in CSV_FIELDS}
    row.update(kw)
    ctx.rows.append(row)


# ── Loading (per-log means over runs) ────────────────────────────────────────

def own_mean(ctx, sub, csv_file, log, metric):
    vals, missing = [], []
    for rn in ctx.runs:
        path = os.path.join(_resolve(sub), f"run_{rn}", csv_file)
        hit = (pd.read_csv(path).loc[lambda d: d["log"] == log, metric]
               if os.path.isfile(path) else pd.Series(dtype=float))
        hit = hit.dropna()
        if hit.empty:
            missing.append(rn)
            continue
        vals.append(float(hit.iloc[0]))
    if missing:
        ctx.emit(f"  WARNING {sub}/{csv_file} [{log}] {metric}: missing runs {missing}")
    return float(np.mean(vals)) if vals else float("nan")


def baseline_mean(ctx, dir_base, log, metric):
    fname, key = _BASELINE_SRC[metric]
    vals, missing = [], []
    for rn in ctx.runs:
        path = os.path.join(_resolve(ctx.baselines_root), log,
                            f"{dir_base}_run{rn}", "TEST_SET_RESULTS", fname)
        if not os.path.isfile(path):
            missing.append(rn)
            continue
        # the pickles hold torch tensors; unpickling them triggers torch's
        # weights_only FutureWarning, which is irrelevant for our own files
        with open(path, "rb") as f, warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            v = pickle.load(f).get(key, "")
        if v == "":
            missing.append(rn)
            continue
        vals.append(float(v))
    if missing:
        ctx.emit(f"  WARNING {dir_base}/{fname} [{log}] {key}: missing runs {missing}")
    return float(np.mean(vals)) if vals else float("nan")


# ── Statistics ───────────────────────────────────────────────────────────────

def friedman(matrix):
    """matrix: (n_blocks, k_models), higher is better. Returns (chi2, p, avg_ranks[k])."""
    matrix = np.asarray(matrix, dtype=np.float64)
    chi2, p = stats.friedmanchisquare(*matrix.T)
    ranks = np.apply_along_axis(lambda row: stats.rankdata(-row), 1, matrix)
    return float(chi2), float(p), ranks.mean(axis=0)


def _friedman_table(ctx, ap, table, metric, models):
    """models: list of (name, fn(log) -> seed-averaged mean). Builds the
    logs x models matrix, prints it, then Friedman -> Nemenyi / CD diagram."""
    ctx.emit("")
    ctx.emit(f"### metric: {METRICS[metric]} (higher is better)")
    model_names = [n for n, _ in models]
    matrix = np.array([[fn(log) for _, fn in models] for log in LOGS], dtype=np.float64)
    print_table(ctx.emit, "per-log mean over runs", ["log"] + model_names,
                ["left"] + ["right"] * len(model_names),
                [[log] + [_f(v) for v in matrix[i]] for i, log in enumerate(LOGS)])

    if np.isnan(matrix).any():
        missing = "; ".join(
            f"{m} ({', '.join(LOGS[i] for i in np.where(np.isnan(matrix[:, j]))[0])})"
            for j, m in enumerate(model_names) if np.isnan(matrix[:, j]).any())
        ctx.emit(f"  Friedman skipped for {metric}: missing for {missing}.")
        return
    if len(model_names) == 2:
        _sign_test(ctx, table, metric, matrix, model_names)
        return
    chi2, p, avg_ranks = friedman(matrix)
    rank_str = ", ".join(f"{m}={r:.3f}" for m, r in zip(model_names, avg_ranks))
    ctx.emit(f"  Friedman (block=log, k={len(model_names)}, n={len(LOGS)}): "
             f"chi2={chi2:.3f}, p={_p(p)}")
    ctx.emit(f"  average ranks (1=best): {rank_str}")
    _add_row(ctx, table=table, metric=metric, comparison="omnibus",
             test="friedman", statistic=chi2, p_value=p, n=len(LOGS),
             note=f"k={len(model_names)}; avg_ranks: {rank_str}")

    if p < ctx.alpha:
        _nemenyi_and_cd(ctx, table, metric, matrix, model_names, avg_ranks, ap)
    else:
        ctx.emit(f"  Friedman not significant at alpha={ctx.alpha}; "
                 f"skipping Nemenyi / CD diagram.")
    ctx.emit("")


def _sign_test(ctx, table, metric, matrix, model_names):
    """k=2: Friedman reduces to the two-sided sign test across the logs (exact
    binomial on wins vs losses, ties dropped). No Nemenyi / CD diagram for k=2."""
    a, b = model_names
    d = matrix[:, 0] - matrix[:, 1]
    wins, losses = int((d > 0).sum()), int((d < 0).sum())
    ties = len(LOGS) - wins - losses
    p = float(stats.binomtest(wins, wins + losses).pvalue) if wins + losses else 1.0
    ctx.emit(f"  k=2 -> sign test (Friedman equivalent), n={len(LOGS)} logs: "
             f"{a} wins/losses/ties={wins}/{losses}/{ties}, p={_p(p)}")
    ctx.emit("  Nemenyi / CD diagram not applicable for 2 models.")
    _add_row(ctx, table=table, metric=metric, comparison=f"{a} vs {b}",
             test="sign_test", model_a=a, model_b=b, statistic=wins, p_value=p,
             n=len(LOGS), note=f"{a} wins/losses/ties={wins}/{losses}/{ties}")
    ctx.emit("")


def _nemenyi_and_cd(ctx, table, metric, matrix, model_names, avg_ranks, ap):
    try:
        import scikit_posthocs as sp
    except ImportError:
        ap.error("scikit-posthocs is required for the Nemenyi post-hoc / CD diagram "
                 "(pip install scikit-posthocs; rebuild the Docker image).")
    pmat = sp.posthoc_nemenyi_friedman(matrix)
    pmat.index = model_names
    pmat.columns = model_names
    ctx.emit("  Nemenyi post-hoc p-values:")
    for a_name in model_names:
        cells = "  ".join(f"{b_name}={_p(pmat.loc[a_name, b_name])}"
                          for b_name in model_names if b_name != a_name)
        ctx.emit(f"    {a_name:<12} {cells}")
    for i, a_name in enumerate(model_names):
        for j, b_name in enumerate(model_names):
            if j <= i:
                continue
            pv = float(pmat.loc[a_name, b_name])
            _add_row(ctx, table=table, metric=metric, comparison="posthoc",
                     test="nemenyi", model_a=a_name, model_b=b_name,
                     statistic=abs(avg_ranks[i] - avg_ranks[j]), p_value=pv,
                     note=f"R_{a_name}={avg_ranks[i]:.3f}; R_{b_name}={avg_ranks[j]:.3f}"
                          + ("; sig" if pv < ctx.alpha else ""))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ranks_series = pd.Series(dict(zip(model_names, avg_ranks)))
    # sig_matrix takes the raw p-value matrix; it thresholds at alpha internally
    # and connects groups that are NOT significantly different.
    plt.figure(figsize=(8, 2.4))
    sp.critical_difference_diagram(ranks_series, pmat, alpha=ctx.alpha)
    plt.title(f"{table} -- {METRICS[metric]}  (Nemenyi, alpha={ctx.alpha})")
    out_png = os.path.join(ctx.out_dir, f"cd_{table}_{metric}.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    ctx.emit(f"  CD diagram -> {out_png}")
    _add_row(ctx, table=table, metric=metric, comparison="cd_diagram",
             test="cd_diagram", note=out_png)


# ── Modes ────────────────────────────────────────────────────────────────────

def _header(ctx, title):
    ctx.emit("=" * 78)
    ctx.emit(title)
    ctx.emit("=" * 78)


def _own_models(ctx, variants):
    return [(name, lambda log, m, s=sub, c=csv_file: own_mean(ctx, s, c, log, m))
            for name, (sub, csv_file) in variants.items()]


def _bind(models, metric):
    return [(name, lambda log, fn=fn: fn(log, metric)) for name, fn in models]


def mode_main(ctx, ap):
    _header(ctx, "MAIN  --  POG vs baselines (GES/BOS from TTNE partial order), Friedman + Nemenyi")
    models = _own_models(ctx, {POG_NAME: (POG_SUB, OWN_CSV)}) + [
        (disp, lambda log, m, b=base: baseline_mean(ctx, b, log, m))
        for disp, base in BASELINES]
    for metric in METRICS:
        _friedman_table(ctx, ap, "main", metric, _bind(models, metric))


def mode_ablation_input(ctx, ap):
    _header(ctx, f"ABLATION INPUT  --  {', '.join(ctx.input_variants)}, Friedman + Nemenyi")
    models = _own_models(ctx, {n: INPUT_VARIANTS[n] for n in ctx.input_variants})
    for metric in METRICS:
        _friedman_table(ctx, ap, "ablation_input", metric, _bind(models, metric))


def mode_ablation_decoder(ctx, ap):
    _header(ctx, "ABLATION DECODER  --  NB vs Block (graph input), Friedman + Nemenyi")
    models = _own_models(ctx, DECODER_VARIANTS)
    for metric in ("ges_exact", "bo_combined"):
        _friedman_table(ctx, ap, "ablation_decoder", metric, _bind(models, metric))


# ── Entry point ──────────────────────────────────────────────────────────────

class Ctx:
    pass


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["main", "ablation-input", "ablation-decoder", "all"])
    ap.add_argument("--runs", default="1,2,3,4,5",
                    help="comma list of run numbers to seed-average (default 1,2,3,4,5)")
    ap.add_argument("--run", type=int, default=None,
                    help="use only this single run (overrides --runs)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--input-variants", default=",".join(INPUT_VARIANTS),
                    help=f"comma list of ablation-input variants, from "
                         f"{', '.join(INPUT_VARIANTS)} (default: all)")
    ap.add_argument("--baselines-root", default=BASELINES_ROOT_DEFAULT)
    ap.add_argument("--out-dir", default="approach_suffix_v2/significance_tests")
    args = ap.parse_args()

    ctx = Ctx()
    ctx.runs = [args.run] if args.run is not None else _parse_runs(args.runs)
    ctx.alpha = args.alpha
    ctx.input_variants = [v.strip() for v in args.input_variants.split(",") if v.strip()]
    unknown = [v for v in ctx.input_variants if v not in INPUT_VARIANTS]
    if unknown:
        ap.error(f"unknown --input-variants {unknown}; choose from {list(INPUT_VARIANTS)}")
    if len(ctx.input_variants) < 2:
        ap.error("--input-variants needs at least 2 variants")
    ctx.baselines_root = args.baselines_root
    ctx.out_dir = _resolve(args.out_dir)
    os.makedirs(ctx.out_dir, exist_ok=True)
    tee = Tee()
    ctx.emit = tee
    ctx.rows = []

    ctx.emit(f"significance_tests.py  mode={args.mode}  runs={ctx.runs}  alpha={ctx.alpha}")
    ctx.emit("")

    modes = {"main": [mode_main], "ablation-input": [mode_ablation_input],
             "ablation-decoder": [mode_ablation_decoder],
             "all": [mode_main, mode_ablation_input, mode_ablation_decoder]}
    for fn in modes[args.mode]:
        fn(ctx, ap)

    txt_path = os.path.join(ctx.out_dir, "significance_tests.txt")
    csv_path = os.path.join(ctx.out_dir, "significance_tests.csv")
    tee.dump(txt_path)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for row in ctx.rows:
            w.writerow(row)
    print(f"\nWrote {txt_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
