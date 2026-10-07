"""Statistical significance tests for the suffix-prediction results.

Runs, on the per-instance (per prefix-suffix pair) scores produced by the eval
pipeline, the tests the paper's comparative claims need:

  main          paper Table 5 -- POG vs each of 5 baselines, GES and DLS.
                Per log: paired Wilcoxon signed-rank on the per-instance scores,
                Holm-Bonferroni over the 5-baseline family, matched-pairs
                rank-biserial effect size, plus a KS normality check on the
                score / paired-difference distributions. Then a Friedman test
                across the 8 logs and, if it is significant, a Nemenyi post-hoc
                with a critical-difference diagram.
  time          paper Table 6 -- POG vs each of the 5 baselines on per-instance
                timestamp MAE (TTNE, minutes). Identical per-log treatment to
                `main`: paired Wilcoxon signed-rank, Holm-Bonferroni over the
                5-baseline family, rank-biserial effect size, KS normality
                check, then a Friedman test across the 8 logs and, if it is
                significant, a Nemenyi post-hoc with a CD diagram.
  trend         suffix-length -- Spearman rho(suffix length, score) per
                (log, model); negative rho => score decreases with length.
  dispersion    seed dispersion -- mean +- std (ddof=1) and 95% CI across the
                5 seeds per (log, model) for GES / DLS / TTNE MAE.
  ablation-input   paper Table 8 -- Graph_v1 / Seq_v1 vs their prefix-flip and
                   prefix-random variants (GraphFlip_v1, GraphRandom_v1,
                   SeqFlip_v1, SeqRandom_v1), pairwise Wilcoxon + rank-biserial
                   per log on GES and DLS. Needs the per-sample files for the
                   seq / prefix-flip / prefix-random variants (the run scripts
                   write them; re-run those variants first).
  ablation-train   paper Table 9 -- v1 vs v2 vs v3 training strategy, same shape.
                   Needs the per-sample files for the v2 / v3 variants.
  all           main + time + trend + dispersion (ablations run only if their
                per-sample files are present).

Run from the repo root, e.g.

    python approach_suffix_v2/significance_tests.py all

Outputs (into --out-dir, default approach_suffix_v2/significance_tests/):
    significance_tests.txt   box tables for every section that ran
    significance_tests.csv   tidy long format, one row per test
    cd_<metric>.png          Table 5 / Table 6 critical-difference diagram (if Friedman p < alpha)

Seed handling: the per-instance vectors are averaged element-wise across the
available runs (run_1..run_5) before testing. This denoises the per-training-run
Monte-Carlo variation while keeping n = #test instances and the paired-test
independence structure intact (stacking the runs would replicate every instance
5x and shrink p-values artificially). --run N uses a single run instead.

The KS normality check estimates mean/std from the sample, so it is the
Lilliefors situation and its p-value is anticonservative; with n in the
hundreds-thousands it will almost always reject. It is reported as descriptive
justification for the non-parametric choice, with the skew / excess-kurtosis
columns carrying the practical story.
"""
import argparse
import csv
import math
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

from analyze_partial_order import (  # noqa: E402
    analyze,
    load_metric_source,
    validate_alignment,
)

# ── Config ───────────────────────────────────────────────────────────────────

LOGS = [
    "Sepsis", "BPI_Challenge_2012_A", "BPI_Challenge_2012_O",
    "BPIC15_1", "BPIC15_2", "BPIC15_3", "BPIC15_4", "BPIC15_5",
]

POG_SUBDIR_DEFAULT = "approach_suffix_v2/results_time_gatv2_gru_nb_v1"
BASELINES_ROOT_DEFAULT = "baselines/results_per_log"
DATASET_ROOT_DEFAULT = "approach_suffix_v2/results_per_log"

POG_NAME = "POG"
POG_FILE_TPL = "{log}_per_sample_metrics.pt"
POG_AGG_CSV = "results_suffix_time_gnn.csv"

# display name -> result-dir base under <baselines_root>/<log>/
TABLE5_BASELINES = [
    ("SEP-LSTM", "SEP_LSTM_results"),
    ("ED-LSTM", "ED_LSTM_results"),
    ("CRTP-LSTM", "CRTP_LSTM_NDA_results"),
    ("SuTraN", "SUTRAN_NDA_results"),
    ("BEST", "BEST_results"),
]

METRIC_LABEL = {
    "ges_approx": "GES up",
    "dl_similarity": "DL similarity up",
    "ttne_mae_minutes": "MAE TTNE (min) down",
}
HIGHER_BETTER = {
    "ges_approx": True,
    "dl_similarity": True,
    "ttne_mae_minutes": False,
}

# baseline averaged_results.pkl keys, for the dispersion table
_AGG_BASELINE_KEYS = {
    "ges_approx": "ges_approx",
    "dl_similarity": "DL sim",
    "ttne_mae_minutes": "MAE TTNE minutes",
}

# Deferred (Tables 8 / 9): name -> (results subdir under approach_suffix_v2/, file template)
ABLATION_INPUT = {
    "Graph_v1": ("approach_suffix_v2/results_time_gatv2_gru_nb_v1", "{log}_per_sample_metrics.pt"),
    "Seq_v1": ("approach_suffix_v2/results_time_gatv2_seq_gru_nb_v1", "{log}_per_sample_metrics.pt"),
    "GraphFlip_v1": ("approach_suffix_v2/results_time_gatv2_gru_nb_v1", "{log}_per_sample_metrics_prefixflip.pt"),
    "GraphRandom_v1": ("approach_suffix_v2/results_time_gatv2_gru_nb_v1", "{log}_per_sample_metrics_prefixrandom.pt"),
    "SeqFlip_v1": ("approach_suffix_v2/results_time_gatv2_seq_gru_nb_v1", "{log}_per_sample_metrics_prefixflip.pt"),
    "SeqRandom_v1": ("approach_suffix_v2/results_time_gatv2_seq_gru_nb_v1", "{log}_per_sample_metrics_prefixrandom.pt"),
}
ABLATION_TRAIN = {
    "v1": ("approach_suffix_v2/results_time_gatv2_gru_nb_v1", "{log}_per_sample_metrics.pt"),
    "v2": ("approach_suffix_v2/results_time_gatv2_gru_nb_v2", "{log}_per_sample_metrics.pt"),
    "v3": ("approach_suffix_v2/results_time_gatv2_gru_nb_v3", "{log}_per_sample_metrics.pt"),
}

CSV_FIELDS = [
    "table", "metric", "log", "comparison", "test", "model_a", "model_b",
    "statistic", "p_value", "p_adj", "effect_r", "effect_type", "n", "n_runs",
    "skew", "kurtosis", "non_normal", "note",
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


# ── Loading / alignment ──────────────────────────────────────────────────────

_SRC_KEYS = ("dl_similarity", "ges_approx", "ttne_mae_minutes", "rrt_mae_minutes",
             "pref_len", "suf_len")


def _seed_average(per_run, log, name, ap):
    """per_run: list of {key: list|None}. Return {key: np.ndarray|None}, ref lens, n."""
    if not per_run:
        ap.error(f"{name} [{log}]: no run directories found")
    if per_run[0]["pref_len"] is None or per_run[0]["suf_len"] is None:
        ap.error(f"{name} [{log}]: pref_len/suf_len missing; cannot align this source")
    ref_pref = np.asarray(per_run[0]["pref_len"], dtype=np.int64)
    ref_suf = np.asarray(per_run[0]["suf_len"], dtype=np.int64)
    n = ref_pref.size
    for r in per_run[1:]:
        if np.asarray(r["pref_len"], dtype=np.int64).shape != ref_pref.shape:
            ap.error(f"{name} [{log}]: runs disagree on sample count")
        if not np.array_equal(np.asarray(r["pref_len"], dtype=np.int64), ref_pref) or \
           not np.array_equal(np.asarray(r["suf_len"], dtype=np.int64), ref_suf):
            ap.error(f"{name} [{log}]: runs are not in the same sample order")

    out = {"pref_len": ref_pref, "suf_len": ref_suf}
    for key in ("dl_similarity", "ges_approx", "ttne_mae_minutes", "rrt_mae_minutes"):
        vecs = [r[key] for r in per_run]
        if any(v is None for v in vecs):
            out[key] = None
            continue
        stack = np.vstack([np.asarray(v, dtype=np.float64) for v in vecs])
        out[key] = stack.mean(axis=0)
    return out, n


def load_source(kind, ident, log, runs, ctx, ap):
    """kind: 'gnn' -> ident is a results subdir; 'baseline' -> ident is a dir base."""
    per_run, found = [], []
    for rn in runs:
        if kind == "gnn":
            path = os.path.join(_resolve(ident), f"run_{rn}", POG_FILE_TPL.format(log=log))
            ok = os.path.isfile(path)
        else:
            path = os.path.join(_resolve(ctx.baselines_root), log,
                                f"{ident}_run{rn}", "TEST_SET_RESULTS")
            ok = os.path.isdir(path)
        if not ok:
            continue
        src = load_metric_source(path)
        per_run.append({k: src.get(k) for k in _SRC_KEYS})
        found.append(rn)
    if not per_run:
        ap.error(f"{ident} [{log}]: none of runs {runs} found "
                 f"(looked under {'gnn subdir ' + ident if kind == 'gnn' else ctx.baselines_root})")
    if len(found) < len(runs):
        ctx.emit(f"  WARNING {ident} [{log}]: using runs {found} (missing "
                 f"{sorted(set(runs) - set(found))})")
    avg, n = _seed_average(per_run, log, ident, ap)
    avg["_n_runs"] = len(found)
    avg["_n"] = n
    return avg


def align_sources(name2src, log, ctx, ap):
    names = list(name2src)
    ref = name2src[names[0]]
    for nm in names[1:]:
        s = name2src[nm]
        if s["_n"] != ref["_n"] or \
           not np.array_equal(s["pref_len"], ref["pref_len"]) or \
           not np.array_equal(s["suf_len"], ref["suf_len"]):
            ap.error(f"[{log}] source {nm!r} is not aligned to {names[0]!r} "
                     f"(sample count / prefix / suffix length mismatch)")
    if not ctx.align_check:
        return
    ds_path = os.path.join(_resolve(ctx.dataset_root), log, "test_graphdataset.pt")
    tss_path = os.path.join(_resolve(ctx.dataset_root), log, "tss_index.txt")
    if not (os.path.isfile(ds_path) and os.path.isfile(tss_path)):
        ctx.emit(f"  WARNING [{log}]: {ctx.dataset_root}/{log}/test_graphdataset.pt not "
                 f"found; skipping canonical dataset-order check")
        return
    import torch  # noqa
    try:
        import torch_geometric  # noqa: F401
    except ImportError:
        pass
    with open(tss_path) as f:
        tss_index = int(f.read().strip())
    dataset = torch.load(ds_path, map_location="cpu", weights_only=False)
    _, masks = analyze(dataset, tss_index)
    for nm, s in name2src.items():
        src_lists = {k: (s[k].tolist() if isinstance(s[k], np.ndarray) else s[k])
                     for k in _SRC_KEYS if k in s}
        validate_alignment(nm, src_lists, masks, ap)


# ── Statistics ───────────────────────────────────────────────────────────────

def signed_rank_parts(d):
    d = np.asarray(d, dtype=np.float64)
    nz = d[d != 0.0]
    m = nz.size
    if m == 0:
        return 0.0, 0.0, 0
    r = stats.rankdata(np.abs(nz))
    w_plus = float(r[nz > 0].sum())
    w_minus = float(r[nz < 0].sum())
    return w_plus, w_minus, m


def paired_wilcoxon(a, b):
    """Wilcoxon signed-rank on a - b, plus the matched-pairs rank-biserial
    effect size r = (W+ - W-) / (W+ + W-)  (Kerby 2014; denom = m(m+1)/2).

    effect_r > 0  =>  a is larger than b on that metric.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = a - b
    w_plus, w_minus, m = signed_rank_parts(d)
    if m == 0:
        return {"statistic": 0.0, "p_value": 1.0, "effect_r": 0.0, "n": 0,
                "median_diff": 0.0, "note": "all-ties"}
    res = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided",
                         method="auto", correction=False)
    return {
        "statistic": float(min(w_plus, w_minus)),
        "p_value": float(res.pvalue),
        "effect_r": (w_plus - w_minus) / (w_plus + w_minus),
        "n": m,
        "median_diff": float(np.median(d)),
        "note": "",
    }


def holm_bonferroni(pvals):
    k = len(pvals)
    order = sorted(range(k), key=lambda i: pvals[i])
    adj = [0.0] * k
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (k - rank) * pvals[idx])
        running = max(running, val)
        adj[idx] = running
    return adj


def ks_normal(x, alpha):
    x = np.asarray(x, dtype=np.float64)
    sd = float(np.std(x, ddof=1))
    if sd == 0.0 or x.size < 3:
        return {"statistic": float("nan"), "p_value": float("nan"),
                "skew": float("nan"), "kurtosis": float("nan"), "non_normal": False}
    d, p = stats.kstest(x, "norm", args=(float(np.mean(x)), sd))
    return {
        "statistic": float(d),
        "p_value": float(p),
        "skew": float(stats.skew(x)),
        "kurtosis": float(stats.kurtosis(x)),  # excess
        "non_normal": bool(p < alpha),
    }


def friedman(matrix, higher_better):
    """matrix: (n_blocks, k_models). Returns (chi2, p, avg_ranks[k])."""
    matrix = np.asarray(matrix, dtype=np.float64)
    chi2, p = stats.friedmanchisquare(*matrix.T)
    if higher_better:
        ranks = np.apply_along_axis(lambda row: stats.rankdata(-row), 1, matrix)
    else:
        ranks = np.apply_along_axis(stats.rankdata, 1, matrix)
    return float(chi2), float(p), ranks.mean(axis=0)


def spearman_trend(x, y):
    rho, p = stats.spearmanr(np.asarray(x, dtype=np.float64),
                             np.asarray(y, dtype=np.float64))
    return float(rho), float(p)


# ── Modes ────────────────────────────────────────────────────────────────────

def _add_row(ctx, **kw):
    row = {k: "" for k in CSV_FIELDS}
    row.update(kw)
    ctx.rows.append(row)


def _direction_legend(ctx, metric):
    if HIGHER_BETTER[metric]:
        ctx.emit(f"  [{METRIC_LABEL[metric]}] positive rank-biserial => {POG_NAME} better; "
                 f"Holm adjusts each log's family of {len(TABLE5_BASELINES)} baselines.")
    else:
        ctx.emit(f"  [{METRIC_LABEL[metric]}] NEGATIVE rank-biserial / median(diff) < 0 "
                 f"=> {POG_NAME} better (lower is better).")


def mode_main(ctx, ap):
    ctx.emit("=" * 78)
    ctx.emit("TABLE 5  --  POG vs baselines, per-instance paired tests + Friedman")
    ctx.emit("=" * 78)
    for metric in ("ges_approx", "dl_similarity"):
        _paired_vs_baselines(ctx, ap, "Table5", metric)


def _paired_vs_baselines(ctx, ap, table, metric, baselines=TABLE5_BASELINES):
    """Per-log POG-vs-baselines paired Wilcoxon + Holm + KS, then Friedman
    across the 8 logs with a Nemenyi post-hoc / CD diagram if it is significant.
    Shared by Table 5 (GES / DLS) and Table 6 (TTNE MAE); `baselines` lets a
    caller (e.g. Table 6, which excludes BEST) use a different baseline set."""
    ctx.emit("")
    ctx.emit(f"### metric: {METRIC_LABEL[metric]}")
    _direction_legend(ctx, metric)
    ctx.emit("")

    per_log_mean = {}  # log -> [POG, then baselines] means
    for log in LOGS:
        name2src = {POG_NAME: load_source("gnn", ctx.pog_subdir, log, ctx.runs, ctx, ap)}
        for disp, base in baselines:
            name2src[disp] = load_source("baseline", base, log, ctx.runs, ctx, ap)
        align_sources(name2src, log, ctx, ap)

        pog = name2src[POG_NAME][metric]
        if pog is None:
            ap.error(f"[{log}] {POG_NAME} has no {metric}")

        # KS on each model's score distribution
        for disp in name2src:
            v = name2src[disp][metric]
            if v is None:
                continue
            ks = ks_normal(v, ctx.alpha)
            _add_row(ctx, table=table, metric=metric, log=log,
                     comparison="scores", test="ks_normal", model_a=disp,
                     statistic=ks["statistic"], p_value=ks["p_value"],
                     skew=ks["skew"], kurtosis=ks["kurtosis"],
                     non_normal=ks["non_normal"], n=v.size)

        # paired Wilcoxon POG vs each baseline
        wilc, diff_nonnormal = [], []
        for disp, _ in baselines:
            b = name2src[disp][metric]
            if b is None:
                wilc.append(None)
                diff_nonnormal.append(None)
                continue
            w = paired_wilcoxon(pog, b)
            wilc.append(w)
            ksd = ks_normal(pog - b, ctx.alpha)
            diff_nonnormal.append(ksd["non_normal"])
            _add_row(ctx, table=table, metric=metric, log=log,
                     comparison="paired_diff", test="ks_normal",
                     model_a=POG_NAME, model_b=disp,
                     statistic=ksd["statistic"], p_value=ksd["p_value"],
                     skew=ksd["skew"], kurtosis=ksd["kurtosis"],
                     non_normal=ksd["non_normal"], n=(pog - b).size)

        valid = [i for i, w in enumerate(wilc) if w is not None]
        adj = holm_bonferroni([wilc[i]["p_value"] for i in valid])
        adj_by_i = {i: a for i, a in zip(valid, adj)}

        trows = []
        for i, (disp, _) in enumerate(baselines):
            w = wilc[i]
            if w is None:
                trows.append([disp, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"])
                continue
            trows.append([
                disp, _f(w["statistic"], 1), _p(w["p_value"]),
                _p(adj_by_i[i]), _f(w["effect_r"]), w["n"],
                "yes" if diff_nonnormal[i] else "no",
            ])
            _add_row(ctx, table=table, metric=metric, log=log,
                     comparison=f"{POG_NAME} vs {disp}", test="wilcoxon",
                     model_a=POG_NAME, model_b=disp,
                     statistic=w["statistic"], p_value=w["p_value"],
                     p_adj=adj_by_i[i], effect_r=w["effect_r"],
                     effect_type="rank_biserial", n=w["n"],
                     n_runs=name2src[disp]["_n_runs"],
                     non_normal=diff_nonnormal[i], note=w["note"])

        print_table(ctx.emit, f"{log}  (n={name2src[POG_NAME]['_n']}, "
                              f"runs avg={name2src[POG_NAME]['_n_runs']})",
                    ["baseline", "W", "p", "p_holm", "r_rb", "n", "diff!=normal"],
                    ["left", "right", "right", "right", "right", "right", "center"],
                    trows)

        per_log_mean[log] = [float(np.mean(name2src[POG_NAME][metric]))] + [
            float(np.mean(name2src[d][metric])) if name2src[d][metric] is not None
            else float("nan") for d, _ in baselines]

    model_names = [POG_NAME] + [d for d, _ in baselines]
    matrix = np.array([per_log_mean[log] for log in LOGS], dtype=np.float64)
    _across_logs(ctx, ap, table, metric, matrix, model_names,
                 [(POG_NAME, d) for d, _ in baselines])


def _across_logs(ctx, ap, table, metric, matrix, model_names, pairs):
    """Friedman across the 8 logs (matrix: logs x models, seed-averaged scores);
    if significant, Nemenyi / CD diagram and an across-log Wilcoxon per pair in
    `pairs`, Holm-corrected over that family."""
    if np.isnan(matrix).any():
        missing = "; ".join(
            f"{m} ({', '.join(LOGS[i] for i in np.where(np.isnan(matrix[:, j]))[0])})"
            for j, m in enumerate(model_names) if np.isnan(matrix[:, j]).any())
        ctx.emit(f"  Friedman skipped for {metric}: missing for {missing}.")
        return
    chi2, p, avg_ranks = friedman(matrix, HIGHER_BETTER[metric])
    rank_str = ", ".join(f"{m}={r:.3f}" for m, r in zip(model_names, avg_ranks))
    ctx.emit(f"  Friedman (block=log, k={len(model_names)}, n=8): "
             f"chi2={chi2:.3f}, p={_p(p)}")
    ctx.emit(f"  average ranks (1=best): {rank_str}")
    _add_row(ctx, table=table, metric=metric, comparison="omnibus",
             test="friedman", statistic=chi2, p_value=p, n=len(LOGS),
             note=f"k={len(model_names)}; avg_ranks: {rank_str}")

    if p < ctx.alpha:
        _nemenyi_and_cd(ctx, table, metric, matrix, model_names, avg_ranks, ap)
        _wilcoxon_across_logs(ctx, table, metric, matrix, model_names, pairs)
    else:
        ctx.emit(f"  Friedman not significant at alpha={ctx.alpha}; "
                 f"skipping Nemenyi / CD diagram.")
    ctx.emit("")


def _wilcoxon_across_logs(ctx, table, metric, matrix, model_names, pairs):
    """Across-log Wilcoxon on the 8 seed-averaged per-log scores for each (a, b)
    in `pairs`, Holm-corrected over the family. Run only after a significant Friedman."""
    higher = HIGHER_BETTER[metric]
    col = {m: i for i, m in enumerate(model_names)}
    res = [paired_wilcoxon(matrix[:, col[a]], matrix[:, col[b]]) for a, b in pairs]
    adj = holm_bonferroni([r["p_value"] for r in res])
    ctx.emit(f"  Across-log Wilcoxon (n={len(LOGS)} logs, Holm over {len(res)} comparisons):")
    trows = []
    for (a, b), r, pa in zip(pairs, res, adj):
        d = matrix[:, col[a]] - matrix[:, col[b]]
        wins = int((d > 0).sum() if higher else (d < 0).sum())
        losses = int((d < 0).sum() if higher else (d > 0).sum())
        wlt = f"{wins}/{losses}/{len(LOGS) - wins - losses}"
        trows.append([f"{a} vs {b}", _f(r["statistic"], 1), _p(r["p_value"]), _p(pa),
                      _f(r["effect_r"]), wlt])
        _add_row(ctx, table=table, metric=metric, comparison=f"{a} vs {b}",
                 test="wilcoxon_across_logs", model_a=a, model_b=b,
                 statistic=r["statistic"], p_value=r["p_value"], p_adj=pa,
                 effect_r=r["effect_r"], effect_type="rank_biserial", n=r["n"],
                 note=f"{a} wins/losses/ties={wlt}")
    print_table(ctx.emit, "across logs",
                ["comparison", "W", "p", "p_holm", "r_rb", "first W/L/T"],
                ["left", "right", "right", "right", "right", "center"], trows)


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
        ctx.emit(f"    {a_name:<10} {cells}")
    for i, a_name in enumerate(model_names):
        for j, b_name in enumerate(model_names):
            if j <= i:
                continue
            pv = float(pmat.loc[a_name, b_name])
            _add_row(ctx, table=table, metric=metric, comparison="posthoc",
                     test="nemenyi", model_a=a_name, model_b=b_name,
                     statistic=abs(avg_ranks[i] - avg_ranks[j]), p_value=pv,
                     non_normal=("sig" if pv < ctx.alpha else ""),
                     note=f"R_{a_name}={avg_ranks[i]:.3f}; R_{b_name}={avg_ranks[j]:.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ranks_series = pd.Series(dict(zip(model_names, avg_ranks)))
    # sig_matrix takes the raw p-value matrix; it thresholds at alpha internally
    # and connects groups that are NOT significantly different.
    plt.figure(figsize=(8, 2.4))
    sp.critical_difference_diagram(ranks_series, pmat, alpha=ctx.alpha)
    plt.title(f"{table[:5]} {table[5:]} -- {METRIC_LABEL[metric]}  (Nemenyi, alpha={ctx.alpha})")
    stem = metric if table in ("Table5", "Table6") else f"{table}_{metric}"
    out_png = os.path.join(ctx.out_dir, f"cd_{stem}.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    ctx.emit(f"  CD diagram -> {out_png}")
    _add_row(ctx, table=table, metric=metric, comparison="cd_diagram",
             test="cd_diagram", note=out_png)


def mode_time(ctx, ap):
    ctx.emit("=" * 78)
    ctx.emit("TABLE 6  --  POG vs baselines, per-instance timestamp MAE + Friedman")
    ctx.emit("=" * 78)
    ctx.emit("  (each per-sample value is already a per-instance mean-abs-error over "
             "suffix positions, minutes)")
    # BEST only supports suffix prediction, not timestamp prediction, and is
    # excluded from tab:results_timestamp_prediction; drop it here too so the
    # Friedman/Nemenyi ranks match the 5 models actually shown in that table.
    baselines = [b for b in TABLE5_BASELINES if b[0] != "BEST"]
    _paired_vs_baselines(ctx, ap, "Table6", "ttne_mae_minutes", baselines=baselines)


def mode_trend(ctx, ap):
    ctx.emit("=" * 78)
    ctx.emit("SUFFIX-LENGTH TREND  --  Spearman rho(suffix length, score) per model")
    ctx.emit("=" * 78)
    ctx.emit("  negative rho => score decreases as the suffix gets longer")
    model_names = [POG_NAME] + [d for d, _ in TABLE5_BASELINES]

    for metric in ("ges_approx", "dl_similarity"):
        ctx.emit("")
        ctx.emit(f"### metric: {METRIC_LABEL[metric]}")
        rho_rows, p_rows = [], []
        for log in LOGS:
            name2src = {POG_NAME: load_source("gnn", ctx.pog_subdir, log, ctx.runs, ctx, ap)}
            for d, base in TABLE5_BASELINES:
                name2src[d] = load_source("baseline", base, log, ctx.runs, ctx, ap)
            align_sources(name2src, log, ctx, ap)
            suf_len = name2src[POG_NAME]["suf_len"]

            rrow, prow = [log], [log]
            for nm in model_names:
                v = name2src[nm][metric]
                if v is None:
                    rrow.append("N/A")
                    prow.append("N/A")
                    continue
                rho, pv = spearman_trend(suf_len, v)
                rrow.append(_f(rho))
                prow.append(_p(pv))
                _add_row(ctx, table="TrendSuffixLen", metric=metric, log=log,
                         comparison="suf_len vs score", test="spearman",
                         model_a=nm, statistic=rho, p_value=pv, n=v.size,
                         n_runs=name2src[nm]["_n_runs"],
                         note="negative rho => score decreases with suffix length")
            rho_rows.append(rrow)
            p_rows.append(prow)

        hdr = ["log"] + model_names
        aligns = ["left"] + ["right"] * len(model_names)
        print_table(ctx.emit, "Spearman rho", hdr, aligns, rho_rows)
        print_table(ctx.emit, "Spearman p-value", hdr, aligns, p_rows)


def _agg_pog_runs(ctx, log, metric):
    key = {"ges_approx": "ges_approx", "dl_similarity": "dl_similarity",
           "ttne_mae_minutes": "ttne_mae_minutes"}[metric]
    vals = []
    for rn in ctx.runs:
        path = os.path.join(_resolve(ctx.pog_subdir), f"run_{rn}", POG_AGG_CSV)
        if not os.path.isfile(path):
            continue
        df = pd.read_csv(path)
        hit = df.loc[df["log"] == log, key]
        if not hit.empty:
            vals.append(float(hit.iloc[0]))
    return vals


def _agg_baseline_runs(ctx, log, base, metric):
    key = _AGG_BASELINE_KEYS[metric]
    vals = []
    for rn in ctx.runs:
        path = os.path.join(_resolve(ctx.baselines_root), log,
                            f"{base}_run{rn}", "TEST_SET_RESULTS", "averaged_results.pkl")
        if not os.path.isfile(path):
            continue
        with open(path, "rb") as f:
            d = pickle.load(f)
        if key in d and d[key] != "":
            vals.append(float(d[key]))
    return vals


def _disp_cell(vals):
    if not vals:
        return "N/A", (float("nan"), float("nan"), float("nan"), 0)
    mean = float(np.mean(vals))
    if len(vals) > 1:
        sd = float(np.std(vals, ddof=1))
        ci = 1.96 * sd / math.sqrt(len(vals))
        return f"{mean:.4f} +- {sd:.4f}", (mean, sd, ci, len(vals))
    return f"{mean:.4f}", (mean, float("nan"), float("nan"), 1)


def mode_dispersion(ctx, ap):
    ctx.emit("=" * 78)
    ctx.emit("SEED DISPERSION  --  mean +- std (ddof=1) across seeds, per (log, model)")
    ctx.emit("=" * 78)
    metrics = ["ges_approx", "dl_similarity", "ttne_mae_minutes"]
    models = [(POG_NAME, None)] + TABLE5_BASELINES
    trows = []
    for log in LOGS:
        for disp, base in models:
            cells = [log, disp]
            for metric in metrics:
                vals = (_agg_pog_runs(ctx, log, metric) if base is None
                        else _agg_baseline_runs(ctx, log, base, metric))
                text, (mean, sd, ci, nr) = _disp_cell(vals)
                cells.append(text)
                _add_row(ctx, table="SeedDispersion", metric=metric, log=log,
                         comparison="across_seeds", test="dispersion", model_a=disp,
                         statistic=mean, n_runs=nr,
                         note=f"std={sd:.6f}; ci95={ci:.6f}; n_runs={nr}"
                         if nr > 1 else f"n_runs={nr}")
            trows.append(cells)
    print_table(ctx.emit, "mean +- std over seeds",
                ["log", "model", "GES", "DL similarity", "MAE TTNE (min)"],
                ["left", "left", "right", "right", "right"], trows)


def _ablation(ctx, ap, table, variants, pairs):
    ctx.emit("=" * 78)
    ctx.emit(f"{table}  --  ablation, per-instance pairwise Wilcoxon")
    ctx.emit("=" * 78)

    v_names = list(variants)
    for name in v_names:
        sub, tpl = variants[name]
        miss = os.path.join(_resolve(sub), f"run_{ctx.runs[0]}",
                            tpl.format(log=LOGS[0]))
        if not os.path.isfile(miss):
            ap.error(
                f"{table} needs per-sample files that are not on disk yet, e.g.\n    {miss}\n"
                f"The run scripts already write them (run_suffix_time_v1_seq/v2/v3.py and "
                f"run_eval_prefix_flip / run_eval_prefix_random -> "
                f"{{log}}_per_sample_metrics[_prefixflip|_prefixrandom].pt); re-run the "
                f"8 logs x 5 seeds for the {name!r} variant via run_all_time.py / "
                f"run_all_suffix.py / run_eval_flipped.py / run_eval_random.py, then retry.")

    for metric in ("ges_approx", "dl_similarity"):
        ctx.emit("")
        ctx.emit(f"### metric: {METRIC_LABEL[metric]}")
        per_log_mean = []
        for log in LOGS:
            name2src = {}
            for name in v_names:
                sub, tpl = variants[name]
                per_run, found = [], []
                for rn in ctx.runs:
                    path = os.path.join(_resolve(sub), f"run_{rn}", tpl.format(log=log))
                    if not os.path.isfile(path):
                        continue
                    s = load_metric_source(path)
                    per_run.append({k: s.get(k) for k in _SRC_KEYS})
                    found.append(rn)
                if not per_run:
                    ap.error(f"{name} [{log}]: none of runs {ctx.runs} found")
                avg, n = _seed_average(per_run, log, name, ap)
                avg["_n_runs"] = len(found)
                avg["_n"] = n
                name2src[name] = avg
            align_sources(name2src, log, ctx, ap)

            trows = []
            for a_name, b_name in pairs:
                a, b = name2src[a_name][metric], name2src[b_name][metric]
                if a is None or b is None:
                    continue
                w = paired_wilcoxon(a, b)
                ksd = ks_normal(a - b, ctx.alpha)
                trows.append([f"{a_name} vs {b_name}", _f(w["statistic"], 1),
                              _p(w["p_value"]), _f(w["effect_r"]),
                              _f(w["median_diff"], 4), w["n"],
                              "yes" if ksd["non_normal"] else "no"])
                _add_row(ctx, table=table, metric=metric, log=log,
                         comparison=f"{a_name} vs {b_name}", test="wilcoxon",
                         model_a=a_name, model_b=b_name, statistic=w["statistic"],
                         p_value=w["p_value"], effect_r=w["effect_r"],
                         effect_type="rank_biserial", n=w["n"],
                         n_runs=name2src[a_name]["_n_runs"],
                         non_normal=ksd["non_normal"],
                         note=f"median({a_name}-{b_name})={w['median_diff']:.4f}")
            print_table(ctx.emit, f"{log}  (n={name2src[v_names[0]]['_n']})",
                        ["comparison", "W", "p", "r_rb", "median(a-b)", "n", "diff!=normal"],
                        ["left", "right", "right", "right", "right", "right", "center"],
                        trows)
            per_log_mean.append([
                float(np.mean(name2src[n][metric])) if name2src[n][metric] is not None
                else float("nan") for n in v_names])

        _across_logs(ctx, ap, table, metric, np.array(per_log_mean, dtype=np.float64),
                     v_names, pairs)


def mode_ablation_input(ctx, ap):
    _ablation(ctx, ap, "Table8", ABLATION_INPUT,
              [("Graph_v1", "Seq_v1"),
               ("Graph_v1", "GraphFlip_v1"), ("Graph_v1", "GraphRandom_v1"),
               ("GraphFlip_v1", "GraphRandom_v1"),
               ("Seq_v1", "SeqFlip_v1"), ("Seq_v1", "SeqRandom_v1"),
               ("SeqFlip_v1", "SeqRandom_v1")])


def mode_ablation_train(ctx, ap):
    _ablation(ctx, ap, "Table9", ABLATION_TRAIN,
              [("v1", "v2"), ("v1", "v3"), ("v2", "v3")])


# ── Entry point ──────────────────────────────────────────────────────────────

class Ctx:
    pass


def _ablation_inputs_present(variants, runs):
    for sub, tpl in variants.values():
        p = os.path.join(_resolve(sub), f"run_{runs[0]}", tpl.format(log=LOGS[0]))
        if not os.path.isfile(p):
            return False
    return True


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["main", "time", "trend", "dispersion",
                                     "ablation-input", "ablation-train", "all"])
    ap.add_argument("--runs", default="1,2,3,4,5",
                    help="comma list of run numbers to seed-average (default 1,2,3,4,5)")
    ap.add_argument("--run", type=int, default=None,
                    help="use only this single run (overrides --runs)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--no-align-check", action="store_true",
                    help="skip the canonical dataset-order check "
                         "(the cross-source length check always runs)")
    ap.add_argument("--pog-root", default=POG_SUBDIR_DEFAULT)
    ap.add_argument("--baselines-root", default=BASELINES_ROOT_DEFAULT)
    ap.add_argument("--dataset-root", default=DATASET_ROOT_DEFAULT)
    ap.add_argument("--out-dir", default="approach_suffix_v2/significance_tests")
    args = ap.parse_args()

    ctx = Ctx()
    ctx.runs = [args.run] if args.run is not None else _parse_runs(args.runs)
    ctx.alpha = args.alpha
    ctx.align_check = not args.no_align_check
    ctx.pog_subdir = args.pog_root
    ctx.baselines_root = args.baselines_root
    ctx.dataset_root = args.dataset_root
    ctx.out_dir = _resolve(args.out_dir)
    os.makedirs(ctx.out_dir, exist_ok=True)
    tee = Tee()
    ctx.emit = tee
    ctx.rows = []

    ctx.emit(f"significance_tests.py  mode={args.mode}  runs={ctx.runs}  alpha={ctx.alpha}")
    ctx.emit("")

    if args.mode == "all":
        mode_main(ctx, ap)
        mode_time(ctx, ap)
        mode_trend(ctx, ap)
        mode_dispersion(ctx, ap)
        for label, variants, fn in (
            ("Table 8 (input format)", ABLATION_INPUT, mode_ablation_input),
            ("Table 9 (training strategy)", ABLATION_TRAIN, mode_ablation_train),
        ):
            if _ablation_inputs_present(variants, ctx.runs):
                fn(ctx, ap)
            else:
                ctx.emit(f"\n[skipped] {label}: per-sample files for the non-v1 "
                         f"variants are not present yet.")
    else:
        {
            "main": mode_main, "time": mode_time, "trend": mode_trend,
            "dispersion": mode_dispersion, "ablation-input": mode_ablation_input,
            "ablation-train": mode_ablation_train,
        }[args.mode](ctx, ap)

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
