"""Utility for collecting next-activity-prediction results across all models."""

import os
import pandas as pd
from definitions import ROOT_DIR

NA_DIR = os.path.join(ROOT_DIR, "evaluation")

_DEFAULT_DIRS = {
    "tax":      os.path.join(NA_DIR, "results_tax"),
    "everman":  os.path.join(NA_DIR, "results_everman"),
    "pydream":  os.path.join(NA_DIR, "results_pydream"),
}


def _collect_tax(log_name, results_dir):
    """Return a DataFrame of tax results for *log_name*.

    Tries the aggregate file ``<results_dir>/<log_name>_results.csv`` first;
    falls back to scanning per-method sub-folders.
    """
    agg_path = os.path.join(results_dir, f"{log_name}_results.csv")
    if os.path.isfile(agg_path):
        df = pd.read_csv(agg_path)
        df["model"] = "tax"
        return df

    # Fallback: walk <results_dir>/<log_name>/<method>/
    rows = []
    log_dir = os.path.join(results_dir, log_name)
    if not os.path.isdir(log_dir):
        return pd.DataFrame()
    for method in os.listdir(log_dir):
        method_dir = os.path.join(log_dir, method)
        if not os.path.isdir(method_dir):
            continue
        csv_name = f"{log_name}_{method}_results.csv"
        csv_path = os.path.join(method_dir, csv_name)
        if os.path.isfile(csv_path):
            rows.append(pd.read_csv(csv_path))
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    df["model"] = "tax"
    return df


def _collect_everman(log_name, results_dir):
    """Return a DataFrame of everman results for *log_name*.

    Scans ``<results_dir>/<log_name>/<method>/`` for per-method CSV files.
    """
    rows = []
    log_dir = os.path.join(results_dir, log_name)
    if not os.path.isdir(log_dir):
        return pd.DataFrame()
    for method in os.listdir(log_dir):
        method_dir = os.path.join(log_dir, method)
        if not os.path.isdir(method_dir):
            continue
        csv_name = f"{log_name}_{method}_results.csv"
        csv_path = os.path.join(method_dir, csv_name)
        if os.path.isfile(csv_path):
            rows.append(pd.read_csv(csv_path))
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    df["model"] = "everman"
    return df


def _collect_pydream(log_name, results_dir):
    """Return a DataFrame of pydream results for *log_name*."""
    csv_path = os.path.join(results_dir, log_name, f"{log_name}_results.csv")
    if not os.path.isfile(csv_path):
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    df["model"] = "pydream"
    # Align column names with the other models.
    df = df.rename(columns={
        "weighted_recall":    "recall",
        "weighted_precision": "precision",
    })
    return df


def get_results_for_log(
    log_name,
    tax_results_dir=None,
    everman_results_dir=None,
    pydream_results_dir=None,
):
    """Collect next-activity-prediction results for *log_name* from all models.

    Reads the result CSV files written by ``run_tax_for_log``,
    ``run_everman_for_log``, and ``run_pydream_for_log`` (or their
    ``__main__`` equivalents) and returns a single combined DataFrame.

    Parameters
    ----------
    log_name : str
        Base name of the event log (e.g. ``'bpic2019'``).
    tax_results_dir : str or None
        Override the default ``results_tax`` directory.
    everman_results_dir : str or None
        Override the default ``results_everman`` directory.
    pydream_results_dir : str or None
        Override the default ``results_pydream`` directory.

    Returns
    -------
    pd.DataFrame
        One row per (model, method) combination.  Columns present depend on
        which metrics each model records, but always include at least
        ``model``, ``log``, ``accuracy``, and ``f1``.  A ``method`` column
        is set to ``'NAP'`` for pydream (which has no embedding variants).
        Rows are sorted by ``model`` then ``method``.

    Examples
    --------
    >>> from results_collector import get_results_for_log
    >>> df = get_results_for_log('bpic2019')
    >>> df[['model', 'method', 'accuracy', 'f1']]
    """
    tax_dir     = tax_results_dir     or _DEFAULT_DIRS["tax"]
    everman_dir = everman_results_dir or _DEFAULT_DIRS["everman"]
    pydream_dir = pydream_results_dir or _DEFAULT_DIRS["pydream"]

    parts = []

    df_tax = _collect_tax(log_name, tax_dir)
    if not df_tax.empty:
        parts.append(df_tax)
    else:
        print(f"[results_collector] No tax results found for '{log_name}' in '{tax_dir}'")

    df_everman = _collect_everman(log_name, everman_dir)
    if not df_everman.empty:
        parts.append(df_everman)
    else:
        print(f"[results_collector] No everman results found for '{log_name}' in '{everman_dir}'")

    df_pydream = _collect_pydream(log_name, pydream_dir)
    if not df_pydream.empty:
        # pydream has no embedding method — label it explicitly.
        if "method" not in df_pydream.columns:
            df_pydream.insert(df_pydream.columns.get_loc("model") + 1, "method", "NAP")
        parts.append(df_pydream)
    else:
        print(f"[results_collector] No pydream results found for '{log_name}' in '{pydream_dir}'")

    if not parts:
        print(f"[results_collector] No results found for log '{log_name}' in any model.")
        return pd.DataFrame()

    df = pd.concat(parts, ignore_index=True)

    # Put model and method first, then sort.
    first_cols = [c for c in ("model", "method", "log") if c in df.columns]
    rest_cols  = [c for c in df.columns if c not in first_cols]
    df = df[first_cols + rest_cols]
    df = df.sort_values(["model", "method"], ignore_index=True)
    return df
