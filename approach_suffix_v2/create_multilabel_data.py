"""Generate multilabel block-grouped datasets for a log.

Usage
-----
Edit the variables at the bottom and run:
    python create_multilabel_data.py

Outputs (to results_per_log/<LOG_NAME>/):
    train_multilabel_dataset.pt
    val_multilabel_dataset.pt
    test_multilabel_dataset.pt
    <LOG_NAME>_T_max.pkl
    <LOG_NAME>_cardin_list_prefix.pkl
    <LOG_NAME>_train_means_dict.pkl  (written by the dataframe pipeline)
    <LOG_NAME>_train_std_dict.pkl    (written by the dataframe pipeline)
"""
import csv
import os
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
from approach_suffix_v2.create_graph_data import (load_log, preprocess_log,
                                  infer_feature_columns, plot_split)
from Preprocessing.from_log_to_tensors_multilabel import log_to_graphs_multilabel


def construct_multilabel_datasets(
    log_path,
    log_name,
    case_id='case:concept:name',
    act_label='concept:name',
    timestamp='time:timestamp',
    timestamp_format=None,
    bool_cols=None,
    str_cols=None,
    cat_casefts=None,
    num_casefts=None,
    cat_eventfts=None,
    num_eventfts=None,
    outcome=None,
    start_date=None,
    start_before_date=None,
    end_date=None,
    max_days=None,
    test_len_share=0.20,
    val_len_share=0.20,
    window_size=None,
    mode='preferred',
    plot=True,
):
    log = load_log(log_path)
    log = preprocess_log(log, timestamp_col=timestamp,
                         timestamp_format=timestamp_format,
                         bool_cols=bool_cols, str_cols=str_cols)

    if plot:
        plot_split(log, log_name=log_name, case_id=case_id, timestamp=timestamp,
                   test_len_share=test_len_share, mode=mode,
                   start_date=start_date, start_before_date=start_before_date,
                   end_date=end_date, max_days=max_days)

    if any(f is None for f in [cat_casefts, num_casefts, cat_eventfts, num_eventfts]):
        inferred     = infer_feature_columns(log, case_id, act_label, timestamp,
                                             exclude_cols=[outcome] if outcome else None)
        cat_casefts  = cat_casefts  if cat_casefts  is not None else inferred[0]
        num_casefts  = num_casefts  if num_casefts  is not None else inferred[1]
        cat_eventfts = cat_eventfts if cat_eventfts is not None else inferred[2]
        num_eventfts = num_eventfts if num_eventfts is not None else inferred[3]

    if window_size is None:
        import pandas as pd
        case_lengths = log.groupby(case_id).size()
        window_size  = int(np.percentile(case_lengths, 98.5))
        print(f"Auto-derived window_size: {window_size}")

    if max_days is None:
        import pandas as pd
        ts  = pd.to_datetime(log[timestamp], utc=True)
        tmp = log.copy()
        tmp['_ts'] = ts
        durations = tmp.groupby(case_id)['_ts'].agg(
            lambda x: (x.max() - x.min()).total_seconds())
        max_days = float(durations.max() / (24 * 3600))

    train_data, val_data, test_data, counts = log_to_graphs_multilabel(
        log, log_name=log_name,
        start_date=start_date, start_before_date=start_before_date,
        end_date=end_date, max_days=max_days,
        test_len_share=test_len_share, val_len_share=val_len_share,
        window_size=window_size, mode=mode,
        case_id=case_id, act_label=act_label, timestamp=timestamp,
        cat_casefts=cat_casefts, num_casefts=num_casefts,
        cat_eventfts=cat_eventfts, num_eventfts=num_eventfts,
        outcome=outcome,
    )

    output_dir = os.path.join('results_per_log', log_name)
    os.makedirs(output_dir, exist_ok=True)

    torch.save(train_data, os.path.join(output_dir, 'train_multilabel_dataset.pt'))
    torch.save(val_data,   os.path.join(output_dir, 'val_multilabel_dataset.pt'))
    torch.save(test_data,  os.path.join(output_dir, 'test_multilabel_dataset.pt'))

    print(f"Datasets saved to '{output_dir}/'")
    return counts


# ─────────────────────────────────────────────
# Batch runner (all logs in a folder)
# ─────────────────────────────────────────────

def _run_one_log(args):
    """Top-level worker for ProcessPoolExecutor: run one log, return counts."""
    log_path, log_name, kw = args
    try:
        counts = construct_multilabel_datasets(log_path, log_name, **kw)
        return {'log': log_name, **counts, 'error': ''}
    except Exception as exc:
        return {'log': log_name, 'n_train': '', 'train_pairs': '',
                'n_val': '', 'val_pairs': '', 'n_test': '', 'test_pairs': '',
                'error': str(exc)}


def run_all_logs(folder, output_file,
                 case_id='case:concept:name',
                 act_label='concept:name',
                 timestamp='time:timestamp',
                 timestamp_format=None,
                 bool_cols=None,
                 str_cols=None,
                 cat_casefts=None,
                 num_casefts=None,
                 cat_eventfts=None,
                 num_eventfts=None,
                 outcome=None,
                 start_date=None,
                 start_before_date=None,
                 end_date=None,
                 max_days=None,
                 test_len_share=0.20,
                 val_len_share=0.20,
                 window_size=None,
                 mode='preferred',
                 plot=False,
                 n_workers=None):
    """Run construct_multilabel_datasets for every log in *folder* and write a summary CSV.

    Parameters
    ----------
    folder : str
        Directory containing .xes, .xes.gz, or .csv event-log files.
    output_file : str
        Path for the output CSV (created or overwritten).
    n_workers : int or None
        Number of parallel worker processes. None = os.cpu_count().
    """
    _SUPPORTED_EXT = {'.xes', '.gz', '.csv'}

    def _stem(fname):
        for ext in ('.xes.gz', '.xes', '.csv'):
            if fname.endswith(ext):
                return fname[:-len(ext)]
        return os.path.splitext(fname)[0]

    files = sorted(
        (os.path.join(folder, f), _stem(f))
        for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
        and os.path.splitext(f)[1].lower() in _SUPPORTED_EXT
    )
    if not files:
        print(f"No log files found in '{folder}'.")
        return

    kw = dict(case_id=case_id, act_label=act_label, timestamp=timestamp,
              timestamp_format=timestamp_format, bool_cols=bool_cols,
              str_cols=str_cols, cat_casefts=cat_casefts, num_casefts=num_casefts,
              cat_eventfts=cat_eventfts, num_eventfts=num_eventfts, outcome=outcome,
              start_date=start_date, start_before_date=start_before_date,
              end_date=end_date, max_days=max_days, test_len_share=test_len_share,
              val_len_share=val_len_share, window_size=window_size, mode=mode,
              plot=plot)

    workers = min(n_workers or os.cpu_count(), len(files))
    args_list = [(path, name, kw) for path, name in files]

    fieldnames = ['log', 'n_train', 'train_pairs', 'n_val', 'val_pairs',
                  'n_test', 'test_pairs', 'error']

    print(f"Processing {len(files)} logs with {workers} workers ...")
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_run_one_log, args): args[1] for args in args_list}
            for future in as_completed(futures):
                r = future.result()
                if r['error']:
                    print(f"  ERROR  {r['log']}: {r['error']}")
                else:
                    print(f"  {r['log']}  "
                          f"train={r['n_train']} ({r['train_pairs']} pairs)  "
                          f"val={r['n_val']} ({r['val_pairs']} pairs)  "
                          f"test={r['n_test']} ({r['test_pairs']} pairs)")
                writer.writerow(r)
                f.flush()

    print(f"\nSummary written to '{output_file}'.")


if __name__ == '__main__':
    LOG_PATH   = 'my_log.xes'
    LOG_NAME   = 'my_log'

    CASE_ID    = 'case:concept:name'
    ACT_LABEL  = 'concept:name'
    TIMESTAMP  = 'time:timestamp'

    CAT_CASEFTS  = None
    NUM_CASEFTS  = None
    CAT_EVENTFTS = None
    NUM_EVENTFTS = None

    START_DATE        = None
    START_BEFORE_DATE = None
    END_DATE          = None
    MAX_DAYS          = None
    WINDOW_SIZE       = None
    TEST_LEN_SHARE    = 0.20
    VAL_LEN_SHARE     = 0.20
    MODE              = 'preferred'
    OUTCOME           = None
    PLOT              = True

    construct_multilabel_datasets(
        log_path=LOG_PATH,
        log_name=LOG_NAME,
        case_id=CASE_ID,
        act_label=ACT_LABEL,
        timestamp=TIMESTAMP,
        cat_casefts=CAT_CASEFTS,
        num_casefts=NUM_CASEFTS,
        cat_eventfts=CAT_EVENTFTS,
        num_eventfts=NUM_EVENTFTS,
        outcome=OUTCOME,
        start_date=START_DATE,
        start_before_date=START_BEFORE_DATE,
        end_date=END_DATE,
        max_days=MAX_DAYS,
        window_size=WINDOW_SIZE,
        test_len_share=TEST_LEN_SHARE,
        val_len_share=VAL_LEN_SHARE,
        mode=MODE,
        plot=PLOT,
    )
