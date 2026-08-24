import csv
import pandas as pd
import numpy as np
import os
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
from Preprocessing.from_log_to_tensors_seq_graph_flip import log_to_seq_graphs_flip


def load_log(log_path):
    ext = os.path.splitext(log_path)[1].lower()
    if ext == '.xes' or ext == '.gz':
        from pm4py.objects.log.importer.xes import importer as xes_importer
        from pm4py.objects.conversion.log import converter
        event_log = xes_importer.apply(log_path)
        log = converter.apply(event_log, variant=converter.Variants.TO_DATA_FRAME)
    elif ext == '.csv':
        log = pd.read_csv(log_path)
    else:
        raise ValueError(f"Unsupported file format '{ext}'. Use '.xes' or '.csv'.")
    return log


def preprocess_log(log, timestamp_col='time:timestamp', timestamp_format=None,
                   bool_cols=None, str_cols=None):
    if timestamp_format is not None:
        log[timestamp_col] = pd.to_datetime(
            log[timestamp_col], format=timestamp_format
        )
    else:
        log[timestamp_col] = pd.to_datetime(log[timestamp_col], utc=True)

    if log[timestamp_col].dt.tz is None:
        log[timestamp_col] = log[timestamp_col].dt.tz_localize('UTC')
    else:
        log[timestamp_col] = log[timestamp_col].dt.tz_convert('UTC')

    for col in (str_cols or []):
        log[col] = log[col].astype('str')
    for col in (bool_cols or []):
        log[col] = log[col].astype('str')

    return log


def infer_feature_columns(log, case_id, act_label, timestamp, exclude_cols=None):
    mandatory = {case_id, act_label, timestamp}
    if exclude_cols:
        mandatory.update(exclude_cols)

    cat_casefts, num_casefts, cat_eventfts, num_eventfts = [], [], [], []

    for col in log.columns:
        if col in mandatory:
            continue
        is_case    = col.startswith('case:')
        is_numeric = pd.api.types.is_numeric_dtype(log[col].dtype)
        if is_case:
            (num_casefts if is_numeric else cat_casefts).append(col)
        else:
            (num_eventfts if is_numeric else cat_eventfts).append(col)

    return cat_casefts, num_casefts, cat_eventfts, num_eventfts


def construct_datasets(
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
):
    log = load_log(log_path)
    log = preprocess_log(
        log,
        timestamp_col=timestamp,
        timestamp_format=timestamp_format,
        bool_cols=bool_cols,
        str_cols=str_cols,
    )

    if any(f is None for f in [cat_casefts, num_casefts, cat_eventfts, num_eventfts]):
        inferred = infer_feature_columns(log, case_id, act_label, timestamp,
                                         exclude_cols=[outcome] if outcome else None)
        cat_casefts  = cat_casefts  if cat_casefts  is not None else inferred[0]
        num_casefts  = num_casefts  if num_casefts  is not None else inferred[1]
        cat_eventfts = cat_eventfts if cat_eventfts is not None else inferred[2]
        num_eventfts = num_eventfts if num_eventfts is not None else inferred[3]

        print("Auto-detected features:")
        print(f"  cat_casefts  : {cat_casefts}")
        print(f"  num_casefts  : {num_casefts}")
        print(f"  cat_eventfts : {cat_eventfts}")
        print(f"  num_eventfts : {num_eventfts}")

    if window_size is None:
        case_lengths = log.groupby(case_id).size()
        window_size  = int(np.percentile(case_lengths, 98.5))
        print(f"Auto-derived window_size (98.5th percentile): {window_size}")

    if max_days is None:
        ts  = pd.to_datetime(log[timestamp], utc=True)
        tmp = log.copy()
        tmp['_ts'] = ts
        durations = tmp.groupby(case_id)['_ts'].agg(
            lambda x: (x.max() - x.min()).total_seconds())
        max_days = float(durations.max() / (24 * 3600))
        print(f"Auto-derived max_days (maximum case duration): {max_days:.2f}")

    counts = log_to_seq_graphs_flip(
        log,
        log_name=log_name,
        start_date=start_date,
        start_before_date=start_before_date,
        end_date=end_date,
        max_days=max_days,
        test_len_share=test_len_share,
        val_len_share=val_len_share,
        window_size=window_size,
        mode=mode,
        case_id=case_id,
        act_label=act_label,
        timestamp=timestamp,
        cat_casefts=cat_casefts,
        num_casefts=num_casefts,
        cat_eventfts=cat_eventfts,
        num_eventfts=num_eventfts,
        outcome=outcome,
    )

    tss_index = len(num_casefts) + len(num_eventfts)
    print(f"tss_index = {tss_index}")

    return counts


# ─────────────────────────────────────────────
# Batch runner (all logs in a folder)
# ─────────────────────────────────────────────

def _run_one_log(args):
    log_path, log_name, kw = args
    try:
        counts = construct_datasets(log_path, log_name, **kw)
        return {'log': log_name, **counts, 'error': ''}
    except Exception as exc:
        return {'log': log_name, 'n_test': '', 'test_pairs': '', 'error': str(exc)}


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
                 n_workers=None):
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
              val_len_share=val_len_share, window_size=window_size, mode=mode)

    workers    = min(n_workers or os.cpu_count(), len(files))
    args_list  = [(path, name, kw) for path, name in files]
    fieldnames = ['log', 'n_test', 'test_pairs', 'error']

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
                    print(f"  {r['log']}  test={r['n_test']} ({r['test_pairs']} pairs)")
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

    construct_datasets(
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
    )
