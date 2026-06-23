import csv
import pandas as pd
import numpy as np
import os
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
from Preprocessing.from_log_to_tensors_sage import log_to_sage_graphs


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
        log[timestamp_col] = pd.to_datetime(log[timestamp_col], format=timestamp_format)
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
            if is_numeric:
                num_casefts.append(col)
            else:
                cat_casefts.append(col)
        else:
            if is_numeric:
                num_eventfts.append(col)
            else:
                cat_eventfts.append(col)

    return cat_casefts, num_casefts, cat_eventfts, num_eventfts


def plot_split(log, log_name, case_id='case:concept:name', timestamp='time:timestamp',
               test_len_share=0.20, mode='preferred', start_date=None,
               start_before_date=None, end_date=None, max_days=None,
               max_cases_shown=60, save_path=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.dates as mdates
    from Preprocessing.create_benchmarks import (
        start_from_date,
        end_before_date as _end_before_date,
        start_before_date_select,
        limited_duration,
    )

    df = log.copy()
    df[timestamp] = pd.to_datetime(df[timestamp], utc=True)
    if start_date:
        df = start_from_date(df, start_date, case_id, timestamp)
    if end_date:
        df = _end_before_date(df, end_date, case_id, timestamp)
    if start_before_date:
        df = start_before_date_select(df, start_before_date, case_id, timestamp)
    df.drop_duplicates(inplace=True, ignore_index=True)
    if max_days is not None:
        df = limited_duration(df, max_days, case_id, timestamp)

    case_starts = df.groupby(case_id)[timestamp].min().sort_values()
    case_ends   = df.groupby(case_id)[timestamp].max()
    n_cases      = len(case_starts)
    first_test_idx = int(n_cases * (1 - test_len_share))
    sep_time = case_starts.iloc[first_test_idx]

    def classify(cid):
        if case_ends[cid] < sep_time:
            return 'train'
        if case_starts[cid] >= sep_time:
            return 'test'
        return 'overlap'

    all_cases = list(case_starts.index)
    if len(all_cases) > max_cases_shown:
        step    = len(all_cases) / max_cases_shown
        sampled = [all_cases[int(i * step)] for i in range(max_cases_shown)]
    else:
        sampled = all_cases

    def to_naive(ts):
        ts = pd.Timestamp(ts)
        if ts.tzinfo is not None:
            ts = ts.tz_convert('UTC').tz_localize(None)
        return ts.to_pydatetime()

    sep_dt = to_naive(sep_time)

    BLUE  = '#5BC0DE'
    GREY  = '#808080'
    RED   = '#D9534F'
    GREEN = '#5CB85C'

    if mode == 'preferred':
        before_color, after_color = RED,   GREEN
        before_label = 'Overlap → discarded from train'
        after_label  = 'Overlap → goes to test'
    else:
        before_color, after_color = GREEN, RED
        before_label = 'Overlap → goes to train'
        after_label  = 'Overlap → discarded from test'

    fig, ax = plt.subplots(figsize=(12, 6))
    BAR_H = 0.7

    for y, cid in enumerate(sampled):
        s_dt = to_naive(case_starts[cid])
        e_dt = to_naive(case_ends[cid])
        cat  = classify(cid)

        s_num   = mdates.date2num(s_dt)
        e_num   = mdates.date2num(e_dt)
        sep_num = mdates.date2num(sep_dt)

        if cat == 'train':
            ax.barh(y, e_num - s_num, left=s_num, height=BAR_H, color=BLUE, linewidth=0)
        elif cat == 'test':
            ax.barh(y, e_num - s_num, left=s_num, height=BAR_H, color=GREY, linewidth=0)
        else:
            ax.barh(y, sep_num - s_num, left=s_num,   height=BAR_H, color=before_color, linewidth=0)
            ax.barh(y, e_num - sep_num, left=sep_num, height=BAR_H, color=after_color,  linewidth=0)

    ax.axvline(mdates.date2num(sep_dt), color='black', linestyle='--', linewidth=1.5)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=30, ha='right', fontsize=9)
    ax.set_yticks([])
    ax.set_ylabel('Cases', fontsize=12)
    ax.set_xlabel('Time', fontsize=12)
    mode_title = 'Preferred' if mode == 'preferred' else 'Workaround'
    ax.set_title(f'{mode_title} Train-Test Split — {log_name}', fontsize=14, fontweight='bold')

    handles = [
        mpatches.Patch(color=BLUE,         label='Train'),
        mpatches.Patch(color=GREY,         label='Test'),
        mpatches.Patch(color=before_color, label=before_label),
        mpatches.Patch(color=after_color,  label=after_label),
        plt.Line2D([0], [0], color='black', linestyle='--', label='Separation time'),
    ]
    ax.legend(handles=handles, loc='upper left', fontsize=9)
    plt.tight_layout()

    if save_path is None:
        out_dir   = os.path.join('results_per_log', log_name)
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, f'{log_name}_{mode}_split.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Split plot saved to '{save_path}'")


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
    plot=True,
):
    log = load_log(log_path)
    log = preprocess_log(log, timestamp_col=timestamp, timestamp_format=timestamp_format,
                         bool_cols=bool_cols, str_cols=str_cols)

    if plot:
        plot_split(log, log_name=log_name, case_id=case_id, timestamp=timestamp,
                   test_len_share=test_len_share, mode=mode, start_date=start_date,
                   start_before_date=start_before_date, end_date=end_date, max_days=max_days)

    if any(f is None for f in [cat_casefts, num_casefts, cat_eventfts, num_eventfts]):
        inferred     = infer_feature_columns(log, case_id, act_label, timestamp,
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
        tmp['_ts']  = ts
        durations   = tmp.groupby(case_id)['_ts'].agg(lambda x: (x.max() - x.min()).total_seconds())
        max_days    = float(durations.max() / (24 * 3600))
        print(f"Auto-derived max_days (maximum case duration): {max_days:.2f}")

    train_data, val_data, test_data, counts = log_to_sage_graphs(
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

    output_directory = os.path.join('results_per_log', log_name)
    os.makedirs(output_directory, exist_ok=True)

    torch.save(train_data, os.path.join(output_directory, 'train_sagedataset.pt'))
    torch.save(val_data,   os.path.join(output_directory, 'val_sagedataset.pt'))
    torch.save(test_data,  os.path.join(output_directory, 'test_sagedataset.pt'))

    # ts_start sits at index len(num_casefts) + len(num_eventfts) in x
    # (same as graph version; ts_prev follows immediately after)
    tss_index = len(num_casefts) + len(num_eventfts)
    print(f"Tensors saved to '{output_directory}/'")
    print(f"tss_index = {tss_index}")
    with open(os.path.join(output_directory, 'tss_index.txt'), 'w') as _f:
        _f.write(str(tss_index))

    N = len(test_data)
    conc_flags = []
    for data in test_data:
        k = data.x.shape[0]
        pref_has_conc = k >= 2 and (data.x[1:, tss_index] == data.x[:-1, tss_index]).any().item()

        eos_tok  = int(data.act_label_seq.max().item())
        real_suf = (data.act_label_seq > 0) & (data.act_label_seq < eos_tok)
        suf_has_conc = (real_suf & (data.new_block_label < 0.5)).any().item()

        conc_flags.append(pref_has_conc or suf_has_conc)
    conc_mask = torch.tensor(conc_flags)
    torch.save(conc_mask, os.path.join(output_directory, 'test_concurrent_mask.pt'))
    conc_count = conc_mask.sum().item()
    print(f"Concurrent-trace test samples: {conc_count} / {N}")
    counts['conc_count'] = conc_count

    return counts


def _run_one_log(args):
    log_path, log_name, kw = args
    try:
        counts = construct_datasets(log_path, log_name, **kw)
        return {'log': log_name, **counts, 'error': ''}
    except Exception as exc:
        return {'log': log_name, 'n_train': '', 'train_pairs': '',
                'n_val': '', 'val_pairs': '', 'n_test': '', 'test_pairs': '',
                'conc_count': '', 'error': str(exc)}


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

    workers   = min(n_workers or os.cpu_count(), len(files))
    args_list = [(path, name, kw) for path, name in files]

    fieldnames = ['log', 'n_train', 'train_pairs', 'n_val', 'val_pairs',
                  'n_test', 'test_pairs', 'conc_count', 'error']

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
                          f"test={r['n_test']} ({r['test_pairs']} pairs)  "
                          f"conc={r['conc_count']}")
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
        plot=PLOT,
    )
