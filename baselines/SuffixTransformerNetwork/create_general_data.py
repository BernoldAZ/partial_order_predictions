import pandas as pd
import numpy as np
import os
import torch
from Preprocessing.from_log_to_tensors import log_to_tensors


def load_log(log_path):
    """Load an event log from a XES or CSV file.

    Parameters
    ----------
    log_path : str
        Path to the event log file. Must have a .xes or .csv extension.

    Returns
    -------
    log : pandas.DataFrame
        Event log as a DataFrame. For XES files, columns follow the
        standard pm4py naming convention (e.g. 'case:concept:name',
        'concept:name', 'time:timestamp').
    """
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
    """Generic preprocessing for an event log DataFrame.

    Converts the timestamp column to a timezone-aware datetime, and
    optionally coerces selected columns to string dtype (e.g. IDs or
    boolean flags that should be treated as categorical).

    Parameters
    ----------
    log : pandas.DataFrame
        Raw event log.
    timestamp_col : str, optional
        Name of the timestamp column. Default 'time:timestamp'.
    timestamp_format : str or None, optional
        strptime format string for the timestamp column. Pass None to let
        pandas infer the format (recommended for most logs).
    bool_cols : list of str or None, optional
        Columns containing boolean values that should be cast to str so
        they are treated as categorical features downstream.
    str_cols : list of str or None, optional
        Columns that should be cast to str dtype (e.g. numeric IDs).

    Returns
    -------
    log : pandas.DataFrame
        Preprocessed event log.
    """
    # Convert timestamp to UTC-aware datetime
    if timestamp_format is not None:
        log[timestamp_col] = pd.to_datetime(
            log[timestamp_col], format=timestamp_format
        )
    else:
        log[timestamp_col] = pd.to_datetime(log[timestamp_col], utc=True)

    # Make timezone-aware if not already (assume UTC)
    if log[timestamp_col].dt.tz is None:
        log[timestamp_col] = log[timestamp_col].dt.tz_localize('UTC')
    else:
        log[timestamp_col] = log[timestamp_col].dt.tz_convert('UTC')

    # Coerce user-specified columns to string
    for col in (str_cols or []):
        log[col] = log[col].astype('str')
    for col in (bool_cols or []):
        log[col] = log[col].astype('str')

    return log


def infer_feature_columns(log, case_id, act_label, timestamp,
                          exclude_cols=None):
    """Automatically infer categorical and numeric feature columns.

    Splits all remaining columns (after removing the mandatory columns and
    any explicitly excluded ones) into categorical case features, numeric
    case features, categorical event features, and numeric event features
    based on column name prefixes and dtype.

    Convention (pm4py / XES standard):
      - Columns prefixed with 'case:' are case-level attributes.
      - All other attribute columns are treated as event-level attributes.

    Parameters
    ----------
    log : pandas.DataFrame
    case_id : str
    act_label : str
    timestamp : str
    exclude_cols : list of str or None
        Additional columns to exclude from feature detection.

    Returns
    -------
    cat_casefts : list of str
    num_casefts : list of str
    cat_eventfts : list of str
    num_eventfts : list of str
    """
    mandatory = {case_id, act_label, timestamp}
    if exclude_cols:
        mandatory.update(exclude_cols)

    cat_casefts, num_casefts, cat_eventfts, num_eventfts = [], [], [], []

    for col in log.columns:
        if col in mandatory:
            continue
        is_case = col.startswith('case:')
        dtype = log[col].dtype
        is_numeric = pd.api.types.is_numeric_dtype(dtype)

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


def plot_split(
    log,
    log_name,
    case_id='case:concept:name',
    timestamp='time:timestamp',
    test_len_share=0.25,
    mode='preferred',
    start_date=None,
    start_before_date=None,
    end_date=None,
    max_days=None,
    max_cases_shown=60,
    save_path=None,
):
    """Visualise the train/test split like the PreferredSplit / WorkaroundSplit
    reference figures.  Each case is drawn as a horizontal bar from its first
    to its last event.  Bar colors encode how the case is partitioned relative
    to the separation time:

    Both modes
    ----------
    Blue  : case fully ends before separation time  → pure training case
    Grey  : case fully starts at/after sep. time    → pure test case

    mode='preferred'
    ----------------
    Red   : overlap portion *before* sep. time (discarded from training)
    Green : overlap portion *after*  sep. time (goes to test)

    mode='workaround'
    -----------------
    Green : overlap portion *before* sep. time (goes to training)
    Red   : overlap portion *after*  sep. time (discarded from test)

    Parameters
    ----------
    log : pd.DataFrame
        Event log, already preprocessed (timestamp column is datetime).
    log_name : str
        Used for the plot title and default file name.
    case_id, timestamp : str
    test_len_share : float
    mode : {'preferred', 'workaround'}
    start_date, start_before_date, end_date : str or None
        Same filtering options as construct_datasets; needed to reproduce the
        exact separation time used during preprocessing.
    max_days : float or None
    max_cases_shown : int
        Max number of cases to draw (uniformly sampled from the chronologically
        sorted list to keep the figure readable).
    save_path : str or None
        Output PNG path.  Defaults to ``<log_name>/<log_name>_<mode>_split.png``.
    """
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

    # ------------------------------------------------------------------
    # 1. Apply the same pre-filtering as remainTimeOrClassifBenchmark
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 2. Compute the separation time (mirrors trainTestSplit logic)
    # ------------------------------------------------------------------
    case_starts = df.groupby(case_id)[timestamp].min().sort_values()
    case_ends   = df.groupby(case_id)[timestamp].max()
    n_cases = len(case_starts)
    first_test_idx = int(n_cases * (1 - test_len_share))
    sep_time = case_starts.iloc[first_test_idx]

    # ------------------------------------------------------------------
    # 3. Classify every case
    # ------------------------------------------------------------------
    def classify(cid):
        if case_ends[cid] < sep_time:
            return 'train'
        if case_starts[cid] >= sep_time:
            return 'test'
        return 'overlap'

    # ------------------------------------------------------------------
    # 4. Sample cases uniformly across the chronologically sorted list
    # ------------------------------------------------------------------
    all_cases = list(case_starts.index)
    if len(all_cases) > max_cases_shown:
        step = len(all_cases) / max_cases_shown
        sampled = [all_cases[int(i * step)] for i in range(max_cases_shown)]
    else:
        sampled = all_cases

    # ------------------------------------------------------------------
    # 5. Convert timestamps to naive datetimes for matplotlib
    # ------------------------------------------------------------------
    def to_naive(ts):
        ts = pd.Timestamp(ts)
        if ts.tzinfo is not None:
            ts = ts.tz_convert('UTC').tz_localize(None)
        return ts.to_pydatetime()

    sep_dt = to_naive(sep_time)

    # ------------------------------------------------------------------
    # 6. Draw the figure
    # ------------------------------------------------------------------
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

        s_num = mdates.date2num(s_dt)
        e_num = mdates.date2num(e_dt)
        sep_num = mdates.date2num(sep_dt)

        if cat == 'train':
            ax.barh(y, e_num - s_num, left=s_num, height=BAR_H,
                    color=BLUE, linewidth=0)
        elif cat == 'test':
            ax.barh(y, e_num - s_num, left=s_num, height=BAR_H,
                    color=GREY, linewidth=0)
        else:  # overlap
            ax.barh(y, sep_num - s_num, left=s_num, height=BAR_H,
                    color=before_color, linewidth=0)
            ax.barh(y, e_num - sep_num, left=sep_num, height=BAR_H,
                    color=after_color,  linewidth=0)

    # Vertical dashed separation line
    ax.axvline(mdates.date2num(sep_dt), color='black', linestyle='--', linewidth=1.5)

    # Axis formatting
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=30, ha='right', fontsize=9)
    ax.set_yticks([])
    ax.set_ylabel('Cases', fontsize=12)
    ax.set_xlabel('Time', fontsize=12)
    mode_title = 'Preferred' if mode == 'preferred' else 'Workaround'
    ax.set_title(f'{mode_title} Train-Test Split — {log_name}',
                 fontsize=14, fontweight='bold')

    # Legend
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
        out_dir = os.path.join('results_per_log', log_name)
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, f'{log_name}_{mode}_split.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Split plot saved to '{save_path}'")


def construct_datasets(
    log_path,
    log_name,
    # Mandatory column names
    case_id='case:concept:name',
    act_label='concept:name',
    timestamp='time:timestamp',
    # Preprocessing options
    timestamp_format=None,
    bool_cols=None,
    str_cols=None,
    # Feature specification (None = auto-detect)
    cat_casefts=None,
    num_casefts=None,
    cat_eventfts=None,
    num_eventfts=None,
    outcome=None,
    # Splitting / filtering parameters
    start_date=None,
    start_before_date=None,
    end_date=None,
    max_days=None,
    test_len_share=0.25,
    val_len_share=0.20,
    window_size=None,
    mode='preferred',
    plot=True,
):
    """Load, preprocess, and convert an event log into train/val/test tensors.

    Parameters
    ----------
    log_path : str
        Path to the event log (.xes or .csv).
    log_name : str
        Identifier for the log; used as the output directory name and in
        intermediate file names written by the preprocessing pipeline.
    case_id : str
        Column name for case identifiers. Default 'case:concept:name'.
    act_label : str
        Column name for activity labels. Default 'concept:name'.
    timestamp : str
        Column name for event timestamps. Default 'time:timestamp'.
    timestamp_format : str or None
        strptime format for timestamp parsing. None = auto-detect.
    bool_cols : list of str or None
        Boolean columns to cast to str before feature detection.
    str_cols : list of str or None
        Columns to cast to str before feature detection (e.g. numeric IDs).
    cat_casefts : list of str or None
        Categorical case features. None = auto-detect from column prefixes
        and dtypes.
    num_casefts : list of str or None
        Numeric case features. None = auto-detect.
    cat_eventfts : list of str or None
        Categorical event features. None = auto-detect.
    num_eventfts : list of str or None
        Numeric event features. None = auto-detect.
    outcome : str or None
        Name of a binary outcome column, or None if not applicable.
    start_date : str or None
        "YYYY-MM" — discard cases starting before this month.
    start_before_date : str or None
        "YYYY-MM" — discard cases starting after this month.
    end_date : str or None
        "YYYY-MM" — discard cases ending after this month.
    max_days : float or None
        Maximum case duration in days; longer cases are discarded.
    test_len_share : float
        Fraction of cases assigned to the test set (chronological split).
    val_len_share : float
        Fraction of training cases assigned to the validation set.
    window_size : int or None
        Maximum sequence length (number of events per case). Cases longer
        than this are discarded. If None, the 98.5th percentile of case
        lengths is used automatically.
    mode : {'workaround', 'preferred'}
        Train-test split mode; see `log_to_tensors` for details.
    plot : bool
        If True (default), save a train/test split visualisation to
        ``<log_name>/<log_name>_<mode>_split.png`` before running the
        full preprocessing pipeline.
    """
    # 1. Load
    log = load_log(log_path)

    # 2. Preprocess
    log = preprocess_log(
        log,
        timestamp_col=timestamp,
        timestamp_format=timestamp_format,
        bool_cols=bool_cols,
        str_cols=str_cols,
    )

    # 2b. Plot the train/test split before heavy preprocessing
    if plot:
        plot_split(
            log,
            log_name=log_name,
            case_id=case_id,
            timestamp=timestamp,
            test_len_share=test_len_share,
            mode=mode,
            start_date=start_date,
            start_before_date=start_before_date,
            end_date=end_date,
            max_days=max_days,
        )

    # 3. Infer features if not provided
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

    # 4. Derive window_size from case-length distribution if not provided
    if window_size is None:
        case_lengths = log.groupby(case_id).size()
        window_size = int(np.percentile(case_lengths, 98.5))
        print(f"Auto-derived window_size (98.5th percentile): {window_size}")

    # 5a. Derive max_days if not provided.
    # limited_duration() in create_benchmarks.py always runs and does
    # `max_days * 1.00000000001`, so None would cause a TypeError.
    # Using the actual maximum duration means no cases are filtered out.
    if max_days is None:
        ts = pd.to_datetime(log[timestamp], utc=True)
        tmp = log.copy()
        tmp['_ts'] = ts
        durations = tmp.groupby(case_id)['_ts'].agg(lambda x: (x.max() - x.min()).total_seconds())
        max_days = float(durations.max() / (24 * 3600))
        print(f"Auto-derived max_days (maximum case duration): {max_days:.2f}")

    # 5b. Run the full preprocessing pipeline
    result = log_to_tensors(
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

    train_data, val_data, test_data = result

    # 6. Unpack and save tensors
    output_directory = os.path.join('results_per_log', log_name)
    os.makedirs(output_directory, exist_ok=True)

    torch.save(train_data, os.path.join(output_directory, 'train_tensordataset.pt'))
    torch.save(val_data,   os.path.join(output_directory, 'val_tensordataset.pt'))
    torch.save(test_data,  os.path.join(output_directory, 'test_tensordataset.pt'))

    tss_index = len(num_casefts) + len(num_eventfts)
    print(f"Tensors saved to '{output_directory}/'")
    print(f"tss_index = {tss_index}  "
          f"(= {len(num_casefts)} num_casefts + {len(num_eventfts)} num_eventfts)")
    print("Use this value for the tss_index parameter in TRAIN_EVAL_*.py scripts.")


if __name__ == '__main__':
    # ------------------------------------------------------------------ #
    # Edit the variables below to match your event log.                   #
    # ------------------------------------------------------------------ #

    LOG_PATH   = 'my_log.xes'   # or 'my_log.csv'
    LOG_NAME   = 'my_log'

    # Standard XES column names — change if your CSV uses different names.
    CASE_ID    = 'case:concept:name'
    ACT_LABEL  = 'concept:name'
    TIMESTAMP  = 'time:timestamp'

    # Set to None to auto-detect from column prefixes / dtypes, or provide
    # explicit lists as in the log-specific create_*.py files.
    CAT_CASEFTS  = None
    NUM_CASEFTS  = None
    CAT_EVENTFTS = None
    NUM_EVENTFTS = None

    # Filtering / splitting parameters — adjust to your log's date range
    # and case-length distribution.
    START_DATE        = None   # e.g. "2018-01"
    START_BEFORE_DATE = None   # e.g. "2018-09"
    END_DATE          = None   # e.g. "2019-02"
    MAX_DAYS          = None   # e.g. 143.33
    WINDOW_SIZE       = None   # None → auto (98.5th percentile)
    TEST_LEN_SHARE    = 0.25
    VAL_LEN_SHARE     = 0.20
    MODE              = 'preferred'
    OUTCOME           = None
    PLOT              = True    # set to False to skip the split visualisation

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
