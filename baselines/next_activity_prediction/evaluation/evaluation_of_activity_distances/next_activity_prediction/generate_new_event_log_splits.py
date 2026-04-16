"""Generate train / val / test XES splits for next-activity prediction
using the DyLoPro-style temporal out-of-time split.

Reference
---------
B. Wuyts, H. Weytjens, S. vanden Broucke, and J. De Weerdt,
"DyLoPro: Profiling the dynamics of event logs," in Business Process
Management, C. Di Francescomarino, A. Burattin, C. Janiesch, and S. Sadiq,
Eds.  Cham: Springer Nature Switzerland, 2023, pp. 146–162.

This script mirrors the split strategy used in
SuffixTransformerNetwork/create_generic_data.py so that next-activity
prediction and suffix prediction share identical data partitions and
results are directly comparable.

Split logic
-----------
The separation time (sep_time) is determined identically to
trainTestSplit() in SuffixTransformerNetwork/Preprocessing/create_benchmarks.py:

    sep_time = start time of the case at position
               floor(N_cases × (1 − TEST_LEN_SHARE))
               in the chronologically sorted case list.

Overlap handling is performed at the *case* level so the exported XES
files are leakage-free without requiring changes to the downstream NAP
pipeline:

mode='workaround'  (default, matches create_generic_data.py default)
    Test  : cases whose first event is at or after sep_time (no overlap).
    Train : cases whose first event is before sep_time. Overlapping cases
            (start before, end at/after sep_time) are TRUNCATED so that
            only events occurring strictly before sep_time are retained.
            This mirrors the effect of last_prefix_dict in the suffix task:
            only prefixes fully within the training window are generated.

mode='preferred'
    Train : cases whose last event is strictly before sep_time (no overlap).
    Test  : cases whose first event is at or after sep_time (no overlap).
            Overlapping cases are excluded from both sets, consistent with
            keeping pure splits that need no downstream prefix filtering.

Validation split
----------------
The validation set is carved from the training portion using the same
temporal logic (val_sep_time derived from the training sub-log with
VAL_LEN_SHARE). Default proportions reproduce the original 64 / 16 / 20
split: TEST_LEN_SHARE=0.20, VAL_LEN_SHARE=0.20 gives train=64 %, val=16 %,
test=20 %.
"""

import os

import pandas as pd
from pm4py.objects.conversion.log import converter
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.exporter.xes import exporter as xes_exporter


# ──────────────────────────────────────────────────────────────────────────────
# Temporal filtering helpers
# (mirrors SuffixTransformerNetwork/Preprocessing/create_benchmarks.py)
# ──────────────────────────────────────────────────────────────────────────────

def _start_from_date(df, start_date, case_id, timestamp):
    """Remove cases whose first event occurs before *start_date*.

    Parameters
    ----------
    df : pd.DataFrame
    start_date : str
        "YYYY-MM" format.  Cases starting before this month are dropped.
    case_id, timestamp : str
    """
    case_starts = df.groupby(case_id)[timestamp].min().reset_index()
    case_starts['_period'] = case_starts[timestamp].dt.to_period('M')
    keep = case_starts[case_starts['_period'].astype(str) >= start_date][case_id].values
    return df[df[case_id].isin(keep)].reset_index(drop=True)


def _end_before_date(df, end_date, case_id, timestamp):
    """Remove cases whose last event occurs after *end_date*.

    Parameters
    ----------
    df : pd.DataFrame
    end_date : str
        "YYYY-MM" format.  Cases ending after this month are dropped.
    case_id, timestamp : str
    """
    case_ends = df.groupby(case_id)[timestamp].max().reset_index()
    case_ends['_period'] = case_ends[timestamp].dt.to_period('M')
    keep = case_ends[case_ends['_period'].astype(str) <= end_date][case_id].values
    return df[df[case_id].isin(keep)].reset_index(drop=True)


def _start_before_date(df, start_before_date, case_id, timestamp):
    """Remove cases whose first event occurs after *start_before_date*.

    Needed for the workaround mode to mitigate extraction bias in the
    test set (mirrors start_before_date_select in create_benchmarks.py).

    Parameters
    ----------
    df : pd.DataFrame
    start_before_date : str
        "YYYY-MM" format.  Cases starting after this month are dropped.
    case_id, timestamp : str
    """
    case_starts = df.groupby(case_id)[timestamp].min().reset_index()
    case_starts['_period'] = case_starts[timestamp].dt.to_period('M')
    keep = case_starts[case_starts['_period'].astype(str) <= start_before_date][case_id].values
    return df[df[case_id].isin(keep)].reset_index(drop=True)


def _limited_duration(df, max_days, case_id, timestamp):
    """Remove cases whose throughput time exceeds *max_days*.

    Parameters
    ----------
    df : pd.DataFrame
    max_days : float
    case_id, timestamp : str
    """
    agg = df.groupby(case_id)[timestamp].agg(['min', 'max']).reset_index()
    agg['_duration'] = (agg['max'] - agg['min']).dt.total_seconds() / 86_400
    keep = agg[agg['_duration'] <= max_days * 1.00000000001][case_id].values
    return df[df[case_id].isin(keep)].reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Core temporal split
# ──────────────────────────────────────────────────────────────────────────────

def _temporal_case_split(df, len_share, mode, case_id, timestamp):
    """Split *df* into (main_df, secondary_df) using a temporal split.

    The separation time is derived identically to trainTestSplit() in
    create_benchmarks.py.  The *secondary* set is the later portion
    (test / val) and *main* is the earlier portion (train+val / train).

    For *mode='workaround'*:
        secondary : cases starting at or after sep_time.
        main      : cases starting before sep_time; overlapping cases
                    (start before, end at/after sep_time) are TRUNCATED
                    to events strictly before sep_time.

    For *mode='preferred'*:
        main      : cases ending strictly before sep_time.
        secondary : cases starting at or after sep_time.
        Overlapping cases are excluded from both sets.

    Cases with fewer than 2 events after truncation are dropped (a single
    event yields no prediction target for NAP).

    Parameters
    ----------
    df : pd.DataFrame
        Event log to split.
    len_share : float
        Fraction of cases assigned to the secondary (later) set.
    mode : {'workaround', 'preferred'}
    case_id, timestamp : str

    Returns
    -------
    main_df, secondary_df : pd.DataFrame
    """
    case_starts = df.groupby(case_id)[timestamp].min().sort_values()
    case_ends   = df.groupby(case_id)[timestamp].max()
    n_cases     = len(case_starts)
    first_sec_idx = int(n_cases * (1 - len_share))
    sep_time = case_starts.iloc[first_sec_idx]

    if mode == 'workaround':
        # Secondary: cases starting at/after sep_time (pure, no overlap)
        secondary_ids = set(case_starts[case_starts >= sep_time].index)
        # Main: cases starting before sep_time
        all_main_ids  = set(case_starts[case_starts < sep_time].index)
        pure_main_ids = set(case_ends[case_ends < sep_time].index)
        overlap_ids   = all_main_ids - pure_main_ids

        secondary_df = df[df[case_id].isin(secondary_ids)].copy()

        pure_main_df    = df[df[case_id].isin(pure_main_ids)].copy()
        overlap_main_df = df[
            df[case_id].isin(overlap_ids) & (df[timestamp] < sep_time)
        ].copy()
        main_df = pd.concat([pure_main_df, overlap_main_df], ignore_index=True)

    elif mode == 'preferred':
        # Main: cases ending strictly before sep_time (pure, no overlap)
        pure_main_ids = set(case_ends[case_ends < sep_time].index)
        # Secondary: cases starting at/after sep_time (pure, no overlap)
        secondary_ids = set(case_starts[case_starts >= sep_time].index)
        # Overlapping cases (start before, end at/after sep_time): excluded

        main_df      = df[df[case_id].isin(pure_main_ids)].copy()
        secondary_df = df[df[case_id].isin(secondary_ids)].copy()

    else:
        raise ValueError(f"mode must be 'workaround' or 'preferred', got '{mode!r}'")

    # Drop truncated cases with fewer than 2 events (no prediction target)
    def _drop_short_cases(frame):
        counts = frame.groupby(case_id)[timestamp].transform('count')
        return frame[counts >= 2].reset_index(drop=True)

    main_df      = _drop_short_cases(main_df)
    secondary_df = _drop_short_cases(secondary_df)

    return main_df.reset_index(drop=True), secondary_df.reset_index(drop=True)


def _val_case_split(df, val_len_share, case_id, timestamp):
    """Carve a validation set from a training DataFrame without truncation.

    Data-leakage prevention (truncation / exclusion of overlapping cases)
    is only necessary at the train/test boundary.  For train/val, the val
    set is used exclusively for monitoring during training, so a simple
    chronological case assignment is sufficient:

    * Val  : the last ``val_len_share`` fraction of cases by start time.
    * Train: all remaining cases — including those whose last event falls
             after the val separation time — kept in full (no truncation).

    Parameters
    ----------
    df : pd.DataFrame
        The train+val portion of the event log (already leakage-free w.r.t.
        the test set).
    val_len_share : float
        Fraction of cases in *df* assigned to the validation set.
    case_id, timestamp : str

    Returns
    -------
    train_df, val_df : pd.DataFrame
    """
    case_starts   = df.groupby(case_id)[timestamp].min().sort_values()
    n_cases       = len(case_starts)
    first_val_idx = int(n_cases * (1 - val_len_share))
    val_sep_time  = case_starts.iloc[first_val_idx]

    val_ids   = set(case_starts[case_starts >= val_sep_time].index)
    train_ids = set(case_starts[case_starts <  val_sep_time].index)

    val_df   = df[df[case_id].isin(val_ids)].copy().reset_index(drop=True)
    train_df = df[df[case_id].isin(train_ids)].copy().reset_index(drop=True)

    return train_df, val_df


def split_log_temporal(
    df,
    test_len_share=0.20,
    val_len_share=0.20,
    mode='workaround',
    case_id='case:concept:name',
    timestamp='time:timestamp',
):
    """Apply a DyLoPro-style temporal split, returning four DataFrames.

    The train/test boundary uses the full DyLoPro overlap-handling logic
    (truncation for ``'workaround'``, exclusion for ``'preferred'``) to
    prevent data leakage.  The train/val boundary uses a simple
    chronological case assignment with no truncation — leakage prevention
    is only critical between train and test.

    Default proportions (test_len_share=0.20, val_len_share=0.20) reproduce
    the original 64 / 16 / 20 train / val / test split.

    Parameters
    ----------
    df : pd.DataFrame
        Pre-filtered event log (timestamp column must be datetime).
    test_len_share : float
        Fraction of cases assigned to the test set.
    val_len_share : float
        Fraction of train+val cases assigned to the validation set.
    mode : {'workaround', 'preferred'}
        Overlap-handling mode for the train/test boundary only.
    case_id, timestamp : str

    Returns
    -------
    train_df, val_df, test_df, train_val_df : pd.DataFrame
    """
    # 1. Train+val vs test — full DyLoPro overlap handling
    train_val_df, test_df = _temporal_case_split(
        df, len_share=test_len_share, mode=mode,
        case_id=case_id, timestamp=timestamp,
    )

    # 2. Train vs val — simple chronological split, no truncation
    train_df, val_df = _val_case_split(
        train_val_df, val_len_share=val_len_share,
        case_id=case_id, timestamp=timestamp,
    )

    return train_df, val_df, test_df, train_val_df


# ──────────────────────────────────────────────────────────────────────────────
# Conversion helper
# ──────────────────────────────────────────────────────────────────────────────

def _df_to_event_log(df, source_log):
    """Convert a filtered DataFrame back to a pm4py EventLog.

    Log-level metadata (extensions, classifiers, attributes) are copied
    from *source_log* so the exported XES files remain well-formed.
    """
    event_log = converter.apply(df, variant=converter.Variants.TO_EVENT_LOG)
    event_log.extensions.update(source_log.extensions)
    event_log.classifiers.update(source_log.classifiers)
    event_log.attributes.update(source_log.attributes)
    return event_log


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────────────────

def plot_split(
    df,
    log_name,
    case_id='case:concept:name',
    timestamp='time:timestamp',
    test_len_share=0.20,
    val_len_share=0.20,
    mode='workaround',
    max_cases_shown=60,
    save_path=None,
):
    """Visualise the train / val / test temporal split.

    Each case is a horizontal bar from its first to its last event.
    Colours encode the partition assignment:

    Blue   : pure train
    Orange : pure validation
    Grey   : pure test
    Green  : overlap portion *kept* (truncated into the earlier split)
    Red    : overlap portion *discarded*

    Two vertical dashed lines mark the val and test separation times.
    The plot is saved to *save_path* (PNG).

    Parameters
    ----------
    df : pd.DataFrame
        Pre-filtered event log (timestamp column already datetime).
    log_name : str
        Used in the plot title and default filename.
    case_id, timestamp : str
    test_len_share, val_len_share : float
    mode : {'workaround', 'preferred'}
    max_cases_shown : int
        Max cases to draw; a uniform chronological sample is taken when
        the log contains more cases than this value.
    save_path : str or None
        Output PNG path.  Defaults to
        split_datasets/<log_name>_<mode>_split.png next to this script.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.dates as mdates

    # 1. Case-level start / end times
    case_starts = df.groupby(case_id)[timestamp].min().sort_values()
    case_ends   = df.groupby(case_id)[timestamp].max()
    n_cases     = len(case_starts)

    # 2. Test separation time  (mirrors _temporal_case_split)
    first_test_idx = int(n_cases * (1 - test_len_share))
    test_sep_time  = case_starts.iloc[first_test_idx]

    # 3. Val separation time (mirrors _val_case_split: simple start-time threshold,
    #    no truncation — leakage prevention is only needed at the train/test boundary).
    #    The train+val pool is all cases whose start time is before test_sep_time,
    #    regardless of mode.
    tv_starts     = case_starts[case_starts < test_sep_time].sort_values()
    n_tv          = len(tv_starts)
    first_val_idx = int(n_tv * (1 - val_len_share))
    val_sep_time  = tv_starts.iloc[first_val_idx]

    # 4. Classify every case
    # At the train/test boundary: overlapping cases are coloured to show truncation
    # or exclusion (depending on mode).
    # At the train/val boundary: no truncation — cases that straddle val_sep_time
    # are kept whole in train (their start time determines assignment).
    def classify(cid):
        s = case_starts[cid]
        e = case_ends[cid]
        if s >= test_sep_time:
            return 'test'
        if e >= test_sep_time:
            return 'overlap_test'
        if s >= val_sep_time:
            return 'val'
        return 'train'

    # 5. Uniform chronological sample for readability
    all_cases = list(case_starts.index)
    if len(all_cases) > max_cases_shown:
        step    = len(all_cases) / max_cases_shown
        sampled = [all_cases[int(i * step)] for i in range(max_cases_shown)]
    else:
        sampled = all_cases

    # 6. Timezone-naive datetimes for matplotlib
    def to_naive(ts):
        ts = pd.Timestamp(ts)
        if ts.tzinfo is not None:
            ts = ts.tz_convert('UTC').tz_localize(None)
        return ts.to_pydatetime()

    test_sep_dt = to_naive(test_sep_time)
    val_sep_dt  = to_naive(val_sep_time)

    # 7. Colour scheme
    BLUE   = '#5BC0DE'   # train
    ORANGE = '#F0AD4E'   # val
    GREY   = '#808080'   # test
    GREEN  = '#5CB85C'   # overlap kept
    RED    = '#D9534F'   # overlap discarded

    if mode == 'workaround':
        # Before sep → kept in the earlier split (green); after → discarded (red)
        before_color = GREEN
        after_color  = RED
        before_label = 'Overlap → kept in train (truncated)'
        after_label  = 'Overlap → discarded'
    else:
        # Before sep → discarded from train (red); after → excluded from test (green)
        before_color = RED
        after_color  = GREEN
        before_label = 'Overlap → discarded from train'
        after_label  = 'Overlap → excluded from test'

    # 8. Draw
    fig_height = max(6, len(sampled) // 6)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    BAR_H = 0.7

    for y, cid in enumerate(sampled):
        s_dt = to_naive(case_starts[cid])
        e_dt = to_naive(case_ends[cid])
        cat  = classify(cid)

        s_num  = mdates.date2num(s_dt)
        e_num  = mdates.date2num(e_dt)
        ts_num = mdates.date2num(test_sep_dt)

        if cat == 'train':
            ax.barh(y, e_num - s_num, left=s_num, height=BAR_H,
                    color=BLUE, linewidth=0)

        elif cat == 'val':
            ax.barh(y, e_num - s_num, left=s_num, height=BAR_H,
                    color=ORANGE, linewidth=0)

        elif cat == 'test':
            ax.barh(y, e_num - s_num, left=s_num, height=BAR_H,
                    color=GREY, linewidth=0)

        elif cat == 'overlap_test':
            # Split bar at test_sep (leakage prevention applies here)
            ax.barh(y, ts_num - s_num,  left=s_num,  height=BAR_H,
                    color=before_color, linewidth=0)
            ax.barh(y, e_num  - ts_num, left=ts_num, height=BAR_H,
                    color=after_color,  linewidth=0)

    # 9. Separation lines
    ax.axvline(mdates.date2num(test_sep_dt), color='black',   linestyle='--', linewidth=1.5)
    ax.axvline(mdates.date2num(val_sep_dt),  color='dimgrey', linestyle=':',  linewidth=1.2)

    # 10. Axis formatting
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=30, ha='right', fontsize=9)
    ax.set_yticks([])
    ax.set_ylabel('Cases', fontsize=12)
    ax.set_xlabel('Time', fontsize=12)
    mode_label = 'Workaround' if mode == 'workaround' else 'Preferred'
    ax.set_title(
        f'{mode_label} Train / Val / Test Split  —  {log_name}',
        fontsize=14, fontweight='bold',
    )

    # 11. Legend
    handles = [
        mpatches.Patch(color=BLUE,         label='Train'),
        mpatches.Patch(color=ORANGE,       label='Validation'),
        mpatches.Patch(color=GREY,         label='Test'),
        mpatches.Patch(color=before_color, label=before_label),
        mpatches.Patch(color=after_color,  label=after_label),
        plt.Line2D([0], [0], color='black',   linestyle='--', label='Test separation time'),
        plt.Line2D([0], [0], color='dimgrey', linestyle=':',  label='Val separation time'),
    ]
    ax.legend(handles=handles, loc='upper left', fontsize=9)
    plt.tight_layout()

    # 12. Save
    if save_path is None:
        save_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'split_datasets',
            f'{log_name}_{mode}_split.png',
        )
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Split plot saved to '{save_path}'")


# ──────────────────────────────────────────────────────────────────────────────
# Public API  ── call this from a Jupyter notebook or any external script
# ──────────────────────────────────────────────────────────────────────────────

def create_nap_splits(
    log_path,
    log_name,
    case_id='case:concept:name',
    timestamp='time:timestamp',
    test_len_share=0.20,
    val_len_share=0.20,
    mode='workaround',
    start_date=None,
    start_before_date=None,
    end_date=None,
    max_days=None,
    plot=True,
    output_dir=None,
):
    """Load one event log, apply a DyLoPro temporal split, and optionally
    export the four partitions as compressed XES files.

    Designed to be called from a Jupyter notebook::

        from generate_new_event_log_splits import create_nap_splits

        splits = create_nap_splits(
            log_path='logs/bpic2019.xes.gz',
            log_name='bpic2019',
            mode='workaround',
            plot=True,
            output_dir='split_datasets',
        )
        train_df     = splits['train']
        val_df       = splits['val']
        test_df      = splits['test']
        train_val_df = splits['train_val']

    Parameters
    ----------
    log_path : str
        Path to a ``.xes``, ``.xes.gz``, or ``.csv`` file.
    log_name : str
        Short identifier for the log; used in plot titles and XES filenames.
    case_id : str
        Column name for case identifiers.  Default ``'case:concept:name'``.
    timestamp : str
        Column name for event timestamps.  Default ``'time:timestamp'``.
    test_len_share : float
        Fraction of cases (chronologically last) assigned to the test set.
        Default ``0.20`` (20 %).
    val_len_share : float
        Fraction of the remaining train+val cases assigned to the validation
        set.  Default ``0.20`` (16 % of the full log when combined with the
        default test_len_share, giving a 64 / 16 / 20 split).
    mode : {'workaround', 'preferred'}
        Overlap-handling strategy.  ``'workaround'`` (default) truncates
        overlapping training cases at the separation time; ``'preferred'``
        excludes overlapping cases from both sides of each boundary.
    start_date : str or None
        ``'YYYY-MM'`` — drop cases whose first event is before this month.
    start_before_date : str or None
        ``'YYYY-MM'`` — drop cases whose first event is after this month
        (workaround extraction-bias fix).
    end_date : str or None
        ``'YYYY-MM'`` — drop cases whose last event is after this month.
    max_days : float or None
        Drop cases whose throughput time exceeds this many days.
    plot : bool
        If ``True`` (default), save a split visualisation PNG to
        *output_dir* (or alongside this script when *output_dir* is None).
    output_dir : str or None
        Directory where the four ``.xes.gz`` files are written.  Pass
        ``None`` to skip XES export and only return the DataFrames.

    Returns
    -------
    dict with keys ``'train'``, ``'val'``, ``'test'``, ``'train_val'``
        Each value is a ``pd.DataFrame`` containing the events for that
        partition.  The ``pm4py`` EventLog objects are *not* returned;
        re-convert with ``pm4py.convert_to_event_log`` if needed.
    """
    # ── 1. Load ──────────────────────────────────────────────────────────────
    ext = os.path.splitext(log_path)[1].lower()
    if ext in ('.xes', '.gz'):
        raw_log = xes_importer.apply(log_path)
        df = converter.apply(raw_log, variant=converter.Variants.TO_DATA_FRAME)
    elif ext == '.csv':
        raw_log = None
        df = pd.read_csv(log_path)
    else:
        raise ValueError(f"Unsupported file format '{ext}'. Use .xes, .xes.gz, or .csv.")

    # ── 2. Timestamp ─────────────────────────────────────────────────────────
    df[timestamp] = pd.to_datetime(df[timestamp], utc=True)

    # ── 3. Chronological-outlier filters ─────────────────────────────────────
    if start_date:
        df = _start_from_date(df, start_date, case_id, timestamp)
    if end_date:
        df = _end_before_date(df, end_date, case_id, timestamp)
    if start_before_date:
        df = _start_before_date(df, start_before_date, case_id, timestamp)
    df.drop_duplicates(inplace=True, ignore_index=True)
    if max_days is not None:
        df = _limited_duration(df, max_days, case_id, timestamp)

    total_cases = df[case_id].nunique()
    print(f"[{log_name}] {total_cases} cases after filtering.")

    # ── 4. Visualise ─────────────────────────────────────────────────────────
    if plot:
        plot_save_path = None
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            plot_save_path = os.path.join(output_dir, f'{log_name}_{mode}_split.png')
        plot_split(
            df,
            log_name=log_name,
            case_id=case_id,
            timestamp=timestamp,
            test_len_share=test_len_share,
            val_len_share=val_len_share,
            mode=mode,
            save_path=plot_save_path,
        )

    # ── 5. Split ─────────────────────────────────────────────────────────────
    train_df, val_df, test_df, train_val_df = split_log_temporal(
        df,
        test_len_share=test_len_share,
        val_len_share=val_len_share,
        mode=mode,
        case_id=case_id,
        timestamp=timestamp,
    )

    print(
        f"[{log_name}]  train={train_df[case_id].nunique()}  "
        f"val={val_df[case_id].nunique()}  "
        f"test={test_df[case_id].nunique()}  "
        f"train_val={train_val_df[case_id].nunique()}"
    )

    # ── 6. Export XES (optional) ──────────────────────────────────────────────
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

        # raw_log may be None when input was CSV; build a minimal source log
        # from train_val_df so metadata copy has something to work with.
        if raw_log is None:
            raw_log = converter.apply(
                train_val_df, variant=converter.Variants.TO_EVENT_LOG
            )

        for tag, part_df in [
            ('train',     train_df),
            ('val',       val_df),
            ('test',      test_df),
            ('train_val', train_val_df),
        ]:
            event_log = _df_to_event_log(part_df, raw_log)
            out_path  = os.path.join(output_dir, f'{tag}_{log_name}.xes.gz')
            xes_exporter.apply(event_log, out_path, parameters={'compress': True})
            print(f"  Saved '{out_path}'")

    return {
        'train':     train_df,
        'val':       val_df,
        'test':      test_df,
        'train_val': train_val_df,
    }


if __name__ == "__main__":

    # ──────────────────────────────────────────────────────────────────────────────
    # Configuration  ── edit these to match your dataset
    # ──────────────────────────────────────────────────────────────────────────────
    
    CASE_ID   = 'case:concept:name'
    TIMESTAMP = 'time:timestamp'
    
    # Splitting parameters (mirrors SuffixTransformerNetwork/create_generic_data.py)
    TEST_LEN_SHARE = 0.20   # fraction of all cases → test set
    VAL_LEN_SHARE  = 0.20   # fraction of train+val cases → validation set
    MODE           = 'preferred'   # 'workaround' or 'preferred' (see module docstring)
    
    # Optional chronological outlier filters — set to None to disable.
    # Format: "YYYY-MM"   e.g. START_DATE = "2016-01"
    START_DATE        = None   # drop cases starting before this month
    START_BEFORE_DATE = None   # drop cases starting after  this month (workaround bias fix)
    END_DATE          = None   # drop cases ending   after  this month
    MAX_DAYS          = None   # drop cases longer than this many days (float)
    
    PLOT = True   # set to False to skip the split visualisation
    
    # ──────────────────────────────────────────────────────────────────────────────
    # Main script
    # ──────────────────────────────────────────────────────────────────────────────
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir    = os.path.join(script_dir, "raw_datasets_that_are_not_evaluated")
    split_dir  = os.path.join(script_dir, "split_datasets")
    os.makedirs(split_dir, exist_ok=True)
    
    for file_name in sorted(os.listdir(raw_dir)):
        if not file_name.endswith(".xes.gz"):
            continue
    
        file_path = os.path.join(raw_dir, file_name)
        print(f"\nProcessing '{file_name}' ...")
    
        # 1. Import XES and convert to DataFrame
        log = xes_importer.apply(file_path)
        df  = converter.apply(log, variant=converter.Variants.TO_DATA_FRAME)
    
        # 2. Ensure timestamp is timezone-aware datetime
        df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP], utc=True)
    
        # 3. Apply optional chronological-outlier filters
        if START_DATE:
            df = _start_from_date(df, START_DATE, CASE_ID, TIMESTAMP)
        if END_DATE:
            df = _end_before_date(df, END_DATE, CASE_ID, TIMESTAMP)
        if START_BEFORE_DATE:
            df = _start_before_date(df, START_BEFORE_DATE, CASE_ID, TIMESTAMP)
        df.drop_duplicates(inplace=True, ignore_index=True)
        if MAX_DAYS is not None:
            df = _limited_duration(df, MAX_DAYS, CASE_ID, TIMESTAMP)
    
        total_cases = df[CASE_ID].nunique()
    
        # 4. Visualise the split before heavy processing
        if PLOT:
            base_name_plot = file_name.replace(".xes.gz", "")
            plot_split(
                df,
                log_name=base_name_plot,
                case_id=CASE_ID,
                timestamp=TIMESTAMP,
                test_len_share=TEST_LEN_SHARE,
                val_len_share=VAL_LEN_SHARE,
                mode=MODE,
            )
    
        # 6. Temporal split
        train_df, val_df, test_df, train_val_df = split_log_temporal(
            df,
            test_len_share=TEST_LEN_SHARE,
            val_len_share=VAL_LEN_SHARE,
            mode=MODE,
            case_id=CASE_ID,
            timestamp=TIMESTAMP,
        )
    
        print(
            f"  total={total_cases} cases  →  "
            f"train={train_df[CASE_ID].nunique()}  "
            f"val={val_df[CASE_ID].nunique()}  "
            f"test={test_df[CASE_ID].nunique()}  "
            f"train_val={train_val_df[CASE_ID].nunique()}"
        )
    
        # 7. Convert DataFrames back to EventLogs
        train_log     = _df_to_event_log(train_df,     log)
        val_log       = _df_to_event_log(val_df,       log)
        test_log      = _df_to_event_log(test_df,      log)
        train_val_log = _df_to_event_log(train_val_df, log)
    
        # 8. Export as .xes.gz
        base_name = file_name.replace(".xes.gz", "")
        xes_exporter.apply(
            train_log,
            os.path.join(split_dir, f"train_{base_name}.xes.gz"),
            parameters={"compress": True},
        )
        xes_exporter.apply(
            val_log,
            os.path.join(split_dir, f"val_{base_name}.xes.gz"),
            parameters={"compress": True},
        )
        xes_exporter.apply(
            test_log,
            os.path.join(split_dir, f"test_{base_name}.xes.gz"),
            parameters={"compress": True},
        )
        xes_exporter.apply(
            train_val_log,
            os.path.join(split_dir, f"train_val_{base_name}.xes.gz"),
            parameters={"compress": True},
        )
    
    print("\nSplitting and saving completed.")
