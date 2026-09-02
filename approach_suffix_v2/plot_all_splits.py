import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from create_graph_data import load_log, preprocess_log, plot_split


def _stem(fname):
    for ext in ('.xes.gz', '.xes', '.csv'):
        if fname.endswith(ext):
            return fname[:-len(ext)]
    return os.path.splitext(fname)[0]


def _plot_one_log(args):
    """Top-level worker for ProcessPoolExecutor: plot the split for one log."""
    log_path, log_name, kw = args
    try:
        log = load_log(log_path)
        log = preprocess_log(
            log,
            timestamp_col=kw['timestamp'],
            timestamp_format=kw['timestamp_format'],
            bool_cols=kw['bool_cols'],
            str_cols=kw['str_cols'],
        )
        out_dir = os.path.join('results_per_log', log_name)
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, f"{log_name}_{kw['mode']}_split.png")
        plot_split(
            log,
            log_name=log_name,
            case_id=kw['case_id'],
            timestamp=kw['timestamp'],
            test_len_share=kw['test_len_share'],
            val_len_share=kw['val_len_share'],
            mode=kw['mode'],
            start_date=kw['start_date'],
            start_before_date=kw['start_before_date'],
            end_date=kw['end_date'],
            max_days=kw['max_days'],
            save_path=save_path,
        )
        return {'log': log_name, 'save_path': save_path, 'error': ''}
    except Exception as exc:
        return {'log': log_name, 'save_path': '', 'error': str(exc)}


def _make_grid(image_paths, grid_path):
    """Arrange the saved per-log split PNGs into a single grid figure."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    n = len(image_paths)
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    axes = axes.flatten()

    for ax, path in zip(axes, image_paths):
        ax.imshow(mpimg.imread(path))
        ax.axis('off')
    for ax in axes[n:]:
        ax.axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(grid_path) or '.', exist_ok=True)
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Grid plot saved to '{grid_path}'")


def plot_all_splits(
    folder,
    case_id='case:concept:name',
    timestamp='time:timestamp',
    timestamp_format=None,
    bool_cols=None,
    str_cols=None,
    start_date=None,
    start_before_date=None,
    end_date=None,
    max_days=None,
    test_len_share=0.20,
    val_len_share=0.20,
    mode='preferred',
    n_workers=None,
    grid_path=None,
):
    """Plot the train/val/test split visualisation (see `plot_split` in
    create_graph_data.py) for every event log in *folder*, in parallel and
    without running the full graph-construction preprocessing pipeline.
    The individual per-log plots are then combined into a single grid figure.

    Parameters mirror the corresponding subset of `construct_datasets`.

    Parameters
    ----------
    folder : str
        Directory containing .xes, .xes.gz, or .csv event-log files.
    n_workers : int or None
        Number of parallel worker processes. None = os.cpu_count().
    grid_path : str or None
        Output path for the combined grid figure. Defaults to
        ``results_per_log/all_splits_grid_<mode>.png``.
    """
    _SUPPORTED_EXT = {'.xes', '.gz', '.csv'}

    files = sorted(
        (os.path.join(folder, f), _stem(f))
        for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
        and os.path.splitext(f)[1].lower() in _SUPPORTED_EXT
    )
    if not files:
        print(f"No log files found in '{folder}'.")
        return

    kw = dict(case_id=case_id, timestamp=timestamp, timestamp_format=timestamp_format,
              bool_cols=bool_cols, str_cols=str_cols, start_date=start_date,
              start_before_date=start_before_date, end_date=end_date, max_days=max_days,
              test_len_share=test_len_share, val_len_share=val_len_share, mode=mode)

    workers = min(n_workers or os.cpu_count(), len(files))
    args_list = [(path, name, kw) for path, name in files]

    print(f"Plotting splits for {len(files)} logs with {workers} workers ...")
    results = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_plot_one_log, args): args[1] for args in args_list}
        for future in as_completed(futures):
            r = future.result()
            if r['error']:
                print(f"  ERROR  {r['log']}: {r['error']}")
            else:
                print(f"  Plotted '{r['log']}' -> {r['save_path']}")
            results[r['log']] = r

    # Assemble the grid, preserving the original folder order
    save_paths = [results[name]['save_path'] for _, name in files if results[name]['save_path']]
    if save_paths:
        _make_grid(save_paths, grid_path or os.path.join('results_per_log', f'all_splits_grid_{mode}.png'))


if __name__ == '__main__':
    # ------------------------------------------------------------------ #
    # Edit the variables below to match your event logs.                  #
    # ------------------------------------------------------------------ #

    FOLDER = 'logs'   # folder containing .xes / .xes.gz / .csv event logs

    CASE_ID   = 'case:concept:name'
    TIMESTAMP = 'time:timestamp'

    START_DATE        = None   # e.g. "2018-01"
    START_BEFORE_DATE = None   # e.g. "2018-09"
    END_DATE           = None   # e.g. "2019-02"
    MAX_DAYS           = None   # e.g. 143.33
    TEST_LEN_SHARE      = 0.20
    VAL_LEN_SHARE       = 0.20
    MODE               = 'preferred'   # or 'workaround'
    N_WORKERS           = None   # None -> os.cpu_count()

    plot_all_splits(
        FOLDER,
        case_id=CASE_ID,
        timestamp=TIMESTAMP,
        start_date=START_DATE,
        start_before_date=START_BEFORE_DATE,
        end_date=END_DATE,
        max_days=MAX_DAYS,
        test_len_share=TEST_LEN_SHARE,
        val_len_share=VAL_LEN_SHARE,
        mode=MODE,
        n_workers=N_WORKERS,
    )
