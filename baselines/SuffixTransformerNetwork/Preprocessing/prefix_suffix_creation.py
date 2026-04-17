
import pandas as pd
import numpy as np


def sort_log(df, case_id = 'case:concept:name', timestamp = 'time:timestamp', act_label = 'concept:name'):
    """Sort events in event log such that cases that occur first are stored
    first, and such that events within the same case are stored based on timestamp.

    Parameters
    ----------
    df : _type_
        _description_
    case_id : str, optional
        _description_, by default 'case:concept:name'
    timestamp : str, optional
        _description_, by default 'time:timestamp'
    act_label : str, optional
        _description_, by default 'concept:name'
    """
    df_help = df.sort_values([case_id, timestamp], ascending = [True, True], kind='mergesort')
    # Now take first row of every case_id: this contains first stamp
    df_first = df_help.drop_duplicates(subset = case_id)[[case_id, timestamp]]
    df_first = df_first.sort_values(timestamp, ascending = True, kind='mergesort')
    # Include integer index to sort on.
    df_first['case_id_int'] = [i for i in range(len(df_first))]
    df_first = df_first.drop(timestamp, axis = 1)
    df = df.merge(df_first, on = case_id, how = 'left')
    df = df.sort_values(['case_id_int', timestamp], ascending = [True, True], kind='mergesort')
    df = df.drop('case_id_int', axis = 1)
    return df

def create_prefix_suffixes(df, window_size, outcome, case_id = 'case:concept:name',
                           timestamp = 'time:timestamp', act_label = 'concept:name',
                           cat_casefts = [], num_casefts = [],
                           cat_eventfts = [], num_eventfts = []):
    """Create dataframes for the prefixes (input encoder), decoder suffix
    (input decoder), activity labels, time till next event labels and
    remaining runtime labels.

    Vectorized implementation: replaces the original O(window_size) groupby
    loop with a single numpy.repeat-based expansion.  The output is
    identical to the original loop-based version.

    For a row at 1-based event index k in a case of length L
    (all cases satisfy L <= window_size after the upstream filter):

      - Prefix:     contributes to nr_events in [k,  L]   → n_reps = L - k + 1
      - Suffix /
        TimeLabel:  contributes to nr_events in [1,  k]   → n_reps = k
      - ActLabel
        (evt_idx_act = k, includes END at k = L+1):
                    contributes to nr_events in [1, k-1]  → n_reps = k - 1
                    (rows at k=1 get n_reps=0 → naturally excluded,
                     matching the original tail(-1) / tail(-n) behaviour)
      - Outcome:    one row per case per nr_events in [1, L]  → n_reps = L

    Parameters
    ----------
    df : _type_
        _description_
    window_size : _type_
        _description_
    outcome : {None, str}
        If a binary outcome column is contained within the event log and
        outcome prediction labels are needed, outcome should be a string
        representing the column name that contains the binary outcome. If
        no binary outcome is present in the event log, or outcome
        prediction is not needed, `outcome` is None.
    case_id : str, optional
        _description_, by default 'case:concept:name'
    timestamp : str, optional
        _description_, by default 'time:timestamp'
    act_label : str, optional
        _description_, by default 'concept:name'
    cat_casefts : list, optional
        _description_, by default []
    num_casefts : list, optional
        _description_, by default []
    cat_eventfts : list, optional
        _description_, by default []
    num_eventfts : list, optional
        _description_, by default []
    """

    # ------------------------------------------------------------------
    # Column subsets
    # ------------------------------------------------------------------
    prefix_cols    = ([case_id, act_label, timestamp, 'ts_start', 'ts_prev',
                       'case_length'] + cat_casefts + num_casefts
                      + cat_eventfts + num_eventfts)
    suffix_cols    = [case_id, act_label, timestamp, 'ts_start', 'ts_prev',
                      'case_length']
    timeLabel_cols = [case_id, timestamp, 'case_length', 'tt_next', 'rtime']
    actLabel_cols  = [case_id, act_label, 'case_length']
    if outcome:
        outcomeLabel_cols = [case_id, 'case_length', outcome]

    # ------------------------------------------------------------------
    # Pre-compute 1-based event index and case length for every row
    # (all cases already have case_length <= window_size at this point)
    # ------------------------------------------------------------------
    df_reset   = df.reset_index(drop=True)
    evt_idx    = df_reset.groupby(case_id, sort=False).cumcount().values + 1  # shape (N,)
    case_len   = df_reset['case_length'].values                                # shape (N,)

    # ------------------------------------------------------------------
    # Helper: compute intra-group offsets for a repeated array
    # ------------------------------------------------------------------
    def _offsets(n_reps):
        """Return 0-based offset within each group after np.repeat."""
        total      = int(n_reps.sum())
        cum_starts = np.concatenate([[0], np.cumsum(n_reps[:-1])])
        return np.arange(total) - np.repeat(cum_starts, n_reps)

    # ------------------------------------------------------------------
    # Helper: expand a sub-dataframe and tag with nr_events / orig_case_id
    # ------------------------------------------------------------------
    def _expand(sub_df, n_reps, nr_events_arr):
        """Repeat rows of sub_df per n_reps, assign prefix_nr and orig_case_id.

        Uses np.repeat on raw numpy arrays instead of pandas iloc so that
        the expansion is a single vectorised C call with no indexing overhead.
        sub_df must have a contiguous RangeIndex (call reset_index first).
        """
        # Repeat every column independently — one np.repeat per column,
        # all in C, no pandas row-indexing overhead.
        expanded = pd.DataFrame(
            {col: np.repeat(sub_df[col].to_numpy(), n_reps)
             for col in sub_df.columns}
        )
        expanded['prefix_nr']    = nr_events_arr
        expanded['orig_case_id'] = expanded[case_id].to_numpy()
        # Build new case_id strings: unchanged for nr_events=1, append '_n' for n>=2
        orig_ids = expanded[case_id].to_numpy().astype(str)
        nr_strs  = np.where(nr_events_arr > 1,
                            np.char.add('_', nr_events_arr.astype(str)),
                            '')
        expanded[case_id] = np.char.add(orig_ids, nr_strs)
        return expanded

    # ------------------------------------------------------------------
    # Prefix
    # Row (evt_idx=k, case_length=L): nr_events in [k, L]
    # n_reps = L - k + 1  (always >= 1 since k <= L)
    # ------------------------------------------------------------------
    n_reps_pref    = (case_len - evt_idx + 1).astype(np.int64)
    offsets_pref   = _offsets(n_reps_pref)
    nr_events_pref = np.repeat(evt_idx, n_reps_pref) + offsets_pref

    prefix_df = _expand(df_reset[prefix_cols].reset_index(drop=True),
                        n_reps_pref, nr_events_pref)

    # ------------------------------------------------------------------
    # Suffix (decoder input) & Time Labels
    # Row (evt_idx=k): nr_events in [1, k]
    # n_reps = k  (since case_length <= window_size, k <= window_size)
    # ------------------------------------------------------------------
    n_reps_suff    = evt_idx.astype(np.int64)
    offsets_suff   = _offsets(n_reps_suff)
    nr_events_suff = offsets_suff + 1   # 1, 2, ..., k for each group

    suffix_df    = _expand(df_reset[suffix_cols].reset_index(drop=True),
                           n_reps_suff, nr_events_suff)
    timeLabel_df = _expand(df_reset[timeLabel_cols].reset_index(drop=True),
                           n_reps_suff, nr_events_suff)

    # ------------------------------------------------------------------
    # Activity Labels
    # Build actLabel_subset with an END_TOKEN appended per case.
    # Row at evt_idx_act=k: nr_events in [1, k-1]   (n_reps = k-1)
    # Rows at k=1 get n_reps=0 → excluded (matches original tail(-1)).
    # ------------------------------------------------------------------
    actLabel_sub = df_reset[actLabel_cols].copy()
    actLabel_sub['_eidx'] = evt_idx  # 1-based

    # END_TOKEN rows: one per case, evt_idx_act = case_length + 1
    case_dedup   = df_reset.drop_duplicates(subset=case_id).reset_index(drop=True)
    case_ids_arr = case_dedup[case_id].values
    case_len_arr = case_dedup['case_length'].values
    # Integer ordering of cases (preserves original case order for sorting)
    case_order   = {cid: i for i, cid in enumerate(case_ids_arr)}

    end_rows = pd.DataFrame({
        case_id    : case_ids_arr,
        act_label  : ['END_TOKEN'] * len(case_ids_arr),
        'case_length': case_len_arr,
        '_eidx'    : case_len_arr + 1,
    })

    actLabel_full = pd.concat([actLabel_sub, end_rows], axis=0, ignore_index=True)
    actLabel_full['_cord'] = actLabel_full[case_id].map(case_order)
    actLabel_full = (actLabel_full
                     .sort_values(['_cord', '_eidx'], kind='mergesort')
                     .reset_index(drop=True))

    act_k       = actLabel_full['_eidx'].values
    n_reps_act  = (act_k - 1).astype(np.int64)          # 0 for evt_idx=1 → excluded
    offsets_act = _offsets(n_reps_act)
    nr_events_act = offsets_act + 1

    actLabel_src = (actLabel_full[[case_id, act_label, 'case_length']]
                    .reset_index(drop=True))
    actLabel_df  = _expand(actLabel_src, n_reps_act, nr_events_act)

    # ------------------------------------------------------------------
    # Outcome Labels (one row per prefix-suffix pair)
    # Each case contributes one row for nr_events in [1, case_length].
    # ------------------------------------------------------------------
    if outcome:
        out_case   = (df_reset.drop_duplicates(subset=case_id)
                              [outcomeLabel_cols]
                              .reset_index(drop=True))
        n_reps_out = case_dedup['case_length'].values.astype(np.int64)
        offsets_out   = _offsets(n_reps_out)
        nr_events_out = offsets_out + 1

        outcomeLabel_df = _expand(out_case, n_reps_out, nr_events_out)

    if outcome:
        return prefix_df, suffix_df, timeLabel_df, actLabel_df, outcomeLabel_df
    else:
        return prefix_df, suffix_df, timeLabel_df, actLabel_df




    


    

def construct_PrefSuff_dfs_pipeline(df, window_size, outcome, case_id = 'case:concept:name', 
                                    timestamp = 'time:timestamp', act_label = 'concept:name', 
                                    cat_casefts = [], num_casefts = [], 
                                    cat_eventfts = [], num_eventfts = []):
    """_summary_

    Parameters
    ----------
    df : _type_
        _description_
    window_size : _type_
        _description_
    outcome : {None, str}
        If a binary outcome column is contained within the event log and 
        outcome prediction labels are needed, outcome should be a string 
        representing the column name that contains the binary outcome. If 
        no binary outcome is present in the event log, or outcome 
        prediction is not needed, `outcome` is None. 
    case_id : str, optional
        _description_, by default 'case:concept:name'
    timestamp : str, optional
        _description_, by default 'time:timestamp'
    act_label : str, optional
        _description_, by default 'concept:name'
    """


    # Create prefixes 
    pref_suf_dataframes = create_prefix_suffixes(df, window_size = window_size, 
                                                 outcome=outcome,
                                                 case_id = case_id, timestamp = timestamp, 
                                                 act_label = act_label, 
                                                 cat_casefts = cat_casefts, 
                                                 num_casefts = num_casefts, 
                                                 cat_eventfts = cat_eventfts, 
                                                 num_eventfts = num_eventfts)

    return pref_suf_dataframes 



def handle_missing_data(df, case_id = 'case:concept:name', timestamp = 'time:timestamp', act_label = 'concept:name', numerical_cols = [], cat_cols = []):
    """Handles missing data. Imputes NANs of numerical columns with 0, and 
    creates an indicator variable = 1 if corresponding column had a nan and 
    0 otherwise. We do not create indicator columns for numerical features 
    not containing any NANs. For the categorical columns, we impute NANs simply 
    by filling the NANs with an additional level 'MISSINGV'. 

    Only the columns provided in the arguments are retained in the resulting df. 

    Parameters
    ----------
    df : 
        _description_
    case_id : str, optional
        _description_, by default 'case:concept:name'
    timestamp : str, optional
        _description_, by default 'time:timestamp'
    act_label : str, optional
        _description_, by default 'concept:name'
    numerical_cols : list, optional
        _description_, by default []
    cat_cols : list, optional
        _description_, by default []
    """


"""REMARKS END OF DAY: """
