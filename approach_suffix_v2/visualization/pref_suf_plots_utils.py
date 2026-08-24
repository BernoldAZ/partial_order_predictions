import os
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

# ---------------------------------------------------------------------------
# Baseline result-directory names (mirrors _RESULT_DIRS in run_all_suffix_baselines.py)
# ---------------------------------------------------------------------------

_BASELINE_PATHS = {
    'SEP_LSTM':     'SEP_LSTM_results',
    'CRTP_LSTM':    'CRTP_LSTM_NDA_results',
    #'CRTP_LSTM_DA': 'CRTP_LSTM_DA_results',
    'ED_LSTM':      'ED_LSTM_results',
    'SuTraN':       'SUTRAN_NDA_results',
    #'SuTraN_DA':    'SUTRAN_DA_results',
    'BEST':         'BEST_results',
}

CONFIG_STRING = {'SEP_LSTM' : 'SEP-LSTM',
                'CRTP_LSTM' : 'CRTP-LSTM',
                #'CRTP_LSTM_DA' : 'CRTP-LSTM',
                'ED_LSTM' : 'ED-LSTM',
                'SuTraN' : 'SuTraN',
                #'SuTraN_DA' : 'SuTraN',
                'BEST' : 'BEST',
                'GATv2_GRU' : 'POG (Ours)'}
CONFIG_STYLES = {
    'CRTP_LSTM': ('#9467bd', '--'),
    #'CRTP_LSTM_DA': ('#9467bd', '-'),
    'SuTraN': ('#2ca02c', '--'),
    #'SuTraN_DA': ('#2ca02c', '-'),
    'SEP_LSTM': ('#d62728', '--'),
    'ED_LSTM': ('#ff7f0e', '--'),
    'BEST': ('#8c564b', ':'),
    'GATv2_GRU': ('#1f77b4', '-'),
}

# Display name for each event log when plotting. Logs not listed here are
# shown under their original (directory/pkl) name.
EVENT_LOGS = {
    'BPIC15_1': 'BPIC15.1',
    'BPIC15_2': 'BPIC15.2',
    'BPIC15_3': 'BPIC15.3',
    'BPIC15_4': 'BPIC15.4',
    'BPIC15_5': 'BPIC15.5',
    'BPI_Challenge_2012_A': 'BPIC12.A',
    'BPI_Challenge_2012_O': 'BPIC12.O',
    'Sepsis': 'Sepsis',
}

# Maps a metric name to where it lives in the tuple returned by
# `create_dataframes()`: (prefix_df_idx, suffix_df_idx, col_suffix, ylabel).
METRIC_CONFIG = {
    'ges':     {'prefix_df_idx': 4, 'suffix_df_idx': 5, 'col_suffix': '_ges', 'ylabel': 'GES'},
    'dl':      {'prefix_df_idx': 0, 'suffix_df_idx': 2, 'col_suffix': '_dls', 'ylabel': 'DL similarity'},
    'mae_rrt': {'prefix_df_idx': 1, 'suffix_df_idx': 3, 'col_suffix': '_mae', 'ylabel': 'MAE RRT (minutes)'},
}


def _load_and_average_runs(pkl_paths):
    """Load per-length result dicts from available run paths and average them.

    Handles both 3-element [DLS, MAE, count] and 4-element [DLS, MAE, count, GES]
    dicts so that old results (without GES) can be mixed with new ones.
    Returns None if no path exists.
    """
    run_dicts = []
    for p in pkl_paths:
        if os.path.isfile(p):
            with open(p, 'rb') as f:
                run_dicts.append(pickle.load(f))
    if not run_dicts:
        return None
    all_keys = set()
    for d in run_dicts:
        all_keys.update(d.keys())
    averaged = {}
    for k in all_keys:
        vals = [d[k] for d in run_dicts if k in d]
        n = max(len(v) for v in vals)
        row = []
        for i in range(n):
            if i == 2:  # instance count is identical across runs (same test set)
                row.append(vals[0][2])
            else:
                col = [v[i] for v in vals if len(v) > i and v[i] is not None]
                row.append(sum(col) / len(col) if col else None)
        averaged[k] = row
    return averaged

def create_dataframes(prefix_dicts, suffix_dicts, string_list_models):
    """
    Create dataframes for average Damerau-Levenshtein similarity (DLS),
    Mean Absolute Error (MAE), and Graph Edit Similarity (GES) based on
    prefix and suffix lengths from N models (N being the number of models
    for which the results should be analyzed, and hence also the number of
    dictionaries in the `prefix_dicts` and `suffix_dicts` lists, as well
    as the number of strings in the `string_list_models` list.

    The order of the prefix and suffix dictionaries contained within the
    `prefix_dicts` and `suffix_dicts` lists respectively, as well as the
    order of the strings in the `string_list_models`, should match.

    Parameters
    ----------
    prefix_dicts : list of dict
        List of dictionaries containing results aggregated over prefix
        lengths for different models. Each dictionary should have keys as
        integer prefix lengths and values as lists of three or four
        elements: [average DLS, average MAE in minutes, total instance
        count] or [average DLS, average MAE in minutes, total instance
        count, average GES].
    suffix_dicts : list of dict
        List of dictionaries containing results aggregated over suffix
        lengths for different models. Each dictionary should have keys as
        integer suffix lengths and values as lists of three or four
        elements: [average DLS, average MAE in minutes, total instance
        count] or [average DLS, average MAE in minutes, total instance
        count, average GES].
    string_list_models : list of str
        List of N model names, with the order of the model names
        corresponding to the order in which the prefix and suffix
        dictionaries (`prefix_dicts` and `suffix_dicts`) are sorted.

    Returns
    -------
    df_prefix_dls : pd.DataFrame
        DataFrame containing prefix lengths, instance counts, and average
        DLS for each model.
    df_prefix_mae : pd.DataFrame
        DataFrame containing prefix lengths, instance counts, and average
        MAE for each model.
    df_suffix_dls : pd.DataFrame
        DataFrame containing suffix lengths, instance counts, and average
        DLS for each model.
    df_suffix_mae : pd.DataFrame
        DataFrame containing suffix lengths, instance counts, and average
        MAE for each model.
    df_prefix_ges : pd.DataFrame
        DataFrame containing prefix lengths, instance counts, and average
        GES for each model (None if GES was not computed).
    df_suffix_ges : pd.DataFrame
        DataFrame containing suffix lengths, instance counts, and average
        GES for each model (None if GES was not computed).
    """
    prefix_lengths = sorted(set.intersection(*[set(d.keys()) for d in prefix_dicts]))
    suffix_lengths = sorted(set.intersection(*[set(d.keys()) for d in suffix_dicts]))

    prefix_instance_counts = [prefix_dicts[0][k][2] for k in prefix_lengths]
    suffix_instance_counts = [suffix_dicts[0][k][2] for k in suffix_lengths]

    prefix_dls = {f'{string_list_models[i]}_dls': [d[k][0] for k in prefix_lengths] for i, d in enumerate(prefix_dicts)}
    prefix_mae = {f'{string_list_models[i]}_mae': [d[k][1] for k in prefix_lengths] for i, d in enumerate(prefix_dicts)}
    prefix_ges = {f'{string_list_models[i]}_ges': [d[k][3] if len(d[k]) > 3 else None for k in prefix_lengths] for i, d in enumerate(prefix_dicts)}

    suffix_dls = {f'{string_list_models[i]}_dls': [d[k][0] for k in suffix_lengths] for i, d in enumerate(suffix_dicts)}
    suffix_mae = {f'{string_list_models[i]}_mae': [d[k][1] for k in suffix_lengths] for i, d in enumerate(suffix_dicts)}
    suffix_ges = {f'{string_list_models[i]}_ges': [d[k][3] if len(d[k]) > 3 else None for k in suffix_lengths] for i, d in enumerate(suffix_dicts)}

    df_prefix_dls = pd.DataFrame({
        'prefix_length': prefix_lengths,
        'instance_count': prefix_instance_counts,
        **prefix_dls
    })

    df_prefix_mae = pd.DataFrame({
        'prefix_length': prefix_lengths,
        'instance_count': prefix_instance_counts,
        **prefix_mae
    })

    df_suffix_dls = pd.DataFrame({
        'suffix_length': suffix_lengths,
        'instance_count': suffix_instance_counts,
        **suffix_dls
    })

    df_suffix_mae = pd.DataFrame({
        'suffix_length': suffix_lengths,
        'instance_count': suffix_instance_counts,
        **suffix_mae
    })

    df_prefix_ges = pd.DataFrame({
        'prefix_length': prefix_lengths,
        'instance_count': prefix_instance_counts,
        **prefix_ges
    })

    df_suffix_ges = pd.DataFrame({
        'suffix_length': suffix_lengths,
        'instance_count': suffix_instance_counts,
        **suffix_ges
    })

    return df_prefix_dls, df_prefix_mae, df_suffix_dls, df_suffix_mae, df_prefix_ges, df_suffix_ges



def create_plots_log(pref_suf_dfs,
                     configs,
                     log_name,
                     include_legend,
                     time_unit='minutes',
                     skip_mae=None):
    """Create four plots:

    #. Average Damerau-Levenstein similarity over the prefix lengths for 
       each of the models (configurations). 

    #. Average MAE RRT over the prefix lengths for 
       each of the models (configurations). 

    #. Average Damerau-Levenstein similarity over the suffix lengths for 
       each of the models (configurations). 

    #. Average MAE RRT over the suffix lengths for 
       each of the models (configurations). 

    Parameters
    ----------
    pref_suf_dfs : list of pd.DataFrame
        Four dataframes:

        #. DataFrame containing prefix lengths, instance counts, and 
           average DLS over each prefix length, for each model.

        #. DataFrame containing prefix lengths, instance counts, and 
           average MAE RRT over each prefix length, for each model.

        #. DataFrame containing suffix lengths, instance counts, and 
           average DLS over each suffix length, for each model.

        #. DataFrame containing suffix lengths, instance counts, and 
           average MAE RRT over each suffix length, for each model.

    configs : list of str
        List of N model names, with the order of the model names 
        corresponding to the order in which the prefix and suffix 
        dictionaries (`prefix_dicts` and `suffix_dicts`) are sorted. 
        Make sure to name them according to one of the keys in the 
        `config_string` dictionary defined below. This should also 
        be accounted for in the `string_list_models` list of the 
        `create_dataframes()` function. 
    log_name : str
        Name of the model for which the plots are created. 
    include_legend : bool 
        If `True`, legend for the different configs will be included. 
    time_unit : str 
        The time unit in which the MAE is displayed. 
    """
    config_string = CONFIG_STRING
    config_styles = CONFIG_STYLES
    _skip_mae = skip_mae or set()

    fontsize = 22
    labelsize = 16
    fig, ax = plt.subplots(2, 2, figsize=(20, 14))
    fig.subplots_adjust(hspace=0.6, wspace=0.4, top=0.88)
    fig.suptitle(EVENT_LOGS.get(log_name, log_name), fontsize=fontsize)

    # Looping over the models / configurations
    for config in configs:
        # Retrieving column names for DLS, MAE and GES for that config
        mae_col = config + '_mae'
        ges_col = config + '_ges'

        df_2 = pref_suf_dfs[1]
        df_4 = pref_suf_dfs[3]
        df_5 = pref_suf_dfs[4]
        df_6 = pref_suf_dfs[5]
        color, linestyle = config_styles[config]
        label = config_string[config]
        if ges_col in df_5.columns and df_5[ges_col].notna().any():
            ax[0, 0].plot(df_5['prefix_length'], df_5[ges_col], label=label, color=color, linestyle=linestyle)
        ax[0, 0].set_title('Average GES over the prefix lengths', fontsize=fontsize)

        if config not in _skip_mae:
            ax[0, 1].plot(df_2['prefix_length'], df_2[mae_col], label=label, color=color, linestyle=linestyle)
        ax[0, 1].set_title('Average MAE ({}) over the prefix lengths'.format(time_unit), fontsize=fontsize)

        if ges_col in df_6.columns and df_6[ges_col].notna().any():
            ax[1, 0].plot(df_6['suffix_length'], df_6[ges_col], label=label, color=color, linestyle=linestyle)
        ax[1, 0].set_title('Average GES over the suffix lengths', fontsize=fontsize)

        if config not in _skip_mae:
            ax[1, 1].plot(df_4['suffix_length'], df_4[mae_col], label=label, color=color, linestyle=linestyle)
        ax[1, 1].set_title('Average MAE ({}) over the suffix lengths'.format(time_unit), fontsize=fontsize)

    ax_ges_pref = ax[0, 0].twinx()
    ax[0, 0].set_ylabel('GES', fontsize=fontsize)
    ax[0, 0].set_xlabel('Prefix Length', fontsize=fontsize)
    ax_ges_pref.plot(df_5['prefix_length'], df_5['instance_count'], label='Number of Instances', color='grey', linestyle='--')
    ax_ges_pref.fill_between(df_5['prefix_length'], 0, df_5['instance_count'], color='grey', alpha=0.3, zorder=0)
    ax_ges_pref.set_ylabel("Instances", color='grey', fontsize=fontsize)
    ax_ges_pref.tick_params('y', colors='grey', labelsize=labelsize)

    ax_mae_pref = ax[0, 1].twinx()
    ax[0, 1].set_ylabel('MAE Remaining Time ({})'.format(time_unit), fontsize=fontsize)
    ax[0, 1].set_xlabel('Prefix Length', fontsize=fontsize)
    ax_mae_pref.plot(df_2['prefix_length'], df_2['instance_count'], label='Number of Instances', color='grey', linestyle='--')
    ax_mae_pref.fill_between(df_2['prefix_length'], 0, df_2['instance_count'], color='grey', alpha=0.3, zorder=0)
    ax_mae_pref.set_ylabel("Instances", color='grey', fontsize=fontsize)
    ax_mae_pref.tick_params('y', colors='grey', labelsize=labelsize)

    ax_ges_suf = ax[1, 0].twinx()
    ax[1, 0].set_ylabel('GES', fontsize=fontsize)
    ax[1, 0].set_xlabel('Suffix Length', fontsize=fontsize)
    ax_ges_suf.plot(df_6['suffix_length'], df_6['instance_count'], label='Number of Instances', color='grey', linestyle='--')
    ax_ges_suf.fill_between(df_6['suffix_length'], 0, df_6['instance_count'], color='grey', alpha=0.3, zorder=0)
    ax_ges_suf.set_ylabel("Instances", color='grey', fontsize=fontsize)
    ax_ges_suf.tick_params('y', colors='grey', labelsize=labelsize)

    ax_mae_suf = ax[1, 1].twinx()
    ax[1, 1].set_ylabel('MAE Remaining Time ({})'.format(time_unit), fontsize=fontsize)
    ax[1, 1].set_xlabel('Suffix Length', fontsize=fontsize)
    ax_mae_suf.plot(df_4['suffix_length'], df_4['instance_count'], label='Number of Instances', color='grey', linestyle='--')
    ax_mae_suf.fill_between(df_4['suffix_length'], 0, df_4['instance_count'], color='grey', alpha=0.3, zorder=0)
    ax_mae_suf.set_ylabel("Instances", color='grey', fontsize=fontsize)
    ax_mae_suf.tick_params('y', colors='grey', labelsize=labelsize)

    for ax_row in ax:
        for axis in ax_row:
            axis.tick_params(axis='both', which='major', labelsize=labelsize)

    if include_legend:
        # Collect handles and labels for the figure's legend
        handles, labels = ax[0, 0].get_legend_handles_labels()

        # Create a single, common legend at the bottom of the figure
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.0002), ncol=6)

    fig.tight_layout()
    plt.show()

################################################################################
# Functionality for Case Length De-noising
################################################################################

def create_pref_suf_dicts(pref_len_tensor,
                          suf_len_tensor,
                          window_size,
                          dam_lev_similarity,
                          MAE_rrt_minutes,
                          ges_tensor=None):
    """Create the prefix and suffix dictionary. This function can be used
    for generating the two dictionaries for a certain model after
    having it evaluated on a test set, or generating the new prefix and
    suffix dictionaries after having performed Case Length De-noising.
    In the latter case, the four input tensors should have been subsetted
    by the de-noising procedure contained within the
    `discard_noisy_cases()` function.

    Parameters
    ----------
    pref_len_tensor : torch.Tensor
        torch.int64 tensor of shape (T,) containing for each of the T
        test set instances (i.e. prefix-suffix pairs) the prefix length
        (in terms of number of events contained within the sequence of
        prefix events).
    suf_len_tensor : torch.Tensor
        torch.int64 tensor of shape (T,) containing for each of the T
        test set instances (i.e. prefix-suffix pairs) the suffix length
        of the ground truth suffix (in terms of number of events
        contained within the sequence of suffix events).
    window_size : int
        The maximum sequence length (both for the prefixes and suffixes)
        corresponding to the event log at hand. Can be found by querying
        the maximum integer value contained within the `pref_len_tensor`
        if needed.
    dam_lev_similarity : torch.Tensor
        torch.float32 tensor of shape (T,) containing the normalized
        Damerau-Levenshtein Similarity score for each of the T test set
        predictions.
    MAE_rrt_minutes : torch.Tensor
        torch.float32 tensor of shape (T,) containing the Mean Absolute
        Error in minutes for each of the T test set predictions.
    ges_tensor : torch.Tensor or None
        torch.float32 tensor of shape (T,) containing the Graph Edit
        Similarity score for each of the T test set predictions. If None,
        the GES entry in each result list will be None.

    Returns
    -------
    results_dict_pref : dict of list
        Dictionary containing the results aggregated over prefix
        lengths. Keys are integer prefix lengths and values are lists of
        four elements:
        [average DLS, average MAE in minutes, total instance count, average GES].
    results_dict_suf : dict of list
        Dictionary containing the results aggregated over suffix
        lengths. Keys are integer suffix lengths and values are lists of
        four elements:
        [average DLS, average MAE in minutes, total instance count, average GES].
    """
    results_dict_pref = {}
    for i in range(1, window_size+1):
        bool_idx = pref_len_tensor==i
        dam_levs = dam_lev_similarity[bool_idx].clone()
        MAE_rrt_i = MAE_rrt_minutes[bool_idx].clone()
        num_inst = dam_levs.shape[0]
        if num_inst > 0:
            avg_dl = (torch.sum(dam_levs) / num_inst).item()
            avg_mae = (torch.sum(MAE_rrt_i) / num_inst).item()
            avg_ges = (torch.sum(ges_tensor[bool_idx]) / num_inst).item() if ges_tensor is not None else None
            results_dict_pref[i] = [avg_dl, avg_mae, num_inst, avg_ges]
    results_dict_suf = {}
    for i in range(1, window_size+1):
        bool_idx = suf_len_tensor==i
        dam_levs = dam_lev_similarity[bool_idx].clone()
        MAE_rrt_i = MAE_rrt_minutes[bool_idx].clone()
        num_inst = dam_levs.shape[0]
        if num_inst > 0:
            avg_dl = (torch.sum(dam_levs) / num_inst).item()
            avg_mae = (torch.sum(MAE_rrt_i) / num_inst).item()
            avg_ges = (torch.sum(ges_tensor[bool_idx]) / num_inst).item() if ges_tensor is not None else None
            results_dict_suf[i] = [avg_dl, avg_mae, num_inst, avg_ges]

    return results_dict_pref, results_dict_suf



def get_corrected_distribution_tensor(pref_len_tensor, suf_len_tensor):
    """Derive a tensor representing the original distribution of case 
    lengths within the test set. I.e. each complete case assigned to the 
    test set, delivers multiple prefix-suffix pairs aka test instances. 
    This method generates a tensor containing the case length (in number 
    of events) distribution from the original set of cases that were 
    used to derive the test set instances, instead of the case length 
    distribution over the ultimately generated test set instances, 
    since the latter would inherently be biased towards the longest 
    case lengths (because a case of length X, will be split up into 
    X instances / prefix-suffix pairs). 

    Parameters
    ----------
    pref_len_tensor : torch.Tensor
        torch.int64 tensor of shape (T,) containing for each of the T 
        test set instances (i.e. prefix-suffix pairs) the prefix length 
        (in terms of number of events contained within the sequence of 
        prefix events). 
    suf_len_tensor : torch.Tensor
        torch.int64 tensor of shape (T,) containing for each of the T 
        test set instances (i.e. prefix-suffix pairs) the suffix length 
        of the ground truth suffix (in terms of number of events 
        contained within the sequence of suffix events). 
    """
    # Deriving a tensor containing the case length for each of the 
    # instances 
    case_len = pref_len_tensor + suf_len_tensor - 1 

    # Get list of unique case lengths present in the test set 
    counts_clen = torch.bincount(case_len)
    unique_caselengths = []
    for integer_value, count in enumerate(counts_clen):
        if count > 0: 
            unique_caselengths.append(integer_value)

    # Initializing dictionary that will store for each case length, the 
    # original amount of cases that were used to create prefix-suffix 
    # pairs pertaining to that total case length. 
    unique_cases_dict = {}
    for uni_len in unique_caselengths:
        # Boolean index. True if prefix-suffix pair corresponding to that 
        # index has case lenght equal to `uni_len` 
        bool_idx = case_len == uni_len

        # Selecting only those prefix lengths of which the indices correspond 
        # to prefix-suffix pairs pertaining to a case of length `uni_len`
        pref_len_subset = pref_len_tensor[bool_idx]

        # Out of that subset, select only the prefix lengths equal 
        # to `uni_len` 
        bool_idx_unilen = pref_len_subset == uni_len 
        # Compute number of unique cases of length `uni_len` used 
        # to generate prefix-suffix pairs
        num_cases_unilen = torch.sum(bool_idx_unilen).item() # integer 

        # Store amount to dictionary 
        unique_cases_dict[uni_len] = num_cases_unilen

    # Deriving list representing original case length distribution 
    # `clen_corr`

    unique_caselens = list(unique_cases_dict.keys())
    # list of number of original cases used to derive 
    # prefix-suffix pairs pertaining to a certain case length
    number_ogcases = list(unique_cases_dict.values())
    clen_corr = []
    for i in range(len(unique_caselens)):
        # retrieve unique case length 
        u_len = unique_caselens[i]

        # retrieve number of original cases of that case length 
        num_og = number_ogcases[i]

        # Add `u_len` `num_og` times 
        clen_corr += [u_len for _ in range(num_og)]
    
    # Making tensor out of it: 
    clen_corr = torch.tensor(data=clen_corr, dtype=torch.float32)

    return clen_corr



def get_subset_bool(pref_len_tensor, 
                    suf_len_tensor, 
                    discard_fraction_lb, 
                    discard_fraction_ub, 
                    return_bounds=False):
    """Create boolean torch.Tensor of shape (T,), with T being the 
    number of test set instances (i.e., prefix-suffix pairs), evaluating 
    to True for those instances of which the case length (in number of 
    events), falls within the range of the percentiles determined by 
    the `discard_fraction_lb` and `discard_fraction_ub` integers. 

    Parameters
    ----------
    pref_len_tensor : torch.Tensor
        torch.int64 tensor of shape (T,) containing for each of the T 
        test set instances (i.e. prefix-suffix pairs) the prefix length 
        (in terms of number of events contained within the sequence of 
        prefix events). 
    suf_len_tensor : torch.Tensor
        torch.int64 tensor of shape (T,) containing for each of the T 
        test set instances (i.e. prefix-suffix pairs) the suffix length 
        of the ground truth suffix (in terms of number of events 
        contained within the sequence of suffix events). 
    discard_fraction_lb : int
        Integer representing the minimum percentile of the case length 
        for which instances are still retained. be an integer between 0 
        and 100, inclusive. 
    discard_fraction_ub : int
        Integer representing the maximum percentile of the case length 
        for which instances are still retained. be an integer between 0 
        and 100, inclusive. 
    return_bounds : bool 
        Whether or not the function should also return the minimum and 
        maximum percentile determined by `discard_fraction_lb` and 
        `discard_fraction_ub`.
    """
    case_len = pref_len_tensor + suf_len_tensor - 1 
    
    # Get tensor mimicing original distribution case length test cases 
    clen_corr = get_corrected_distribution_tensor(pref_len_tensor, suf_len_tensor)

    # Create percentiles tensor corrected distribution case length 
    percentiles = torch.arange(0, 101) / 100.0

    # Compute percentiles 
    corr_distr_percentiles = torch.quantile(clen_corr, percentiles) # shape (101, )

    # Deriving lower bound and upper bound case length for which instances are still 
    # retained 
    lb_case_length = corr_distr_percentiles[discard_fraction_lb].item()
    lb_case_length = int(lb_case_length)
    ub_case_length = corr_distr_percentiles[discard_fraction_ub].item()
    ub_case_length = int(ub_case_length)
    
    retain_bool = (case_len >= lb_case_length) & (case_len <= ub_case_length)
    if return_bounds:
        return retain_bool, lb_case_length, ub_case_length
    else:
        return retain_bool


# ---------------------------------------------------------------------------
# Combined loading and plotting
# ---------------------------------------------------------------------------

def _load_model_dicts_for_log(
    log_name,
    baselines_root,
    approach_results_base,
    approach_key,
    run_ids,
    baselines_to_include,
):
    """Load per-length result dicts for all baselines and one approach model
    for a single event log, averaged across available runs.

    Returns
    -------
    prefix_dicts, suffix_dicts, model_keys : lists as expected by
        `create_dataframes()`. Models with no runs found on disk are
        silently skipped (with a printed [SKIP] notice).
    """
    if baselines_to_include is None:
        baselines_to_include = list(_BASELINE_PATHS.keys())

    prefix_dicts, suffix_dicts, model_keys = [], [], []

    for key in baselines_to_include:
        result_dir = _BASELINE_PATHS[key]
        pref_paths = [
            os.path.join(baselines_root, 'results_per_log', log_name,
                         f'{result_dir}_run{r}', 'TEST_SET_RESULTS',
                         'prefix_length_results_dict.pkl')
            for r in run_ids
        ]
        suf_paths = [p.replace('prefix_length', 'suffix_length') for p in pref_paths]
        pref_avg = _load_and_average_runs(pref_paths)
        suf_avg  = _load_and_average_runs(suf_paths)
        if pref_avg is None or suf_avg is None:
            print(f"[SKIP] {key}: no runs found for {log_name}")
            continue
        prefix_dicts.append(pref_avg)
        suffix_dicts.append(suf_avg)
        model_keys.append(key)

    pref_paths = [
        os.path.join(approach_results_base, f'run_{r}',
                     f'{log_name}_prefix_length_results_dict.pkl')
        for r in run_ids
    ]
    suf_paths = [p.replace('prefix_length', 'suffix_length') for p in pref_paths]
    pref_avg = _load_and_average_runs(pref_paths)
    suf_avg  = _load_and_average_runs(suf_paths)
    if pref_avg is not None and suf_avg is not None:
        prefix_dicts.append(pref_avg)
        suffix_dicts.append(suf_avg)
        model_keys.append(approach_key)
    else:
        print(f"[SKIP] {approach_key}: no runs found for {log_name}")

    return prefix_dicts, suffix_dicts, model_keys


def load_and_plot(
    log_name,
    baselines_root,
    approach_results_base,
    approach_key='GATv2_GRU',
    run_ids=(1, 2, 3, 4, 5),
    include_legend=True,
    baselines_to_include=None,
):
    """Load per-length result dicts for all baselines and one approach model,
    average across available runs, and produce the 2×3 comparison plot.

    Parameters
    ----------
    log_name : str
        Event log name (e.g. 'Sepsis').
    baselines_root : str
        Absolute path to the baselines/SuffixTransformerNetwork/ directory.
        Per-length pickles are expected at:
        {baselines_root}/results_per_log/{log_name}/{MODEL}_results_run{N}/
            TEST_SET_RESULTS/prefix_length_results_dict.pkl
    approach_results_base : str
        Absolute path to the approach results directory (e.g.
        .../results_time_gatv2_gru_nb_v5/). Per-length pickles are expected at:
        {approach_results_base}/run_{N}/{log_name}_prefix_length_results_dict.pkl
    approach_key : str
        Config key for the approach model. Must match an entry in
        config_string/config_styles inside create_plots_log (default 'GATv2_GRU').
    run_ids : tuple of int
        Run IDs to look for. Missing runs are silently skipped.
    include_legend : bool
        Passed to create_plots_log.
    baselines_to_include : list of str or None
        Subset of _BASELINE_PATHS keys to include. None = all seven.
    """
    prefix_dicts, suffix_dicts, model_keys = _load_model_dicts_for_log(
        log_name, baselines_root, approach_results_base, approach_key,
        run_ids, baselines_to_include,
    )

    if not model_keys:
        print(f"No results found for {log_name}.")
        return

    pref_suf_dfs = create_dataframes(prefix_dicts, suffix_dicts, model_keys)
    skip_mae = {'BEST'} & set(model_keys)
    create_plots_log(pref_suf_dfs, model_keys, log_name, include_legend,
                     skip_mae=skip_mae)


def plot_all_logs(
    log_names,
    baselines_root,
    approach_results_base,
    save_path,
    metric='ges',
    length_axis='both',
    logs_per_row=1,
    approach_key='GATv2_GRU',
    run_ids=(1, 2, 3, 4, 5),
    baselines_to_include=None,
):
    """Build a single figure with the chosen metric plotted over prefix
    and/or suffix length, for every log in `log_names`, and save it to
    disk as a PNG.

    Logs are laid out `logs_per_row` at a time per grid row (default 1,
    i.e. one log per row as before). Each log occupies 1 sub-column
    (`length_axis='prefix'` or `'suffix'`) or 2 sub-columns
    (`length_axis='both'`), so the figure has `logs_per_row * (1 or 2)`
    grid columns and `ceil(len(log_names) / logs_per_row)` grid rows.
    The model/approach legend is drawn once at the bottom of the whole
    figure, in a large font.

    Parameters
    ----------
    log_names : list of str
        Event log names (e.g. ['Sepsis', 'BPIC17', ...]).
    baselines_root : str
        Absolute path to the baselines/SuffixTransformerNetwork/ directory.
    approach_results_base : str
        Absolute path to the approach results directory.
    save_path : str
        File path the PNG figure is written to (parent dirs are created
        if needed).
    metric : str
        One of 'ges', 'dl', 'mae_rrt' (keys of METRIC_CONFIG).
    length_axis : str
        One of 'prefix', 'suffix', 'both'. Selects which column(s) are drawn.
    logs_per_row : int
        How many logs to place side by side per grid row, to reduce the
        number of rows (default 1: one log per row).
    approach_key : str
        Config key for the approach model (default 'GATv2_GRU').
    run_ids : tuple of int
        Run IDs to look for. Missing runs are silently skipped.
    baselines_to_include : list of str or None
        Subset of _BASELINE_PATHS keys to include. None = all seven.
    """
    if metric not in METRIC_CONFIG:
        raise ValueError(f"metric must be one of {list(METRIC_CONFIG)}, got {metric!r}")
    if length_axis not in ('prefix', 'suffix', 'both'):
        raise ValueError(f"length_axis must be one of 'prefix', 'suffix', 'both', got {length_axis!r}")

    metric_cfg = METRIC_CONFIG[metric]
    col_suffix = metric_cfg['col_suffix']

    columns = []
    if length_axis in ('prefix', 'both'):
        columns.append({
            'df_idx': metric_cfg['prefix_df_idx'],
            'length_col': 'prefix_length',
            'title': 'Prefix Length',
        })
    if length_axis in ('suffix', 'both'):
        columns.append({
            'df_idx': metric_cfg['suffix_df_idx'],
            'length_col': 'suffix_length',
            'title': 'Suffix Length',
        })

    fontsize = 30
    labelsize = 25
    legend_fontsize = 35

    n_logs = len(log_names)
    logs_per_row = min(logs_per_row, n_logs)
    n_metric_cols = len(columns)
    n_grid_rows = (n_logs + logs_per_row - 1) // logs_per_row
    n_grid_cols = logs_per_row * n_metric_cols
    fig, axes = plt.subplots(n_grid_rows, n_grid_cols, figsize=(10 * n_grid_cols, 8 * n_grid_rows), squeeze=False)
    #fig.suptitle(f"{metric_cfg['ylabel']} across event logs", fontsize=fontsize)

    legend_handles = {}

    for i, log_name in enumerate(log_names):
        grid_row = i // logs_per_row
        log_col_group = i % logs_per_row

        prefix_dicts, suffix_dicts, model_keys = _load_model_dicts_for_log(
            log_name, baselines_root, approach_results_base, approach_key,
            run_ids, baselines_to_include,
        )
        if not model_keys:
            print(f"No results found for {log_name}.")
            continue

        pref_suf_dfs = create_dataframes(prefix_dicts, suffix_dicts, model_keys)

        for col_idx, column in enumerate(columns):
            ax = axes[grid_row, log_col_group * n_metric_cols + col_idx]
            df = pref_suf_dfs[column['df_idx']]

            for model in model_keys:
                if metric == 'mae_rrt' and model == 'BEST':
                    continue  # BEST has no real RRT predictions
                data_col = f'{model}{col_suffix}'
                color, linestyle = CONFIG_STYLES[model]
                label = CONFIG_STRING[model]
                if data_col in df.columns and df[data_col].notna().any():
                    line, = ax.plot(df[column['length_col']], df[data_col],
                                     color=color, linestyle=linestyle, label=label)
                    legend_handles.setdefault(label, line)

            ax_twin = ax.twinx()
            ax_twin.plot(df[column['length_col']], df['instance_count'],
                         color='grey', linestyle='--')
            ax_twin.fill_between(df[column['length_col']], 0, df['instance_count'],
                                 color='grey', alpha=0.3, zorder=0)
            ax_twin.tick_params('y', colors='grey', labelsize=labelsize)

            ax.set_title(EVENT_LOGS.get(log_name, log_name), fontsize=fontsize+10)
            ax.set_ylabel(metric_cfg['ylabel'], fontsize=fontsize)

            #if row == 0:
            ax_twin.set_ylabel('Instances', color='grey', fontsize=fontsize)

            #if row == n_logs - 1:
            ax.set_xlabel(column['title'], fontsize=fontsize)

            ax.tick_params(axis='both', which='major', labelsize=labelsize)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.5, bottom=0.20)
    fig.legend(legend_handles.values(), legend_handles.keys(),
               loc='lower center', bbox_to_anchor=(0.5, 0.0),
               ncol=min(len(legend_handles), 3), fontsize=legend_fontsize)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()