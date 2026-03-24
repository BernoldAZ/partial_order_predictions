# =============================================================================
# Based on:
# Bose, R.P. Jagadeesh Chandra, and Wil M.P. van der Aalst.
# "Context Aware Trace Clustering: Towards Improving Process Mining Results."
# Proceedings of the 2009 SIAM International Conference on Data Mining.
# Society for Industrial and Applied Mathematics, 2009.
# https://doi.org/10.1137/1.9781611972795.35
# =============================================================================

from typing import List, Tuple, Dict
from collections import defaultdict
from math import log

def get_same_symbols_cooccurrence_counts(context_dict: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, str], Dict[Tuple[str, ...], int]]:
    cooccurrence_counts_dict = defaultdict(lambda: defaultdict(int))  # Nested defaultdict for co-occurance frequencies
    #co-occurrence combinations is for ‘like’ acitivities
    for activity in context_dict:
        for context in context_dict[activity]:
            cooccurrence_counts_dict[activity][context] = (context_dict[activity].get(context) * (
                            context_dict[activity].get(context) - 1)) / 2
    return {k: dict(v) for k, v in cooccurrence_counts_dict.items()}  # Convert inner defaultdicts to regular dicts

def get_right_given_left_count(alphabet, context_dict, cooccurrence_counts_dict, ngram_size):
    right_given_left_count = defaultdict(lambda: defaultdict(float))
    middle_index = ngram_size // 2
    for activity in context_dict.keys():
        for context in context_dict[activity]:
            left_context = context[:middle_index]
            right_context = context[middle_index:]
            right_given_left_count[activity][left_context] += cooccurrence_counts_dict[
                activity].get(left_context + right_context, 0)
    return right_given_left_count

def get_norm_of_activities(right_given_left_count_dict, context_dict):
    norm_of_activity_dict = defaultdict(int)
    for activity in right_given_left_count_dict.keys():
        norm = 0
        for left_context in right_given_left_count_dict[activity].keys():
            norm += right_given_left_count_dict[activity][left_context]
        norm_of_activity_dict[activity] = norm
    return norm_of_activity_dict

def get_norm_right_given_left_count(right_given_left_count_dict, norm_of_activity_dict):
    for activity in right_given_left_count_dict.keys():
        for left_context in right_given_left_count_dict[activity].keys():
            right_given_left_count_dict[activity][left_context] /= norm_of_activity_dict[activity]
    return right_given_left_count_dict

def get_insertion_scores_right_given_left(norm_right_given_left_count_dict, probabilities_of_symbol_occurrence_dict):
    insertion_scores_dict = defaultdict(float)
    for activity in norm_right_given_left_count_dict.keys():
        for left_context in norm_right_given_left_count_dict[activity].keys():
            if left_context[-1] != ".":
                value = norm_right_given_left_count_dict[activity][left_context]/probabilities_of_symbol_occurrence_dict[activity]*probabilities_of_symbol_occurrence_dict[left_context[-1]]
                if value != 0:
                    insertion_scores_dict[(activity, left_context)] = int(log(value))

def get_insertion_scores(alphabet, context_dict, ngram_size, probabilities_of_symbol_occurrence_dict):
    #Line 4
    same_symbols_cooccurrence_counts_dict = get_same_symbols_cooccurrence_counts(context_dict)
    #Line 5
    right_given_left_count_dict = get_right_given_left_count(alphabet, context_dict, same_symbols_cooccurrence_counts_dict, ngram_size)
    #Line 6
    norm_of_activity_dict = get_norm_of_activities(right_given_left_count_dict, context_dict)
    #Line 8
    norm_right_given_left_count_dict = get_norm_right_given_left_count(right_given_left_count_dict, norm_of_activity_dict)
    #Line 9
    insertion_scores_right_given_left_dict = get_insertion_scores_right_given_left(norm_right_given_left_count_dict, probabilities_of_symbol_occurrence_dict)
    return insertion_scores_right_given_left_dict