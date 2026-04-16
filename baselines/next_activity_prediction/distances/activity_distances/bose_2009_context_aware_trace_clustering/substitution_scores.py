# =============================================================================
# Based on:
# Bose, R.P. Jagadeesh Chandra, and Wil M.P. van der Aalst.
# "Context Aware Trace Clustering: Towards Improving Process Mining Results."
# Proceedings of the 2009 SIAM International Conference on Data Mining.
# Society for Industrial and Applied Mathematics, 2009.
# https://doi.org/10.1137/1.9781611972795.35
# =============================================================================

from collections import defaultdict
from typing import List, Tuple, Dict
from math import log


def get_common_context_dict(context_dict: Dict[str, Dict[Tuple[str, ...], int]]) -> Dict[
    Tuple[str, str], List[Tuple[str, ...]]]:
    common_context_dict = defaultdict(list)

    grams = list(context_dict.keys())
    for i in range(len(grams)):
        for j in range(len(grams)):
            if i !=j:
                gram1 = grams[i]
                gram2 = grams[j]

                contexts1 = set(context_dict[gram1].keys())
                contexts2 = set(context_dict[gram2].keys())

                common_contexts = contexts1.intersection(contexts2)
                if common_contexts:
                    common_context_dict[(gram1, gram2)] = list(common_contexts)

    return dict(common_context_dict)

def get_cooccurrence_counts(context_dict: Dict[Tuple[str, ...], int], common_context_dict: Dict[Tuple[str, str], List[Tuple[str, ...]]], unique_activities) -> Dict[Tuple[str, str], Dict[Tuple[str, ...], int]]:
    cooccurrence_counts_dict = defaultdict(int) # Nested defaultdict for co-occurance frequencies
    #co-occurrence combinations is for ‘like’ acitivities
    for activity in context_dict.keys():
        cooccurrence_counts = 0
        for context in context_dict[activity]:
            cooccurrence_counts_dict[(activity, activity)] += (context_dict[activity].get(context) * (context_dict[activity].get(context)-1)) / 2
            #cooccurrence_counts += int((context_dict[activity].get(entry) * (context_dict[activity].get(entry)-1)) / 2) # n over 2
        #cooccurrence_counts_dict[entry][(activity, activity)] = cooccurrence_counts
    #co-occurrence combinations is for ‘un-like’ acitivities
    for activity_pair in common_context_dict.keys():
        cooccurrence_counts = 0
        for context in common_context_dict[activity_pair]:
            cooccurrence_counts_dict[activity_pair] += context_dict[activity_pair[0]][context] * context_dict[activity_pair[1]][context]
        #cooccurrence_counts_dict[cooccurrence][activity_pair] =  cooccurrence_counts
    return dict(cooccurrence_counts_dict) # Convert to regular dict

def get_normalized_co_occurrence_counts(cooccurrence_counts:Dict[Tuple[str, str], Dict[Tuple[str, ...], int]]) -> Dict[Tuple[str, str], float]:
    normalized_co_occurrence_counts_dict = defaultdict(float)
    sum_of_all = 0
    for value in cooccurrence_counts.values():
        sum_of_all += value
    for activity_pair in cooccurrence_counts.keys():
        normalized_co_occurrence_counts_dict[activity_pair] = cooccurrence_counts[activity_pair]/sum_of_all
    return dict(normalized_co_occurrence_counts_dict)

    # normalized_co_occurrence_counts_dict = defaultdict(float)
    # sum_of_cooccurrence_combinations_of_allcontext_dict = defaultdict(float)
    # all_cooccurrence_set = set()
    # for activity_pair in cooccurrence_counts.keys():
    #     for cooccurrence in list(cooccurrence_counts[activity_pair].keys()):
    #         all_cooccurrence_set.add(cooccurrence)
    # for activity_pair in all_cooccurrence_set:
    #     sum = 0
    #     for cooccurrence in cooccurrence_counts.keys():
    #         sum += cooccurrence_counts[cooccurrence].get(activity_pair, 0)
    #     sum_of_cooccurrence_combinations_of_allcontext_dict[activity_pair] = sum
    # sum_of_all = 0
    # for value in sum_of_cooccurrence_combinations_of_allcontext_dict.values():
    #     sum_of_all += value
    # for activity_pair in sum_of_cooccurrence_combinations_of_allcontext_dict.keys():
    # return dict(normalized_co_occurrence_counts_dict)

def get_probabilities_of_symbol_occurrence(normalized_cooccurrence_counts_of_allcontext_dict: Dict[Tuple[str, str], float], unique_activities: List[str]) -> Dict[str, float]:
    probabilities_of_symbol_occurrence = defaultdict(float)
    sum = 0
    for activity_1 in unique_activities:
        probability = 0
        #probability = sum_of_cooccurrence_combinations_of_allcontext_dict.get((activity,activity), 0.0)
        for activity_2 in unique_activities:
            probability += normalized_cooccurrence_counts_of_allcontext_dict.get((activity_1, activity_2), 0.0)
        probabilities_of_symbol_occurrence[activity_1] = probability
        sum += probability
    #print(sum)
    return dict(probabilities_of_symbol_occurrence)

def compute_substitution_scores(probabilities_of_symbol_occurrence, unique_activities: List[str], sum_of_cooccurrence_combinations_of_allcontext_dict) -> Dict[Tuple[str, str], float]:
    substitution_scores = defaultdict(float)
    for activity_1 in unique_activities:
        for activity_2 in unique_activities:
            if activity_1 == activity_2:
                expected_value = probabilities_of_symbol_occurrence.get(activity_1, 0.0) * probabilities_of_symbol_occurrence.get(activity_1, 0.0)
            else:
                expected_value = 2 * probabilities_of_symbol_occurrence[activity_1] * probabilities_of_symbol_occurrence[activity_2]
            sum_of_cooccurrence = sum_of_cooccurrence_combinations_of_allcontext_dict.get((activity_1, activity_2), 0.0)
            if expected_value == 0.0:
                substitution_scores[(activity_1, activity_2)] = 0.0
            elif sum_of_cooccurrence == 0.0:
                #substitution_scores[(activity_1, activity_2)] = -log2(1/expected_value)
                #substitution_scores[(activity_1, activity_2)] = int(-1*log(1/expected_value))
                substitution_scores[(activity_1, activity_2)] = -1*log(1/expected_value)

            else:
                #substitution_scores[(activity_1, activity_2)] = int(log(sum_of_cooccurrence/expected_value))
                substitution_scores[(activity_1, activity_2)] = log(sum_of_cooccurrence/expected_value)

    return dict(substitution_scores)

#Algorithm 1: Algorithm to derive substitution scores
def get_substitution_scores(unique_activity_set, context_dict):
    #Line 4
    common_context_dict = get_common_context_dict(context_dict)
    #print(common_context_dict)
    #Line 5
    cooccurrence_counts = get_cooccurrence_counts(context_dict, common_context_dict, unique_activity_set)
    #print(cooccurrence_counts)
    #Line 6-8
    normalized_cooccurrence_counts_of_allcontext_dict = get_normalized_co_occurrence_counts(cooccurrence_counts)
    #print(normalized_cooccurrence_counts_of_allcontext_dict)
    #Line 9
    probabilities_of_symbol_occurrence = get_probabilities_of_symbol_occurrence(normalized_cooccurrence_counts_of_allcontext_dict, unique_activity_set)
    #print(probabilities_of_symbol_occurrence)
    #Line 10-11
    substitution_scores = compute_substitution_scores(probabilities_of_symbol_occurrence, unique_activity_set, normalized_cooccurrence_counts_of_allcontext_dict)
    return substitution_scores, probabilities_of_symbol_occurrence

#print(substitution_scores.get(('Repair (Simple)', 'Repair (Complex)')))
#print(substitution_scores.get(('Repair (Simple)', 'Repair (Simple)')))
#print(substitution_scores.get(('Test Repair', 'Archive Repair')))
#print(substitution_scores.get(('Inform User', 'Archive Repair')))



"""
Expected output:
{
    ('b', 'c'): [('a', 'c')],
    ('c', 'd'): [('b', 'c')]
}

Explanation:
- The input list of lists is [['a', 'b', 'c', 'd'], ['b', 'c', 'd', 'e'], ['a', 'b', 'c']].
- For ngram size of 3:
  - In the first sublist ['a', 'b', 'c', 'd']:
    - The n-grams are ('a', 'b', 'c') and ('b', 'c', 'd').
  - In the second sublist ['b', 'c', 'd', 'e']:
    - The n-grams are ('b', 'c', 'd') and ('c', 'd', 'e').
  - In the third sublist ['a', 'b', 'c']:
    - The n-gram is ('a', 'b', 'c').
- The n-gram counts are:
  {('a', 'b', 'c'): 2, ('b', 'c', 'd'): 2, ('c', 'd', 'e'): 1}.
- The context dictionary is:
  - 'b' is the middle gram in ('a', 'b', 'c') and occurs twice with context ('a', 'c').
  - 'c' is the middle gram in ('b', 'c', 'd') and occurs twice with context ('b', 'd').
  - 'd' is the middle gram in ('c', 'd', 'e') with context ('c', 'e') once and in ('b', 'c') with context ('b', 'c') once.
- Common contexts are:
  - 'b' and 'c' share context ('a', 'c').
  - 'c' and 'd' share context ('b', 'c').
"""
"""
log = [['a', 'b', 'c', 'd'], ['b', 'c', 'd', 'e'], ['a', 'b', 'c'], ['a', 'd', 'c']]
ngram_size = 3
ngrams_dict = get_ngrams_dict(log, ngram_size)
ngrams_dict: 
{
    ('a', 'b', 'c'): 2,
    ('b', 'c', 'd'): 2,
    ('c', 'd', 'e'): 1,
    ('a', 'd', 'c'): 1
}
context_dict = get_context_dict(ngrams_dict)
context_dict: 
{
    'b': {('a', 'c'): 2},
    'c': {('b', 'd'): 2},
    'd': {('c', 'e'): 1, ('a', 'c'): 1}
}
common_context_dict = get_common_context_dict(context_dict)
{
    ('b', 'd'): [('a', 'c')] # activity b and d both happend as "abc" and "adc"
}
"""