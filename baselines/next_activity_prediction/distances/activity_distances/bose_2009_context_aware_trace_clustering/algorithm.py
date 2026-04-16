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

from distances.activity_distances.data_util.algorithm import give_log_padding, get_ngrams_dict, get_context_dict
from distances.activity_distances.bose_2009_context_aware_trace_clustering.substitution_scores import get_substitution_scores
import numpy as np

def get_substitution_and_insertion_scores(log, alphabet, ngram_size):
    log = give_log_padding(log, ngram_size)

    #Line 2 in Algoirthm 1 & 2
    ngrams_dict = get_ngrams_dict(log, ngram_size)
    #Line 3 in Algoirthm 1 & 2
    context_dict = get_context_dict(ngrams_dict)

    #Algorithm 1: Algorithm to derive substitution scores
    substitution_scores, probabilities_of_symbol_occurrence  = get_substitution_scores(alphabet, context_dict)
    #Algorithm 2: Algorithm to derive insertion scores
    #insertion_scores = get_insertion_scores(alphabet, context_dict, ngram_size, probabilities_of_symbol_occurrence)
    # 1. Extract unique row labels (and assume the same set applies for columns)
    keys = sorted({key[0] for key in substitution_scores.keys()})
    # Optionally, verify that the same set exists in the second element of each key tuple:
    assert keys == sorted({key[1] for key in substitution_scores.keys()}), "Matrix is not square."

    # 2. Build the embedding dictionary where each row maps to its corresponding numpy array.
    embedding_dict = {
        row: np.array([substitution_scores[(row, col)] for col in keys])
        for row in keys
    }
    return substitution_scores, embedding_dict #, insertion_scores