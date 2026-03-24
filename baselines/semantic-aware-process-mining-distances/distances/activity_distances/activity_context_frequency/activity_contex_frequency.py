from collections import defaultdict, Counter

import numpy as np

from distances.activity_distances.data_util.algorithm import give_log_padding, get_ngrams_dict, get_context_dict, \
    get_cosine_distance_dict
from math import floor
#from additional_scripts.embedding_as_heatmap import plot_heatmap
from distances.activity_distances.activity_activity_co_occurence.activity_activity_co_occurrence import get_bag_of_words_context_dict
def get_activity_embeddings(context_dict):
    del context_dict["."]
    all_contexts = sorted(set(context for activity in context_dict.values() for context in activity))
    context_index = {context: i for i, context in enumerate(all_contexts)}

    # Create embeddings
    embeddings = {}
    for activity, contexts in context_dict.items():
        vector = np.zeros(len(all_contexts))
        for context, freq in contexts.items():
            vector[context_index[context]] = freq
        embeddings[activity] = vector

    # Print results
    for activity, vector in embeddings.items():
        print(f"{activity}: {vector}")


def get_activity_context_frequency_matrix(log, alphabet, ngram_size, bag_of_words):
    # bag_of_words = 0, n-gram l r
    # bag_of_words = 1,  n-gram but count as bag of words
    # bag_of_words = 2,  n-grams collapse to multi sets
    activity_freq_dict = Counter(word for sublist in log for word in sublist)
    log = give_log_padding(log, ngram_size)
    ngrams_dict = get_ngrams_dict(log, ngram_size)
    context_dict = get_context_dict(ngrams_dict)

    # common_context_dict = get_common_context_dict(context_dict)
    if bag_of_words == 2:
        context_dict = get_bag_of_words_context_dict(context_dict)

        # Collect all unique keys from the inner dictionaries
        all_contexts = {key for inner_dict in context_dict.values() for key in inner_dict}
        context_index = {context: i for i, context in enumerate(all_contexts)}
        context_freq_dict = defaultdict(int)

        # Iterate over inner dictionaries
        for inner_dict in context_dict.values():
            for key, value in inner_dict.items():
                context_freq_dict[key] += value  # Sum values for each key

        embeddings = {}
        for activity in alphabet:
            vector = np.zeros(len(all_contexts))
            embeddings[activity] = vector

        for activity in context_dict.keys():
            for context in context_dict[activity].keys():
                embeddings[activity][context_index[context]] += context_dict[activity][context]

        """ 
        # New dictionary where keys are multisets (Counter objects)
        context_freq_dict = defaultdict(int)

        index_to_remove = floor((ngram_size - 1) / 2)

        for key, value in ngrams_dict.items():
            key_as_multiset = frozenset(Counter(key).items())  # Convert Counter to a hashable frozenset
            context_freq_dict[key_as_multiset] += value  # Sum values for identical multisets

        # Create embeddings
        embeddings = {}
        for activity in alphabet:
            vector = np.zeros(len(context_freq_dict.keys()))
            embeddings[activity] = vector

        context_index = {context: i for i, context in enumerate(context_freq_dict.keys())}

        for multi_set, freq in context_freq_dict.items():
            for activity, count in multi_set:
                if activity != ".":
                    embeddings[activity][context_index[multi_set]] += count * freq
        """

    if bag_of_words == 1:
        # Create embeddings
        embeddings = {}
        for activity in alphabet:
            vector = np.zeros(len(ngrams_dict.keys()))
            embeddings[activity] = vector

        context_freq_dict = ngrams_dict
        context_index = {context: i for i, context in enumerate(ngrams_dict.keys())}

        for ngram, freq in ngrams_dict.items():
            for activity in ngram:
                if activity != ".":
                    embeddings[activity][context_index[ngram]] += freq

    if bag_of_words == 0:
        # Collect all unique keys from the inner dictionaries
        all_contexts = {key for inner_dict in context_dict.values() for key in inner_dict}
        context_index = {context: i for i, context in enumerate(all_contexts)}
        context_freq_dict = defaultdict(int)

        # Iterate over inner dictionaries
        for inner_dict in context_dict.values():
            for key, value in inner_dict.items():
                context_freq_dict[key] += value  # Sum values for each key


        embeddings = {}
        for activity in alphabet:
            vector = np.zeros(len(all_contexts))
            embeddings[activity] = vector

        for activity in context_dict.keys():
            for context in context_dict[activity].keys():
                embeddings[activity][context_index[context]] += context_dict[activity][context]


    distance_matrix = get_cosine_distance_dict(embeddings)
    return distance_matrix, embeddings, activity_freq_dict, context_freq_dict, context_index
