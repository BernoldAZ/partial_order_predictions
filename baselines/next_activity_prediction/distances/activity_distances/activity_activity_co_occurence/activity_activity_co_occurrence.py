from distances.activity_distances.data_util.algorithm import give_log_padding, get_ngrams_dict, get_context_dict, get_cosine_distance_dict
from distances.activity_distances.bose_2009_context_aware_trace_clustering.substitution_scores import get_common_context_dict, get_cooccurrence_counts
import numpy as np
from collections import defaultdict, Counter
from typing import List, Tuple, Dict
from math import floor

def get_cooccurrence_counts(context_dict: Dict[Tuple[str, ...], int], common_context_dict: Dict[Tuple[str, str], List[Tuple[str, ...]]], unique_activities) -> Dict[Tuple[str, str], Dict[Tuple[str, ...], int]]:
    cooccurrence_counts_dict = defaultdict(int) # Nested defaultdict for co-occurance frequencies
    #co-occurrence combinations is for ‘like’ acitivities
    for activity in context_dict.keys():
        cooccurrence_counts = 0
        for context in context_dict[activity]:
            cooccurrence_counts_dict[(activity, activity)] += (context_dict[activity].get(context) + (context_dict[activity].get(context)))
            #cooccurrence_counts += int((context_dict[activity].get(entry) * (context_dict[activity].get(entry)-1)) / 2) # n over 2
        #cooccurrence_counts_dict[entry][(activity, activity)] = cooccurrence_counts
    #co-occurrence combinations is for ‘un-like’ acitivities
    for activity_pair in common_context_dict.keys():
        cooccurrence_counts = 0
        for context in common_context_dict[activity_pair]:
            cooccurrence_counts_dict[activity_pair] += context_dict[activity_pair[0]][context] + context_dict[activity_pair[1]][context]
        #cooccurrence_counts_dict[cooccurrence][activity_pair] =  cooccurrence_counts
    return dict(cooccurrence_counts_dict) # Convert to regular dict

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

def get_bag_of_words_context_dict(context_dict):
    bag_of_words_context_dict = defaultdict(lambda: defaultdict(int))  # Nested defaultdict for context frequencies
    for activity, contexts in context_dict.items():
        for context, freq in context_dict[activity].items():
            key_as_multiset = frozenset(Counter(context).items())
            bag_of_words_context_dict[activity][key_as_multiset] += freq
    return {k: dict(v) for k, v in bag_of_words_context_dict.items()}  # Convert inner defaultdicts to regular



def get_bag_of_words_common_dict(context_dict: Dict[str, Dict[Tuple[str, ...], int]]) -> Dict[
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


def get_activity_activity_co_occurence_matrix(log, alphabet, ngram_size, bag_of_words ):
    activity_freq_dict = Counter(word for sublist in log for word in sublist)
    log = give_log_padding(log, ngram_size)
    ngrams_dict = get_ngrams_dict(log, ngram_size)
    context_dict = get_context_dict(ngrams_dict)
    #Line 4

    activity_index = {activity: i for i, activity in enumerate(alphabet)}

    # Create embeddings
    embeddings = {}
    for activity in alphabet:
        vector = np.zeros(len(alphabet))
        embeddings[activity] = vector

    if bag_of_words:
        context_dict = get_bag_of_words_context_dict(context_dict)

        """ 
        context_dict = get_bag_of_words_context_dict(context_dict)
        common_bag_of_words_dict = get_common_bag_of_words_dict(alphabet)
        for n_gram, freq in ngrams_dict.items():
            for word in n_gram:
                if word != ".":
                    for other_word in n_gram:
                        if other_word != "." and other_word != word:
                            embeddings[word][activity_index[other_word]] += freq
        """
    common_context_dict = get_common_context_dict(context_dict)
    cooccurrence_counts = get_cooccurrence_counts(context_dict, common_context_dict, alphabet)
    for activity_pair in cooccurrence_counts.keys():
        embeddings[activity_pair[0]][activity_index[activity_pair[1]]] = cooccurrence_counts[activity_pair]

    distance_matrix = get_cosine_distance_dict(embeddings)

    return distance_matrix, embeddings, activity_freq_dict, activity_index
