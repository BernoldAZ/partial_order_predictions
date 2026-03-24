import numpy as np
import math
from distances.activity_distances.data_util.algorithm import give_log_padding, get_ngrams_dict, get_context_dict, \
    get_cosine_distance_dict

def get_activity_context_frequency_matrix_pmi(embeddings, activity_freq_dict, context_freq_dict, context_index, ppmi):
    pmi_embeddings = {}
    for activity in embeddings.keys():
        vector = np.zeros(len(embeddings[activity]))
        pmi_embeddings[activity] = vector

    activity_count = sum(activity_freq_dict.values())
    #transform dict from context -> id to id -> context
    index_context = {i: context for context, i in context_index.items()}

    for activity in embeddings.keys():
        for context_id, context_freq in enumerate(embeddings[activity]):
            pij = context_freq / activity_count
            pi = activity_freq_dict[activity] / activity_count
            pj = context_freq_dict[index_context[context_id]] / activity_count
            if pij == 0:
                pmi = 0
            else:
                pmi = math.log(pij /(pi * pj))
            if ppmi == 0:
                pmi_embeddings[activity][context_id] = pmi
            else:
                pmi_embeddings[activity][context_id] = max(pmi, 0)

    distance_matrix = get_cosine_distance_dict(pmi_embeddings)
    return distance_matrix, pmi_embeddings

def get_activity_activity_frequency_matrix_pmi(embeddings, activity_freq_dict, activity_index, ppmi):
    pmi_embeddings = {}
    for activity in embeddings.keys():
        vector = np.zeros(len(embeddings[activity]))
        pmi_embeddings[activity] = vector

    activity_count = sum(activity_freq_dict.values())
    #transform dict from activity -> id to id -> activity
    index_activity = {i: activity for activity, i in activity_index.items()}


    for activity in embeddings.keys():
        for activity_id, co_occurrence_count in enumerate(embeddings[activity]):
            pij = co_occurrence_count / activity_count
            pi = activity_freq_dict[activity] / activity_count
            pj = activity_freq_dict[index_activity[activity_id]] / activity_count
            if pij == 0:
                pmi = 0
            else:
                pmi = math.log(pij / (pi * pj))
            if ppmi == 0:
                pmi_embeddings[activity][activity_id] = pmi
            else:
                pmi_embeddings[activity][activity_id] = max(pmi, 0)

    distance_matrix = get_cosine_distance_dict(pmi_embeddings)
    return distance_matrix, pmi_embeddings