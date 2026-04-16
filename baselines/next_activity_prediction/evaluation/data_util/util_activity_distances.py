import gc
import math
import os
import shutil
import sys
from collections import defaultdict
from typing import List
import re
import psutil
import numpy as np
from definitions import ROOT_DIR
from distances.activity_distances.activity_activity_co_occurence.activity_activity_co_occurrence import \
    get_activity_activity_co_occurence_matrix
from distances.activity_distances.bose_2009_context_aware_trace_clustering.algorithm import \
    get_substitution_and_insertion_scores
from distances.activity_distances.chiorrini_2022_embedding_process_structure.embedding_process_structure import \
    get_embedding_process_structure_distance_matrix
from distances.activity_distances.de_koninck_2018_act2vec.algorithm import get_act2vec_distance_matrix
from distances.activity_distances.de_koninck_2018_act2vec.our_hyperparas import get_act2vec_distance_matrix_our
from distances.activity_distances.gamallo_fernandez_2023_context_based_representations.src.embeddings_generator.main_new import \
    get_context_based_distance_matrix
from distances.activity_distances.activity_context_frequency.activity_contex_frequency import get_activity_context_frequency_matrix
from distances.activity_distances.pmi.pmi import get_activity_context_frequency_matrix_pmi, get_activity_activity_frequency_matrix_pmi

def get_alphabet(log: List[List[str]]) -> List[str]:
    unique_activities = set()
    for trace in log:
        for activity in trace:
            unique_activities.add(activity)
    return list(unique_activities)


def get_normalized_activity_distance_matrix_dict_list(args):
    log_control_flow_perspective, activity_distance_function, alphabet = args

    get_distance_matrix = get_activity_distance_matrix(log_control_flow_perspective, activity_distance_function,
                                                       alphabet)

    # Step 1: Find the minimum and maximum values
    min_value = min(get_distance_matrix[activity_distance_function].values())
    max_value = max(get_distance_matrix[activity_distance_function].values())

    # Step 2: Normalize the values
    normalized_data = {key: (value - min_value) / (max_value - min_value) for key, value in
                       get_distance_matrix[activity_distance_function].items()}

    distance_dict = dict()
    distance_dict[activity_distance_function] = normalized_data

    return distance_dict


def get_activity_distance_matrix(log_control_flow_perspective, activity_distance_function, alphabet):
    window_size = 3
    activity_distance_matrix_dict = defaultdict()
    if "Bose 2009 Substitution Scores" == activity_distance_function:
        activity_distance_matrix = get_substitution_and_insertion_scores(
            log_control_flow_perspective,
            alphabet, window_size)
        activity_distance_matrix_dict[activity_distance_function] = activity_distance_matrix
    elif "De Koninck 2018 act2vec" == activity_distance_function[:23]:
        if activity_distance_function[24:] == "CBOW":
            sg = 0
        else:
            sg = 1
        act2vec_distance_matrix = get_act2vec_distance_matrix(log_control_flow_perspective,
                                                              alphabet, sg)
        activity_distance_matrix_dict[activity_distance_function] = act2vec_distance_matrix
    return dict(activity_distance_matrix_dict)


def get_activity_distance_matrix_dict(activity_distance_functions, logs_with_replaced_activities_dict):
    activity_distance_matrix_dict = defaultdict(lambda: defaultdict())
    for activity_distance_function in activity_distance_functions:

        window_size = extract_window_size(activity_distance_function)

        if activity_distance_function.startswith("Bose 2009 Substitution Scores"):
            for key in logs_with_replaced_activities_dict:
                activity_distance_matrix, embedding = get_substitution_and_insertion_scores(
                    logs_with_replaced_activities_dict[key],
                    get_alphabet(
                        logs_with_replaced_activities_dict[
                            key]), window_size)
                activity_distance_matrix_dict[activity_distance_function][key] = activity_distance_matrix
        elif activity_distance_function.startswith("De Koninck 2018 act2vec"):
            if "CBOW" in activity_distance_function:
                sg = 0
            else:
                sg = 1
            for key in logs_with_replaced_activities_dict:
                act2vec_distance_matrix, embedding = get_act2vec_distance_matrix(logs_with_replaced_activities_dict[key],
                                                                      get_alphabet(
                                                                          logs_with_replaced_activities_dict[key]), sg, window_size)
                activity_distance_matrix_dict[activity_distance_function][key] = act2vec_distance_matrix
        elif activity_distance_function.startswith("Our act2vec"):
            for key in logs_with_replaced_activities_dict:
                activity_distance_matrix_dict[activity_distance_function][key] =get_act2vec_distance_matrix_our(logs_with_replaced_activities_dict[key], get_alphabet(
                                                                              logs_with_replaced_activities_dict[key]), window_size)
        elif activity_distance_function.startswith("Unit Distance"):
            for key in logs_with_replaced_activities_dict:
                unit_distance_matrix, emb = get_unit_cost_activity_distance_matrix(logs_with_replaced_activities_dict[key],
                                                                              get_alphabet(
                                                                                  logs_with_replaced_activities_dict[
                                                                                      key]))
                activity_distance_matrix_dict[activity_distance_function][key] = unit_distance_matrix
        elif "Chiorrini 2022 Embedding Process Structure" in activity_distance_function:
            for key in logs_with_replaced_activities_dict:
                embedding_process_structure_distance_matrix, embedding = get_embedding_process_structure_distance_matrix(
                    logs_with_replaced_activities_dict[key],
                    get_alphabet(
                        logs_with_replaced_activities_dict[key]), False)
                activity_distance_matrix_dict[activity_distance_function][
                    key] = embedding_process_structure_distance_matrix
        elif  "Gamallo Fernandez 2023 Context Based" in activity_distance_function:
            for key in logs_with_replaced_activities_dict:
                auto_encoder_cosine_matrix, embedding = get_context_based_distance_matrix(
                    logs_with_replaced_activities_dict[key], window_size)
                activity_distance_matrix_dict[activity_distance_function][
                    key] = auto_encoder_cosine_matrix
        elif activity_distance_function.startswith("Activity-Activitiy Co Occurrence Bag Of Words"):
            for key in logs_with_replaced_activities_dict:
                activity_distance_matrix_dict[activity_distance_function][
                    key], embedding, activity_freq_dict, activity_index = get_activity_activity_co_occurence_matrix(logs_with_replaced_activities_dict[key],
                                                                                get_alphabet(
                                                                                    logs_with_replaced_activities_dict[
                                                                                        key]), window_size, True)
                if "PPMI" in activity_distance_function:
                    activity_distance_matrix_dict[activity_distance_function][
                        key], embedding = get_activity_activity_frequency_matrix_pmi(embedding, activity_freq_dict, activity_index, 1)
                elif "PMI" in activity_distance_function:
                    activity_distance_matrix_dict[activity_distance_function][
                        key], embedding = get_activity_activity_frequency_matrix_pmi(embedding, activity_freq_dict, activity_index, 0)

        elif activity_distance_function.startswith("Activity-Activitiy Co Occurrence N-Gram"):
            for key in logs_with_replaced_activities_dict:
                activity_distance_matrix_dict[activity_distance_function][
                    key], embedding, activity_freq_dict, activity_index = get_activity_activity_co_occurence_matrix(logs_with_replaced_activities_dict[key],
                                                                                get_alphabet(
                                                                                    logs_with_replaced_activities_dict[
                                                                                        key]), window_size, False)
                if "PPMI" in activity_distance_function:
                    activity_distance_matrix_dict[activity_distance_function][
                        key], embedding = get_activity_activity_frequency_matrix_pmi(embedding, activity_freq_dict, activity_index, 1)
                elif "PMI" in activity_distance_function:
                    activity_distance_matrix_dict[activity_distance_function][
                        key], embedding = get_activity_activity_frequency_matrix_pmi(embedding, activity_freq_dict, activity_index, 0)

        elif activity_distance_function.startswith("Activity-Context Bag Of Words "):
            for key in logs_with_replaced_activities_dict:
                activity_distance_matrix_dict[activity_distance_function][
                    key], embedding, activity_freq_dict, context_freq_dict, context_index  = get_activity_context_frequency_matrix(logs_with_replaced_activities_dict[key],
                                                                                get_alphabet(
                                                                                    logs_with_replaced_activities_dict[
                                                                                        key]), window_size, 2)
                if "PPMI" in activity_distance_function:
                    activity_distance_matrix_dict[activity_distance_function][
                        key], embedding = get_activity_context_frequency_matrix_pmi(embedding, activity_freq_dict, context_freq_dict, context_index, 1)
                elif "PMI" in activity_distance_function:
                    activity_distance_matrix_dict[activity_distance_function][
                        key], embedding = get_activity_context_frequency_matrix_pmi(embedding, activity_freq_dict, context_freq_dict, context_index, 0)



        elif "Activity-Context as Bag of Words as N-Grams" in activity_distance_function:
            for key in logs_with_replaced_activities_dict:
                activity_distance_matrix_dict[activity_distance_function][
                    key], embedding, activity_freq_dict, context_freq_dict, context_index = get_activity_context_frequency_matrix(logs_with_replaced_activities_dict[key],
                                                                                get_alphabet(
                                                                                    logs_with_replaced_activities_dict[
                                                                                        key]), window_size, 1)
                if "PPMI" in activity_distance_function:
                    activity_distance_matrix_dict[activity_distance_function][
                        key], embedding = get_activity_context_frequency_matrix_pmi(embedding, activity_freq_dict, context_freq_dict, context_index, 1)
                elif "PMI" in activity_distance_function:
                    activity_distance_matrix_dict[activity_distance_function][
                        key], embedding = get_activity_context_frequency_matrix_pmi(embedding, activity_freq_dict, context_freq_dict, context_index, 0)


        elif "Activity-Context N-Grams" in activity_distance_function:
            for key in logs_with_replaced_activities_dict:
                activity_distance_matrix_dict[activity_distance_function][
                    key], embedding, activity_freq_dict, context_freq_dict, context_index = get_activity_context_frequency_matrix(logs_with_replaced_activities_dict[key],
                                                                                get_alphabet(
                                                                                    logs_with_replaced_activities_dict[
                                                                                        key]), window_size, 0)
                if "PPMI" in activity_distance_function:
                    activity_distance_matrix_dict[activity_distance_function][
                        key], embedding = get_activity_context_frequency_matrix_pmi(embedding, activity_freq_dict, context_freq_dict, context_index, 1)
                elif "PMI" in activity_distance_function:
                    activity_distance_matrix_dict[activity_distance_function][
                        key], embedding = get_activity_context_frequency_matrix_pmi(embedding, activity_freq_dict, context_freq_dict, context_index, 0)




    return dict(activity_distance_matrix_dict)


def get_log_control_flow_perspective_with_short_activity_names(log_control_flow_perspective, alphabet):
    short_activity_names_dict = dict()
    i = 0
    for activity in alphabet:
        short_activity_names_dict[activity] = str(i)
        i += 1
    for trace in log_control_flow_perspective:
        i = 0
        for activity in trace:
            trace[i] = short_activity_names_dict[activity]
            i += 1
    return log_control_flow_perspective, short_activity_names_dict


def get_obj_size(obj):
    marked = {id(obj)}
    obj_q = [obj]
    sz = 0

    while obj_q:
        sz += sum(map(sys.getsizeof, obj_q))

        # Lookup all the object referred to by the object in obj_q.
        # See: https://docs.python.org/3.7/library/gc.html#gc.get_referents
        all_refr = ((id(o), o) for o in gc.get_referents(*obj_q))

        # Filter object that are already marked.
        # Using dict notation will prevent repeated objects.
        new_refr = {o_id: o for o_id, o in all_refr if o_id not in marked and not isinstance(o, type)}

        # The new obj_q will be the ones that were not marked,
        # and we will update marked with their ids so we will
        # not traverse them again.
        obj_q = new_refr.values()
        marked.update(new_refr.keys())

    return sz


def unresponsiveness_prediction(log_size, alphabet_size, r, w, sampling_size=None):
    memory_info = psutil.virtual_memory()
    # Get the amount of free memory
    free_memory = memory_info.free

    # Initialize the sum
    total_sum = 0
    for r_val in range(1, r + 1):
        inner_sum = 0
        for w_val in range(2, w + 1):
            # Calculate (alphabet_size choose r_val) * log_size
            combination = math.comb(alphabet_size, r_val)
            if sampling_size is not None:
                combination = sampling_size
            inner_sum += combination * log_size
        total_sum += inner_sum

    if total_sum > free_memory / 2:
        # print("System might run out of memory.")
        return True
    else:
        return False


def print_log_stats(log, alphabet):
    print("Alphabet Size: " + str(len(alphabet)))
    print("Number of Traces: " + str(len(log)))

    # Convert each list to a tuple and then use a set to find unique tuples
    trace_variants = set(tuple(trace) for trace in log)

    # Get the number of unique lists
    print("Number of Trace Variants: " + str(len(trace_variants)))

    # Get the lengths of all lists
    lengths = [len(trace) for trace in log]

    # Calculate min, max, and average length
    min_length = min(lengths)
    max_length = max(lengths)
    avg_length = sum(lengths) / len(lengths)

    print("Trace Length Min: " + str(min_length))
    print("Trace Length Avg: " + str(avg_length))
    print("Trace Length Max: " + str(max_length))


def get_unit_cost_activity_distance_matrix(log, alphabet):
    distances = {}
    for activity1 in alphabet:
        for activity2 in alphabet:
            if activity1 == activity2:
                distances[(activity1, activity2)] = 0
            else:
                distances[(activity1, activity2)] = 1
    emb = {activity: np.eye(len(alphabet))[i] for i, activity in enumerate(alphabet)}

    return distances, emb


def delete_temporary_files():
    folder_path = ROOT_DIR + "/evaluation/evaluation_of_activity_distances/modles"
    try:
        if os.path.exists(folder_path):  # Check if folder exists
            shutil.rmtree(folder_path)
            # print("Folder deleted successfully.")
        # else:
        # print("Folder does not exist.")
    except Exception as e:
        print(f"Error: {e}")

    folder_path = ROOT_DIR + "/evaluation/evaluation_of_activity_distances/lightning_logs/"
    try:
        if os.path.exists(folder_path):  # Check if folder exists
            shutil.rmtree(folder_path)
            # print("Folder deleted successfully.")
        # else:
        # print("Folder does not exist.")
    except Exception as e:
        print(f"Error: {e}")


def extract_window_size(s):
    match = re.search(r"w_(\d+)", s)
    return int(match.group(1)) if match else 3

