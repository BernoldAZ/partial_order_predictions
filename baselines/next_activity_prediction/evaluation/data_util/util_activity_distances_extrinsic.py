import os
import random
import numpy as np
from pm4py.objects.log.importer.xes import importer as xes_importer
import pm4py
from definitions import ROOT_DIR
from distances.trace_distances.edit_distance.levenshtein.algorithm import compute_levenshtein_distance
from evaluation.data_util.util_activity_distances_intrinsic import get_log_control_flow_perspective
from sklearn.manifold import MDS
import hdbscan


def get_sublog_list(folder):
    #i = 0
    sublog_list = list()
    for sublog_name in os.listdir(ROOT_DIR + '/event_logs/' + folder):
        sublog = xes_importer.apply(ROOT_DIR + '/event_logs/' + folder + "/" + sublog_name)
        #pt = pm4py.discover_process_tree_inductive(sublog)
        #pm4py.view_process_tree(pt)
        sublog_control_flow_perspective = get_log_control_flow_perspective(sublog)
        sublog_list.append(sublog_control_flow_perspective)
        #i += 1
        #if i == 10:
         #   break
    return sublog_list


def get_trace_distances(trace, all_trace_list, activity_distance_matrix_dict, activity_clustering):
    trace_distance_list = list()
    for trace_tuple in all_trace_list:
        if trace[0] != trace_tuple[0]:  # do not compute distance between same trace
            trace_distance_list.append(
                (compute_levenshtein_distance(trace[2], trace_tuple[2], activity_distance_matrix_dict, activity_clustering),) + trace_tuple)
    return trace_distance_list


# get_sublog_list("PDC 2019")

def get_log_with_trace_ids(log_control_flow_perspective, sublogsize_list):
    trace_sublog_pair_list = list()
    trace_sublog_list_all_list = list()
    trace_sublog_list_all_list_flat = list()
    i = 0
    sublog_id = 0
    trace_id = 0
    for trace in log_control_flow_perspective:
        trace_sublog_pair_list.append((trace_id, sublog_id, trace))
        if i == sublogsize_list[sublog_id] - 1:
            sublog_id += 1
            trace_sublog_list_all_list.append(trace_sublog_pair_list)
            trace_sublog_list_all_list_flat.extend(trace_sublog_pair_list)
            trace_sublog_pair_list = list()
            i = 0
        else:
            i += 1
        trace_id += 1
    return trace_sublog_list_all_list, trace_sublog_list_all_list_flat


def get_precision_values(trace_distance_list, trace, sublogsize_list):
    # Sort distances by the similarity score (distance)
    trace_distance_list.sort(key=lambda x: x[0], reverse=False)

    # precision@1 (nn)
    if trace_distance_list[0][2] == trace[1]:
        precision_at_1 = 1
    else:
        precision_at_1 = 0

    k = sublogsize_list[trace[1]]
    # precision_at_k
    same_sub_log_traces_count = 0
    for trace_tuple in trace_distance_list[:k]:
        if trace_tuple[2] == trace[1]:
            same_sub_log_traces_count += 1
    precision_at_k = same_sub_log_traces_count / k

    k = 10
    # precision_at_k
    same_sub_log_traces_count = 0
    for trace_tuple in trace_distance_list[:k]:
        if trace_tuple[2] == trace[1]:
            same_sub_log_traces_count += 1
    precision_at_10 = same_sub_log_traces_count / k


    return precision_at_1, precision_at_k, precision_at_10


def print_avg_values(results, activity_distance_functions):
    for activity_distance_function in activity_distance_functions:
        print(activity_distance_function)
        nn_for_activity_distance_function_list = list()
        pre_for_activity_distance_function_list = list()
        pre10_for_activity_distance_function_list = list()
        for result in results:
            if result[0] == activity_distance_function:
                nn_for_activity_distance_function_list.append(result[1])
                pre_for_activity_distance_function_list.append(result[2])
                pre10_for_activity_distance_function_list.append(result[3])
        print(" NN: " + str(
            sum(nn_for_activity_distance_function_list) / len(nn_for_activity_distance_function_list)) + " Pre10: " + str(
            sum(pre10_for_activity_distance_function_list) / len(pre10_for_activity_distance_function_list)) + " Pre: " + str(
            sum(pre_for_activity_distance_function_list) / len(pre_for_activity_distance_function_list)))


def get_sampled_sublogs(list_of_lists, percentage):
    sampled_lists = []

    for lst in list_of_lists:
        # Calculate % of the length of the list
        one_percent = max(int(len(lst) * percentage), 10)

        # Randomly sample 1% or at least 10 elements from the list
        if len(lst) >= one_percent:
            sampled = random.sample(lst, one_percent)
        else:
            sampled = random.sample(lst, len(lst))

        sampled_lists.append(sampled)

    return sampled_lists


def get_activity_clustering(activity_distance_matrix_dict_list):

    # Dictionary to store activity clusters for each distance function
    activity_clusters = {}

    for distance_data_ in activity_distance_matrix_dict_list:
        distance_function_name = list(distance_data_.keys())[0]
        distance_data = distance_data_[distance_function_name]

        # Step 1: Extract the activity names
        activities = sorted(
            list(set([key[0] for key in distance_data.keys()] + [key[1] for key in distance_data.keys()])))

        # Step 2: Create the distance matrix
        n = len(activities)
        distance_matrix = np.zeros((n, n))

        for i, act1 in enumerate(activities):
            for j, act2 in enumerate(activities):
                if (act1, act2) in distance_data:
                    distance_matrix[i, j] = distance_data[(act1, act2)]
                elif (act2, act1) in distance_data:
                    distance_matrix[i, j] = distance_data[(act2, act1)]

        # Step 3: Perform MDS
        mds = MDS(dissimilarity='precomputed', random_state=42)
        mds_result = mds.fit_transform(distance_matrix)

        # Step 3: Perform HDBSCAN clustering
        hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
        cluster_labels = hdbscan_clusterer.fit_predict(mds_result)

        # Store clusters for each activity in the `activity_clusters` dictionary
        activity_clusters[distance_function_name] = {
            activity: cluster_labels[i] for i, activity in enumerate(activities)
        }
    return activity_clusters
