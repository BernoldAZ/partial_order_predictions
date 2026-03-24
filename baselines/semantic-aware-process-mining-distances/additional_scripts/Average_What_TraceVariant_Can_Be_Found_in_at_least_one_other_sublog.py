import multiprocessing
import os
from multiprocessing import Pool
import seaborn as sns
import matplotlib.pyplot as plt
from evaluation.data_util.util_activity_distances import get_alphabet, get_normalized_activity_distance_matrix_dict_list, print_log_stats
from evaluation.data_util.util_activity_distances_extrinsic import get_sublog_list, get_log_with_trace_ids, get_activity_clustering
from definitions import ROOT_DIR
from collections import defaultdict


def find_shared_sublists(nested_list, sublog_list):
    # Dictionary to store the shared sublists for each list of lists
    shared_sublists = defaultdict(list)

    # Go through each list of lists
    for i, sublist in enumerate(nested_list):
        # Track unique sublists in the current list as tuples for comparison
        unique_sublists = set(tuple(item) for item in sublist)

        # Compare with other lists in nested_list
        for j, other_sublist in enumerate(nested_list):
            if i != j:  # Skip comparing the list with itself
                other_unique_sublists = set(tuple(item) for item in other_sublist)
                # Find the intersection of unique lists between the current list and another list
                common_sublists = unique_sublists.intersection(other_unique_sublists)

                # Add these common sublists to the dictionary for the current list
                for item in common_sublists:
                    # Convert tuples back to lists and store in the dictionary
                    if list(item) not in shared_sublists[i]:
                        shared_sublists[i].append(list(item))

    # Remove duplicates in shared_sublists entries
    #shared_sublists = {k: [list(x) for x in set(tuple(i) for i in v)] for k, v in shared_sublists.items()}

    # Calculate the average percentage of shared sublists - trace variants
    total_shared_percentage = 0  # Initialize the total percentage accumulator

    for i, sublist in enumerate(nested_list):
        # Skip if there are no shared sublists for this index
        if i not in shared_sublists:
            continue

        # Calculate the number of shared traces in sublist
        total_shared_trace = 0
        trace_list = sublog_list[i]  # Assuming sublog_list is nested_list here

        for trace in trace_list:
            # Increment total_shared_trace if trace is in shared_sublists[i]
            if trace in shared_sublists[i]:
                total_shared_trace += 1

        # Calculate the shared percentage for the current sublist
        shared_percentage = (total_shared_trace / len(trace_list)) * 100
        total_shared_percentage += shared_percentage  # Accumulate shared percentage

    # Calculate the average amount of traces where there exists another trace that is the same in another sublog
    average_shared_percentage_traces = total_shared_percentage / len(nested_list) if nested_list else 0



    # Calculate the average percentage of shared sublists


    total_shared_percentage = 0
    for i, sublist in enumerate(nested_list):

        # Calculate the percentage of shared lists
        num_unique = len(sublist)
        num_shared = len(shared_sublists[i])
        if num_unique > 0:
            shared_percentage = (num_shared / num_unique) * 100
            total_shared_percentage += shared_percentage

    # Calculate the average amount of traces where there exists another trace that is the same in another sublog
    average_shared_percentage_trace_variants = total_shared_percentage / len(nested_list) if nested_list else 0

    return shared_sublists, average_shared_percentage_traces, average_shared_percentage_trace_variants



# Function to flatten lists of lists
def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]

# Calculate the percentage of similarity
def similarity_percentage(list1, list2):
    number_of_same_traces = 0
    matched_indices = set()

    for trace1 in list1:
        for idx, trace2 in enumerate(list2):
            if idx in matched_indices:
                continue
            if len(trace1) != len(trace2):
                continue
            if all(trace1[i] == trace2[i] for i in range(len(trace1))):
                number_of_same_traces += 1
                matched_indices.add(idx)
                break

    return number_of_same_traces / len(list1) if list1 else 0


def replace_activities_with_clusters(trace, activity_clustering):
    newTrace = []
    for activity in trace:
        newActivity = activity_clustering[activity]
        if newActivity == -1:
            newActivity = activity
        newTrace.append(newActivity)
    return newTrace


def remove_duplicate_lists(sublog_list):
    # Iterate through each list of lists in the input
    unique_nested_list = []
    for sublist in sublog_list:
        # Use a set to remove duplicates within each sublist by converting lists to tuples
        unique_sublist = list(set(tuple(item) for item in sublist))
        # Convert each tuple back to a list
        unique_sublist = [list(item) for item in unique_sublist]
        # Append the unique sublist to the result
        unique_nested_list.append(unique_sublist)
    return unique_nested_list


def log_similarity(event_log_list):

    for log_name in event_log_list:
        if log_name[:4] == "bpic" or log_name[:3] == "pdc":
            sublog_list = get_sublog_list(log_name)
            sublogsize_list = [len(sublog) for sublog in sublog_list]
            log_control_flow_perspective = [inner for outer in sublog_list for inner in outer]

        #alphabet = get_alphabet(log_control_flow_perspective)

        trace_variant_sublogs = remove_duplicate_lists(sublog_list)

        shared_sublists, average_shared_percentage = find_shared_sublists(trace_variant_sublogs, sublog_list)

        #print_log_stats(log_control_flow_perspective, alphabet)

        remove_duplicate_lists(sublog_list)


        percentage_sum = 0





        combinations = [
            (log_control_flow_perspective, activity_distance_function, alphabet)
            for activity_distance_function in activity_distance_functions
        ]

        total_cores = multiprocessing.cpu_count()
        cores_to_use = int(total_cores * 0.80)
        cores_to_use = max(1, cores_to_use)

        with Pool(processes=cores_to_use) as pool:
            activity_distance_matrix_dict_list = pool.map(get_normalized_activity_distance_matrix_dict_list, combinations)

        print("Distances Computed")

        trace_sublog_list_all_list, trace_sublog_list_all_list_flat = get_log_with_trace_ids(log_control_flow_perspective,
                                                                                             sublogsize_list)

        # Convert similarities to distances
        for admd in activity_distance_matrix_dict_list:
            if "Bose 2009 Substitution Scores" in admd:
                for key in admd["Bose 2009 Substitution Scores"].keys():
                    admd["Bose 2009 Substitution Scores"][key] = 1 - admd["Bose 2009 Substitution Scores"][key]

        activity_clustering = get_activity_clustering(activity_distance_matrix_dict_list)

        # Display original similarity matrix
        display_sim_matrix(sublog_list, "original activities")

        # Prepare data for display_sim_matrix in parallel
        task_data = []
        for adf in activity_clustering:
            sublog_cluster_list = []
            for log in sublog_list:
                sublog_cluster = []
                for trace in log:
                    sublog_cluster.append(replace_activities_with_clusters(trace, activity_clustering.get(adf)))
                sublog_cluster_list.append(sublog_cluster)
            task_data.append((sublog_cluster_list, adf))

        # Display similarity matrices in parallel
        with Pool(processes=cores_to_use) as pool:
            pool.starmap(display_sim_matrix, task_data)


    def display_sim_matrix(sublog_cluster_list, adf):
        n = len(sublog_cluster_list)
        similarity_matrix = [[0] * n for _ in range(n)]  # Matrix of zeros without numpy

        for i in range(n):
            print(i)
            for j in range(n):
                similarity_matrix[i][j] = similarity_percentage(sublog_cluster_list[i], sublog_cluster_list[j])

        # Adjust figure size
        plt.figure(figsize=(n / 2, n / 2))  # Adjust sizing based on matrix size
        sns.heatmap(similarity_matrix, annot=(n <= 20), cmap='coolwarm', fmt=".1f",
                    xticklabels=[f'List {i+1}' for i in range(n)] if n <= 50 else False,
                    yticklabels=[f'List {i+1}' for i in range(n)] if n <= 50 else False)
        plt.title('Pairwise Similarity Percentage Heatmap ' + adf)
        plt.show()

if __name__ == '__main__':

    # List only directories within the specified path
    folder_names = [name for name in os.listdir(ROOT_DIR + "/event_logs") if os.path.isdir(os.path.join(ROOT_DIR + "/event_logs", name))]
    log_similarity(folder_names)
