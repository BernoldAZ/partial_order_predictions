import multiprocessing
from multiprocessing import Pool
import seaborn as sns
import matplotlib.pyplot as plt
from evaluation.data_util.util_activity_distances import get_alphabet, get_normalized_activity_distance_matrix_dict_list, print_log_stats
from evaluation.data_util.util_activity_distances_extrinsic import get_sublog_list, get_log_with_trace_ids, get_activity_clustering

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


def log_similarity(log_name, activity_distance_functions):
    if log_name[:4] == "bpic" or log_name[:3] == "pdc":
        sublog_list = get_sublog_list(log_name)
        sublogsize_list = [len(sublog) for sublog in sublog_list]
        log_control_flow_perspective = [inner for outer in sublog_list for inner in outer]

    alphabet = get_alphabet(log_control_flow_perspective)

    print_log_stats(log_control_flow_perspective, alphabet)

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
    activity_distance_functions = [
        "Bose 2009 Substitution Scores",
        "De Koninck 2018 act2vec CBOW"
    ]
    event_log_folder = "pdc_2019"
    log_similarity(event_log_folder, activity_distance_functions)
