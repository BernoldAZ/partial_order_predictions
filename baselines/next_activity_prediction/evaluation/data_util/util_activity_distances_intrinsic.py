import itertools
from collections import defaultdict
from pathlib import Path
from typing import List
import gc
import copy
import random
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from networkx.algorithms.approximation import diameter

from evaluation.data_util.util_activity_distances import get_obj_size
import pandas as pd
from definitions import ROOT_DIR
import sys
from itertools import combinations
import pickle
import os
import re


def get_log_control_flow_perspective(log):
    log_list = list()
    for trace in log:
        trace_list = list()
        for event in trace._list:
            #trace_list.append(event._dict.get('concept:name')+"-"+event._dict.get('lifecycle:transition'))
            trace_list.append(event._dict.get('concept:name'))
        for _ in range(1, 1+1):
            log_list.append(copy.copy(trace_list))
    return log_list

def reservoir_sampling(iterator, sample_size):
    """Perform reservoir sampling from the given iterator."""
    reservoir = []
    for index, item in enumerate(iterator):
        if len(reservoir) < sample_size:
            reservoir.append(item)
        else:
            # Randomly replace elements in the reservoir with decreasing probability
            replace_index = random.randint(0, index)
            if replace_index < sample_size:
                reservoir[replace_index] = item
    return reservoir


def get_activities_to_replace(alphabet: List[str], different_activities_to_replace_count: int, sample_size: int):
    activity_dict = {i: activity for i, activity in enumerate(alphabet)}
    id_set = set(activity_dict.keys())
    alphabet_len = len(alphabet)
    alphabet_len_minus_one = alphabet_len -1
    sampled_combinations = set()
    comb_number = math.comb(alphabet_len, different_activities_to_replace_count)
    #sampling to not run out of memory
    while len(sampled_combinations) < sample_size and len(sampled_combinations) < comb_number:
        activity_index_set = set()
        while len(activity_index_set) < different_activities_to_replace_count:
            activity_index_set.add(random.randint(0, alphabet_len_minus_one))
        activity_index_list = list(activity_index_set)
        activity_index_list.sort()
        activity_index_set_frozen = frozenset(activity_index_list)
        """
        acitvity_list = list()
        for index in activity_index_list:
            acitvity_list.append(alphabet[index])
        """
        sampled_combinations.add(activity_index_set_frozen)

    # Convert set of frozenset to list of tuples
    sampled_combinations_id_list = [tuple(sampled_combination) for sampled_combination in sampled_combinations]
    # Convert IDs in tuples to corresponding values from alphabet_dict
    sampled_combinations_list = [tuple(activity_dict[i] for i in tpl) for tpl in sampled_combinations_id_list]
    return sampled_combinations_list


def get_logs_with_replaced_activities_dict(activities_to_replace_in_each_run_list, log_control_flow_perspective,
                                           different_activities_to_replace_count, activities_to_replace_with_count):
    logs_with_replaced_activities_dict = dict()
    for activities_to_replace_tuple in activities_to_replace_in_each_run_list:
        # log_with_replaced_activities = deepcopy(log_control_flow_perspective)
        replacing_activities_dict = dict()
        for activity in activities_to_replace_tuple:
            replacing_activities_dict[activity] = set(range(activities_to_replace_with_count))
        log_with_replaced_activities = []
        for trace in log_control_flow_perspective:
            i = 0
            # to replace all activites we want to replace in a trace with the same new activity we need to store the activity we choose for this trace
            activities_to_replace_with = [None] * different_activities_to_replace_count
            trace_with_replaced_activities = []
            for activity in trace:
                trace_with_replaced_activities.append(activity)
                if activity in replacing_activities_dict:
                    activity_index = activities_to_replace_tuple.index(activity)
                    if activities_to_replace_with[activity_index] is None:
                        if len(replacing_activities_dict[activity]) == 0:
                            replacing_activities_dict[activity] = set(range(activities_to_replace_with_count))
                        activities_to_replace_with[activity_index] = activity + ':' + str(
                            replacing_activities_dict[activity].pop())
                    trace_with_replaced_activities[i] = activities_to_replace_with[activity_index]
                i += 1
            log_with_replaced_activities.append(trace_with_replaced_activities)
        logs_with_replaced_activities_dict[activities_to_replace_tuple] = log_with_replaced_activities
    return logs_with_replaced_activities_dict


def get_n_nearest_neighbors(n, replaced_activities, similarity_scores_of_activities, activities_to_replace_with_count,
                            reverse):
    neighbors = dict()
    similarity_scores_of_activities = get_nested_dict(similarity_scores_of_activities)
    for replaced_activity in replaced_activities:
        for i in range(activities_to_replace_with_count):
            replaced_activity_i = replaced_activity + ':' + str(i)
            # Collect all similarity scores for the current replaced_activity
            distances = []
            if replaced_activity_i in similarity_scores_of_activities:
                for activity in similarity_scores_of_activities[replaced_activity_i]:
                    if activity != replaced_activity_i:
                        distances.append((activity, similarity_scores_of_activities[replaced_activity_i][activity]))

                if len(distances) > 0:
                    # Sort distances by the similarity score (distance)
                    distances.sort(key=lambda x: x[1], reverse=reverse)

                    # Get the top n nearest neighbors
                    nearest_neighbors = [activity for activity, _ in distances[:n]]

                    # Store the nearest neighbors in the dictionary
                    neighbors[replaced_activity_i] = nearest_neighbors

    return neighbors


def get_knn_dict(activity_distance_matrix_dict,
                 activities_to_replace_with_count, reverse, knn_count):
    knn_dict = defaultdict(lambda: defaultdict())
    for activity_distance_function in activity_distance_matrix_dict:
        for replaced_activities in activity_distance_matrix_dict[activity_distance_function].keys():
            nearest_neighbors = get_n_nearest_neighbors(knn_count, replaced_activities,
                                                        activity_distance_matrix_dict[activity_distance_function][
                                                            replaced_activities], activities_to_replace_with_count,
                                                        reverse)

            knn_dict[activity_distance_function][replaced_activities] = nearest_neighbors
    return dict(knn_dict)


def get_precision_at_k(knn_dict, activity_distances):
    precision_at_k_dict = defaultdict(float)
    for activity_distance in activity_distances:
        precision_replaced_activity_at_k = 0
        for replaced_activities in knn_dict[activity_distance].keys():
            precision_at_k_sum = 0
            for replaced_activity in replaced_activities:
                for replaced_activity_with in knn_dict[activity_distance][replaced_activities].keys():
                    replaced_activity_with_split = replaced_activity_with.split(':')
                    if not replaced_activity_with_split[1].isdigit():
                        sys.exit("Naming Error")
                    if replaced_activity_with_split[0] == replaced_activity:
                        precision_sum = 0
                        for activity in knn_dict[activity_distance][replaced_activities][replaced_activity_with]:
                            activity_split = activity.split(':')
                            if activity_split[0] == replaced_activity_with_split[0]:
                                precision_sum += 1
                        a = len(knn_dict[activity_distance][replaced_activities][replaced_activity_with])
                        precision_at_k_sum += precision_sum / a
            precision_replaced_activity_at_k += precision_at_k_sum / len(
                knn_dict[activity_distance][replaced_activities].keys())
        precision_at_k_dict[activity_distance] = precision_replaced_activity_at_k / len(
            knn_dict[activity_distance].keys())
    return dict(precision_at_k_dict)


def save_intrinsic_results(activity_distance_functions, results, log_name, r, w, sampling_size):
    activity_distance_function_index = 0
    df_average_values = pd.DataFrame(columns=["Log Name", "Distance Function", 'diameter', 'precision@w-1', 'precision@1', 'triplet'])

    for activity_distance_function in activity_distance_functions:
        results_per_activity_distance_function = list()
        for result in results:
            results_per_activity_distance_function.append(result[activity_distance_function_index])

        # Create DataFrame from results
        Path(ROOT_DIR + "/results/activity_distances/intrinsic/" + log_name).mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(results_per_activity_distance_function, columns=['r', 'w', 'diameter', 'precision@w-1', 'precision@1', 'triplet'])

        # Generate the file name incorporating all function arguments if no output_file is provided
        csv = f"{log_name}_distfunc_{activity_distance_function}_r{r}_w{w}_samplesize_{sampling_size}.csv"

        # Ensure that the file name is valid and does not contain invalid characters (especially for file systems)
        csv = csv.replace("/", "_").replace("\\", "_")

        # Save the DataFrame to a CSV file
        df.to_csv(ROOT_DIR + "/results/activity_distances/intrinsic/" + log_name + "/"+ csv, index=False)

        # Print average results
        print_average_results(df, activity_distance_function)

        # diameter
        diameter_result = df.pivot(index='w', columns='r', values='diameter')
        diameter_average = diameter_result.values.mean()

        # precision@w-1
        prec_result = df.pivot(index='w', columns='r', values='precision@w-1')
        prec_average = prec_result.values.mean()

        # precision@1
        nn_result = df.pivot(index='w', columns='r', values='precision@1')
        nn_average = nn_result.values.mean()

        # triplet
        triplet_result = df.pivot(index='w', columns='r', values='triplet')
        triplet_average = triplet_result.values.mean()

        new_row = pd.DataFrame([{
            "Log Name": log_name,
            "Distance Function": activity_distance_function,
            'diameter': diameter_average,
            'precision@w-1': prec_average,
            'precision@1': nn_average,
            'triplet': triplet_average
        }])

        df_average_values = pd.concat([df_average_values, new_row], ignore_index=True)

        activity_distance_function_index += 1
    # Plot the results
    df_avg_dir = os.path.join(ROOT_DIR, "results", "activity_distances", "intrinsic_df_avg", log_name)
    os.makedirs(df_avg_dir, exist_ok=True)
    file_name = f"dfavg_r{r}_w{w}_samplesize_{sampling_size}.pkl"
    file_path =os.path.join(df_avg_dir, file_name)
    with open(file_path, "wb") as file:
        pickle.dump(df_average_values, file)

    plot_results(df_average_values, log_name)

def plot_results(df_average_values, log_name):
    # Create a folder for saving plots if it doesn't exist
    plot_dir = ROOT_DIR + "/results/activity_distances/intrinsic/" + log_name + "/plots/"
    os.makedirs(plot_dir, exist_ok=True)

    # List of columns to plot
    metrics = ['diameter', 'precision@w-1', 'precision@1', 'triplet']

    # Set plot style
    sns.set(style="whitegrid")

    # Generate and save a plot for each metric
    for metric in metrics:
        plt.figure(figsize=(20, 6))

        # Create the plot: Barplot for each distance function
        sns.barplot(data=df_average_values, x="Distance Function", y=metric, palette="viridis")

        # Set plot labels and title
        plt.title(f"{metric} for {log_name}", fontsize=16)
        plt.xlabel('Distance Function', fontsize=14)
        plt.ylabel(metric, fontsize=14)
        plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for readability

        # Save the plot to the directory
        plot_filename = f"{log_name}_{metric}_plot.png"
        plt.tight_layout()
        plt.savefig(plot_dir + plot_filename)

        # Show the plot (optional)
        plt.show()

# visualization_intrinsic_evaluation(results_per_activity_distance_function, activity_distance_function,
   #                                    log_name, r, w, sampling_size, output_file)
   # load_visualization_intrinsic_evaluation(results_per_activity_distance_function, activity_distance_function,
   #                                         log_name, r, w, sampling_size, output_file)
   # activity_distance_function_index = activity_distance_function_index + 1

def print_average_results(df, activity_distance_function):

    #diameter
    result = df.pivot(index='w', columns='r', values='diameter')
    average_value = result.values.mean()
    print("The average diameter is: " + str(average_value) + " " + activity_distance_function)


    #precision@w-1
    result = df.pivot(index='w', columns='r', values='precision@w-1')
    average_value = result.values.mean()
    print("The average precision@w-1 is: " + str(average_value) + " " + activity_distance_function)

    #precision@1
    result = df.pivot(index='w', columns='r', values='precision@1')
    average_value = result.values.mean()
    print("The average Nearest Neighbor is: " + str(average_value) + " " + activity_distance_function)

    #precision@1
    result = df.pivot(index='w', columns='r', values='triplet')
    average_value = result.values.mean()
    print("The average triplet value is: " + str(average_value) + " " + activity_distance_function)


#'''
def get_triplet(activity_distance_matrix_dict,
                 activities_to_replace_with_count, reverse, alphabet):
    triplet_list = list()
    for activity_distance_function in activity_distance_matrix_dict:
        for replaced_activities in activity_distance_matrix_dict[activity_distance_function].keys():
            avg_triplet_value = get_avg_triplet_value(replaced_activities, activity_distance_matrix_dict[activity_distance_function][
                                                            replaced_activities], activities_to_replace_with_count, reverse, alphabet)
            triplet_list.append(avg_triplet_value)

    return sum(triplet_list) / len(triplet_list)

def get_avg_triplet_value(replaced_activities, similarity_scores_of_activities, activities_to_replace_with_count,
                            reverse, alphabet):
    #high bose similiarity scores mean low distance
    if reverse:
        similarity_scores_of_activities = {key: -value for key, value in similarity_scores_of_activities.items()}

    #for performance reasons transform matrix into nested dict
    similarity_scores_of_activities = get_nested_dict(similarity_scores_of_activities)

    replaced_activities_dict = get_replaced_activities_dict(replaced_activities, similarity_scores_of_activities)
    triplet_value_for_class = list()
    for replaced_activity in replaced_activities:
        out_of_class_activities = get_out_of_class_activities(similarity_scores_of_activities, replaced_activity)
        in_class_combinations = get_in_class_combinations(replaced_activities_dict[replaced_activity])
        triplet_value_for_pairs = list()
        for replaced_activity_pair in in_class_combinations:
            triplet_value_for_pair = list()
            for out_of_class_activity in out_of_class_activities:
                triplet_value_for_pair.append(triplet_is_in_class(replaced_activity_pair[0], replaced_activity_pair[1], out_of_class_activity, similarity_scores_of_activities))
            triplet_value_for_pairs.append(sum(triplet_value_for_pair) / len(triplet_value_for_pair))
        if len(triplet_value_for_pairs) == 0:
            triplet_value_for_class.append(0)
        else:
            triplet_value_for_class.append(sum(triplet_value_for_pairs) / len(triplet_value_for_pairs))
    triplet_value_for_log = sum(triplet_value_for_class) / len(triplet_value_for_class)
    return triplet_value_for_log





def triplet_is_in_class(activity_1_in_class, activity_2_in_class, activity_3_out_of_class, activity_distance_matrix):
    if activity_distance_matrix[activity_1_in_class][activity_2_in_class] < activity_distance_matrix[activity_1_in_class][activity_3_out_of_class]:
        return 1
    else: return 0




def get_nested_dict(similarity_scores_of_activities):
    # Initialize the nested dictionary
    nested_dict = {}

    for (outer_key, inner_key), value in similarity_scores_of_activities.items():
        # Ensure the outer key exists in the nested dictionary
        if outer_key not in nested_dict:
            nested_dict[outer_key] = {}
        # Assign the value to the inner key
        nested_dict[outer_key][inner_key] = value
    return nested_dict


def get_replaced_activities_dict(replaced_activities, activity_distance_matrix):
    replaced_activities_dict = {key: list() for key in replaced_activities}
    for activity in activity_distance_matrix:
        replaced_activity_with_split = activity.split(':')
        if len(replaced_activity_with_split) > 1:
            if not replaced_activity_with_split[1].isdigit():
                sys.exit("Naming Error")
            else:
                replaced_activities_dict[replaced_activity_with_split[0]].append(activity)
    return replaced_activities_dict

def get_out_of_class_activities(activity_distance_matrix, replaced_activity):
    out_of_class_activities = list()
    for activity in activity_distance_matrix.keys():
        if activity.split(':')[0] != replaced_activity:
            out_of_class_activities.append(activity)
    return out_of_class_activities

def get_in_class_combinations(replaced_activities_of_class):
    in_class_combinations = list(combinations(replaced_activities_of_class, 2))
    result = []
    for comb in in_class_combinations:
        result.append(comb)  # Add original combination
        result.append(comb[::-1])  # Add reversed combination
    return result


def get_diameter(activity_distance_matrix_dict,
                 activities_to_replace_with_count, reverse, alphabet):
    diameter_list = list()
    for activity_distance_function in activity_distance_matrix_dict:
        for replaced_activities in activity_distance_matrix_dict[activity_distance_function].keys():
            avg_triplet_value = get_avg_diameter_value(replaced_activities, activity_distance_matrix_dict[activity_distance_function][
                                                            replaced_activities], activities_to_replace_with_count, reverse, alphabet)
            diameter_list.append(avg_triplet_value)


    return sum(diameter_list) / len(diameter_list)


def get_avg_diameter_value(replaced_activities, similarity_scores_of_activities, activities_to_replace_with_count,
                            reverse, alphabet):
    #high bose similiarity scores mean low distance
    if reverse:
        similarity_scores_of_activities = {key: -value for key, value in similarity_scores_of_activities.items()}

    normalized_similarity_scores_of_activities = get_normalized_similarity_scores_of_activities(similarity_scores_of_activities)

    #for performance reasons transform matrix into nested dict
    normalized_similarity_scores_of_activities = get_nested_dict(normalized_similarity_scores_of_activities)
    replaced_activities_dict = get_replaced_activities_dict(replaced_activities, normalized_similarity_scores_of_activities)
    diameter_per_class = list()
    for replaced_activity in replaced_activities:
        in_class_combinations = get_in_class_combinations(replaced_activities_dict[replaced_activity])
        diameter_in_class = list()
        for in_class_combination in in_class_combinations:
            diameter_in_class.append(normalized_similarity_scores_of_activities[in_class_combination[0]][in_class_combination[1]])
        #if activity got only by one other activity replaced
        if len(in_class_combinations) != 0:
            diameter_per_class.append(sum(diameter_in_class) / len(diameter_in_class))
        else:
            diameter_per_class.append(0)
    avg_diameter_for_log = sum(diameter_per_class) / len(diameter_per_class)
    return avg_diameter_for_log

def get_normalized_similarity_scores_of_activities(similarity_scores_of_activities):
    min_value = min(similarity_scores_of_activities.values())
    max_value = max(similarity_scores_of_activities.values())

    # Normalize the values
    normalized_data = {
        key: (value - min_value) / (max_value - min_value)
        for key, value in similarity_scores_of_activities.items()
    }

    return normalized_data

def save_results(results, log_name, activity_distance_function, different_activities_to_replace_count, activities_to_replace_with_count, sampling_size):
    file_name = f"r_{different_activities_to_replace_count}_w_{activities_to_replace_with_count}_s_{sampling_size}.pkl"
    path_to_file = os.path.join(
        ROOT_DIR, "evaluation", "evaluation_of_activity_distances",
        "intrinsic_evaluation", "results", log_name, activity_distance_function, file_name
    )

    # Ensure the directory exists before saving
    os.makedirs(os.path.dirname(path_to_file), exist_ok=True)  # Create all missing directories

    # Save results
    with open(path_to_file, "wb") as f:
        pickle.dump(results, f)


def load_results(log_name, activity_distance_function, different_activities_to_replace_count, activities_to_replace_with_count, sampling_size):

    file_name = f"r_{different_activities_to_replace_count}_w_{activities_to_replace_with_count}_s_{sampling_size}.pkl"

    path_to_file = os.path.join(ROOT_DIR, "evaluation", "evaluation_of_activity_distances",
                                "intrinsic_evaluation",
                                "results", log_name, activity_distance_function, file_name)

    # Check if the file exists and load it
    if os.path.exists(path_to_file):
        with open(path_to_file, "rb") as f:
            return pickle.load(f)
    else:
        return None


def add_window_size_evaluation(activity_distance_functions, window_size_list):

    new_activity_distance_function_list = list()
    for activity_distance_function in activity_distance_functions:
        if activity_distance_function.startswith(
                ("Bose", "De Koninck", "Activity-Activitiy", "Activity-Context", "Our")):
            for window_size in window_size_list:
                new_activity_distance_function_list.append(activity_distance_function + " w_" + str(window_size))
        else:
            new_activity_distance_function_list.append(activity_distance_function)
    return new_activity_distance_function_list

def get_triplet_for_example():
    distance_matrix = {
    ("a:1", "a:1"): 0, ("a:1", "a:2"): 5, ("a:1", "a:3"): 5, ("a:1", "b:1"): 2, ("a:1", "b:2"): 2, ("a:1", "c"): 8, ("a:1", "d"): 7, ("a:1", "e"): 7,
    ("a:2", "a:1"): 5, ("a:2", "a:2"): 0, ("a:2", "a:3"): 1, ("a:2", "b:1"): 3, ("a:2", "b:2"): 4, ("a:2", "c"): 3, ("a:2", "d"): 2, ("a:2", "e"): 3,
    ("a:3", "a:1"): 5, ("a:3", "a:2"): 1, ("a:3", "a:3"): 0, ("a:3", "b:1"): 4, ("a:3", "b:2"): 3, ("a:3", "c"): 3, ("a:3", "d"): 3, ("a:3", "e"): 2,
    ("b:1", "a:1"): 2, ("b:1", "a:2"): 3, ("b:1", "a:3"): 4, ("b:1", "b:1"): 0, ("b:1", "b:2"): 1, ("b:1", "c"): 6, ("b:1", "d"): 5, ("b:1", "e"): 6,
    ("b:2", "a:1"): 2, ("b:2", "a:2"): 4, ("b:2", "a:3"): 3, ("b:2", "b:1"): 1, ("b:2", "b:2"): 0, ("b:2", "c"): 6, ("b:2", "d"): 6, ("b:2", "e"): 5,
    ("c",   "a:1"): 8, ("c",   "a:2"): 3, ("c",   "a:3"): 3, ("c",   "b:1"): 6, ("c",   "b:2"): 6, ("c",   "c"):   0, ("c",   "d"): 1, ("c",   "e"): 1,
    ("d",   "a:1"): 7, ("d",   "a:2"): 2, ("d",   "a:3"): 3, ("d",   "b:1"): 5, ("d",   "b:2"): 6, ("d",   "c"):   1, ("d",   "d"):   0, ("d",   "e"): 2,
    ("e",   "a:1"): 7, ("e",   "a:2"): 3, ("e",   "a:3"): 2, ("e",   "b:1"): 6, ("e",   "b:2"): 5, ("e",   "c"):   1, ("e",   "d"): 2, ("e",   "e"): 0,
}

    avg_triplet_value = get_avg_triplet_value(("a", "b"), distance_matrix, 3, False, [])
    print(avg_triplet_value)
