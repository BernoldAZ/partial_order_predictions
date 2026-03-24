import multiprocessing
import random
from multiprocessing import Pool
import os
import pickle
import time

from pm4py.objects.log.importer.xes import importer as xes_importer

from definitions import ROOT_DIR
from evaluation.data_util.util_activity_distances import get_alphabet, get_activity_distance_matrix_dict, \
    get_log_control_flow_perspective_with_short_activity_names, delete_temporary_files
from evaluation.data_util.util_activity_distances_extrinsic import get_sublog_list
from evaluation.data_util.util_activity_distances_intrinsic import (
    get_log_control_flow_perspective, get_activities_to_replace,
    get_logs_with_replaced_activities_dict,
    get_knn_dict, get_precision_at_k, save_intrinsic_results, get_triplet, get_diameter, load_results, save_results, add_window_size_evaluation
)


def evaluate_intrinsic(activity_distance_functions, log_list, r_min, w, sampling_size, load_ground_truth_logs):
    # Load event logs from files
    for log_name in log_list:

        path_to_dict = os.path.join(ROOT_DIR, "evaluation", "evaluation_of_activity_distances", "intrinsic_evaluation",
                                    "newly_created_logs", log_name)
        if not os.path.exists(path_to_dict):
            os.makedirs(path_to_dict)
        log = xes_importer.apply(ROOT_DIR + '/event_logs/' + log_name + '.xes.gz')
        log_control_flow_perspective = get_log_control_flow_perspective(log)

        alphabet = get_alphabet(log_control_flow_perspective)

        # Transform activity labels into sequential numbers to improve performance.
        log_control_flow_perspective, short_activity_names_dict = get_log_control_flow_perspective_with_short_activity_names(log_control_flow_perspective, alphabet)
        alphabet = get_alphabet(log_control_flow_perspective)

        # Limit the number of activity replacements to 'r_min' when the total number of activities in the log exceeds 'r_min' to enhance performance.
        r = min(r_min, len(alphabet))

        # Define all subproblems required for our intrinsic evaluation, allowing each subproblem to be assigned to a separate core for parallel processing.
        combinations = [
            (
                different_activities_to_replace_count, activities_to_replace_with_count, log_control_flow_perspective,
                alphabet, activity_distance_functions, sampling_size, load_ground_truth_logs, log_name)
            for different_activities_to_replace_count in range(1, r + 1)
            for activities_to_replace_with_count in range(2, w + 1)
        ]

        # Limit used cores, to ensure system responsiveness
        total_cores = multiprocessing.cpu_count()

        # Calculate 75% of the available cores
        cores_to_use = int(total_cores * 0.8)

        # Ensure at least one core is used
        cores_to_use = max(1, cores_to_use)

        # Set MP to 1 to use multiprocessing
        mp = 0
        if mp == 1:
            with Pool(processes=cores_to_use) as pool:
                results = pool.map(intrinsic_evaluation, combinations)
            
        else:
            results = list()
            for combination in combinations:
                results.append(intrinsic_evaluation(combination))

        #delete tempory files
        delete_temporary_files()

        # Combine and save results
        save_intrinsic_results(activity_distance_functions, results, log_name, r, w, sampling_size)


def intrinsic_evaluation(args):
    different_activities_to_replace_count, activities_to_replace_with_count, log_control_flow_perspective, alphabet, activity_distance_function_list, sampling_size, load_logs, log_name = args
    # if we want to load created event logs but corresponding file does not exist, save new event logs to disk
    create_logs_to_then_save = False
    # load created event logs
    path_to_dict = os.path.join(ROOT_DIR, "evaluation", "evaluation_of_activity_distances", "intrinsic_evaluation",
                                "newly_created_logs", log_name)
    file_name = f"r_{different_activities_to_replace_count}_w_{activities_to_replace_with_count}_s_{sampling_size}.pkl"
    file_path = os.path.join(path_to_dict, file_name)

    if load_logs:
        if os.path.isfile(file_path):
            logs_with_replaced_activities_dict = pickle.load(open(file_path, "rb"))
            activities_to_replace_in_each_run_list = [key for key in logs_with_replaced_activities_dict.keys()]
        else:
            create_logs_to_then_save = True
    # create new event logs, if we wanted to load them from disk but file does not exist, save them to disk
    if (not load_logs) or create_logs_to_then_save:
        # 1: get the activities that we want to replace in each run
        activities_to_replace_in_each_run_list = get_activities_to_replace(alphabet, different_activities_to_replace_count,
                                                                       sampling_size)

        #2: create dict that maps activities_to_replace_in_each_run to new event logs
        logs_with_replaced_activities_dict = get_logs_with_replaced_activities_dict(
                    activities_to_replace_in_each_run_list, log_control_flow_perspective,
                    different_activities_to_replace_count, activities_to_replace_with_count
                )

        with open(file_path, "wb") as file:
            pickle.dump(logs_with_replaced_activities_dict, file)

    print("start ---- r:" + str(different_activities_to_replace_count) + " w: " + str(activities_to_replace_with_count))


    results_list = list()

    for activity_distance_function in activity_distance_function_list:

        #try to load old results if they exist, then no need to redo computation
        run_experiment = 1
        if load_logs:
            results = load_results(log_name, activity_distance_function, different_activities_to_replace_count, activities_to_replace_with_count, sampling_size)
            if results is not None:
                results_list.append(results)
                run_experiment = 0

        if run_experiment:
            activity_distance_function = [activity_distance_function]

            diameter_list = list()
            precision_at_w_minus_1_list = list()
            precision_at_1_list = list()
            triplet_list = list()

            for activities_to_replace in activities_to_replace_in_each_run_list:
                # 2: replace activities
                logs_with_replaced_activities = dict()
                logs_with_replaced_activities[activities_to_replace] = logs_with_replaced_activities_dict[activities_to_replace]

                # 3: Compute distances between activities
                activity_distance_matrix_dict = get_activity_distance_matrix_dict(
                    activity_distance_function, logs_with_replaced_activities
                )

                if "Bose 2009 Substitution Scores" in activity_distance_function[0]:
                    reverse = True  # high values = high similarity
                else:
                    reverse = False  # high values = high distances

                # To store computation times
                computation_times = {}

                # Measure time for Average Diameter Distance
                start_time = time.time()
                diameter_list.append(
                    get_diameter(activity_distance_matrix_dict, activities_to_replace_with_count, reverse, alphabet))
                computation_times['Average Diameter'] = time.time() - start_time

                # Measure time for Precision@w-1
                start_time = time.time()
                w_minus_one_nn_dict = get_knn_dict(activity_distance_matrix_dict, activities_to_replace_with_count,
                                                   reverse,
                                                   activities_to_replace_with_count - 1)
                precision_at_w_minus_1_dict = get_precision_at_k(w_minus_one_nn_dict, activity_distance_function)
                precision_at_w_minus_1_list.append(precision_at_w_minus_1_dict[activity_distance_function[0]])
                computation_times['Precision@w-1'] = time.time() - start_time

                # Measure time for Nearest Neighbor
                start_time = time.time()
                one_nn_dict = get_knn_dict(activity_distance_matrix_dict, activities_to_replace_with_count, reverse, 1)
                precision_at_1_dict = get_precision_at_k(one_nn_dict, activity_distance_function)
                precision_at_1_list.append(precision_at_1_dict[activity_distance_function[0]])
                computation_times['Nearest Neighbor'] = time.time() - start_time

                # Measure time for Triplet
                start_time = time.time()
                triplet_list.append(
                    get_triplet(activity_distance_matrix_dict, activities_to_replace_with_count, reverse, alphabet))
                computation_times['Triplet'] = time.time() - start_time

                # Print the computation times
                #for key, value in computation_times.items():
                #    print(f"{key}: {value:.4f} seconds")

            diameter = sum(diameter_list) / len(diameter_list)
            precision_at_w_minus_1 = sum(precision_at_w_minus_1_list) / len(precision_at_w_minus_1_list)
            precision_at_1 = sum(precision_at_1_list) / len(precision_at_1_list)
            triplet = sum(triplet_list) / len(triplet_list)
            results_list.append((different_activities_to_replace_count, activities_to_replace_with_count, diameter,
                                 precision_at_w_minus_1, precision_at_1, triplet))
            #save results
            results = (different_activities_to_replace_count, activities_to_replace_with_count, diameter,
                             precision_at_w_minus_1, precision_at_1, triplet)
            save_results(results, log_name, activity_distance_function[0], different_activities_to_replace_count, activities_to_replace_with_count, sampling_size)

    print("end ---- r:" + str(different_activities_to_replace_count) + " w: " + str(
        activities_to_replace_with_count) + " sampling size: " + str(sampling_size))
    return results_list


if __name__ == '__main__':
    # ==============================================================================
    # Similarity Methods to Evaluate
    # ==============================================================================

    activity_distance_functions = []

    # --- Our New Methods ---
    activity_distance_functions.append("Activity-Activitiy Co Occurrence Bag Of Words")
    activity_distance_functions.append("Activity-Activitiy Co Occurrence N-Gram")
    activity_distance_functions.append("Activity-Activitiy Co Occurrence Bag Of Words PMI")
    activity_distance_functions.append("Activity-Activitiy Co Occurrence N-Gram PMI")
    activity_distance_functions.append("Activity-Activitiy Co Occurrence Bag Of Words PPMI")
    activity_distance_functions.append("Activity-Activitiy Co Occurrence N-Gram PPMI")

    activity_distance_functions.append("Activity-Context Bag Of Words")
    activity_distance_functions.append("Activity-Context N-Grams")
    activity_distance_functions.append("Activity-Context Bag Of Words PMI")
    activity_distance_functions.append("Activity-Context N-Grams PMI")
    activity_distance_functions.append("Activity-Context Bag Of Words PPMI")
    activity_distance_functions.append("Activity-Context N-Grams PPMI")

    # --- Existing Methods ---
    # activity_distance_functions.append("Unit Distance")
    activity_distance_functions.append("Bose 2009 Substitution Scores")
    activity_distance_functions.append("De Koninck 2018 act2vec CBOW")
    activity_distance_functions.append("De Koninck 2018 act2vec skip-gram")
    activity_distance_functions.append("Chiorrini 2022 Embedding Process Structure")
    #activity_distance_functions.append("Gamallo Fernandez 2023 Context Based w_3")  # set x in w_x to desired window size

    # ==============================================================================
    # Parameters
    # ==============================================================================

    # Parameters for ground truth log creation
    r_min         = 10
    w             = 5
    sampling_size = 5
    load_ground_truth_logs = False
    # If True, load existing ground truth logs if available; otherwise, create and save new ones.
    # If False, always create new logs without loading existing ones.
    # If True, if results of the corresponding parameters exist for the evaluated original log and method, load them from evaluation / evaluation_of_activity_distances / intrinsic_evaluation / results, if results file daoes not exist do the experiemnt.
    # To rerun the experiemnts with new event logs to store them, delete the evaluation / evaluation_of_activity_distances / intrinsic_evaluation / results folder.

    # Pre-generated logs: https://box.hu-berlin.de/d/7a97101239654eae8e6c/
    # Unzip and place in 'evaluation/evaluation_of_activity_distances/intrinsic_evaluation/newly_created_logs'

    # Parameters for Similarity / Embedding Methods
    window_size_list = [3, 5, 9]
    activity_distance_functions = add_window_size_evaluation(activity_distance_functions, window_size_list)

    # ==============================================================================
    # Intrinsic Evaluation - Event Logs to Evaluate
    # ==============================================================================

    log_list = []
    log_list.append('Sepsis')

    # Optional: Add more logs below as needed
    """
    log_list.append('BPIC12')
    log_list.append('BPIC12_A')
    log_list.append('BPIC12_Complete')
    log_list.append('BPIC12_O')
    log_list.append('BPIC12_W')
    log_list.append('BPIC12_W_Complete')
    log_list.append('BPIC13_closed_problems')
    log_list.append('BPIC13_incidents')
    log_list.append('BPIC13_open_problems')
    log_list.append('BPIC15_1')
    log_list.append('BPIC15_2')
    log_list.append('BPIC15_3')
    log_list.append('BPIC15_4')
    log_list.append('BPIC15_5')
    log_list.append('BPIC17')
    log_list.append('BPIC18')
    log_list.append('BPIC19')
    log_list.append('BPIC20_DomesticDeclarations')
    log_list.append('BPIC20_InternationalDeclarations')
    log_list.append('BPIC20_PermitLog')
    log_list.append('BPIC20_PrepaidTravelCost')
    log_list.append('BPIC20_RequestForPayment')
    log_list.append('CCC19')
    log_list.append('Env Permit')
    log_list.append('Helpdesk')
    log_list.append('Hospital Billing')
    log_list.append('RTFM')
    """

    # ==============================================================================
    # Overview of Configuration
    # ==============================================================================

    print("Selected Similarity Methods:")
    for method in activity_distance_functions:
        print(f" - {method}")

    print("\nEvaluation Parameters:")
    print(f" - r_min         : {r_min}")
    print(f" - w             : {w}")
    print(f" - sampling_size : {sampling_size}")
    print(f" - load_ground_truth_logs    : {load_ground_truth_logs}")
    print(f" - logs used     : {log_list}")

    # ==============================================================================
    # Start Benchmark
    # ==============================================================================

    evaluate_intrinsic(activity_distance_functions, log_list, r_min, w, sampling_size, load_ground_truth_logs)
