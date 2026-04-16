from pathlib import Path

import pandas as pd
import os
import seaborn as sns
from matplotlib import pyplot as plt, rcParams, rc

from definitions import ROOT_DIR
import pickle
def visualization_intrinsic_evaluation_from_csv(log_name):

    '''
    #w_index_nn_list = list()
    #w_index_prec_list = list()
    for file in os.listdir(ROOT_DIR + '/results/activity_distances/intrinsic/' + log_name):
        # Load the DataFrame from the CSV file
        df = pd.read_csv(ROOT_DIR + '/results/activity_distances/intrinsic/' + log_name + "/" + file)
        # Load results
        result_prec = df.pivot(index='w', columns='r', values='precision@w-1')
        result_nn = df.pivot(index='w', columns='r', values='precision@1')

        # Find the index of the first rows where the average is below 0.05
        row_means_prec = result_prec.mean(axis=1)
        w_index_prec_list.append(row_means_prec[row_means_prec < 0.2].index[0])
        row_means_nn = result_nn.mean(axis=1)
        w_index_nn_list.append(row_means_nn[row_means_prec < 0.2].index[0])
    '''

    for file in os.listdir(ROOT_DIR + '/results/activity_distances/intrinsic/' + log_name):
       
        # Step 1: Remove the ".csv" extension
        input_str = file.replace(".csv", "")

        # Step 2: Split the string by underscores
        parts = input_str.split("_")

        # Step 3: Assign the parts to the variables
        log_name = parts[0]
        activity_distance_function = parts[2]
        r = int(parts[3][1:])  # Extract the number after 'r'
        w = int(parts[4][1:])  # Extract the number after 'w'
        sampling_size = int(parts[6])  # Extract the number after 'samplesize'
        print(f"r: {r}, w: {w}, sampling_size: {sampling_size}")

        # Load the DataFrame from the CSV file
        df = pd.read_csv(ROOT_DIR + '/results/activity_distances/intrinsic/' + log_name + "/" + file)

        # Load results
        #max_w = max(w_index_prec_list)
        max_w = 30

        print(str(max_w) + "prec")

        #max_w = 30
        df_filtered = df[df['w'] <= max_w]
        result_prec = df_filtered.pivot(index='w', columns='r', values='precision@w-1')
        average_value_prec = result_prec.values.mean()
        rc('font', **{'family': 'serif', 'serif': ['Times New Roman'], 'size': 12})
        cmap = sns.cm.rocket_r
        ax = sns.heatmap(result_prec, cmap=cmap, vmin=0, vmax=1, linewidth=.5)
        ax.invert_yaxis()
        ax.set_title("precision@w-1 for " + log_name + "\n" + activity_distance_function + " - Sampling Size: " + str(
            sampling_size))
        Path(ROOT_DIR + "/results/activity_distances/intrinsic/precision_at_k").mkdir(parents=True, exist_ok=True)
        plt.savefig(
            ROOT_DIR + "/results/activity_distances/intrinsic/precision_at_k/" + "pre_" + activity_distance_function + "_" + log_name + "_r:" + str(
                r) + "_w:" + str(w) + "_sampling:" + str(sampling_size) + ".pdf", format="pdf", transparent=True)
        plt.show()

        # heat map precision@1
        #max_w = max(w_index_nn_list)
        print(str(max_w) + "nn")
        max_w = 31

        result_nn = df_filtered.pivot(index='w', columns='r', values='precision@1')

        average_value_nn = result_nn.values.mean()
        print("Average Nearest Neighbor is: " + str(average_value_nn) + " " + activity_distance_function)
        # Plotting
        #rc('font', **{'family': 'serif', 'size': 20})
        #f, ax = plt.subplots()
        cmap = sns.cm.rocket_r
        ax = sns.heatmap(result_nn, cmap=cmap, vmin=0, vmax=1, linewidth=.5)
        ax.invert_yaxis()
        ax.set_title("Nearest Neighbor for " + log_name + "\n" + activity_distance_function + " - Sampling Size: " + str(
            sampling_size))
        Path(ROOT_DIR + "/results/activity_distances/intrinsic/nn").mkdir(parents=True, exist_ok=True)
        plt.savefig(
            ROOT_DIR + "/results/activity_distances/intrinsic/nn/" + "nn" + activity_distance_function + "_" + log_name + "_r:" + str(
                r) + "_w:" + str(w) + "_sampling:" + str(sampling_size) + ".pdf", format="pdf", transparent=True)
        plt.show()
        print("Average precision@w-1 is: " + str(average_value_prec) + " " + activity_distance_function)

def visualization_intrinsic_evaluation_from_pkl(log_name):
    path_to_log = os.path.join(ROOT_DIR, "evaluation", "evaluation_of_activity_distances", "intrinsic_evaluation",
                                "results", log_name)
    # Iterate over all activity distance directories
    for activity_distance_name in os.listdir(path_to_log):
        result_list = []

        # Iterate over result files in each directory
        for result in os.listdir(os.path.join(path_to_log, activity_distance_name)):
            path_to_result = os.path.join(path_to_log, activity_distance_name, result)

            # Load only .pkl files
            if result.endswith(".pkl") and os.path.isfile(path_to_result):
                with open(path_to_result, "rb") as file:
                    data = pickle.load(file)
                    result_list.append(data)

        # Convert list of dictionaries to DataFrame
        df = pd.DataFrame(result_list, columns=['r', 'w', "diameter", "precision@w-1", "nn", "triplet"])

        # Plot heatmaps for each metric
        metrics = ["diameter", "precision@w-1", "nn", "triplet"]
        for metric in metrics:
            plt.figure(figsize=(8, 6))

            # Create pivot table and sort w values in descending order
            pivot_table = df.pivot(index="w", columns="r", values=metric).sort_index(ascending=False)

            sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="viridis")
            plt.title(f"Heatmap of {metric} - {activity_distance_name}")  # Activity distance in title
            plt.xlabel("r")
            plt.ylabel("w")
            plt.show()


if __name__ == '__main__':
    #log_name = "BPI Challenge 2018"
    #log_name = "Road Traffic Fine Management Process"
    #log_name = "BPI Challenge 2015 1"
    #log_name = "BPI Challenge 2017"
    #log_name = "BPI Challenge 2018"
    #log_name = "BPI Challenge 2019"
    #log_name = "PDC 2016" # take for visualization
    #log_name = "PDC 2017" # visualization
    #log_name = "PDC 2019"
    #log_name = "Sepsis"
    #log_name = "WABO"
    log_name = "Sepsis"
    visualization_intrinsic_evaluation_from_pkl(log_name)