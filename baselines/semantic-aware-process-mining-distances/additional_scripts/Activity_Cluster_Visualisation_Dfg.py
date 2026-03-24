import multiprocessing
import os
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx  # For DFG visualization
from sklearn.manifold import MDS
import hdbscan
from scipy.spatial import ConvexHull
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from evaluation.data_util.util_activity_distances import get_alphabet, get_normalized_activity_distance_matrix_dict_list, get_obj_size, print_log_stats
from evaluation.data_util.util_activity_distances_intrinsic import get_log_control_flow_perspective
from evaluation.data_util.util_activity_distances_extrinsic import (
    get_sublog_list, get_trace_distances, get_precision_values, get_log_with_trace_ids, print_avg_values, get_sampled_sublogs
)

from definitions import ROOT_DIR

def plot_dfg(dfg):
    """
    Function to visualize the Directly Follows Graph (DFG) using networkx.
    """
    G = nx.DiGraph()

    # Add edges from DFG (directly follows graph)
    for (activity1, activity2), count in dfg.items():
        G.add_edge(activity1, activity2, weight=count)

    plt.figure(figsize=(10, 10))

    # Draw the network graph
    pos = nx.spring_layout(G, seed=42)  # Position the nodes using a spring layout
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, font_size=10, font_weight='bold', arrows=True, arrowstyle='->', arrowsize=20)

    # Add edge labels (frequencies)
    edge_labels = {(u, v): f'{d["weight"]}' for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title('Directly Follows Graph (DFG)')
    plt.show()


def run_mds():
    activity_distance_functions = list()
    activity_distance_functions.append("Bose 2009 Substitution Scores")
    activity_distance_functions.append("De Koninck 2018 act2vec CBOW")

    log_name = "Sepsis"
    if log_name[:4] == "bpic" or log_name[:3] == "pdc":
        sublog_list = get_sublog_list(log_name)
        sublogsize_list = [len(sublog) for sublog in sublog_list]
        log_control_flow_perspective = [inner for outer in sublog_list for inner in outer]
    else:

        log = xes_importer.apply(ROOT_DIR + '/event_logs/' + log_name + '.xes')
        #pm4py.view_process_tree(pm4py.discover_process_tree_inductive(log))
        log_control_flow_perspective = get_log_control_flow_perspective(log)
        sublog_list = [log]
        sublogsize_list = [len(sublog) for sublog in sublog_list]
        #print(get_obj_size(log_control_flow_perspective))


    ''' 
    log = xes_importer.apply(ROOT_DIR + '/event_logs/' + log_name + '.xes')
    log_control_flow_perspective = get_log_control_flow_perspective(log)
    '''
    alphabet = get_alphabet(log_control_flow_perspective)
    #'''

    combinations = [
        (log_control_flow_perspective, activity_distance_function, alphabet)
        for activity_distance_function in activity_distance_functions
    ]

    # limit used cores, for system responsiveness
    total_cores = multiprocessing.cpu_count()

    # Calculate 20% of the available cores
    cores_to_use = int(total_cores * 0.20)

    # Ensure at least one core is used
    cores_to_use = max(1, cores_to_use)

    with Pool(processes=cores_to_use) as pool:
        activity_distance_matrix_dict_list = pool.map(get_normalized_activity_distance_matrix_dict_list, combinations)

    # Step 1: Discover and plot Directly Follows Graph (DFG)
    dfg_list = list()
    if log_name[:4] == "bpic" or log_name[:3] == "pdc":
        for sublog_name in os.listdir(ROOT_DIR + '/event_logs/' + log_name):
            sublog = xes_importer.apply(ROOT_DIR + '/event_logs/' + log_name + "/" + sublog_name)
            dfg = dfg_discovery.apply(sublog)
            dfg_list.append(dfg)
    else:
        dfg = dfg_discovery.apply(ROOT_DIR + '/event_logs/' + log_name)
        dfg_list.append(dfg)
    #plot_dfg(dfg)  # Plot DFG at the beginning

    for distance_data_ in activity_distance_matrix_dict_list:
        distance_data = list(distance_data_.values())[0]

        # Step 2: Extract the activity names
        activities = sorted(list(set([key[0] for key in distance_data.keys()] + [key[1] for key in distance_data.keys()])))

        # Step 3: Create the distance matrix
        n = len(activities)
        distance_matrix = np.zeros((n, n))

        for i, act1 in enumerate(activities):
            for j, act2 in enumerate(activities):
                if (act1, act2) in distance_data:
                    distance_matrix[i, j] = distance_data[(act1, act2)]
                elif (act2, act1) in distance_data:
                    distance_matrix[i, j] = distance_data[(act2, act1)]

        # Step 4: Perform MDS
        mds = MDS(dissimilarity='precomputed', random_state=42)
        mds_result = mds.fit_transform(distance_matrix)

        # Step 5: Perform HDBSCAN clustering
        hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
        cluster_labels = hdbscan_clusterer.fit_predict(mds_result)

        # Step 6: Create MDS plot with colored clusters
        plt.figure(figsize=(5, 5))
        unique_labels = set(cluster_labels)

        # Define colors for clusters and assign a specific color for noise
        colors = plt.cm.get_cmap('tab10', len(unique_labels) - (1 if -1 in unique_labels else 0))  # Exclude noise
        noise_color = 'black'  # Distinct color for noise

        # Store coordinates of clusters for convex hulls
        cluster_points = {label: [] for label in unique_labels}

        # Draw points, color based on clusters
        activity_to_mds_mapping = {}
        plotted_clusters = set()  # To ensure we only add one legend entry per cluster

        for i, activity in enumerate(activities):
            label = cluster_labels[i]
            if label == -1:  # Noise points (label -1)
                color = noise_color
                label_text = 'Noise'
            else:
                color = colors(label)
                label_text = f'Cluster {label}'

            # Plot the activity point with corresponding cluster color
            plt.scatter(mds_result[i, 0], mds_result[i, 1], color=color, label=label_text if label not in plotted_clusters else '')
            plt.text(mds_result[i, 0], mds_result[i, 1], activity, fontsize=9)

            activity_to_mds_mapping[activity] = (mds_result[i, 0], mds_result[i, 1])

            # Add the label to plotted_clusters to avoid multiple legend entries for the same cluster
            plotted_clusters.add(label)

        # Step 8: Add arrows for directly follows relations with smaller size
        for dfg in dfg_list:
            for (activity1, activity2), count in dfg.items():
                if activity1 in activity_to_mds_mapping and activity2 in activity_to_mds_mapping:
                    x_start, y_start = activity_to_mds_mapping[activity1]
                    x_end, y_end = activity_to_mds_mapping[activity2]

                    # Smaller arrows
                    plt.arrow(x_start, y_start, (x_end - x_start) * 0.9, (y_end - y_start) * 0.9,  # Scale down the arrow length
                              head_width=0.02, head_length=0.02, fc='green', ec='green', alpha=0.6)
        #'''
        # Step 9: Add legend for clusters (only one entry per cluster)
        plt.legend(title="Cluster Labels", loc='upper right')

        plt.title('MDS Visualization with HDBSCAN Clusters, DFG Arrows: ' + list(distance_data_.keys())[0])
        plt.xlabel('MDS Dimension 1')
        plt.ylabel('MDS Dimension 2')
        plt.grid(True)
        plt.show()

if __name__ == '__main__':
    run_mds()
