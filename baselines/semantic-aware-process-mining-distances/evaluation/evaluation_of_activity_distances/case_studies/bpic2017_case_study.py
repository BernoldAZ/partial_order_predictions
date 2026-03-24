from pm4py.objects.log.importer.xes import importer as xes_importer

from definitions import ROOT_DIR
from distances.activity_distances.bose_2009_context_aware_trace_clustering.algorithm import \
    get_substitution_and_insertion_scores
from distances.activity_distances.de_koninck_2018_act2vec.algorithm import get_act2vec_distance_matrix
from evaluation.data_util.util_activity_distances import get_alphabet, get_unit_cost_activity_distance_matrix
from evaluation.data_util.util_activity_distances_intrinsic import (
    get_log_control_flow_perspective
)
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA




def remap_labels(cluster_labels, true_labels):
    """
    Remap cluster labels to match the true labels as closely as possible.

    Parameters:
    - cluster_labels: List of cluster labels predicted by the clustering algorithm.
    - true_labels: List of true labels.

    Returns:
    - remapped_labels: List of remapped cluster labels.
    """
    # Create a confusion matrix
    max_label = max(max(cluster_labels), max(true_labels)) + 1
    confusion_matrix = np.zeros((max_label, max_label), dtype=int)

    for pred, true in zip(cluster_labels, true_labels):
        confusion_matrix[pred, true] += 1

    # Use the Hungarian algorithm to find the optimal label assignment
    row_ind, col_ind = linear_sum_assignment(-confusion_matrix)  # Negative because it's a maximization problem

    # Create a mapping from old to new labels
    label_mapping = {row: col for row, col in zip(row_ind, col_ind)}

    # Remap the cluster labels
    remapped_labels = [label_mapping[label] for label in cluster_labels]

    return remapped_labels


def evaluate_overlap_with_remapping(activities, cluster_labels, color_groups):
    """
    Evaluate overlap between clusters and predefined color groups, using remapped labels.
    """
    # Create true labels based on color groups
    activity_to_color = {}
    for idx, color_group in enumerate(color_groups):
        for activity in color_group:
            activity_to_color[activity] = idx

    true_labels = [activity_to_color.get(activity, -1) for activity in activities]

    # Remove activities not in the predefined groups
    valid_indices = [i for i, label in enumerate(true_labels) if label != -1]
    filtered_true_labels = [true_labels[i] for i in valid_indices]
    filtered_cluster_labels = [cluster_labels[i] for i in valid_indices]

    # Remap cluster labels
    remapped_labels = remap_labels(filtered_cluster_labels, filtered_true_labels)

    print(remapped_labels)
    print(filtered_true_labels)

    # Evaluate clustering performance
    ari = adjusted_rand_score(filtered_true_labels, remapped_labels)
    nmi = normalized_mutual_info_score(filtered_true_labels, remapped_labels)

    return ari, nmi


def cluster_activities(activity_distance_matrix, n_clusters, method="agglomerative"):
    """
    Cluster activities using the specified clustering method.

    Parameters:
    - activity_distance_matrix: dict of distance values
    - n_clusters: number of clusters
    - method: clustering method ('agglomerative', 'kmeans', 'dbscan', 'spectral')

    Returns:
    - activities: list of activity names
    - cluster_labels: list of cluster labels
    """
    activities = list({a for pair in activity_distance_matrix.keys() for a in pair})
    activity_indices = {activity: idx for idx, activity in enumerate(activities)}

    # Create a square matrix from the distance dictionary
    distance_matrix = np.zeros((len(activities), len(activities)))
    for (act1, act2), distance in activity_distance_matrix.items():
        idx1, idx2 = activity_indices[act1], activity_indices[act2]
        distance_matrix[idx1, idx2] = distance
        distance_matrix[idx2, idx1] = distance

    if method == "agglomerative":
        clustering_model = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="precomputed",
            linkage="average",
        )
    elif method == "kmeans":
        # Convert the distance matrix to a similarity matrix for K-Means
        similarity_matrix = 1 - distance_matrix / np.max(distance_matrix)
        clustering_model = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = clustering_model.fit_predict(similarity_matrix)
    elif method == "dbscan":
        clustering_model = DBSCAN(eps=0.5, metric="precomputed", min_samples=2)
        cluster_labels = clustering_model.fit_predict(distance_matrix)
    elif method == "spectral":
        clustering_model = SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            random_state=42,
        )
        cluster_labels = clustering_model.fit_predict(distance_matrix)
    else:
        raise ValueError(f"Unsupported clustering method: {method}")

    if method != "kmeans":  # K-Means already computes cluster_labels
        cluster_labels = clustering_model.fit_predict(distance_matrix)

    return activities, cluster_labels


def evaluate_overlap(activities, cluster_labels, color_groups):
    """
    Evaluate overlap between clusters and predefined color groups.
    """
    # Create true labels based on color groups
    activity_to_color = {}
    for idx, color_group in enumerate(color_groups):
        for activity in color_group:
            activity_to_color[activity] = idx

    true_labels = [activity_to_color.get(activity, -1) for activity in activities]

    # Remove activities not in the predefined groups
    valid_indices = [i for i, label in enumerate(true_labels) if label != -1]
    filtered_true_labels = [true_labels[i] for i in valid_indices]
    filtered_cluster_labels = [cluster_labels[i] for i in valid_indices]

    # Evaluate clustering performance
    ari = adjusted_rand_score(filtered_true_labels, filtered_cluster_labels)
    nmi = normalized_mutual_info_score(filtered_true_labels, filtered_cluster_labels)

    return ari, nmi


def reverse_activity_distance_matrix(activity_distance_matrix):
    """
    Reverse all values in the activity distance matrix and ensure positivity.
    """
    # Get the maximum value in the matrix
    max_distance = max(activity_distance_matrix.values())

    # Reverse the values and make them positive
    reversed_matrix = {
        key: max_distance - value
        for key, value in activity_distance_matrix.items()
    }
    return reversed_matrix


def get_activity_distances(activity_distance_function, log,
                           n_gram_size_bose_2009=3):
    if "Bose 2009 Substitution Scores" == activity_distance_function:
        activity_distance_matrix = get_substitution_and_insertion_scores(
            log,
            get_alphabet(
                log), n_gram_size_bose_2009)
    elif "De Koninck 2018 act2vec" == activity_distance_function[:23]:
        if activity_distance_function[24:] == "CBOW":
            sg = 0
        else:
            sg = 1
        activity_distance_matrix = get_act2vec_distance_matrix(log,
                                                               get_alphabet(
                                                                   log), sg)
    elif "Unit Distance" == activity_distance_function:
        activity_distance_matrix = get_unit_cost_activity_distance_matrix(log, get_alphabet(
            log))

    return activity_distance_matrix


def remove_lists_with_extra_activities(log_control_flow_perspective, extra_activities):
    # Filter out lists that contain any activity in extra_activities
    log_control_flow_perspective = [
        activity_list for activity_list in log_control_flow_perspective
        if not any(activity in extra_activities for activity in activity_list)
    ]
    return log_control_flow_perspective

def visualize_activities_in_2d(activities, activity_distance_matrix, color_groups):
    """
    Visualize activities in a 2D plane based on the distance matrix.

    Parameters:
    - activities: List of activity names.
    - activity_distance_matrix: Dictionary of activity pair distances.
    - color_groups: List of lists, where each sublist represents activities with the same color.
    """
    # Convert activity_distance_matrix (dict) into a square numpy array
    activity_indices = {activity: idx for idx, activity in enumerate(activities)}
    distance_matrix = np.zeros((len(activities), len(activities)))

    for (act1, act2), distance in activity_distance_matrix.items():
        idx1, idx2 = activity_indices[act1], activity_indices[act2]
        distance_matrix[idx1, idx2] = distance
        distance_matrix[idx2, idx1] = distance  # Ensure symmetry

    # Map activities to colors
    activity_to_color = {}
    color_palette = ['gray', 'yellow', 'blue', 'green', 'orange']  # Predefined colors for groups
    for idx, color_group in enumerate(color_groups):
        for activity in color_group:
            activity_to_color[activity] = color_palette[idx]

    # Get colors for the activities
    colors = [activity_to_color.get(activity, 'black') for activity in activities]

    # Dynamically set perplexity based on the number of activities
    perplexity = min(30, len(activities) - 1)

    # Reduce dimensionality to 2D using t-SNE with random initialization
    tsne = TSNE(n_components=2, metric='precomputed', perplexity=perplexity, init='random', random_state=42)
    coordinates = tsne.fit_transform(distance_matrix)

    # Plot the activities in 2D
    plt.figure(figsize=(10, 8))
    for i, activity in enumerate(activities):
        plt.scatter(
            coordinates[i, 0],
            coordinates[i, 1],
            color=colors[i],
            s=100,
            alpha=0.8,
            edgecolors='k'
        )
        plt.text(coordinates[i, 0] + 0.03, coordinates[i, 1] + 0.03, activity, fontsize=9)

    # Add legend
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=color_group[0])
        for color, color_group in zip(color_palette, color_groups)
    ]
    plt.legend(handles=handles, loc='best', title='Activity Groups')

    # Add labels and title
    plt.title('2D Visualization of Activities by Distance')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True)
    plt.tight_layout()
    plt.show()




if __name__ == '__main__':
    ##############################################################################
    # intrinsic - activity_distance_functions we want to evaluate
    activity_distance_functions = list()
    activity_distance_functions.append("Bose 2009 Substitution Scores")
    activity_distance_functions.append("De Koninck 2018 act2vec CBOW")
    activity_distance_functions.append("Unit Distance")

    log_name = "BPIC2017"

    log = xes_importer.apply(ROOT_DIR + '/event_logs/' + log_name + '.xes')
    log_control_flow_perspective = get_log_control_flow_perspective(log)

    n_gram_size_bose_2009 = 3

    for activity_distance_function in activity_distance_functions:

        print(activity_distance_function)

        # 3: Compute distances between activities
        activity_distance_matrix = get_activity_distances(
            activity_distance_function, log_control_flow_perspective, n_gram_size_bose_2009
        )

        alphabet = get_alphabet(log_control_flow_perspective)

        alphabet_set = set(alphabet)

        # Gray activities
        gray_list = ["A_Create Application", "A_Submitted", "A_Concept", "W_Handle leads"]

        # Yellow activities
        yellow_list = [
            "W_Complete application",
            "A_Accepted",
            "O_Create Offer",
            "O_Created",
            "W_Call after offers",
            "A_Complete",
            "O_Sent (online only)",
            "O_Sent (mail and online)"

        ]

        # Blue activities
        blue_list = ["W_Validate application", "A_Validating", "O_Returned"]

        # Green activities
        green_list = ["O_Accepted", "A_Pending", "O_Cancelled", "A_Cancelled", "O_Refused", "A_Denied"]

        # Orange activities
        orange_list = ["W_Call incomplete files", "A_Incomplete"]

        all_activities = set(
            gray_list + yellow_list + blue_list + green_list + orange_list
        )

        print(len(all_activities))

        # Check if all activities are in alphabet
        missing_activities = all_activities - alphabet_set
        extra_activities = alphabet_set - all_activities

        ''' 
        if missing_activities:
            print(f"Missing activities in alphabet: {missing_activities}")
        else:
            print("All activities are in the alphabet.")

        if extra_activities:
            print(f"Extra activities in alphabet: {extra_activities}")
        else:
            print("No extra activities in the alphabet.")
        '''

        log_control_flow_perspective = remove_lists_with_extra_activities(log_control_flow_perspective,
                                                                          extra_activities)

        activity_distance_matrix = get_activity_distances(
            activity_distance_function, log_control_flow_perspective, n_gram_size_bose_2009
        )

        if "Bose 2009 Substitution Scores" == activity_distance_function:
            reverse = True  # high values = high similarity
        else:
            reverse = False  # high values = high distances

        if reverse is True:
            activity_distance_matrix = reverse_activity_distance_matrix(activity_distance_matrix)

        # Predefined color groups
        color_groups = [gray_list, yellow_list, blue_list, green_list, orange_list]
        n_clusters = len(color_groups)

        clustering_methods = ["kmeans"] # "agglomerative", "dbscan", "spectral"]

        # Replace the original `evaluate_overlap` function call with the remapped version
        for method in clustering_methods:
            print(f"Clustering method: {method}")
            activities, cluster_labels = cluster_activities(activity_distance_matrix, n_clusters, method=method)
            ari, nmi = evaluate_overlap_with_remapping(activities, cluster_labels, color_groups)
            print(f"Adjusted Rand Index (ARI): {ari}")
            #print(f"Normalized Mutual Information (NMI): {nmi}")
            # Visualize the activities in 2D
            visualize_activities_in_2d(activities, activity_distance_matrix, color_groups)




