import os
import re
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE   # Uncomment if you wish to use TSNE instead of PCA
from mpl_toolkits.mplot3d import Axes3D  # Registers the 3D projection
from matplotlib import rc

# Set LaTeX-style fonts (using Computer Modern, which works well)
rc('text', usetex=True)
rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})

try:
    from adjustText import adjust_text

    ADJUST_TEXT_AVAILABLE = True
except ImportError:
    print("adjustText module not found; labels might overlap. Install it via 'pip install adjustText'.")
    ADJUST_TEXT_AVAILABLE = False


###################################
# Grouping and Color Mapping
###################################
def get_group(activity):
    if activity.startswith("Release"):
        return "Release"
    elif activity in ["IV Antibiotics", "IV Liquid", "LacticAcid", "Leucocytes"]:
        return "IV"
    else:
        return "Other"


# Fixed colors for "Release" and "IV" groups.
GROUP_COLORS = {
    "Release": "tab:orange",
    "IV": "tab:blue"
}


# For "Other" activities, we assign unique colors using the viridis colormap.

###################################
# 2D Multi-Plot Function With Global Axes Limits
###################################
def visualize_embeddings_2d_multi(distance_functions, embedding_input, reduction_method="pca",
                                  title="2D Comparison of Embeddings", output_file=None):
    """
    Creates a figure with side-by-side 2D scatter plots for each distance function.
    All plots are forced to share the same x and y limits (computed from all embeddings)
    so that the unit distance embedding is "zoomed out" to the same level as the others.
    Activities are grouped so that:
      - All "Release" activities share a fixed color but use different markers.
      - All IV-related activities share a fixed color.
      - All "Other" activities are each assigned a unique color.
    A shared legend is placed below the subplots.

    Parameters:
      distance_functions (list of str): List of distance function names (to be passed to get_embeddings_for_method).
      embedding_input: Input log/control-flow perspective for computing embeddings.
      reduction_method (str): Dimensionality reduction method ("pca" or "tsne"). Default is "pca".
      title (str): Overall title for the figure.
      output_file (str): If provided, the figure is saved as a PDF.
    """
    # Get a consistent ordering of activity names.
    alphabet = sorted(set(token for trace in embedding_input for token in trace))
    num_activities = len(alphabet)

    # Compute a color mapping for "Other" activities.
    other_activities = [act for act in alphabet if get_group(act) == "Other"]
    num_other = len(other_activities)
    other_color_map = {}
    for j, act in enumerate(other_activities):
        other_color_map[act] = plt.cm.viridis(j / (num_other - 1)) if num_other > 1 else plt.cm.viridis(0.5)

    n = len(distance_functions)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]

    # First, compute all 2D embeddings and collect all x and y values.
    all_vectors_2d = []
    for func in distance_functions:
        embedding = get_embeddings_for_method(func, embedding_input)
        vectors = np.array([embedding[name] for name in alphabet])
        if reduction_method.lower() == "pca":
            reducer = PCA(n_components=2)
        elif reduction_method.lower() == "tsne":
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42)
        else:
            raise ValueError("Unknown reduction method. Use 'pca' or 'tsne'.")
        vectors_2d = reducer.fit_transform(vectors)
        all_vectors_2d.append(vectors_2d)
    # Compute global limits.
    all_x = np.hstack([v[:, 0] for v in all_vectors_2d])
    all_y = np.hstack([v[:, 1] for v in all_vectors_2d])
    global_xmin, global_xmax = all_x.min(), all_x.max()
    global_ymin, global_ymax = all_y.min(), all_y.max()

    # Define marker styles.
    marker_styles = ['o', 's', '^', 'D', 'v', 'p', '*', 'H', 'X', 'd']
    # Accumulate legend handles.
    legend_handles = []
    legend_labels = []

    # Plot each function's 2D embedding.
    for idx, (func, vectors_2d) in enumerate(zip(distance_functions, all_vectors_2d)):
        ax = axes[idx]
        group_marker_counter = {}  # Counter for markers within each group.
        for i, name in enumerate(alphabet):
            group = get_group(name)
            if group not in group_marker_counter:
                group_marker_counter[group] = 0
            marker = marker_styles[group_marker_counter[group] % len(marker_styles)]
            group_marker_counter[group] += 1

            if group in GROUP_COLORS:
                color = GROUP_COLORS[group]
            else:
                color = other_color_map[name]
            sc = ax.scatter(vectors_2d[i, 0], vectors_2d[i, 1], s=80, alpha=0.8,
                            color=color, marker=marker)
            if idx == 0:
                legend_handles.append(sc)
                legend_labels.append(name)
        ax.set_title(func, fontsize=14)
        ax.set_xlabel("Component 1", fontsize=12)
        ax.set_ylabel("Component 2", fontsize=12)
        ax.grid(True)
        # Set the same axis limits for all subplots.
        ax.set_xlim(global_xmin, global_xmax)
        ax.set_ylim(global_ymin, global_ymax)

    fig.suptitle(title, fontsize=16)
    # Place a shared legend below the subplots.
    fig.legend(legend_handles, legend_labels, loc='lower center', ncol=min(num_activities, 5),
               fontsize=10, bbox_to_anchor=(0.5, -0.05))
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    if output_file:
        plt.savefig(output_file, format="pdf", dpi=300)
        print(f"2D combined figure saved to {output_file}")
    plt.show()


###################################
# 3D Multi-Plot Function With Fixed Viewing Angle
###################################
def visualize_embeddings_3d_multi(distance_functions, embedding_input, reduction_method="pca",
                                  title="3D Comparison of Embeddings", output_file=None,
                                  elev=30, azim=45):
    """
    Creates a figure with side-by-side 3D scatter plots for each distance function.
    All 3D subplots use the same viewing angle (elevation and azimuth).
    Activities are grouped so that:
      - "Release" activities share a fixed color.
      - IV-related activities share a fixed color.
      - "Other" activities each receive a unique color.
    A shared legend is placed below the subplots.

    Parameters:
      distance_functions (list of str): List of distance function names.
      embedding_input: Input for computing embeddings.
      reduction_method (str): "pca" or "tsne" (default "pca").
      title (str): Overall title for the figure.
      output_file (str): If provided, the figure is saved as a PDF.
      elev (float): Elevation angle for the 3D view.
      azim (float): Azimuth angle for the 3D view.
    """
    alphabet = sorted(set(token for trace in embedding_input for token in trace))
    num_activities = len(alphabet)

    # Build color mapping for "Other" activities.
    other_activities = [act for act in alphabet if get_group(act) == "Other"]
    num_other = len(other_activities)
    other_color_map = {}
    for j, act in enumerate(other_activities):
        other_color_map[act] = plt.cm.viridis(j / (num_other - 1)) if num_other > 1 else plt.cm.viridis(0.5)

    n = len(distance_functions)
    fig = plt.figure(figsize=(6 * n, 6))
    axes = []
    for idx in range(n):
        ax = fig.add_subplot(1, n, idx + 1, projection='3d')
        # Set the same view angle for each subplot.
        ax.view_init(elev=elev, azim=azim)
        axes.append(ax)

    marker_styles = ['o', 's', '^', 'D', 'v', 'p', '*', 'H', 'X', 'd']
    legend_handles = []
    legend_labels = []

    # For each distance function, compute its embedding.
    for idx, func in enumerate(distance_functions):
        embedding = get_embeddings_for_method(func, embedding_input)
        vectors = np.array([embedding[name] for name in alphabet])
        if reduction_method.lower() == "pca":
            reducer = PCA(n_components=3)
        elif reduction_method.lower() == "tsne":
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=3, random_state=42)
        else:
            raise ValueError("Unknown reduction method. Use 'pca' or 'tsne'.")
        vectors_3d = reducer.fit_transform(vectors)

        ax = axes[idx]
        group_marker_counter = {}
        for i, name in enumerate(alphabet):
            group = get_group(name)
            if group not in group_marker_counter:
                group_marker_counter[group] = 0
            marker = marker_styles[group_marker_counter[group] % len(marker_styles)]
            group_marker_counter[group] += 1

            if group in GROUP_COLORS:
                color = GROUP_COLORS[group]
            else:
                color = other_color_map[name]
            sc = ax.scatter(vectors_3d[i, 0], vectors_3d[i, 1], vectors_3d[i, 2],
                            s=80, alpha=0.8, color=color, marker=marker)
            if idx == 0:
                legend_handles.append(sc)
                legend_labels.append(name)
        ax.set_title(func, fontsize=14)
        ax.set_xlabel("Component 1", fontsize=12)
        ax.set_ylabel("Component 2", fontsize=12)
        ax.set_zlabel("Component 3", fontsize=12)

    fig.suptitle(title, fontsize=16)
    fig.legend(legend_handles, legend_labels, loc='lower center', ncol=min(num_activities, 5),
               fontsize=10, bbox_to_anchor=(0.5, -0.05))
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    if output_file:
        plt.savefig(output_file, format="pdf", dpi=300)
        print(f"3D combined figure saved to {output_file}")
    plt.show()


###################################
# Example Usage
###################################
if __name__ == '__main__':
    from pm4py.objects.log.importer.xes import importer as xes_importer
    from definitions import ROOT_DIR
    import pandas as pd
    from distances.activity_distances.pmi.pmi import (
        get_activity_context_frequency_matrix_pmi,
        get_activity_activity_frequency_matrix_pmi
    )
    from distances.activity_distances.bose_2009_context_aware_trace_clustering.algorithm import \
        get_substitution_and_insertion_scores
    from distances.activity_distances.de_koninck_2018_act2vec.algorithm import get_act2vec_distance_matrix
    from distances.activity_distances.de_koninck_2018_act2vec.our_hyperparas import get_act2vec_distance_matrix_our
    from distances.activity_distances.activity_activity_co_occurence.activity_activity_co_occurrence import \
        get_activity_activity_co_occurence_matrix
    from distances.activity_distances.activity_context_frequency.activity_contex_frequency import \
        get_activity_context_frequency_matrix
    from distances.activity_distances.chiorrini_2022_embedding_process_structure.embedding_process_structure import \
        get_embedding_process_structure_distance_matrix
    from distances.activity_distances.gamallo_fernandez_2023_context_based_representations.src.embeddings_generator.main_new import \
        get_context_based_distance_matrix
    from evaluation.data_util.util_activity_distances_intrinsic import (
        get_log_control_flow_perspective, get_activities_to_replace, get_logs_with_replaced_activities_dict,
        get_knn_dict, get_precision_at_k, save_intrinsic_results, get_triplet, get_diameter, load_results, save_results,
        add_window_size_evaluation
    )


    def extract_window_size(s):
        match = re.search(r"w_(\d+)", s)
        return int(match.group(1)) if match else 3


    def get_embeddings_for_method(method, embedding_input):
        alphabet = sorted(set(token for trace in embedding_input for token in trace))
        win_size = extract_window_size(method)
        if method == "Unit Distance":
            emb = {activity: np.eye(len(alphabet))[i] for i, activity in enumerate(alphabet)}
            return emb
        elif method.startswith("Activity-Activitiy Co Occurrence"):
            bag = True if "Bag Of Words" in method else False
            _, emb, activity_freq_dict, activity_index = get_activity_activity_co_occurence_matrix(
                embedding_input, alphabet, win_size, bag_of_words=bag)
            if "PMI" in method:
                _, emb = get_activity_activity_frequency_matrix_pmi(emb, activity_freq_dict, activity_index, 0)
            return emb


    # Define the list of distance functions to compare.
    distance_functions = [
        "Unit Distance",
        "Activity-Activitiy Co Occurrence N-Gram PMI w_3",
    ]

    # Load the log (using the Sepsis log in this example).
    log = xes_importer.apply(os.path.join(ROOT_DIR, "event_logs", "Sepsis.xes.gz"))
    log_control_flow_perspective = get_log_control_flow_perspective(log)

    # Create a combined 2D plot for all distance functions.
    visualize_embeddings_2d_multi(distance_functions, log_control_flow_perspective, reduction_method="pca",
                                  title="2D Comparison of Distance Functions",
                                  output_file="../../../Downloads/additional_scripts-20250514T072927Z-1-001/additional_scripts/combined_2d.pdf")

    # Create a combined 3D plot for all distance functions with fixed viewing angle.
    visualize_embeddings_3d_multi(distance_functions, log_control_flow_perspective, reduction_method="pca",
                                  title="3D Comparison of Distance Functions",
                                  output_file="../../../Downloads/additional_scripts-20250514T072927Z-1-001/additional_scripts/combined_3d.pdf", elev=30, azim=45)
