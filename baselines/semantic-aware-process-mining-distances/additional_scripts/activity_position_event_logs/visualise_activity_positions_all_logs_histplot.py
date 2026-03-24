import numpy as np
import os
import collections
from pm4py.objects.log.importer.xes import importer as xes_importer
from evaluation.data_util.util_activity_distances_intrinsic import get_log_control_flow_perspective
from definitions import ROOT_DIR
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rc
import pickle

# Set LaTeX-style fonts
rc('text', usetex=True)
rc('font', **{'family': 'serif', 'serif': ['Times']})

# Define event log directory
EVENT_LOGS_DIR = os.path.join(ROOT_DIR, "event_logs")
PERSPECTIVE_DIR = os.path.join(ROOT_DIR, "event_logs_control_flow_perspective")


def compute_relative_positions(log_control_flow_perspective):
    """
    Computes the relative positions of the top 5 most frequent activities.

    :param log_control_flow_perspective: List of traces, where each trace is a list of activities.
    :return: Dictionary with activities as keys and lists of relative positions.
    """
    all_activities = [event for trace in log_control_flow_perspective for event in trace]
    activity_counts = collections.Counter(all_activities)

    top_activities = [act for act, _ in activity_counts.most_common(5)]  # Select top 5 activities
    activity_positions = {activity: [] for activity in top_activities}

    for trace in log_control_flow_perspective:
        trace_length = len(trace)
        for idx, activity in enumerate(trace):
            if activity in top_activities:
                relative_pos = idx / trace_length
                activity_positions[activity].append(relative_pos)

    return activity_positions, activity_counts


def load_log_perspective(log_name):
    """Loads log_control_flow_perspective from a file."""
    file_path = os.path.join(PERSPECTIVE_DIR, log_name)
    with open(file_path, "rb") as f:
        return pickle.load(f)


def process_event_logs(log_dir):
    """
    Processes all event logs in the given directory and computes relative positions.

    :param log_dir: Directory containing event logs.
    :return: Dictionary with log names as keys and their respective activity position statistics.
    """
    all_logs = sorted([f for f in os.listdir(PERSPECTIVE_DIR)])
    log_statistics = {}
    activity_occurrences = {}

    for log_file in all_logs:
        log_name = os.path.splitext(log_file)[0]
        log_control_flow_perspective = load_log_perspective(log_file)
        log_statistics[log_name], activity_occurrences[log_name] = compute_relative_positions(
            log_control_flow_perspective)
    return log_statistics, activity_occurrences


log_statistics, activity_occurrences = process_event_logs(PERSPECTIVE_DIR)

data = []
for log_name, stats in log_statistics.items():
    for activity, activity_positions in stats.items():
        for pos in activity_positions:
            bucket = int(pos * 20)  # Assign bucket number from 0-19
            data.append([log_name, activity, pos, bucket])

df = pd.DataFrame(data, columns=["Log", "Activity", "Relative Position", "Bucket"])

df["Normalized Count"] = df.groupby(["Log", "Activity", "Bucket"])["Relative Position"].transform(
    lambda x: len(x) / len(df[df["Activity"] == x.name[1]]))

num_logs = len(df["Log"].unique())
ncols = 7  # Two plots per row
nrows = (num_logs + ncols - 1) // ncols  # Calculate number of rows needed

fig, axes = plt.subplots(nrows, ncols, figsize=(34, 5 * nrows))  # Adjust figure size
axes = axes.flatten()  # Flatten axes array for easy iteration

for i, (ax, (log_name, log_data)) in enumerate(zip(axes, df.groupby("Log"))):
    top_activities = log_data["Activity"].unique()[:5]  # Ensure only top 5 activities are plotted
    log_data = log_data[log_data["Activity"].isin(top_activities)]

    for activity in top_activities:
        activity_data = log_data[log_data["Activity"] == activity]
        sns.histplot(activity_data, x="Bucket", weights=activity_data["Normalized Count"], bins=20, ax=ax,
                     label=activity, alpha=0.6)

    ax.set_title(log_name, fontsize=20)
    #ax.legend(title="Activity", fontsize=14)

    row, col = divmod(i, ncols)  # Get row and column indices

    if col == 0:
        ax.set_ylabel("Normalized Frequency", fontsize=20)
    else:
        ax.set_ylabel("")

    if row == nrows - 1:
        ax.set_xlabel("Relative Activity Positions (Buckets)", fontsize=20)
    else:
        ax.set_xlabel("")

    ax.tick_params(axis="both", labelsize=14)  # Adjust tick label font size

# Hide unused axes if num_logs is not a multiple of ncols
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig("rel_positions.pdf", format="pdf", transparent=True)
plt.show()
