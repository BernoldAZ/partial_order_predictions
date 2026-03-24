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

# Set LaTeX-style fonts
rc('text', usetex=True)
rc('font', **{'family': 'serif', 'serif': ['Times']})

# Define event log directory
EVENT_LOGS_DIR = os.path.join(ROOT_DIR, "event_logs")


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

    return activity_positions


def process_event_logs(log_dir):
    """
    Processes all event logs in the given directory and computes relative positions.

    :param log_dir: Directory containing event logs.
    :return: Dictionary with log names as keys and their respective activity position statistics.
    """
    all_logs = sorted([f for f in os.listdir(log_dir) if f.endswith(".xes.gz")])
    log_statistics = {}

    for log_file in all_logs:
        log_name = os.path.splitext(os.path.splitext(log_file)[0])[0]
        log_path = os.path.join(log_dir, log_file)
        log = xes_importer.apply(log_path)
        log_control_flow_perspective = get_log_control_flow_perspective(log)
        log_statistics[log_name] = compute_relative_positions(log_control_flow_perspective)

    return log_statistics


log_statistics = process_event_logs(EVENT_LOGS_DIR)

data = []
for log_name, stats in log_statistics.items():
    for activity, activity_positions in stats.items():
        for pos in activity_positions:
            data.append([log_name, activity, pos])

df = pd.DataFrame(data, columns=["Log", "Activity", "Relative Position"])

num_logs = len(df["Log"].unique())
ncols = 4  # Two plots per row
nrows = 1  # Calculate number of rows needed

fig, axes = plt.subplots(nrows, ncols, figsize=(20, max(5, 5 * nrows)), constrained_layout=True)
axes = axes.flatten()  # Flatten axes array for easy iteration

for i, (ax, (log_name, log_data)) in enumerate(zip(axes, df.groupby("Log"))):
    top_activities = log_data["Activity"].unique()[:5]  # Ensure only top 5 activities are plotted
    log_data = log_data[log_data["Activity"].isin(top_activities)]

    sns.boxplot(x="Activity", y="Relative Position", data=log_data, ax=ax, whis=[0, 100], palette="Set2", width=.6)

    row, col = divmod(i, ncols)  # Get row and column indices

    ax.set_title(log_name, fontsize=20)

    # Y-axis label only for the first column
    if col == 0:
        ax.set_ylabel("Relative Activity Positions", fontsize=20)
    else:
        ax.set_ylabel("")

    # X-axis label only for the last row
    if row == nrows - 1:
        ax.set_xlabel("Five most frequent activities", fontsize=20)
    else:
        ax.set_xlabel("Five most frequent activities", fontsize=20)

    # Hide x-axis activity labels for all plots
    ax.set_xticklabels([])

    # **Ensure y-ticks are from 0 to 1.0 with font size 14**
    ax.set_yticks(np.linspace(0, 1.0, num=6))  # Set ticks at intervals between 0 and 1
    ax.tick_params(axis="y", labelsize=20)  # Adjust y-tick label font size

# Hide unused axes if num_logs is not a multiple of ncols
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

fig.subplots_adjust(hspace=0.3)  # Reduce vertical spacing between subplots
plt.savefig("activity_positions_selected_logs_boxplots.pdf", format="pdf", transparent=True)
plt.show()
