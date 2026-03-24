import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import collections
from pm4py.objects.log.importer.xes import importer as xes_importer
from evaluation.data_util.util_activity_distances_intrinsic import get_log_control_flow_perspective
from definitions import ROOT_DIR
from matplotlib import rc
import matplotlib.ticker as ticker

# Set LaTeX-style fonts
rc('text', usetex=True)
rc('font', **{'family': 'serif', 'serif': ['Times']})

# Define event log directory
EVENT_LOGS_DIR = os.path.join(ROOT_DIR, "event_logs")

# Define colors for each log category
category_colors = {
    "BPIC12": "tab:blue",
    "BPIC13": "tab:purple",
    "BPIC15": "tab:green",
    "BPIC17": "tab:red",
    "BPIC18": "tab:orange",
    "BPIC19": "tab:brown",
    "BPIC20": "tab:pink",
    "Helpdesk": "tab:cyan",
    "Sepsis": "tab:olive",
    "CCC19": "xkcd:light green",
    "Env Permit": "xkcd:yellow",
    "Hospital Billing": "xkcd:beige"
}

# Define different line styles and markers for uniqueness
line_styles = ["-", "--", ":", "-."]
marker_styles = [".", "s"]  # Extra markers for variety


def get_category(log_name):
    """Extracts the category from a log name."""
    for category in category_colors:
        if log_name.startswith(category):
            return category
    return "Other"


def print_high_probability_activities(log_name, activity_counts, total_activities, log_control_flow_perspective):
    """Prints activities that exceed the probability threshold and their occurrence in traces."""
    threshold = 0.01 if "BPIC15" in log_name else 0.09  # Lower threshold for BPIC15
    print(f"\nActivities for {log_name} (Probability > {threshold}):")

    filtered_activities = {activity: count / total_activities for activity, count in activity_counts.items() if
                           count / total_activities > threshold}

    sorted_activities = sorted(filtered_activities.items(), key=lambda x: x[1], reverse=True)

    num_traces = len(log_control_flow_perspective)

    for activity, prob in sorted_activities:
        traces_with_activity = sum(1 for trace in log_control_flow_perspective if activity in trace)
        trace_percentage = (traces_with_activity / num_traces) * 100
        print(f"  {activity}: Probability = {prob:.4f}, Appears in {trace_percentage:.2f}% of traces")


def plot_zipf_for_event_logs(log_dir):
    sns.set_theme(style="whitegrid", rc={"font.family": "serif", "font.serif": ["Times"]})
    plt.figure(figsize=(12, 8))

    all_logs = sorted([f for f in os.listdir(log_dir) if f.endswith(".xes.gz")])
    style_index = {}  # Dictionary to track linestyles & markers per category
    max_rank = 1  # To determine the longest log's rank

    # Process each log file
    for log_file in all_logs:
        log_name = os.path.splitext(os.path.splitext(log_file)[0])[0]
        log_path = os.path.join(log_dir, log_file)

        log = xes_importer.apply(log_path)
        log_control_flow_perspective = get_log_control_flow_perspective(log)

        all_activities = [event for trace in log_control_flow_perspective for event in trace]
        activity_counts = collections.Counter(all_activities)
        total_activities = sum(activity_counts.values())
        sorted_counts = sorted(activity_counts.values(), reverse=True)

        ranks = np.arange(1, len(sorted_counts) + 1)
        probabilities = np.array(sorted_counts) / total_activities

        max_rank = max(max_rank, len(sorted_counts))

        # Get category and color
        category = get_category(log_name)
        color = category_colors.get(category, "tab:gray")

        # Initialize category index tracking
        if category not in style_index:
            style_index[category] = {"linestyle": 0, "marker": 0}

        # Assign linestyle and marker
        linestyle = line_styles[style_index[category]["linestyle"] % len(line_styles)]
        marker = marker_styles[style_index[category]["marker"] % len(marker_styles)]

        # Update indices: switch marker after cycling through all linestyles
        style_index[category]["linestyle"] += 1
        if style_index[category]["linestyle"] % len(line_styles) == 0:
            style_index[category]["marker"] += 1  # Move to the next marker

        # Plot with linestyle, marker, and ensure markers appear in the legend
        plt.loglog(
            ranks, probabilities, marker=marker, linestyle=linestyle, markersize=4,
            color=color, label=log_name, linewidth=1, markerfacecolor=color
        )

    # Add ideal Zipf’s Law reference line
    ideal_ranks = np.linspace(1, max_rank, max_rank)
    ideal_probabilities = ideal_ranks ** -1
    plt.loglog(ideal_ranks, ideal_probabilities, "k--", label="Ideal Zipf Law", linewidth=2)

    # Labels and legend
    plt.xlabel("Rank of Activities", fontsize=12)
    plt.ylabel("Relative Frequency of Activities", fontsize=12)

    # Format axes with decimal scaling
    plt.xscale("log")
    plt.yscale("log")
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))

    # Adjust legend for better visibility
    legend = plt.legend(title="Event Logs", loc="upper center", bbox_to_anchor=(0.5, -0.15),
                        ncol=min(len(all_logs), 4), fontsize=11, handlelength=4)

    for handle in legend.legend_handles:
        handle.set_markersize(4)  # Ensure legend markers are visible
        handle.set_markerfacecolor(handle.get_color())  # Set marker face color

    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig("zipf.pdf", format="pdf", transparent=True)
    plt.show()


# Example usage
plot_zipf_for_event_logs(EVENT_LOGS_DIR)
