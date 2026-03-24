import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from pm4py.objects.log.importer.xes import importer as xes_importer
from definitions import ROOT_DIR
from evaluation.data_util.util_activity_distances_intrinsic import get_log_control_flow_perspective
from matplotlib import rc
from itertools import cycle  # For cycling markers per category

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
    "Hospital Billing": "xkcd:beige",
    "RTFM": "tab:gray"
}

# Define marker styles
marker_styles = ["o", "s", "D", "^", "v", "P", "*", "X", "<", ">"]

# Function to get category
def get_category(log_name):
    for category in category_colors:
        if log_name.startswith(category):
            return category
    return "Other"

# Collect activity data
data = []
all_logs = sorted([f for f in os.listdir(EVENT_LOGS_DIR) if f.endswith(".xes.gz")])  # Ensure logs are sorted alphabetically

# Group logs by category
log_by_category = {}
for log_file in all_logs:
    log_name = os.path.splitext(os.path.splitext(log_file)[0])[0]
    category = get_category(log_name)
    if category not in log_by_category:
        log_by_category[category] = []
    log_by_category[category].append(log_name)

# Assign markers per category (reset marker cycle for each category)
log_marker_map = {}
for category, logs in log_by_category.items():
    logs.sort()  # Ensure logs within a category are sorted alphabetically
    marker_cycle = cycle(marker_styles)  # Reset marker cycle for each category
    for log in logs:
        log_marker_map[log] = next(marker_cycle)

# Load data
for log_file in all_logs:
    log_name = os.path.splitext(os.path.splitext(log_file)[0])[0]
    log_path = os.path.join(EVENT_LOGS_DIR, log_file)

    log = xes_importer.apply(log_path)
    log_control_flow_perspective = get_log_control_flow_perspective(log)
    unique_activities = set(activity for trace in log_control_flow_perspective for activity in trace)

    data.append({
        "Log": log_name,
        "Num Activities": len(unique_activities),
        "Category": get_category(log_name),
        "Marker": log_marker_map[log_name]  # Store marker
    })

# Convert to DataFrame
df = pd.DataFrame(data)

# Sort DataFrame to maintain alphabetical order for categories & logs
categories_sorted = sorted(df["Category"].unique(), reverse=True)  # Reverse order for correct display

df = df.sort_values(by=["Category", "Log"])
df["Category"] = pd.Categorical(df["Category"], categories=categories_sorted, ordered=True)

# Set font style
rc('text', usetex=True)
rc('font', **{'family': 'serif', 'serif': ['Times']})

fig, ax = plt.subplots(figsize=(12, 8))
sns.set_theme(style="whitegrid", rc={"font.family": "serif", "font.serif": ["Times"]})

# Plot points manually with transparent fill and thicker borders
# Create mapping from category to y-axis position
category_to_y = {category: idx for idx, category in enumerate(categories_sorted)}

# Plot points manually with transparent fill and thicker borders
for category in df["Category"].unique():
    subset = df[df["Category"] == category]
    for log_name in subset["Log"]:
        log_subset = subset[subset["Log"] == log_name]
        ax.scatter(
            log_subset["Num Activities"],
            category_to_y[category],  # Use mapped y-axis positions
            marker=log_marker_map[log_name],  # Different markers for each category
            edgecolors=category_colors.get(category, "tab:gray"),  # Colored border
            facecolors='none',  # Transparent inside
            linewidths=2.5,  # Thicker Border
            s=100,  # Marker size
            label=log_name  # For legend
        )

# Adjust spacing
plt.subplots_adjust(bottom=0.3)

# Create legend with unique markers
handles, labels = [], []
for category in df["Category"].unique():
    subset = df[df["Category"] == category]
    for log_name in subset["Log"]:
        color = category_colors.get(category, "tab:gray")
        marker = log_marker_map[log_name]
        handles.append(plt.Line2D([0], [0], marker=marker, color=color, linestyle='', markersize=10, markerfacecolor='none', markeredgewidth=2.5))
        labels.append(log_name)

legend = plt.legend(
    handles, labels, title="Event Logs", loc="upper center", bbox_to_anchor=(0.5, -0.10),
    ncol=min(len(all_logs), 4), fontsize=11, handlelength=4
)

plt.grid(True, which="both", linestyle="--", linewidth=0.5)

# Set labels and title
ax.set_xlabel("Number of Activities", fontsize=14)
ax.set_ylabel("Event Log Dataset", fontsize=14)  # Ensure category is labeled
# Force y-axis (Event Log Dataset) to follow correct order (BPIC12 at top)
#categories_sorted = sorted(df["Category"].unique(), reverse=True)  # Reverse order for correct display
ax.set_yticks(range(len(categories_sorted)))
ax.set_yticklabels(categories_sorted)

# Save figure
plt.savefig("number_of_activities.pdf", format="pdf", transparent=True, bbox_inches="tight", bbox_extra_artists=[legend])
plt.show()
