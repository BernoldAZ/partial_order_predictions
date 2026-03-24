import numpy as np
import os
import collections
from pm4py.objects.log.importer.xes import importer as xes_importer
from evaluation.data_util.util_activity_distances_intrinsic import get_log_control_flow_perspective
from definitions import ROOT_DIR

# Define event log directory
EVENT_LOGS_DIR = os.path.join(ROOT_DIR, "event_logs")

def compute_relative_positions(log_control_flow_perspective):
    """
    Computes the average relative position, standard deviation, and probability of the 5% most frequent activities.

    :param log_control_flow_perspective: List of traces, where each trace is a list of activities.
    :return: Dictionary with activities as keys and (mean relative position, std deviation, probability) as values.
    """
    # Flatten the event log and count activity occurrences
    all_activities = [event for trace in log_control_flow_perspective for event in trace]
    activity_counts = collections.Counter(all_activities)

    # Compute total activity occurrences for probability calculation
    total_activities = sum(activity_counts.values())

    # Determine the top 5% most frequent activities
    num_top_activities = max(5, int(len(activity_counts) * 0.03))  # Ensure at least 1 activity
    top_activities = [act for act, _ in activity_counts.most_common(num_top_activities)]

    # Store relative positions of top activities
    activity_positions = {activity: [] for activity in top_activities}
    activity_probabilities = {activity: activity_counts[activity] / total_activities for activity in top_activities}

    for trace in log_control_flow_perspective:
        trace_length = len(trace)
        for idx, activity in enumerate(trace):
            if activity in top_activities:
                relative_pos = idx / trace_length  # Compute relative position
                activity_positions[activity].append(relative_pos)

    # Compute mean and standard deviation
    result = {
        act: (np.mean(pos_list), np.std(pos_list), activity_probabilities[act])
        for act, pos_list in activity_positions.items()
    }
    return result

def process_event_logs(log_dir):
    """
    Processes all event logs in the given directory and computes relative positions and probabilities.

    :param log_dir: Directory containing event logs.
    :return: Dictionary with log names as keys and their respective activity position statistics.
    """
    all_logs = sorted([f for f in os.listdir(log_dir) if f.endswith(".xes.gz")])
    log_statistics = {}

    for log_file in all_logs:
        log_name = os.path.splitext(os.path.splitext(log_file)[0])[0]
        log_path = os.path.join(log_dir, log_file)

        # Load event log
        log = xes_importer.apply(log_path)
        log_control_flow_perspective = get_log_control_flow_perspective(log)

        # Compute activity positions
        log_statistics[log_name] = compute_relative_positions(log_control_flow_perspective)

    return log_statistics

# Run for all logs
log_results = process_event_logs(EVENT_LOGS_DIR)

# Print results
for log_name, stats in log_results.items():
    print(f"\nLog: {log_name}")
    for activity, (mean_pos, std_dev, probability) in stats.items():
        print(f"  Activity: {activity}, Mean Position: {mean_pos:.4f}, Std Dev: {std_dev:.4f}, Probability: {probability:.4f}")


