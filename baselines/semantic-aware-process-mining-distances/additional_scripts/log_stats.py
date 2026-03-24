#!/usr/bin/env python3
import os
import csv
from pm4py.objects.log.importer.xes import importer as xes_importer
from definitions import ROOT_DIR
from evaluation.data_util.util_activity_distances_intrinsic import get_log_control_flow_perspective
from evaluation.data_util.util_activity_distances import get_alphabet

# Directory with the logs
LOGS_DIR = "/vol/home-vol3/dbisteam/kirchmhe/PycharmProjects/semantic-aware-process-mining-distances/evaluation/evaluation_of_activity_distances/next_activity_prediction/raw_datasets"

# Find all XES logs (gzipped)
raw_logs = [f for f in os.listdir(LOGS_DIR) if f.endswith(".xes.gz")]

# Path for the output CSV file
output_csv = os.path.join(ROOT_DIR, "log_stats.csv")

with open(output_csv, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    # Write header row
    writer.writerow([
        "log_name",
        "unique_activities",
        "num_traces",
        "num_trace_variants",
        "ratio_trace_variants",
        "min_trace_length",
        "avg_trace_length",
        "max_trace_length"
    ])

    # Process each log file
    for log_file in raw_logs:
        # Derive the log name (remove .xes.gz suffix)
        log_name = os.path.splitext(os.path.splitext(log_file)[0])[0]
        full_path = os.path.join(LOGS_DIR, log_file)
        print(f"\n========== Processing log: {log_name} ==========")

        # Import the log using pm4py
        log_full = xes_importer.apply(full_path)

        # Get the control-flow perspective (list of traces)
        log_control_flow_perspective = get_log_control_flow_perspective(log_full)

        # Get the alphabet (unique activities) from the log
        alphabet = get_alphabet(log_control_flow_perspective)

        # Number of traces in the log
        num_traces = len(log_control_flow_perspective)

        # Calculate unique trace variants by converting each trace to a tuple
        trace_variants = set(tuple(trace) for trace in log_control_flow_perspective)
        num_trace_variants = len(trace_variants)

        # Compute the ratio of unique trace variants to the total number of traces
        ratio_trace_variants = num_trace_variants / num_traces if num_traces > 0 else 0

        # Get lengths of all traces
        trace_lengths = [len(trace) for trace in log_control_flow_perspective]
        min_length = min(trace_lengths)
        max_length = max(trace_lengths)
        avg_length = sum(trace_lengths) / num_traces

        # Write the computed stats for this log to the CSV file
        writer.writerow([
            log_name,
            len(alphabet),
            num_traces,
            num_trace_variants,
            ratio_trace_variants,
            min_length,
            avg_length,
            max_length
        ])

print(f"Log statistics saved to {output_csv}")
