import os
import pickle
from pm4py.objects.log.importer.xes import importer as xes_importer
from definitions import ROOT_DIR
from evaluation.data_util.util_activity_distances_intrinsic import get_log_control_flow_perspective

# Define event log directory
EVENT_LOGS_DIR = os.path.join(ROOT_DIR, "event_logs")
PERSPECTIVE_DIR = os.path.join(ROOT_DIR, "event_logs_control_flow_perspective")

# Ensure the directory exists
os.makedirs(PERSPECTIVE_DIR, exist_ok=True)


def save_log_perspective(log_name, log_control_flow_perspective):
    """Saves log_control_flow_perspective to a file for fast reading."""
    file_path = os.path.join(PERSPECTIVE_DIR, f"{log_name}.pkl")
    with open(file_path, "wb") as f:
        pickle.dump(log_control_flow_perspective, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_log_perspective(log_name):
    """Loads log_control_flow_perspective from a file."""
    file_path = os.path.join(PERSPECTIVE_DIR, log_name)
    with open(file_path, "rb") as f:
        return pickle.load(f)


print("All logs processed and saved successfully.")

""" 
all_logs = sorted([f for f in os.listdir(EVENT_LOGS_DIR) if f.endswith(".xes.gz")])
for log_file in all_logs:
    log_name = os.path.splitext(os.path.splitext(log_file)[0])[0]
    log_path = os.path.join(EVENT_LOGS_DIR, log_file)

    # Import event log
    log = xes_importer.apply(log_path)

    # Extract control flow perspective
    log_control_flow_perspective = get_log_control_flow_perspective(log)

    # Save to file
    save_log_perspective(log_name, log_control_flow_perspective)

all_logs = sorted([f for f in os.listdir(EVENT_LOGS_DIR) if f.endswith(".xes.gz")])
"""
all_logs = sorted([f for f in os.listdir(PERSPECTIVE_DIR)])
for log_file in all_logs:
    log_control_flow_perspective = load_log_perspective(log_file)
    print(log_control_flow_perspective)
