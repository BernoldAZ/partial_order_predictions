from pm4py.objects.log.importer.xes import importer as xes_importer
import os, sys, copy, random, time, re

from definitions import ROOT_DIR

NA_DIR = os.path.join(ROOT_DIR, "evaluation", "evaluation_of_activity_distances", "next_activity_prediction")
RAW_DATASETS_DIR = os.path.join(NA_DIR, "raw_datasets")
SPLIT_DATASETS_DIR = os.path.join(NA_DIR, "split_datasets")
RESULTS_DIR = os.path.join(NA_DIR, "results")
MODELS_DIR = os.path.join(NA_DIR, "models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def load_file_xes(file_path):
    log = xes_importer.apply(file_path)
    lines, timeseqs, timeseqs2, timeseqs3, timeseqs4, caseids = [], [], [], [], [], []
    for trace in log:
        caseid = trace.attributes.get("concept:name", "case_" + str(len(caseids)))
        caseids.append(caseid)
        tokens, times, times2, times3, times4 = [], [], [], [], []
        casestart, lastevent = None, None
        for event in trace:
            activity = event["concept:name"]
            timestamp = event["time:timestamp"]
            tokens.append(activity)
            if casestart is None:
                casestart = timestamp
                lastevent = timestamp
            diff = (timestamp - lastevent).total_seconds()
            diff2 = (timestamp - casestart).total_seconds()
            midnight = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
            diff3 = (timestamp - midnight).total_seconds()
            diff4 = timestamp.weekday()
            times.append(diff)
            times2.append(diff2)
            times3.append(diff3)
            times4.append(diff4)
            lastevent = timestamp
        tokens.append('!')
        lines.append(tokens)
        timeseqs.append(times)
        timeseqs2.append(times2)
        timeseqs3.append(times3)
        timeseqs4.append(times4)
    return lines, timeseqs, timeseqs2, timeseqs3, timeseqs4, caseids

raw_logs = [f for f in os.listdir(RAW_DATASETS_DIR) if f.endswith(".xes.gz")]
results_summary = []

for raw_log in raw_logs:
    log_name = os.path.splitext(os.path.splitext(raw_log)[0])[0]  # remove ".xes.gz"
    full_path = os.path.join(RAW_DATASETS_DIR, raw_log)
    train_path = os.path.join(SPLIT_DATASETS_DIR, f"train_{log_name}.xes.gz")
    val_path = os.path.join(SPLIT_DATASETS_DIR, f"val_{log_name}.xes.gz")
    test_path = os.path.join(SPLIT_DATASETS_DIR, f"test_{log_name}.xes.gz")
    print("\n========== Processing log:", log_name, "==========")
    print("Full log:", full_path)
    print("Train:", train_path)
    print("Val:", val_path)
    print("Test:", test_path)

    lines_full_log, ts_full_log, ts2_full_log, ts3_full_log, ts4_full_log, _ = load_file_xes(full_path)
    lines_train, ts_train, ts2_train, ts3_train, ts4_train, _ = load_file_xes(train_path)
    lines_val, ts_val, ts2_val, ts3_val, ts4_val, _ = load_file_xes(val_path)
    lines_test, ts_test, ts2_test, ts3_test, ts4_test, _ = load_file_xes(test_path)

    # Create a set of all tokens in lines_train
    train_tokens = {token for line in lines_train for token in line}


    # Function to check for unknown tokens in a list of lines
    def check_unknown_tokens(lines, label):
        # Collect tokens that are in 'lines' but not in 'train_tokens'
        unknown_tokens = {token for line in lines for token in line if token not in train_tokens}
        if unknown_tokens:
            print(f"Unknown tokens in {label}:")
            for token in unknown_tokens:
                print(token)
        else:
            print(f"No unknown tokens found in {label}.")


    # Check for unknown tokens in validation and test datasets
    check_unknown_tokens(lines_val, "Validation")
    check_unknown_tokens(lines_test, "Test")
