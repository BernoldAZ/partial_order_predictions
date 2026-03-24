"""
DATA PROCESSOR:
Read a XES file, convert it to a CSV file and make the splits.
It can create holdout or cross-validation partitions.
"""

from distances.activity_distances.gamallo_fernandez_2023_context_based_representations.src.data_processor.args_reader import read_input_args
from distances.activity_distances.gamallo_fernandez_2023_context_based_representations.src.data_processor.logger import DataProcessorLogger
import distances.activity_distances.gamallo_fernandez_2023_context_based_representations.src.data_processor.utils as utils
import sys
import os
import pm4py

def write_csvs(xes_file_name, xes_log_path):

    sys.argv = [
        "script.py",  # Fake script name (needed for argparse)
        "--dataset", xes_log_path,  # Single dataset mode
        "--holdout",  # Enables cross-validation
        "--activity",  # Extract ActivityID column
        "--timestamp",  # Extract Timestamp column
        "--stats",  # Print dataset statistics
        "--print_console_file"  # Print output to both console and file
    ]

    args = read_input_args()

    logger = DataProcessorLogger(args.print_mode)


    # Select the specified columns
    output_columns = utils.select_output_columns(True, True,
                                                False)

    csv_dataset, config_csv_path, attr_dict  = utils.convert_xes_to_csv(xes_log_path, output_columns)


    utils.make_holdout(csv_dataset)

        # Calculate the statistics of the dataset
        #if args.stats:
        #    stats = utils.calculate_stats(csv_dataset, logger)
    return csv_dataset, config_csv_path, attr_dict

