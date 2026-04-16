import argparse
from dataclasses import dataclass
from distances.activity_distances.gamallo_fernandez_2023_context_based_representations.src.data_processor.utils import PrintMode


@dataclass
class DataProcessorInputArgs:
    xes_path: str
    batch: bool
    crossvalidation: bool
    activity: bool
    timestamp: bool
    resource: bool
    stats: bool
    print_mode: PrintMode


def read_input_args() -> DataProcessorInputArgs:
    """
    Read the user input arguments
    :return: DataProcessorInputArgs object with the input arguments
    """
    parser = argparse.ArgumentParser(description="Prepare datasets from .xes.gz to .csv and "
                                                 "partition them")
    processing_mode = parser.add_mutually_exclusive_group(required=True)
    processing_mode.add_argument("--dataset", help="Raw dataset to prepare")
    processing_mode.add_argument("--batch", help="Batch process the selected folder")
    partition_mode = parser.add_mutually_exclusive_group(required=True)
    partition_mode.add_argument("--holdout", help="Split in train/validation/test",
                                action='store_true')
    partition_mode.add_argument("--crossvalidation", help="5-fold cross validation",
                                action='store_true')
    parser.add_argument("--activity", help="Extract ActivityID column", action='store_true')
    parser.add_argument("--timestamp", help="Extract Timestamp column", action='store_true')
    parser.add_argument("--resource", help="Extract Resource column", action='store_true')
    parser.add_argument("--stats", help="Print statistics about dataset and its cases", action='store_true')
    print_mode = parser.add_mutually_exclusive_group(required=False)
    print_mode.add_argument("--print_console", help="Print output to the console", action='store_true')
    print_mode.add_argument("--print_file", help="Print output to a file", action='store_true')
    print_mode.add_argument("--print_console_file", help="Print output to the console and file", action='store_true')
    args = parser.parse_args()

    # Check if batch mode or only one dataset
    if args.dataset:
        batch = False
        xes_path = args.dataset
    else:
        batch = True
        xes_path = args.batch

    # Split mode
    if args.holdout:
        crossvalidation = False
    else:
        crossvalidation = True

    # Store the activities from the XES file
    activity = args.activity
    # Store the timestamps from the XES file
    timestamp = args.timestamp
    # Store the resources from the XES file
    resource = args.resource

    # Print dataset statistics
    stats = args.stats

    # Check the print mode
    if args.print_console:
        print_mode = PrintMode.CONSOLE
    elif args.print_file:
        print_mode = PrintMode.TO_FILE
    elif args.print_console_file:
        print_mode = PrintMode.CONSOLE_AND_FILE
    else:
        print_mode = PrintMode.NONE

    return DataProcessorInputArgs(xes_path, batch, crossvalidation, activity, timestamp,
                                  resource, stats, print_mode)

