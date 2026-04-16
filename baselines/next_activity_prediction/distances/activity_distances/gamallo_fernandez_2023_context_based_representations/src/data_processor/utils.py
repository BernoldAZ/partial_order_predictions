import os
import pm4py
import pandas as pd
from enum import Enum
from pathlib import Path
from sklearn.model_selection import KFold

class PrintMode(Enum):
    NONE = 0
    CONSOLE = 1
    TO_FILE = 2
    CONSOLE_AND_FILE = 3


class Config:
    current_path = Path(__file__).parent
    LOG_PATH = os.path.abspath(os.path.join(current_path, '../../logs'))
    CSV_PATH = os.path.abspath(os.path.join(current_path, '../../data'))
    STATS_FILE = 'DatasetStats.csv'
    CV_FOLDS = 5
    TRAIN_SIZE = 80
    VAL_SIZE_FROM_TRAIN = 20


class XesFields:
    """
    Supported xes fields that may be present in a xes log.
    """
    CASE_COLUMN = "case:concept:name"
    ACTIVITY_COLUMN = "concept:name"
    TIMESTAMP_COLUMN = "time:timestamp"
    LIFECYCLE_COLUMN = "lifecycle:transition"
    RESOURCE_COLUMN = "org:resource"


class DataFrameFields:
    CASE_COLUMN = "CaseID"
    ACTIVITY_COLUMN = "Activity"
    TIMESTAMP_COLUMN = "Timestamp"
    RESOURCE_COLUMN = "Resource"


def get_datasets_list(path: str, batch_mode: bool, logger) -> list:
    """
    Get list of paths to datasets to be processed
    :param path: Path to the dataset or folder
    :param batch_mode: If batch mode is used or only one dataset
    :param logger: DataProcessorLogger to print the stats
    :return: A list of the path to datasets
    """

    dataset_list = []
    if batch_mode:
        files = os.listdir(path)
        for dataset in files:
            dataset_list.append(os.path.join(path, dataset))
    else:
        dataset_list.append(path)

    logger.log_console(f'Dataset list: {dataset_list}')

    return dataset_list


def select_output_columns(activity: bool, timestamp: bool,
                          resource: bool) -> dict:
    """
    Select and keep from the dataset only the specified columns
    :param activity: Boolean indicating if keep the activity column
    :param timestamp: Boolean indicating if keep the timestamp column
    :param resource: Boolean indicating if keep the resource column
    :return: Dictionary with the specified columns
    """

    output_columns = {XesFields.CASE_COLUMN: DataFrameFields.CASE_COLUMN}
    if activity:
        output_columns[XesFields.ACTIVITY_COLUMN] = DataFrameFields.ACTIVITY_COLUMN
    if timestamp:
        output_columns[XesFields.TIMESTAMP_COLUMN] = DataFrameFields.TIMESTAMP_COLUMN
    #if resource:
    #    output_columns[XesFields.RESOURCE_COLUMN] = DataFrameFields.RESOURCE_COLUMN

    return output_columns


def categorize_attribute(attr: pd.Series) -> list:
    """
    Convert the dataset column (pd.Series) type to categorical
    :param attr: Pandas Series of the column in the dataset
    :return: List with the values in the new categorical type
    """
    unique_attr = attr.unique()
    attr_dict = {act: idx for idx, act in enumerate(unique_attr)}
    attr_cat = list(map(lambda x: attr_dict[x], attr.values))

    return attr_cat

def return_id_to_attribute_mapping(attr: pd.Series):
    unique_attr = attr.unique()
    attr_dict = {act: idx for idx, act in enumerate(unique_attr)}
    return attr_dict


def convert_xes_to_csv(xes_path: str, output_columns: dict) -> str:
    """
    Convert the XES file with the dataset to a CSV format file
    :param xes_path: Full path to the XES file
    :param output_columns: Dictionary with the selected columns to keep
    :return: Full path to the CSV file
    """
    print(f"Looking for XES file at: {xes_path}")

    df_log = pm4py.read_xes(xes_path, return_legacy_log_object = False)
    #df_log = pm4py.convert_to_dataframe(log)

    # Get real activities
    if XesFields.LIFECYCLE_COLUMN in df_log:
        unique_lifecycle = df_log[XesFields.LIFECYCLE_COLUMN].unique()
        if len(unique_lifecycle) > 1:
            df_log[XesFields.ACTIVITY_COLUMN] = df_log[XesFields.ACTIVITY_COLUMN].astype(str) + "+"\
                                                + df_log[XesFields.LIFECYCLE_COLUMN]

    # Correct timestamp format
    if XesFields.TIMESTAMP_COLUMN in df_log:
        df_log[XesFields.TIMESTAMP_COLUMN] = pd.to_datetime(
            df_log[XesFields.TIMESTAMP_COLUMN], utc=True)

    # Select relevant columns
    if output_columns:
        df_log = df_log[list(output_columns.keys())]

        if XesFields.CASE_COLUMN in output_columns:
            df_log[XesFields.CASE_COLUMN] = categorize_attribute(
                df_log[XesFields.CASE_COLUMN])
            df_log[XesFields.CASE_COLUMN].astype(str)

        if XesFields.ACTIVITY_COLUMN in output_columns:
            attr_dict = return_id_to_attribute_mapping(df_log[XesFields.ACTIVITY_COLUMN])
            df_log[XesFields.ACTIVITY_COLUMN] = categorize_attribute(
                df_log[XesFields.ACTIVITY_COLUMN])
            df_log[XesFields.ACTIVITY_COLUMN].astype(str)

        if XesFields.RESOURCE_COLUMN in output_columns:
            df_log[XesFields.RESOURCE_COLUMN] = categorize_attribute(
                df_log[XesFields.RESOURCE_COLUMN])
            df_log[XesFields.RESOURCE_COLUMN].astype(str)

        df_log.rename(columns=output_columns, inplace="True")

    # Write in CSV
    csv_file = Path(xes_path).stem.split(".")[0] + ".csv"
    if Config.CSV_PATH:
        csv_path = Path(Config.CSV_PATH)
    else:
        csv_path = Path(xes_path).parent.parent
    csv_fullpath = os.path.join(csv_path, csv_file)

    # Ensure the directory exists
    os.makedirs(csv_path, exist_ok=True)  # Creates the directory if it doesn't exist

    df_log.to_csv(csv_fullpath, index=False)

    return csv_fullpath, Config.CSV_PATH, attr_dict


def make_crossvalidation(full_csv_path: str):
    """
    Create the k-fold cross-validation and store the folds
    :param full_csv_path: Full path to the CSV file with the dataset
    """
    full_df = pd.read_csv(full_csv_path)

    unique_case_ids = list(full_df[DataFrameFields.CASE_COLUMN].unique())
    kfold = KFold(n_splits=Config.CV_FOLDS, random_state=42, shuffle=True)
    indexes = sorted(unique_case_ids)
    splits = kfold.split(indexes)

    filename = Path(full_csv_path).stem + ".csv"
    if Config.CSV_PATH:
        folder = Config.CSV_PATH
    else:
        folder = str(Path(full_csv_path).parent)
    cv_path = folder + "/crossvalidation/"
    if not os.path.exists(cv_path):
        os.makedirs(cv_path)

    fold = 0
    for train_index, test_index in splits:
        val_cut = round(len(train_index) * ((100 - Config.VAL_SIZE_FROM_TRAIN) / 100))

        val_index = train_index[val_cut:]
        train_index = train_index[:val_cut]

        train_groups = [full_df[full_df[DataFrameFields.CASE_COLUMN] == train_g] for train_g in train_index]
        val_groups = [full_df[full_df[DataFrameFields.CASE_COLUMN] == val_g] for val_g in val_index]
        test_groups = [full_df[full_df[DataFrameFields.CASE_COLUMN] == test_g] for test_g in test_index]

        train_df = pd.concat(train_groups)
        val_df = pd.concat(val_groups)
        test_df = pd.concat(test_groups)

        train_df.to_csv(cv_path + "fold" + str(fold) + "_train_" + filename)
        val_df.to_csv(cv_path + "fold" + str(fold) + "_val_" + filename)
        test_df.to_csv(cv_path + "fold" + str(fold) + "_test_" + filename)

        fold += 1


def make_holdout(full_csv_path: str):
    full_df = pd.read_csv(full_csv_path)

    df_groupby = full_df.groupby(DataFrameFields.CASE_COLUMN, sort=False)
    groups = [group for _, group in df_groupby]

    # Split groups into 80% training and 20% validation
    first_cut = round(len(groups) * 0.8)
    train_groups = groups[:first_cut]
    val_groups = groups[first_cut:]

    train_df = pd.concat(train_groups)
    val_df = pd.concat(val_groups)

    filename = Path(full_csv_path).stem + ".csv"
    if Config.CSV_PATH:
        folder = Config.CSV_PATH
    else:
        folder = str(Path(full_csv_path).parent)
    holdout_path = os.path.join(folder, "holdout")
    if not os.path.exists(holdout_path):
        os.makedirs(holdout_path)

    train_df.to_csv(os.path.join(holdout_path, "train_" + filename), index=False)
    val_df.to_csv(os.path.join(holdout_path, "val_" + filename), index=False)

def make_holdout_old(full_csv_path: str):
    """
    Create the train-val-test splits and store them
    :param full_csv_path: Full path to the CSV file with the dataset
    """
    full_df = pd.read_csv(full_csv_path)

    df_groupby = full_df.groupby(DataFrameFields.CASE_COLUMN, sort=False)
    groups = [group for _, group in df_groupby]

    real_val_size = (Config.TRAIN_SIZE / 100) * (Config.VAL_SIZE_FROM_TRAIN / 100)
    real_train_size = (Config.TRAIN_SIZE / 100) - real_val_size

    first_cut = round(len(groups) * real_train_size)
    second_cut = round(len(groups) * (real_train_size + real_val_size))

    train_groups = groups[:first_cut]
    val_groups = groups[first_cut:second_cut]
    test_groups = groups[second_cut:]

    train_df = pd.concat(train_groups)
    val_df = pd.concat(val_groups)
    test_df = pd.concat(test_groups)

    filename = Path(full_csv_path).stem + ".csv"
    if Config.CSV_PATH:
        folder = Config.CSV_PATH
    else:
        folder = str(Path(full_csv_path).parent)
    cv_path = folder + "/holdout/"
    if not os.path.exists(cv_path):
        os.makedirs(cv_path)

    train_df.to_csv(cv_path + "train_" + filename)
    val_df.to_csv(cv_path + "val_" + filename)
    test_df.to_csv(cv_path + "test_" + filename)


def calculate_stats(path_dataset: str, logger):
    """
    Calculate the statistics of the dataset stored in a CSV file
    :param path_dataset: Path to the CSV file with the dataset
    :param logger: DataProcessorLogger to print the stats
    :return: List with the statistics of the dataset
    """

    data = pd.read_csv(path_dataset)
    filename = Path(path_dataset).stem
    logger.log_console(f'********** {filename} stats **********')

    num_events = len(data)
    logger.log_console(f'Num events: {num_events}')

    activities = len(data.groupby(DataFrameFields.ACTIVITY_COLUMN))
    logger.log_console(f'Number of activities: {activities}')

    data[DataFrameFields.ACTIVITY_COLUMN] = data[
        DataFrameFields.ACTIVITY_COLUMN].astype(str)
    cases = data.groupby(DataFrameFields.CASE_COLUMN)
    num_cases = len(cases)
    avg_case_length = cases[DataFrameFields.ACTIVITY_COLUMN].count().mean()
    max_case_length = cases[DataFrameFields.ACTIVITY_COLUMN].count().max()
    logger.log_console(f'Number of cases: {num_cases}\n'
                       f'Avg case length: {avg_case_length:.2f}\n'
                       f'Max case length: {max_case_length:.2f}')

    variants = cases[DataFrameFields.ACTIVITY_COLUMN]\
        .agg("->".join).nunique()
    logger.log_console(f'Number of variants: {variants}\n')

    logger.log_metrics([filename, num_events, activities, num_cases,
                        avg_case_length, max_case_length, variants])





