import os
import pandas as pd
import numpy as np
from pathlib import Path
from distances.activity_distances.gamallo_fernandez_2023_context_based_representations.src.embeddings_generator.utils import \
    DataFrameFields


class EventlogDataset:
    """
    Custom Dataset for .csv eventlogs (Pandas DataFrame).
    Supports both cross-validation (train, val, test) and holdout (train, val only).
    """
    filename: str
    directory: Path

    df_train: pd.DataFrame
    df_val: pd.DataFrame
    df_test: pd.DataFrame

    num_activities: int
    num_resources: int

    def __init__(self, csv_path: str, cv_fold: int = None, read_test: bool = True):
        """
        Creates the EventlogDataset and reads the eventlog from the specified path.
        :param csv_path: Path to the .csv file with the eventlog.
        :param cv_fold: Fold number if a cross-validation split is used; use None for holdout.
        :param read_test: Boolean indicating if the test split should be read (set False for train–val holdout).
        """
        self.filename = Path(csv_path).stem
        self.directory = Path(csv_path).parent

        self.df_train = self.read_split('train', cv_fold)
        self.df_val = self.read_split('val', cv_fold)
        if read_test:
            self.df_test = self.read_split('test', cv_fold)

        self.num_activities = self.get_num_activities(use_test=read_test)
        if DataFrameFields.RESOURCE_COLUMN in self.df_train.columns:
            self.num_resources = self.get_num_resources(use_test=read_test)

    def read_split(self, split: str, cv_fold: int) -> pd.DataFrame:
        """
        Reads the .csv file corresponding to the specified split.
        :param split: Partition to read: 'train', 'val', or 'test'.
        :param cv_fold: Fold number if using cross-validation; None for holdout.
        :return: Pandas DataFrame with the eventlog data.
        """
        # Set the data directory (assumed to be two levels up in the 'data' folder)
        self.directory = Path(__file__).resolve().parents[2] / "data"

        if cv_fold is not None:
            # Use cross-validation folder
            path_to_eventlog = Path(self.directory) / "crossvalidation" / f"fold{cv_fold}_{split}_{self.filename}.csv"
        else:
            # Use holdout folder for train–val only
            path_to_eventlog = Path(self.directory) / "holdout" / f"{split}_{self.filename}.csv"

        # Read CSV without setting an index so that all columns (including 'CaseID') remain intact
        df = pd.read_csv(str(path_to_eventlog))
        df[DataFrameFields.ACTIVITY_COLUMN] = df[DataFrameFields.ACTIVITY_COLUMN].astype('category')
        return df

    def get_num_activities(self, use_test: bool = False) -> int:
        """
        Gets the number of unique activities. Uses train and validation sets by default.
        :param use_test: Boolean indicating if the test set should also be considered.
        :return: Number of unique activities.
        """
        all_activities = pd.concat([self.df_train[DataFrameFields.ACTIVITY_COLUMN],
                                    self.df_val[DataFrameFields.ACTIVITY_COLUMN]])
        if use_test and hasattr(self, 'df_test'):
            all_activities = pd.concat([all_activities, self.df_test[DataFrameFields.ACTIVITY_COLUMN]])
        all_activities = np.sort(np.unique(all_activities))
        num_activities = all_activities[-1].item() + 1
        return num_activities

    def get_num_resources(self, use_test: bool = False) -> int:
        """
        Gets the number of unique resources. Uses train and validation sets by default.
        :param use_test: Boolean indicating if the test set should also be considered.
        :return: Number of unique resources.
        """
        all_resources = pd.concat([self.df_train[DataFrameFields.RESOURCE_COLUMN],
                                   self.df_val[DataFrameFields.RESOURCE_COLUMN]])
        if use_test and hasattr(self, 'df_test'):
            all_resources = pd.concat([all_resources, self.df_test[DataFrameFields.RESOURCE_COLUMN]])
        all_resources = np.sort(np.unique(all_resources))
        num_resources = all_resources[-1].item() + 1
        return num_resources

    def get_num_events(self, split: str) -> int:
        """
        Gets the number of events in the eventlog for the specified split.
        :param split: 'train', 'val', or 'test'.
        :return: The number of events in the split.
        """
        if split == 'train':
            return len(self.df_train.index)
        elif split == 'val':
            return len(self.df_val.index)
        elif split == 'test' and hasattr(self, 'df_test'):
            return len(self.df_test.index)
        else:
            return 0

    def get_num_cases(self, split: str) -> int:
        """
        Gets the number of cases in the eventlog for the specified split.
        :param split: 'train', 'val', or 'test'.
        :return: The number of cases in the split.
        """
        if split == 'train':
            return len(self.df_train.groupby(DataFrameFields.CASE_COLUMN))
        elif split == 'val':
            return len(self.df_val.groupby(DataFrameFields.CASE_COLUMN))
        elif split == 'test' and hasattr(self, 'df_test'):
            return len(self.df_test.groupby(DataFrameFields.CASE_COLUMN))
        else:
            return 0
