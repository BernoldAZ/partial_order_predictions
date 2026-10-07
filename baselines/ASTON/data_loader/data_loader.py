import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
import datetime as dt
from math import log

from datasets.datasets import TracesDatasetSeqTime


def __load_dataset_dict_simple_fold(dataset):
    if dataset == 'SEPSIS':
        # Maximum length trace in dataset
        max_length_trace = 185
        # Name of the file with the eventlog
        eventlog = 'SEPSIS.csv'
    elif dataset == 'bpi_challenge_2013_incidents':
        # Maximum length trace in dataset
        max_length_trace = 123
        # Name of the file with the eventlog
        eventlog = 'bpi_challenge_2013_incidents.csv'
    elif dataset == 'BPI_Challenge_2012_O':
        # Maximum length trace in dataset
        max_length_trace = 30
        # Name of the file with the eventlog
        eventlog = 'BPI_Challenge_2012_O.csv'
    elif dataset == 'BPI_Challenge_2012_Complete':
        # Maximum length trace in dataset
        max_length_trace = 96
        # Name of the file with the eventlog
        eventlog = 'BPI_Challenge_2012_Complete.csv'
    elif dataset == 'BPI_Challenge_2012_A':
        # Maximum length trace in dataset
        max_length_trace = 8
        # Name of the file with the eventlog
        eventlog = 'BPI_Challenge_2012_A.csv'
    elif dataset == 'BPI_Challenge_2012':
        # Maximum length trace in dataset
        max_length_trace = 175
        # Name of the file with the eventlog
        eventlog = 'BPI_Challenge_2012.csv'
    elif dataset == 'nasa':
        # Maximum length trace in dataset
        max_length_trace = 50
        # Name of the file with the eventlog
        eventlog = 'nasa.csv'
    elif dataset == 'BPI_Challenge_2012_W_Complete':
        # Maximum length trace in dataset
        max_length_trace = 74
        # Name of the file with the eventlog
        eventlog = 'BPI_Challenge_2012_W_Complete.csv'
    elif dataset == 'env_permit':
        # Maximum length trace in dataset
        max_length_trace = 25
        # Name of the file with the eventlog
        eventlog = 'env_permit.csv'
    elif dataset == 'Helpdesk':
        # Maximum length trace in dataset
        max_length_trace = 15
        # Name of the file with the eventlog
        eventlog = 'Helpdesk.csv'
    elif dataset == 'BPI_Challenge_2012_W':
        # Maximum length trace in dataset
        max_length_trace = 156
        # Name of the file with the eventlog
        eventlog = 'BPI_Challenge_2012_W.csv'
    elif dataset == 'BPI_Challenge_2013_closed_problems':
        max_length_trace = 35
        # Name of the file with the eventlog
        eventlog = 'BPI_Challenge_2013_closed_problems.csv'
    elif dataset == 'test':
        case_id_col = "Case ID"
        activity_col = "Activity"
        timestamp_col = "Complete Timestamp"
        # Maximum length trace in dataset
        max_length_trace = 19
        # Name of the file with the eventlog
        eventlog = 'helpdesk_mini.csv'
    else:
        # The input dataset is unknown
        print('Not known dataset')
        exit()

    """return {
        "case_id_col": "CaseID",
        "activity_col": "ActivityID",
        "timestamp_col": "CompleteTimestamp",
        "resource_col": "Resource",
        "max_length_trace": max_length_trace,
        "eventlog": eventlog
    }"""

    return {
        "case_id_col": "caseid",
        "activity_col": "task",
        "timestamp_col": "end_timestamp",
        "resource_col": "user",
        "max_length_trace": max_length_trace,
        "eventlog": eventlog
    }


def __load_dataset_dict_simple_resource_fold(dataset):
    if dataset == 'SEPSIS':
        # Maximum length trace in dataset
        max_length_trace = 185
        # Name of the file with the eventlog
        eventlog = 'SEPSIS.csv'
    elif dataset == 'bpi_challenge_2013_incidents':
        # Maximum length trace in dataset
        max_length_trace = 123
        # Name of the file with the eventlog
        eventlog = 'bpi_challenge_2013_incidents.csv'
    elif dataset == 'BPI_Challenge_2012_O':
        # Maximum length trace in dataset
        max_length_trace = 30
        # Name of the file with the eventlog
        eventlog = 'BPI_Challenge_2012_O.csv'
    elif dataset == 'BPI_Challenge_2012_Complete':
        # Maximum length trace in dataset
        max_length_trace = 96
        # Name of the file with the eventlog
        eventlog = 'BPI_Challenge_2012_Complete.csv'
    elif dataset == 'BPI_Challenge_2012_A':
        # Maximum length trace in dataset
        max_length_trace = 8
        # Name of the file with the eventlog
        eventlog = 'BPI_Challenge_2012_A.csv'
    elif dataset == 'BPI_Challenge_2012':
        # Maximum length trace in dataset
        max_length_trace = 175
        # Name of the file with the eventlog
        eventlog = 'BPI_Challenge_2012.csv'
    elif dataset == 'nasa':
        # Maximum length trace in dataset
        max_length_trace = 50
        # Name of the file with the eventlog
        eventlog = 'nasa.csv'
    elif dataset == 'BPI_Challenge_2012_W_Complete':
        # Maximum length trace in dataset
        max_length_trace = 74
        # Name of the file with the eventlog
        eventlog = 'BPI_Challenge_2012_W_Complete.csv'
    elif dataset == 'env_permit':
        # Maximum length trace in dataset
        max_length_trace = 25
        # Name of the file with the eventlog
        eventlog = 'env_permit.csv'
    elif dataset == 'Helpdesk':
        # Maximum length trace in dataset
        max_length_trace = 15
        # Name of the file with the eventlog
        eventlog = 'Helpdesk.csv'
    elif dataset == 'BPI_Challenge_2012_W':
        # Maximum length trace in dataset
        max_length_trace = 156
        # Name of the file with the eventlog
        eventlog = 'BPI_Challenge_2012_W.csv'
    elif dataset == 'BPI_Challenge_2013_closed_problems':
        max_length_trace = 35
        # Name of the file with the eventlog
        eventlog = 'BPI_Challenge_2013_closed_problems.csv'
    elif dataset == 'test':
        case_id_col = "Case ID"
        activity_col = "Activity"
        timestamp_col = "Complete Timestamp"
        # Maximum length trace in dataset
        max_length_trace = 19
        # Name of the file with the eventlog
        eventlog = 'helpdesk_mini.csv'
    else:
        # The input dataset is unknown
        print('Not known dataset')
        exit()

    return {
        "case_id_col": "caseid",
        "activity_col": "task",
        "timestamp_col": "end_timestamp",
        "resource_col": "user",
        "max_length_trace": max_length_trace,
        "eventlog": eventlog
    }


def __label_encode_column(df_train, df_val, df_test, col_name):
    # We instantiate a label encoder
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=int)
    # We train the encoder over our training set
    df_train[col_name] = encoder.fit_transform(df_train[[col_name]])
    df_train[col_name] = df_train[col_name].apply(lambda x: 1 if x == -1 else x + 2)
    # We apply the encoder over our validation and test set
    df_val[col_name] = encoder.transform(df_val[[col_name]])
    df_val[col_name] = df_val[col_name].apply(lambda x: 1 if x == -1 else x + 2)
    df_test[col_name] = encoder.transform(df_test[[col_name]])
    df_test[col_name] = df_test[col_name].apply(lambda x: 1 if x == -1 else x + 2)
    # We return the modified datasets
    return df_train, df_val, df_test, len(encoder.categories_[0])


def __label_encode_column_list(df_train, df_val, df_test, cols_name):
    num_categories = []
    for col_name in cols_name:
        # We instantiate a label encoder
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=int)
        # We train the encoder over our training set
        df_train[col_name] = encoder.fit_transform(df_train[[col_name]])
        df_train[col_name] = df_train[col_name].apply(lambda x: 1 if x == -1 else x + 2)
        # We apply the encoder over our validation and test set
        df_val[col_name] = encoder.transform(df_val[[col_name]])
        df_val[col_name] = df_val[col_name].apply(lambda x: 1 if x == -1 else x + 2)
        df_test[col_name] = encoder.transform(df_test[[col_name]])
        df_test[col_name] = df_test[col_name].apply(lambda x: 1 if x == -1 else x + 2)
        # We save the number of categories
        num_categories.append(len(encoder.categories_[0]))
    # We return the modified datasets
    return df_train, df_val, df_test, num_categories


def __vectorize_fold_seq_seq_time(df, num_activities, case_id_col, activity_col, max_length_trace, time_features):
    # We aggregate the cases together to generate the traces
    tracesGroupedDf = df.groupby(case_id_col).aggregate(lambda x: (list(x))).reset_index(drop=True)
    # We are only interested in the activity column
    tracesDf = tracesGroupedDf[[activity_col] + time_features]

    tracesInput = []
    validInput = []
    tracesOutput = []
    validOutput = []
    # Every case will generate as many traces as events it has
    for _, row in tracesDf.iterrows():
        # 2 is the minimum length of a trace
        for i in range(1, len(row[0]) + 1):
            tracesInput.append(list(map(lambda x: x[:i], row)))
            # tracesInput.append(row[0][:i])
            validInput.append(i)
            tracesOutput.append(row[0][i:])
            # tracesOutput.append(row[0][i:i+window_size])
            validOutput.append(len(row[0]) - i + 1)

    # We pad the traces so they all have the same length
    tracesInput = [[[0] * (max_length_trace - len(trace)) + trace for trace in sample] for sample in tracesInput]
    # We pad the traces so they all have the same length
    tracesOutput = [trace + [num_activities + 2] * (max_length_trace - len(trace)) for trace in tracesOutput]
    # We return the input, output and its corresponding valid lengths
    return tracesInput, validInput, tracesOutput, validOutput


def __vectorize_fold_seq_seq_time_resouce(df, num_activities, case_id_col, activity_col, resource_col, max_length_trace,
                                          time_features):
    # We aggregate the cases together to generate the traces
    tracesGroupedDf = df.groupby(case_id_col).aggregate(lambda x: (list(x))).reset_index(drop=True)
    # We are only interested in the activity column
    if resource_col in tracesGroupedDf.columns:
        tracesDf = tracesGroupedDf[[activity_col, resource_col] + time_features]
    else:
        tracesDf = tracesGroupedDf[[activity_col] + time_features]

    tracesInput = []
    validInput = []
    tracesOutput = []
    validOutput = []
    # Every case will generate as many traces as events it has
    for _, row in tracesDf.iterrows():
        # 2 is the minimum length of a trace
        for i in range(1, len(row[0]) + 1):
            tracesInput.append(list(map(lambda x: x[:i], row)))
            # tracesInput.append(row[0][:i])
            validInput.append(i)
            tracesOutput.append(row[0][i:])
            # tracesOutput.append(row[0][i:i+window_size])
            validOutput.append(len(row[0]) - i + 1)

    # We pad the traces so they all have the same length
    tracesInput = [[[0] * (max_length_trace - len(trace)) + trace for trace in sample] for sample in tracesInput]
    # We pad the traces so they all have the same length
    tracesOutput = [trace + [num_activities + 2] * (max_length_trace - len(trace)) for trace in tracesOutput]
    # We return the input, output and its corresponding valid lengths
    return tracesInput, validInput, tracesOutput, validOutput


def __load_dataset_time_resource_log(filename, timestamp_col, case_id_col, activity_col, resource_col):
    # Format of the datetime
    format_timestamp = '%Y-%m-%dT%H:%M:%S.%f'
    # We cant do log(0)
    epsilon = 0.001
    # We load the dataset
    df = pd.read_csv(filename, header=0, delimiter=',', keep_default_na=False, dtype=object)
    # We select only the columns we are interested in
    if resource_col in df.columns:
        df = df[[case_id_col, activity_col, resource_col, timestamp_col]]
    else:
        df = df[[case_id_col, activity_col, timestamp_col]]
    # Convert timestamps to datetime format
    df[timestamp_col] = df[timestamp_col].apply(lambda x: dt.datetime.strptime(x, format_timestamp))
    # Add column with the time diffference between the previous event in the trace and the current event
    df['timesincelastevent'] = [0] + [
        (df[timestamp_col][i] - df[timestamp_col][i - 1]).total_seconds() if df[case_id_col][i - 1] == df[case_id_col][
            i] else 0 for i in range(1, len(df.index))]
    # We want to get the start date of every case
    df_cases_start = df[[case_id_col, timestamp_col]].groupby(case_id_col).min()
    # We rename the column timestamp
    df_cases_start = df_cases_start.rename(columns={timestamp_col: 'timesincecasestart'})
    # We add the column with the start time for every case
    df = df.reset_index().merge(df_cases_start, left_on=case_id_col, right_index=True).set_index(df.index)
    # We drop the residual column generated on merge
    df = df.drop(['index'], axis=1)
    # We compute the time since case start
    df['timesincecasestart'] = df[timestamp_col] - df['timesincecasestart']
    # We convert time deltas to number of seconds
    df['timesincecasestart'] = df['timesincecasestart'].apply(lambda x: log(x.total_seconds() + epsilon))

    # Add column to get time within day
    df['timesincemidnight'] = df[timestamp_col].apply(
        lambda x: log((x - x.replace(hour=0, minute=0, second=0)).total_seconds() + epsilon))
    # Add column to get the month
    df['month'] = df[timestamp_col].apply(lambda x: x.month)
    # Add column to get the day of the week
    df['weekday'] = df[timestamp_col].apply(lambda x: x.weekday())
    # Add column to get the day of the week
    df['hour'] = df[timestamp_col].apply(lambda x: x.hour)
    # We return the processed dataset
    return df


def __load_dataset_time_log(filename, timestamp_col, case_id_col, activity_col):
    # Format of the datetime
    # format_timestamp = '%Y-%m-%d %H:%M:%S'
    format_timestamp = '%Y-%m-%dT%H:%M:%S.%f'
    # We cant do log(0)
    epsilon = 0.001
    # We load the dataset
    df = pd.read_csv(filename, header=0, delimiter=',', keep_default_na=False, dtype=object)
    # We select only the columns we are interested in
    df = df[[case_id_col, activity_col, timestamp_col]]
    # Convert timestamps to datetime format
    df[timestamp_col] = df[timestamp_col].apply(lambda x: dt.datetime.strptime(x, format_timestamp))
    # Add column with the time diffference between the previous event in the trace and the current event
    df['timesincelastevent'] = [0] + [
        (df[timestamp_col][i] - df[timestamp_col][i - 1]).total_seconds() if df[case_id_col][i - 1] == df[case_id_col][
            i] else 0 for i in range(1, len(df.index))]
    # We want to get the start date of every case
    df_cases_start = df[[case_id_col, timestamp_col]].groupby(case_id_col).min()
    # We rename the column timestamp
    df_cases_start = df_cases_start.rename(columns={timestamp_col: 'timesincecasestart'})
    # We add the column with the start time for every case
    df = df.reset_index().merge(df_cases_start, left_on=case_id_col, right_index=True).set_index(df.index)
    # We drop the residual column generated on merge
    df = df.drop(['index'], axis=1)
    # We compute the time since case start
    df['timesincecasestart'] = df[timestamp_col] - df['timesincecasestart']
    # We convert time deltas to number of seconds
    df['timesincecasestart'] = df['timesincecasestart'].apply(lambda x: log(x.total_seconds() + epsilon))

    # Add column to get time within day
    df['timesincemidnight'] = df[timestamp_col].apply(
        lambda x: log((x - x.replace(hour=0, minute=0, second=0)).total_seconds() + epsilon))
    # Add column to get the month
    df['month'] = df[timestamp_col].apply(lambda x: x.month)
    # Add column to get the day of the week
    df['weekday'] = df[timestamp_col].apply(lambda x: x.weekday())
    # Add column to get the day of the week
    df['hour'] = df[timestamp_col].apply(lambda x: x.hour)
    # We return the processed dataset
    return df


def __normalize_numerical_features(dfTrain, dfVal, dfTest):
    # We apply different normalizations for the artificial features
    divisors = [60 * 60 * 24, 12, 7, 24]
    divisorsCols = ['timesincemidnight', 'month', 'weekday', 'hour']

    # We make sure our columns have the adequate types to perform the scaling
    typeDict = dict(zip(divisorsCols, [float] * len(divisorsCols)))
    dfTrain = dfTrain.astype(typeDict)
    dfVal = dfVal.astype(typeDict)
    dfTest = dfTest.astype(typeDict)

    #for divisor, divisorCol in zip(divisors, divisorsCols):
    #    dfTrain[divisorCol] /= divisor
    #    dfVal[divisorCol] /= divisor
    #    dfTest[divisorCol] /= divisor

    # We apply standard scaler for time difference features
    for col in ['timesincelastevent', 'timesincecasestart']:
        scaler = StandardScaler()
        dfTrain[col] = scaler.fit_transform(dfTrain[[col]])
        dfVal[col] = scaler.transform(dfVal[[col]])
        dfTest[col] = scaler.transform(dfTest[[col]])

    # We return the train, validation and test dataframes
    return dfTrain, dfVal, dfTest


def load_dataset_seq_seq_time_fold_csv(dataset, num_folds):
    dataset_config = __load_dataset_dict_simple_fold(dataset)
    # Datasets with different splits
    train_dataset_fold, val_dataset_fold, test_dataset_fold = [], [], []
    # Num activities in each fold
    num_activities_fold = []
    for i in range(num_folds):
        # We load the dataset including time features
        fold_train = __load_dataset_time_log('data/train_fold' + str(i) + '_variation0_' + dataset_config['eventlog'],
                                             dataset_config['timestamp_col'], dataset_config['case_id_col'],
                                             dataset_config['activity_col'])
        fold_val = __load_dataset_time_log('data/val_fold' + str(i) + '_variation0_' + dataset_config['eventlog'],
                                           dataset_config['timestamp_col'], dataset_config['case_id_col'],
                                           dataset_config['activity_col'])
        fold_test = __load_dataset_time_log('data/test_fold' + str(i) + '_variation0_' + dataset_config['eventlog'],
                                            dataset_config['timestamp_col'], dataset_config['case_id_col'],
                                            dataset_config['activity_col'])
        # The following features will be associated to timestamp data
        time_features = ['timesincemidnight', 'month', 'weekday', 'hour', 'timesincelastevent', 'timesincecasestart']
        # We one-hot encode the categorical features
        print('Encode categorical features------')
        df_train, df_val, df_test, num_activities = __label_encode_column(fold_train, fold_val, fold_test,
                                                                          dataset_config['activity_col'])
        # Normalize numerical features
        print('Normalize numerical features------')
        df_train, df_val, df_test = __normalize_numerical_features(df_train, df_val, df_test)
        # We add to the configuration the number of activities
        num_activities_fold.append(num_activities)
        # We vectorize our sets
        print('Vectorize training set-----')
        X_train, X_train_valid_len, Y_train, Y_train_valid_len = __vectorize_fold_seq_seq_time(df_train, num_activities,
                                                                                               dataset_config[
                                                                                                   'case_id_col'],
                                                                                               dataset_config[
                                                                                                   'activity_col'],
                                                                                               dataset_config[
                                                                                                   'max_length_trace'],
                                                                                               time_features)
        print('Vectorize validation set-----')
        X_val, X_val_valid_len, Y_val, Y_val_valid_len = __vectorize_fold_seq_seq_time(df_val, num_activities,
                                                                                       dataset_config['case_id_col'],
                                                                                       dataset_config['activity_col'],
                                                                                       dataset_config[
                                                                                           'max_length_trace'],
                                                                                       time_features)
        print('Vectorize test set-----')
        X_test, X_test_valid_len, Y_test, Y_test_valid_len = __vectorize_fold_seq_seq_time(df_test, num_activities,
                                                                                           dataset_config[
                                                                                               'case_id_col'],
                                                                                           dataset_config[
                                                                                               'activity_col'],
                                                                                           dataset_config[
                                                                                               'max_length_trace'],
                                                                                           time_features)
        # We create the dataset with the traces
        train_dataset = TracesDatasetSeqTime(X_train, X_train_valid_len, Y_train, Y_train_valid_len)
        val_dataset = TracesDatasetSeqTime(X_val, X_val_valid_len, Y_val, Y_val_valid_len)
        test_dataset = TracesDatasetSeqTime(X_test, X_test_valid_len, Y_test, Y_test_valid_len)
        # We append the datasets to the different folds
        train_dataset_fold.append(train_dataset)
        val_dataset_fold.append(val_dataset)
        test_dataset_fold.append(test_dataset)
        # We return the datasets
    return train_dataset_fold, val_dataset_fold, test_dataset_fold, dataset_config, num_activities_fold


def load_dataset_seq_seq_time_resource_fold_csv(dataset, fold_num, train_dataset_fold_grnn, val_dataset_fold_grnn, test_dataset_fold_grnn, N, F, adjacency_matrix):
    dataset_config = __load_dataset_dict_simple_resource_fold(dataset)
    # Datasets with different splits

    # We load the dataset including time features
    fold_train = __load_dataset_time_resource_log(
        'data/train_fold' + str(fold_num) + '_variation0_' + dataset_config['eventlog'], dataset_config['timestamp_col'],
        dataset_config['case_id_col'], dataset_config['activity_col'], dataset_config['resource_col'])
    fold_val = __load_dataset_time_resource_log(
        'data/val_fold' + str(fold_num) + '_variation0_' + dataset_config['eventlog'], dataset_config['timestamp_col'],
        dataset_config['case_id_col'], dataset_config['activity_col'], dataset_config['resource_col'])
    fold_test = __load_dataset_time_resource_log(
        'data/test_fold' + str(fold_num) + '_variation0_' + dataset_config['eventlog'], dataset_config['timestamp_col'],
        dataset_config['case_id_col'], dataset_config['activity_col'], dataset_config['resource_col'])
    # The following features will be associated to timestamp data
    time_features = ['timesincemidnight', 'month', 'weekday', 'hour', 'timesincelastevent', 'timesincecasestart']
    # We label encode the activities
    print('Encode categorical features------')
    categorical_columns = [dataset_config["activity_col"]]

    # TODO: technically, we should open the dataset and inspect the columns
    if dataset != "SEPSIS" and dataset != "nasa":
        categorical_columns.append(dataset_config["resource_col"])
    df_train, df_val, df_test, num_categories = __label_encode_column_list(fold_train, fold_val, fold_test,
                                                                           categorical_columns)
    # We add to the configuration the number of activities
    num_activities = num_categories[0]

    # TODO: technically, we should open the dataset and inspect the columns
    if dataset != "SEPSIS" and dataset != "nasa":
        num_resources = num_categories[1]
    else:
        num_resources = -3

    # print(num_activities)
    # We label encode the resources
    # df_train, df_val, df_test, num_resources = __label_encode_column(df_train, df_val, df_test, )
    # Normalize numerical features
    # print(num_resources)
    print('Normalize numerical features------')
    # TODO: deactivate normalization to use with time2vec
    df_train, df_val, df_test = __normalize_numerical_features(df_train, df_val, df_test)

    # We vectorize our sets
    print('Vectorize training set-----')
    X_train, X_train_valid_len, Y_train, Y_train_valid_len = __vectorize_fold_seq_seq_time_resouce(df_train,
                                                                                                   num_activities,
                                                                                                   dataset_config[
                                                                                                       'case_id_col'],
                                                                                                   dataset_config[
                                                                                                       'activity_col'],
                                                                                                   dataset_config[
                                                                                                       'resource_col'],
                                                                                                   dataset_config[
                                                                                                       'max_length_trace'],
                                                                                                   time_features)
    # print(X_train[2])
    print('Vectorize validation set-----')
    X_val, X_val_valid_len, Y_val, Y_val_valid_len = __vectorize_fold_seq_seq_time_resouce(df_val, num_activities,
                                                                                           dataset_config[
                                                                                               'case_id_col'],
                                                                                           dataset_config[
                                                                                               'activity_col'],
                                                                                           dataset_config[
                                                                                               'resource_col'],
                                                                                           dataset_config[
                                                                                               'max_length_trace'],
                                                                                           time_features)
    print('Vectorize test set-----')
    X_test, X_test_valid_len, Y_test, Y_test_valid_len = __vectorize_fold_seq_seq_time_resouce(df_test,
                                                                                               num_activities,
                                                                                               dataset_config[
                                                                                                   'case_id_col'],
                                                                                               dataset_config[
                                                                                                   'activity_col'],
                                                                                               dataset_config[
                                                                                                   'resource_col'],
                                                                                               dataset_config[
                                                                                                   'max_length_trace'],
                                                                                               time_features)
    # We create the dataset with the traces
    train_dataset = TracesDatasetSeqTime(X_train, X_train_valid_len, Y_train, Y_train_valid_len, train_dataset_fold_grnn, N, F, dataset_config["max_length_trace"])
    val_dataset = TracesDatasetSeqTime(X_val, X_val_valid_len, Y_val, Y_val_valid_len, val_dataset_fold_grnn, N, F, dataset_config["max_length_trace"])
    test_dataset = TracesDatasetSeqTime(X_test, X_test_valid_len, Y_test, Y_test_valid_len, test_dataset_fold_grnn, N, F, dataset_config["max_length_trace"])
    # We return the datasets
    return train_dataset, val_dataset, test_dataset, dataset_config, num_activities, num_resources
