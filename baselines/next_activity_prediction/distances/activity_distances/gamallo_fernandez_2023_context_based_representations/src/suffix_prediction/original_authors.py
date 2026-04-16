"""
ORIGINAL AUTHORS PREDICTION MODELS
"""
import argparse
import itertools
import math
from pathlib import Path
from pandas_dataset import EventlogDataset
from utils import PrintMode, AuthorModel, Config, DataFrameFields
from args_reader import SuffixPredInputArgs
from logger import SuffixPredLogger
import torch
from scipy.stats import pearsonr
import networkx as nx
import pandas as pd
import numpy as np
from operator import itemgetter
from nltk.util import ngrams
from sklearn.metrics import f1_score
from jellyfish._jellyfish import damerau_levenshtein_distance
from datetime import timedelta
from datetime import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import pathlib
import tensorflow.keras.utils as ku
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Conv1D, GlobalAveragePooling1D, \
    GlobalMaxPooling1D, Reshape, MaxPooling1D, Flatten, Dense, Embedding, Dropout, \
    LSTM, BatchNormalization, Dot
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import f1_score, accuracy_score
from tensorflow.keras.preprocessing.sequence import pad_sequences

tf.compat.v1.set_random_seed(42)
import random
random.seed(42)
np.random.seed(42)



def read_arguments():
    parser = argparse.ArgumentParser(description="Execute the training of the original "
                                                 "prediction models")
    parser.add_argument("-d", "--dataset", required=True, type=str,
                        help="Full path to the dataset")
    partition_mode = parser.add_mutually_exclusive_group(required=True)
    partition_mode.add_argument("--holdout", help="Split in train/validation/test",
                                action='store_true')
    partition_mode.add_argument("--crossvalidation", help="5-fold cross validation",
                                action='store_true')
    author_model = parser.add_mutually_exclusive_group(required=True)
    author_model.add_argument("--tax", help="Execute the Tax model prediction",
                              action='store_true')
    author_model.add_argument("--camargo", help="Execute the Camargo model prediction",
                              action='store_true')
    author_model.add_argument("--evermann", help="Execute the Evermann model prediction",
                              action='store_true')
    print_mode = parser.add_mutually_exclusive_group(required=False)
    print_mode.add_argument("--print_console", help="Print output to the console", action='store_true')
    print_mode.add_argument("--print_file", help="Print output to a file", action='store_true')
    print_mode.add_argument("--print_console_file", help="Print output to the console and file", action='store_true')
    args = parser.parse_args()

    # Full path to the dataset
    dataset = args.dataset

    # Split mode
    if args.holdout:
        crossvalidation = False
    else:
        crossvalidation = True

    # Author model
    if args.tax:
        author = AuthorModel.TAX
    elif args.camargo:
        author = AuthorModel.CAMARGO
    elif args.evermann:
        author = AuthorModel.EVERMANN
    else:
        author = None

    # Check the print mode
    if args.print_console:
        print_mode = PrintMode.CONSOLE
    elif args.print_file:
        print_mode = PrintMode.TO_FILE
    elif args.print_console_file:
        print_mode = PrintMode.CONSOLE_AND_FILE
    else:
        print_mode = PrintMode.NONE

    return SuffixPredInputArgs(dataset, crossvalidation, author, None,
                                None, None, print_mode)


def start(eventlog, author, logger):
    if author == AuthorModel.TAX:
        dl_score = tax(eventlog, logger)
    elif author == AuthorModel.CAMARGO:
        dl_score = camargo(eventlog, logger)
    elif author == AuthorModel.EVERMANN:
        dl_score = evermann(eventlog, logger)
    else:
        dl_score = None
        logger.print_error('Not correct author selected')
        exit(-1)

    return dl_score


def tax(eventlog, logger):
    set_device(logger)

    eventlog.df_train = create_tax_time_variables(eventlog.df_train)
    eventlog.df_val = create_tax_time_variables(eventlog.df_val)
    eventlog.df_test = create_tax_time_variables(eventlog.df_test)

    divisor = eventlog.df_train['times1'].mean()
    divisor2 = eventlog.df_train['times2'].mean()
    divisor3 = get_divisor3_tax(eventlog.df_train)

    eventlog.df_train = add_eoc_tax(eventlog.df_train, eventlog.num_activities)
    eventlog.df_val = add_eoc_tax(eventlog.df_val, eventlog.num_activities)
    # eventlog.df_test = add_eoc_tax(eventlog.df_test, eventlog.num_activities)

    X_train, y_a_train, y_t_train = vectorize_fold_tax(eventlog.df_train, divisor, divisor2,
                                                       eventlog.max_len_case, eventlog.num_activities)
    X_val, y_a_val, y_t_val = vectorize_fold_tax(eventlog.df_val, divisor, divisor2,
                                                 eventlog.max_len_case, eventlog.num_activities)
    X_test, y_a_test, y_t_test = vectorize_fold_tax(eventlog.df_test, divisor, divisor2,
                                                    eventlog.max_len_case, eventlog.num_activities)


    main_input = Input(shape=(eventlog.max_len_case, eventlog.num_activities + 5), name='main_input')

    # train a 2-layer LSTM with one shared layer
    l1 = LSTM(100, implementation=2, kernel_initializer='glorot_uniform',
              return_sequences=True, dropout=0.2)(main_input)  # the shared layer
    b1 = BatchNormalization()(l1)
    l2_1 = LSTM(100, implementation=2, kernel_initializer='glorot_uniform',
                return_sequences=False, dropout=0.2)(b1)  # the layer specialized in activity prediction
    b2_1 = BatchNormalization()(l2_1)
    l2_2 = LSTM(100, implementation=2, kernel_initializer='glorot_uniform',
                return_sequences=False, dropout=0.2)(b1)  # the layer specialized in time prediction
    b2_2 = BatchNormalization()(l2_2)
    act_output = Dense(eventlog.num_activities + 1, activation='softmax',
                       kernel_initializer='glorot_uniform', name='act_output')(b2_1)
    time_output = Dense(1, kernel_initializer='glorot_uniform', name='time_output')(b2_2)

    model = Model(inputs=[main_input], outputs=[act_output, time_output])

    opt = Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004, clipvalue=3)

    model.compile(loss={'act_output': 'categorical_crossentropy', 'time_output': 'mae'}, optimizer=opt,
                  metrics={"act_output": "acc", "time_output": "mae"})
    early_stopping = EarlyStopping(monitor='val_loss', patience=42)

    path = Config.MODEL_PATH + '/' + eventlog.filename + '/suffix/' + str(AuthorModel.TAX)
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    name = 'ORIGINAL' + 'embsize' + str(0) + '_winsize' + str(0) + '.m'
    model_checkpoint = ModelCheckpoint(filepath=path + '/' + name, monitor='val_loss',
                                       save_best_only=True, save_weights_only=False, mode='auto')
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto',
                                   min_delta=0.0001, cooldown=0, min_lr=0)

    # Training
    if logger.print_mode == PrintMode.CONSOLE or \
            logger.print_mode == PrintMode.CONSOLE_AND_FILE:
        verbose = 2
    else:
        verbose = 0
    model.fit(X_train, {'act_output': y_a_train, 'time_output': y_t_train},
              validation_data=(X_val, {"act_output": y_a_val, "time_output": y_t_val}), verbose=verbose,
              callbacks=[early_stopping, model_checkpoint, lr_reducer],
              batch_size=eventlog.max_len_case, epochs=200)

    # Test
    model.load_weights(path + '/' + name)
    model.compile(loss={'act_output': 'categorical_crossentropy', 'time_output': 'mae'}, optimizer=opt,
                  metrics={"act_output": "acc", "time_output": "mae"})
    predict_size = eventlog.max_len_case

    dl_score = []
    for prefix_size in range(1, eventlog.max_len_case):
        data_group = eventlog.df_test.groupby(DataFrameFields.CASE_COLUMN)
        for name, gr in data_group:
            cropped_line = gr[DataFrameFields.ACTIVITY_COLUMN][:prefix_size].values.tolist()
            cropped_times = gr['times1'][:prefix_size].values.tolist()
            cropped_times3 = gr[DataFrameFields.TIMESTAMP_COLUMN][:prefix_size].tolist()
            cropped_times3 = list(map(lambda x: x.to_pydatetime(), cropped_times3))

            if prefix_size >= len(gr):
                continue  # make no prediction for this case, since this case has ended already
            ground_truth = gr[DataFrameFields.ACTIVITY_COLUMN][prefix_size:prefix_size+predict_size].values
            ground_truth_t = gr['times2'][prefix_size - 1]
            case_end_time = gr['times2'][len(gr) - 1]
            ground_truth_t = case_end_time - ground_truth_t
            predicted = []
            total_predicted_time = 0
            for i in range(predict_size):
                enc = encode_tax(cropped_line, cropped_times, cropped_times3,
                                 divisor, divisor2, eventlog.max_len_case, eventlog.num_activities)
                y = model.predict(enc, verbose=0)  # make predictions
                # split predictions into separate activity and time predictions
                y_char = y[0][0]
                y_t = y[1][0][0]
                prediction = np.argmax(y_char)
                cropped_line.append(prediction)
                if y_t < 0:
                    y_t = 0.0
                cropped_times.append(y_t)
                if prediction == eventlog.num_activities:
                    break  # end of case was just predicted, therefore, stop prediction further into the future
                y_t = y_t * divisor3
                cropped_times3.append(cropped_times3[-1] + timedelta(seconds=y_t))
                total_predicted_time = total_predicted_time + y_t
                predicted.append(prediction)
            if len(ground_truth) > 0:
                predicted = list(map(lambda x: chr(x+161), predicted))
                ground_truth = list(map(lambda x: chr(x+161), ground_truth))
                dls = 1 - (damerau_levenshtein_distance(''.join(predicted), ''.join(ground_truth)) / max(
                    len(predicted), len(ground_truth)))
                if dls < 0:
                    dls = 0
                dl_score.append(dls)

    dl_score = np.mean(np.array(dl_score))
    return dl_score


def set_device(logger):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            logger.print_error('No GPU device found')


def create_tax_time_variables(data):
    data_augment = pd.DataFrame()

    # Group by case
    data_group = data.groupby(DataFrameFields.CASE_COLUMN)
    # Iterate over case
    for name, gr in data_group:
        gr = gr.reset_index(drop=True)

        timesincelastevent = gr.loc[:, DataFrameFields.TIMESTAMP_COLUMN].diff() / np.timedelta64(1, 's')
        timesincelastevent.iloc[0] = 0.0

        casestart = gr.loc[0, DataFrameFields.TIMESTAMP_COLUMN]
        timesincecasestart = (gr.loc[:, DataFrameFields.TIMESTAMP_COLUMN] - casestart) / np.timedelta64(1, 's')

        midnight = gr[DataFrameFields.TIMESTAMP_COLUMN].apply(lambda x: x.replace(hour=00, minute=00, second=00))
        timesincemidnight = (gr.loc[:, DataFrameFields.TIMESTAMP_COLUMN] - midnight) / np.timedelta64(1, 's')

        weekday = gr.loc[:, DataFrameFields.TIMESTAMP_COLUMN].dt.dayofweek

        gr['times1'] = timesincelastevent
        gr['times2'] = timesincecasestart
        gr['times3'] = timesincemidnight
        gr['times4'] = weekday

        # gr = gr.drop(columns=[DataFrameFields.TIMESTAMP_COLUMN])

        data_augment = pd.concat([data_augment, gr])

    return data_augment


def get_divisor3_tax(data):
    list_times = []

    # Group by case
    data_group = data.groupby(DataFrameFields.CASE_COLUMN)
    # Iterate over case
    for name, gr in data_group:
        gr = gr.reset_index(drop=True)
        caseend = gr.loc[len(gr)-1, 'times2']
        timeuntilend = caseend - gr.loc[:, 'times2']

        list_times.append(timeuntilend.mean())

    divisor3 = np.mean(np.array(list_times))
    return divisor3


def add_eoc_tax(data, num_activities):
    data_augment = pd.DataFrame()

    # Group by case
    data_group = data.groupby(DataFrameFields.CASE_COLUMN)

    for name, gr in data_group:
        gr = gr.reset_index(drop=True)

        eoc_row = pd.DataFrame({DataFrameFields.CASE_COLUMN: [gr[DataFrameFields.CASE_COLUMN][0]],
                                DataFrameFields.ACTIVITY_COLUMN: [num_activities]})
        gr = pd.concat([gr, eoc_row])
        gr = gr.reset_index(drop=True)

        data_augment = pd.concat([data_augment, gr])

    return data_augment


def vectorize_fold_tax(data, divisor, divisor2, max_len, num_activities):
    sentences = []
    sentences_t = []
    sentences_t2 = []
    sentences_t3 = []
    sentences_t4 = []
    next_chars = []
    next_chars_t = []

    data_group = data.groupby(DataFrameFields.CASE_COLUMN)
    for name, gr in data_group:
        for i in range(0, len(gr), 1):
            # This would be an empty prefix, and it doesn't make much sense to predict based on nothing
            if i == 0:
                continue
            sentences.append(gr[DataFrameFields.ACTIVITY_COLUMN][0:i].values)
            sentences_t.append(gr['times1'][0:i].values)
            sentences_t2.append(gr['times2'][0:i].values)
            sentences_t3.append(gr['times3'][0:i].values)
            sentences_t4.append(gr['times4'][0:i].values)
            # Store the desired prediction
            next_chars.append(gr[DataFrameFields.ACTIVITY_COLUMN][i])
            if i == len(gr) - 1:
                next_chars_t.append(0)
            else:
                next_chars_t.append(gr['times1'][i])

    # Matrix containing the training data
    X = np.zeros((len(sentences), max_len, num_activities+5), dtype=np.float32)
    # Target event prediction data
    y_a = np.zeros((len(sentences), num_activities+1), dtype=np.float32)
    # Target time prediction data
    y_t = np.zeros((len(sentences)), dtype=np.float32)
    for i, sentence in enumerate(sentences):
        leftpad = max_len - len(sentence)
        next_t = next_chars_t[i]
        sentence_t = sentences_t[i]
        sentence_t2 = sentences_t2[i]
        sentence_t3 = sentences_t3[i]
        sentence_t4 = sentences_t4[i]
        for t, act in enumerate(sentence):
            X[i, t + leftpad, act] = 1
            X[i, t + leftpad, num_activities] = t + 1
            X[i, t + leftpad, num_activities + 1] = sentence_t[t] / divisor
            X[i, t + leftpad, num_activities + 2] = sentence_t2[t] / divisor2
            X[i, t + leftpad, num_activities + 3] = sentence_t3[t] / 86400
            X[i, t + leftpad, num_activities + 4] = sentence_t4[t] / 7

        y_a[i, next_chars[i]] = 1
        y_t[i] = next_t / divisor

    return X, y_a, y_t


def encode_tax(sentence, times, times3,
               divisor, divisor2, maxlen, num_activities):
    num_features = num_activities + 5
    X = np.zeros((1, maxlen, num_features), dtype=np.float32)
    leftpad = maxlen - len(sentence)
    times2 = np.cumsum(times)
    for t, act in enumerate(sentence):
        midnight = times3[t].replace(hour=0, minute=0, second=0, microsecond=0)
        timesincemidnight = times3[t] - midnight
        X[0, t + leftpad, act] = 1
        X[0, t + leftpad, num_activities] = t + 1
        X[0, t + leftpad, num_activities + 1] = times[t] / divisor
        X[0, t + leftpad, num_activities + 2] = times2[t] / divisor2
        X[0, t + leftpad, num_activities + 3] = timesincemidnight.seconds / 86400
        X[0, t + leftpad, num_activities + 4] = times3[t].weekday() / 7

    return X


#####################################################################################################

if __name__ == "__main__":
    args = read_arguments()

    logger = SuffixPredLogger(args.print_mode)

    if args.crossvalidation:
        dl_scores = []

        for i in range(Config.CV_FOLDS):
            eventlog = EventlogDataset(args.dataset, cv_fold=i, read_test=True)

            dl_score = start(eventlog, args.author, logger)

            logger.log_console(f'{str(args.author)} TEST FOLD {i}:\n'
                               f'\tDL_SCORE: {dl_score}')

            dl_scores.append(dl_score)
        logger.log_metrics_cv(Path(args.dataset).stem, str(args.author),
                              "ORIGINAL", 0, 0, dl_scores)

    else:
        eventlog = EventlogDataset(args.dataset, read_test=True)

        dl_score = start(eventlog, args.author, logger)
        logger.log_console(f'TEST DL_SCORE: {dl_score}')
        logger.log_metrics_holdout(Path(args.dataset).stem, str(args.author),
                                   "ORIGINAL", 0, 0, dl_score)
