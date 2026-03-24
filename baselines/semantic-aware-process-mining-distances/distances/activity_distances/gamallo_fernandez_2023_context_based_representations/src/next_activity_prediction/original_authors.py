"""
ORIGINAL AUTHORS PREDICTION MODELS
"""
import argparse
import itertools
import math
from pathlib import Path
from pandas_dataset import EventlogDataset
from utils import PrintMode, AuthorModel, Config, DataFrameFields
from args_reader import NextActPredInputArgs
from logger import NextActPredLogger
import torch
from scipy.stats import pearsonr
import networkx as nx
import pandas as pd
import numpy as np
from operator import itemgetter
from nltk.util import ngrams
from sklearn.metrics import f1_score
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

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
    author_model.add_argument("--mauro", help="Execute the Mauro model prediction",
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
    elif args.mauro:
        author = AuthorModel.MAURO
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

    return NextActPredInputArgs(dataset, crossvalidation, author, None,
                                None, None, print_mode)


def start(eventlog, author, logger):
    if author == AuthorModel.TAX:
        accuracy, f1_score = tax(eventlog, logger)
    elif author == AuthorModel.CAMARGO:
        accuracy, f1_score = camargo(eventlog, logger)
    elif author == AuthorModel.EVERMANN:
        accuracy, f1_score = evermann(eventlog, logger)
    elif author == AuthorModel.MAURO:
        accuracy, f1_score = mauro(eventlog, logger)
    else:
        accuracy = None
        f1_score = None
        logger.print_error('Not correct author selected')
        exit(-1)

    return accuracy, f1_score


def tax(eventlog, logger):
    set_device(logger)

    eventlog.df_train = create_tax_time_variables(eventlog.df_train)
    eventlog.df_val = create_tax_time_variables(eventlog.df_val)
    eventlog.df_test = create_tax_time_variables(eventlog.df_test)

    divisor = eventlog.df_train['times1'].mean()
    divisor2 = eventlog.df_train['times2'].mean()

    eventlog.df_train = add_eoc_tax(eventlog.df_train, eventlog.num_activities)
    eventlog.df_val = add_eoc_tax(eventlog.df_val, eventlog.num_activities)
    eventlog.df_test = add_eoc_tax(eventlog.df_test, eventlog.num_activities)


    X_train, y_a_train, y_t_train = vectorize_fold_tax(eventlog.df_train, divisor, divisor2,
                                                       eventlog.max_len_case, eventlog.num_activities)
    X_val, y_a_val, y_t_val = vectorize_fold_tax(eventlog.df_val, divisor, divisor2,
                                                 eventlog.max_len_case, eventlog.num_activities)
    X_test, y_a_test, y_t_test = vectorize_fold_tax(eventlog.df_test, divisor, divisor2,
                                                    eventlog.max_len_case, eventlog.num_activities)

    main_input = Input(shape=(eventlog.max_len_case, eventlog.num_activities+5), name='main_input')

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
    act_output = Dense(eventlog.num_activities+1, activation='softmax',
                       kernel_initializer='glorot_uniform', name='act_output')(b2_1)
    time_output = Dense(1, kernel_initializer='glorot_uniform', name='time_output')(b2_2)

    model = Model(inputs=[main_input], outputs=[act_output, time_output])

    opt = Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004, clipvalue=3)

    model.compile(loss={'act_output': 'categorical_crossentropy', 'time_output': 'mae'}, optimizer=opt,
                  metrics={"act_output": "acc", "time_output": "mae"})
    early_stopping = EarlyStopping(monitor='val_loss', patience=42)


    path = Config.MODEL_PATH + '/' + eventlog.filename + '/prediction/' + str(AuthorModel.TAX)
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    name = 'ORIGINAL' + 'embsize' + str(0) + '_winsize' + str(0) + '.m'
    model_checkpoint = ModelCheckpoint(filepath=path + '/' + name, monitor='val_loss',
                                       save_best_only=True, save_weights_only=True, mode='auto')
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
    preds = model.predict([X_test])
    y_a_pred_probs = preds[0]
    y_t_pred_probs = preds[1]
    y_a_pred = np.argmax(y_a_pred_probs, axis=1)
    y_true = np.argmax(y_a_test, axis=1)

    acc = accuracy_score(y_true, y_a_pred)
    f1 = f1_score(y_true, y_a_pred, average="weighted")

    return acc, f1


def evermann(eventlog, logger):
    set_device(logger)

    seq_len = 20
    batch_size = 20
    embedding_dim = 32
    rnn_units = 32
    maxGradNorm = 5
    lrDecay = 0.75
    lr = 1.0

    X_train = vectorize_fold_evermann(eventlog.df_train, eventlog.num_activities)
    X_val = vectorize_fold_evermann(eventlog.df_val, eventlog.num_activities)
    X_test = vectorize_fold_evermann(eventlog.df_test, eventlog.num_activities)

    train_dataset = to_dataset_evermann(X_train, seq_len, batch_size)
    val_dataset = to_dataset_evermann(X_val, seq_len, batch_size)
    test_dataset = to_dataset_evermann(X_test, seq_len, batch_size)

    model = build_model_evermann(eventlog.num_activities+1, embedding_dim, rnn_units, batch_size)

    optimizer = tf.keras.optimizers.SGD(learning_rate=lr, decay=lrDecay, clipnorm=maxGradNorm, momentum=0.9, nesterov=True)

    path = Config.MODEL_PATH + '/' + eventlog.filename + '/prediction/' + str(AuthorModel.EVERMANN)
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    name = 'ORIGINAL' + 'embsize' + str(0) + '_winsize' + str(0) + '.m'
    checkpoint_callback = ModelCheckpoint(
        os.path.join(path, name),
        monitor="val_loss",
        save_weights_only=True, save_best_only=True)

    model.compile(
        loss=loss_evermann, optimizer=optimizer,
        metrics=[tf.keras.metrics.sparse_categorical_accuracy]
    )

    # Training
    if logger.print_mode == PrintMode.CONSOLE or \
            logger.print_mode == PrintMode.CONSOLE_AND_FILE:
        verbose = 2
    else:
        verbose = 0
    model.fit(train_dataset, epochs=100, verbose=verbose,
              callbacks=[checkpoint_callback], validation_data=val_dataset)

    # Test
    tf.compat.v1.set_random_seed(42)
    random.seed(42)
    np.random.seed(42)
    model = build_model_evermann(eventlog.num_activities+1, embedding_dim, rnn_units, batch_size=1)
    model.load_weights(os.path.join(path, name))
    model.build(tf.TensorShape([1, None]))
    y_pred = []
    y_true = []
    for trace in X_test:
        for i, event in enumerate(trace):
            model.reset_states()
            inp = trace[:i + 1]
            next_event = trace[i + 1]
            full_preds = model(tf.expand_dims(inp, 0))
            probs = tf.nn.softmax(tf.squeeze(full_preds, 0).numpy()[-1])
            y_pred.append(probs)
            y_true.append(np.eye(eventlog.num_activities+1)[next_event])
            if next_event == eventlog.num_activities:
                break

    y_pred_a = np.argmax(y_pred, axis=1)
    y_true_a = np.argmax(y_true, axis=1)

    acc = accuracy_score(y_true_a, y_pred_a)
    f1 = f1_score(y_true_a, y_pred_a, average="weighted")

    return acc, f1


def camargo(eventlog, logger):
    set_device(logger)

    # Get roles
    log_df_resources = pd.DataFrame.from_records(get_resource_table(eventlog))
    log_df_resources = log_df_resources.rename(index=str, columns={"resource": DataFrameFields.RESOURCE_COLUMN})

    eventlog.df_train = eventlog.df_train.merge(log_df_resources,
                                                on=[DataFrameFields.RESOURCE_COLUMN], how='left')
    eventlog.df_val = eventlog.df_val.merge(log_df_resources,
                                            on=[DataFrameFields.RESOURCE_COLUMN], how='left')
    eventlog.df_test = eventlog.df_test.merge(log_df_resources,
                                              on=[DataFrameFields.RESOURCE_COLUMN], how='left')

    unique_activities = [*range(eventlog.num_activities)]
    unique_roles = list(log_df_resources['role'].unique())
    dim_number = math.ceil(len(list(itertools.product(*[unique_activities, unique_roles])))**0.25)

    # Train embeddings
    ac_weights, rl_weights = train_embedded_camargo(eventlog.df_train, len(unique_activities),
                                                    len(unique_roles), dim_number)

    eventlog.df_train = prepare_data_camargo(eventlog.df_train)
    eventlog.df_val = prepare_data_camargo(eventlog.df_val)
    eventlog.df_test = prepare_data_camargo(eventlog.df_test)
    # Here Camargo calculates again the roles, but we already have them
    eventlog.df_train = add_calculated_times_camargo(eventlog.df_train)
    eventlog.df_val = add_calculated_times_camargo(eventlog.df_val)
    eventlog.df_test = add_calculated_times_camargo(eventlog.df_test)
    # Here Camargo creates indexes for activities and roles, but we already have them
    examples_train = vectorize_camargo(eventlog.df_train, eventlog.num_activities, len(unique_roles))
    examples_val = vectorize_camargo(eventlog.df_val, eventlog.num_activities, len(unique_roles))

    model, path = training_model_camargo(examples_train, ac_weights, rl_weights, examples_val, logger)
    model = load_model(path)

    examples_test = sample_next_event_camargo(eventlog.df_test, eventlog.num_activities, len(unique_roles))
    preds = predict_next_event_camargo(model, examples_test)

    acc = accuracy_score(examples_test['next_evt'][DataFrameFields.ACTIVITY_COLUMN], preds)
    f1 = f1_score(examples_test['next_evt'][DataFrameFields.ACTIVITY_COLUMN], preds, average="weighted")

    return acc, f1


def mauro(eventlog, logger):
    set_device(logger)

    eventlog.df_train = get_mauro_cases_times(eventlog.df_train)
    eventlog.df_val = get_mauro_cases_times(eventlog.df_val)
    eventlog.df_test = get_mauro_cases_times(eventlog.df_test)

    (X_a_train, X_t_train), (y_a_train, y_t_train), divisor, max_len = load_data(eventlog.df_train)
    (X_a_val, X_t_val), (y_a_val, y_t_val), _, _ = load_data(eventlog.df_val, divisor, max_len)
    (X_a_test, X_t_test), (y_a_test, y_t_test), _, _ = load_data(eventlog.df_test, divisor, max_len)

    y_a_train = to_categorical(y_a_train, num_classes=eventlog.num_activities)
    y_a_val = to_categorical(y_a_val, num_classes=eventlog.num_activities)
    y_a_test = to_categorical(y_a_test, num_classes=eventlog.num_activities)

    model = get_model(input_length=max_len, emb_size=16,
                      n_classes=eventlog.num_activities, n_modules=3,
                      learning_rate=Config.LEARNING_RATE)
    early_stopping = EarlyStopping(monitor='val_loss', patience=20)
    path = Config.MODEL_PATH + '/' + eventlog.filename + '/prediction/' + str(AuthorModel.MAURO)
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    checkpoint_filepath = path + '/' + 'ORIGINAL' + 'embsize' + str(0) + '_winsize' + str(0) + '.m'
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath, save_weights_only=True, monitor='val_accuracy', mode='max')

    if logger.print_mode == PrintMode.CONSOLE or \
            logger.print_mode == PrintMode.CONSOLE_AND_FILE:
        verbose = 2
    else:
        verbose = 0
    model.fit([X_a_train, X_t_train], y_a_train, epochs=Config.EPOCHS, verbose=verbose,
              validation_data=([X_a_val, X_t_val], y_a_val),
              callbacks=[early_stopping, model_checkpoint_callback], batch_size=32)

    model.load_weights(checkpoint_filepath)
    preds_a = model.predict([X_a_test, X_t_test], verbose=0)
    test_loss, test_acc = model.evaluate([X_a_test, X_t_test], y_a_test, verbose=0)
    y_a_test_max = np.argmax(y_a_test, axis=1)
    preds_a_max = np.argmax(preds_a, axis=1)
    f1 = f1_score(y_a_test_max, preds_a_max, average='weighted')

    return test_acc, f1


def get_device(gpu=True):
    if gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    return device


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

        gr = gr.drop(columns=[DataFrameFields.TIMESTAMP_COLUMN])

        data_augment = pd.concat([data_augment, gr])

    return data_augment


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


def vectorize_fold_evermann(data, num_activities):
    vectorized_log = []

    data_group = data.groupby(DataFrameFields.CASE_COLUMN)
    for name, gr in data_group:
        trace_ids = gr[DataFrameFields.ACTIVITY_COLUMN].tolist()
        # Add end-Of-Case
        trace_ids.append(num_activities)

        vectorized_log.append(trace_ids)
    return vectorized_log


def split_input_target_evermann(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


def to_dataset_evermann(vectorized_log, seq_len, batch_size):
    vectorized_log = np.array(list(itertools.chain(*vectorized_log)))
    # List of ids
    char_dataset = tf.data.Dataset.from_tensor_slices(vectorized_log)
    # Compact them on a list of traces of length seq_length+1
    sequences = char_dataset.batch(seq_len + 1, drop_remainder=True)

    dataset = sequences.map(split_input_target_evermann)

    dataset = dataset.shuffle(10000).batch(batch_size, drop_remainder=True)
    return dataset


def build_model_evermann(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(rnn_units,
                             return_sequences=True,
                             stateful=True,
                             recurrent_initializer='glorot_uniform',
                             dropout=0.2),
        tf.keras.layers.Dense(vocab_size),
    ])
    return model


def loss_evermann(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


def get_resource_table(eventlog, sim_threshold=0.7):
    data = pd.concat([eventlog.df_train[[DataFrameFields.ACTIVITY_COLUMN, DataFrameFields.RESOURCE_COLUMN]],
                      eventlog.df_val[[DataFrameFields.ACTIVITY_COLUMN, DataFrameFields.RESOURCE_COLUMN]],
                      eventlog.df_test[[DataFrameFields.ACTIVITY_COLUMN, DataFrameFields.RESOURCE_COLUMN]]])

    tasks = {val: i for i, val in enumerate(data[DataFrameFields.ACTIVITY_COLUMN].unique())}
    users = {val: i for i, val in enumerate(data[DataFrameFields.RESOURCE_COLUMN].unique())}
    associations = lambda x: (tasks[x[DataFrameFields.ACTIVITY_COLUMN]],
                              users[x[DataFrameFields.RESOURCE_COLUMN]])
    data['ac_rl'] = data.apply(associations, axis=1)

    freq_matrix = (data.groupby(by='ac_rl')[DataFrameFields.ACTIVITY_COLUMN]
                   .count().reset_index()
                   .rename(columns={DataFrameFields.ACTIVITY_COLUMN: 'freq'}))
    freq_matrix = {x['ac_rl']: x['freq'] for x in freq_matrix.to_dict('records')}

    profiles = list()
    for user, idx in users.items():
        profile = [0, ] * len(tasks)
        for ac_rl, freq in freq_matrix.items():
            if idx == ac_rl[1]:
                profile[ac_rl[0]] = freq
        profiles.append({'user': idx, 'profile': profile})

    correl_matrix = list()
    for profile_x in profiles:
        for profile_y in profiles:
            x = np.array(profile_x['profile'])
            y = np.array(profile_y['profile'])
            r_row, p_value = pearsonr(x, y)
            correl_matrix.append(({'x': profile_x['user'],
                                   'y': profile_y['user'],
                                   'distance': r_row}))

    g = nx.Graph()
    for user in users.values():
        g.add_node(user)
    for rel in correl_matrix:
        # creation of edges between nodes excluding the same elements
        # and those below the similarity threshold
        if rel['distance'] > sim_threshold and rel['x'] != rel['y']:
            g.add_edge(rel['x'],
                       rel['y'],
                       weight=rel['distance'])

    sub_graphs = list(nx.connected_components(g))

    user_index = {v: k for k, v in users.items()}
    records = list()
    for i in range(0, len(sub_graphs)):
        users_names = [user_index[x] for x in sub_graphs[i]]
        records.append({'role': i,
                        'quantity': len(sub_graphs[i]),
                        'members': users_names})
    # Sort roles by number of resources
    records = sorted(records, key=itemgetter('quantity'), reverse=True)
    for i in range(0, len(records)):
        records[i]['role'] = i
    resource_table = list()
    for record in records:
        for member in record['members']:
            resource_table.append({'role': record['role'],
                                   'resource': member})

    return resource_table


def train_embedded_camargo(log_df, num_activities, num_roles, dim_number):
    pairs = list()
    for i in range(0, len(log_df)):
        pairs.append((log_df.iloc[i][DataFrameFields.ACTIVITY_COLUMN], log_df.iloc[i]['role']))

    # Both inputs are 1-dimensional
    activity = Input(name='activity', shape=[1])
    role = Input(name='role', shape=[1])

    # Embedding the activity (shape will be (None, 1, embedding_size))
    activity_embedding = Embedding(name='activity_embedding',
                                    input_dim=num_activities+2,
                                    output_dim=dim_number)(activity)

    # Embedding the role (shape will be (None, 1, embedding_size))
    role_embedding = Embedding(name='role_embedding',
                               input_dim=num_roles+2,
                               output_dim=dim_number)(role)

    # Merge the layers with a dot product along the second axis (shape will be (None, 1, 1))
    merged = Dot(name='dot_product', normalize=True, axes=2)([activity_embedding, role_embedding])

    # Reshape to be a single number (shape will be (None, 1))
    merged = Reshape(target_shape=[1])(merged)

    # Loss function is mean squared error
    model = Model(inputs=[activity, role], outputs=merged)
    model.compile(optimizer='Adam', loss='mse')

    n_positive = 1024
    gen = generate_bacth_embs_camargo(pairs, n_positive, 2, num_activities, num_roles)
    if logger.print_mode == PrintMode.CONSOLE or \
            logger.print_mode == PrintMode.CONSOLE_AND_FILE:
        verbose = 2
    else:
        verbose = 0
    model.fit_generator(gen, epochs=100,
                        steps_per_epoch=len(pairs) // n_positive,
                        verbose=verbose)

    ac_layer = model.get_layer('activity_embedding')
    rl_layer = model.get_layer('role_embedding')

    ac_weights = ac_layer.get_weights()[0]
    rl_weights = rl_layer.get_weights()[0]

    return ac_weights, rl_weights


def generate_bacth_embs_camargo(pairs, n_positive, negative_ratio, num_activities, num_roles):
    batch_size = n_positive * (1 + negative_ratio)
    batch = np.zeros((batch_size, 3))
    pairs_set = set(pairs)

    # This creates a generator
    while True:
        # randomly choose positive examples
        idx = 0
        for idx, (activity, role) in enumerate(random.sample(pairs, n_positive)):
            batch[idx, :] = (activity, role, 1)
        # Increment idx by 1
        idx += 1

        # Add negative examples until reach batch size
        while idx < batch_size:
            # random selection
            random_ac = random.randrange(num_activities)
            random_rl = random.randrange(num_roles)

            # Check to make sure this is not a positive example
            if (random_ac, random_rl) not in pairs_set:
                # Add to batch and increment index, label 0 due classification task
                batch[idx, :] = (random_ac, random_rl, 0)
                idx += 1

        # Make sure to shuffle order
        np.random.shuffle(batch)
        yield {'activity': batch[:, 0], 'role': batch[:, 1]}, batch[:, 2]


def prepare_data_camargo(log):
    data = log.to_dict('records')
    new_data = list()
    data = sorted(data, key=lambda x: x[DataFrameFields.CASE_COLUMN])
    # Camargo do this to add Start and End events, but then he removes these rows
    # TODO: Maybe it is possible to remove this part of the code
    for key, group in itertools.groupby(data, key=lambda x: x[DataFrameFields.CASE_COLUMN]):
        trace = list(group)
        for new_event in ['Start', 'End']:
            idx = 0 if new_event == 'Start' else -1
            temp_event = dict()
            temp_event[DataFrameFields.CASE_COLUMN] = trace[idx][DataFrameFields.CASE_COLUMN]
            temp_event[DataFrameFields.ACTIVITY_COLUMN] = new_event
            temp_event[DataFrameFields.RESOURCE_COLUMN] = new_event
            temp_event['role'] = new_event
            temp_event[DataFrameFields.TIMESTAMP_COLUMN] = trace[idx][DataFrameFields.TIMESTAMP_COLUMN]
            if new_event == 'Start':
                trace.insert(0, temp_event)
            else:
                trace.append(temp_event)
        new_data.extend(trace)
    data = new_data

    log_df = pd.DataFrame(data)
    log_df = log_df[~log_df[DataFrameFields.ACTIVITY_COLUMN].isin(['Start', 'End'])]
    log_df = log_df.reset_index(drop=True)
    return log_df


def add_calculated_times_camargo(log):
    log['dur'] = 0
    log = log.to_dict('records')
    log = sorted(log, key=lambda x: x[DataFrameFields.CASE_COLUMN])
    for _, group in itertools.groupby(log, key=lambda x: x[DataFrameFields.CASE_COLUMN]):
        events = list(group)
        events = sorted(events, key=itemgetter(DataFrameFields.TIMESTAMP_COLUMN))
        for i in range(0, len(events)):
            if i == 0:
                events[i]['dur'] = 0
            else:
                dur = (events[i][DataFrameFields.TIMESTAMP_COLUMN] -
                       events[i-1][DataFrameFields.TIMESTAMP_COLUMN]).total_seconds()
                events[i]['dur'] = dur
    return pd.DataFrame.from_dict(log)


def vectorize_camargo(log, num_activities, num_roles):
    # Normalize times with lognorm
    log['dur_log'] = np.log1p(log['dur'])
    max_value = np.max(log['dur'])
    log['dur_norm'] = np.divide(log['dur_log'], max_value) if max_value > 0 else 0
    log = log.drop(('dur_log'), axis=1)

    vec = {'prefixes': dict(),
           'next_evt': dict(),
           'max_dur': np.max(log.dur)}
    log = reformat_events(log, num_activities, num_roles)
    # n-gram definition
    for i, _ in enumerate(log):
        for x in [DataFrameFields.ACTIVITY_COLUMN, 'role', 'dur_norm']:
            serie = list(ngrams(log[i][x], 5, pad_left=True, left_pad_symbol=0))
            y_serie = [x[-1] for x in serie]
            serie = serie[:-1]
            y_serie = y_serie[1:]
            vec['prefixes'][x] = vec['prefixes'][x] + serie if i > 0 else serie
            vec['next_evt'][x] = vec['next_evt'][x] + y_serie if i > 0 else y_serie
    # Transform task, dur and role prefixes in vectors
    for value in [DataFrameFields.ACTIVITY_COLUMN, 'role', 'dur_norm']:
        vec['prefixes'][value] = np.array(vec['prefixes'][value])
        vec['next_evt'][value] = np.array(vec['next_evt'][value])
    # Reshape dur (prefixes, n-gram size, 1) i.e. time distribute
    vec['prefixes']['dur_norm'] = vec['prefixes']['dur_norm'].reshape(
        (vec['prefixes']['dur_norm'].shape[0],
         vec['prefixes']['dur_norm'].shape[1], 1))
    # one-hot encode target values
    vec['next_evt'][DataFrameFields.ACTIVITY_COLUMN] = ku.to_categorical(
        vec['next_evt'][DataFrameFields.ACTIVITY_COLUMN], num_classes=num_activities+2)
    vec['next_evt']['role'] = ku.to_categorical(
        vec['next_evt']['role'], num_classes=num_roles+2)

    return vec


def reformat_events(log, num_activities, num_roles):
    temp_data = list()
    log_df = log.to_dict('records')
    log_df = sorted(log_df, key=lambda x: (x[DataFrameFields.CASE_COLUMN],
                                           DataFrameFields.TIMESTAMP_COLUMN))
    for key, group in itertools.groupby(log_df, key=lambda x: x[DataFrameFields.CASE_COLUMN]):
        trace = list(group)
        temp_dict = dict()
        for x in [DataFrameFields.ACTIVITY_COLUMN, 'role', 'dur_norm']:
            serie = [y[x] for y in trace]
            if x == DataFrameFields.ACTIVITY_COLUMN:
                serie.insert(0, num_activities)
                serie.append(num_activities+1)
            elif x == 'role':
                serie.insert(0, num_roles)
                serie.append(num_roles+1)
            else:
                serie.insert(0, 0)
                serie.append(0)
            temp_dict = {**{x: serie}, **temp_dict}
        temp_dict = {**{DataFrameFields.CASE_COLUMN: key}, **temp_dict}
        temp_data.append(temp_dict)

    return temp_data


def training_model_camargo(vec, ac_weights, rl_weights, vec_val, logger):

    ac_input = Input(shape=(vec['prefixes'][DataFrameFields.ACTIVITY_COLUMN].shape[1],), name='ac_input')
    rl_input = Input(shape=(vec['prefixes']['role'].shape[1],), name='rl_input')
    t_input = Input(shape=(vec['prefixes']['dur_norm'].shape[1], 1), name='t_input')

    ac_embedding = Embedding(ac_weights.shape[0],
                             ac_weights.shape[1],
                             weights=[ac_weights],
                             input_length=vec['prefixes'][DataFrameFields.ACTIVITY_COLUMN].shape[1],
                             trainable=False, name='ac_embedding')(ac_input)

    rl_embedding = Embedding(rl_weights.shape[0],
                             rl_weights.shape[1],
                             weights=[rl_weights],
                             input_length=vec['prefixes']['role'].shape[1],
                             trainable=False, name='rl_embedding')(rl_input)

    merged = Concatenate(name='concatenated', axis=2)([ac_embedding, rl_embedding])

    l1_c1 = LSTM(100,
                 kernel_initializer='glorot_uniform',
                 return_sequences=True,
                 dropout=0.2,
                 implementation=1)(merged)

    l1_c3 = LSTM(100,
                 activation='sigmoid',
                 kernel_initializer='glorot_uniform',
                 return_sequences=True,
                 dropout=0.2,
                 implementation=1)(t_input)

    batch1 = BatchNormalization()(l1_c1)
    batch3 = BatchNormalization()(l1_c3)

    l2_c1 = LSTM(100,
                 kernel_initializer='glorot_uniform',
                 return_sequences=False,
                 dropout=0.2,
                 implementation=1)(batch1)

    l2_c2 = LSTM(100,
                 kernel_initializer='glorot_uniform',
                 return_sequences=False,
                 dropout=0.2,
                 implementation=1)(batch1)

    l2_3 = LSTM(100,
                activation='sigmoid',
                kernel_initializer='glorot_uniform',
                return_sequences=False,
                dropout=0.2,
                implementation=1)(batch3)

    act_output = Dense(ac_weights.shape[0],
                       activation='softmax',
                       kernel_initializer='glorot_uniform',
                       name='act_output')(l2_c1)

    role_output = Dense(rl_weights.shape[0],
                        activation='softmax',
                        kernel_initializer='glorot_uniform',
                        name='role_output')(l2_c2)

    time_output = Dense(1, activation='sigmoid',
                        kernel_initializer='glorot_uniform',
                        name='time_output')(l2_3)

    model = Model(inputs=[ac_input, rl_input, t_input],
                  outputs=[act_output, role_output, time_output])

    opt = Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999)

    model.compile(loss={'act_output': 'categorical_crossentropy',
                        'role_output': 'categorical_crossentropy',
                        'time_output': 'mae'}, optimizer=opt,
                  metrics={"act_output": "acc", "role_output": "acc",
                           "time_output": "mae"})

    early_stopping = EarlyStopping(monitor='val_loss', patience=50)
    output_folder = Config.MODEL_PATH + '/' + eventlog.filename + '/prediction/' + str(AuthorModel.CAMARGO)
    pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)

    name = 'ORIGINAL' + 'embsize' + str(0) + '_winsize' + str(0) + '.m'
    output_file_path = os.path.join(output_folder, name)

    model_checkpoint = ModelCheckpoint(output_file_path,
                                       monitor='val_loss',
                                       verbose=0,
                                       save_best_only=True,
                                       save_weights_only=False,
                                       mode='auto')
    lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.5,
                                   patience=10,
                                   verbose=0,
                                   mode='auto',
                                   min_delta=0.0001,
                                   cooldown=0,
                                   min_lr=0)

    batch_size = 32
    if logger.print_mode == PrintMode.CONSOLE or \
            logger.print_mode == PrintMode.CONSOLE_AND_FILE:
        verbose = 2
    else:
        verbose = 0
    history = model.fit({'ac_input': vec['prefixes'][DataFrameFields.ACTIVITY_COLUMN],
                         'rl_input': vec['prefixes']['role'],
                         't_input': vec['prefixes']['dur_norm']},
                        {'act_output': vec['next_evt'][DataFrameFields.ACTIVITY_COLUMN],
                         'role_output': vec['next_evt']['role'],
                         'time_output': vec['next_evt']['dur_norm']},
                        validation_data=({'ac_input': vec_val['prefixes'][DataFrameFields.ACTIVITY_COLUMN],
                                          'rl_input': vec_val['prefixes']['role'],
                                          't_input': vec_val['prefixes']['dur_norm']},
                                         {'act_output': vec_val['next_evt'][DataFrameFields.ACTIVITY_COLUMN],
                                          'role_output': vec_val['next_evt']['role'],
                                          'time_output': vec_val['next_evt']['dur_norm']}),
                        verbose=verbose,
                        callbacks=[early_stopping, model_checkpoint, lr_reducer],
                        batch_size=batch_size,
                        epochs=100)

    return model, output_file_path


def sample_next_event_camargo(log, num_activities, num_roles):
    # Normalize times with lognorm
    log['dur_log'] = np.log1p(log['dur'])
    max_value = np.max(log['dur'])
    log['dur_norm'] = np.divide(log['dur_log'], max_value) if max_value > 0 else 0
    log = log.drop(('dur_log'), axis=1)

    log = reformat_events(log, num_activities, num_roles)
    examples = {'prefixes': dict(), 'next_evt': dict()}
    for i, _ in enumerate(log):
        for x in [DataFrameFields.ACTIVITY_COLUMN, 'role', 'dur_norm']:
            serie = [log[i][x][:idx] for idx in range(1, len(log[i][x]))]
            y_serie = [x[-1] for x in serie]
            serie = serie[:-1]
            y_serie = y_serie[1:]
            examples['prefixes'][x] = (examples['prefixes'][x] + serie if i > 0 else serie)
            examples['next_evt'][x] = (examples['next_evt'][x] + y_serie if i > 0 else y_serie)

    return examples


def predict_next_event_camargo(model, examples):
    acts_preds = []
    for i, _ in enumerate(examples['prefixes'][DataFrameFields.ACTIVITY_COLUMN]):
        x_ac_ngram = np.append(
            np.zeros(5),
            np.array(examples['prefixes'][DataFrameFields.ACTIVITY_COLUMN][i]), axis=0)[-5:].reshape(
            (1, 5))
        x_rl_ngram = np.append(
            np.zeros(5),
            np.array(examples['prefixes']['role'][i]),
            axis=0)[-5:].reshape((1, 5))
        x_t_ngram = np.array([np.append(
            np.zeros(5),
            np.array(examples['prefixes']['dur_norm'][i]),
            axis=0)[-5:].reshape((5, 1))])

        inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram]
        predictions = model.predict(inputs, verbose=0)
        pos = np.argmax(predictions[0][0])
        pos1 = np.argmax(predictions[1][0])
        acts_preds.append(pos)

    return acts_preds


def calculate_correct_preds(pred, real):
    pred = torch.softmax(pred, dim=1)

    pred_indices = torch.max(pred, 1).indices
    return pred_indices == real, pred_indices.tolist()


def save_model(model, author, filename):
    path = Config.MODEL_PATH + '/' + filename + '/prediction/' + author
    name = 'ORIGINAL' + 'embsize' + str(0) + '_winsize' + str(0) + '.m'

    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    torch.save(model, path + '/' + name)


def load_best_model(author, filename):
    path = Config.MODEL_PATH + '/' + filename + '/prediction/' + author
    name = 'ORIGINAL' + 'embsize' + str(0) + '_winsize' + str(0) + '.m'

    model = torch.load(path + '/' + name)
    return model


def set_device(logger):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            logger.print_error('No GPU device found')


def get_mauro_cases_times(data):
    cases_times = pd.DataFrame()

    cases = data.groupby(DataFrameFields.CASE_COLUMN)
    for _, case in cases:
        case = case.reset_index(drop=True)

        timesincelastevent = case.loc[:, DataFrameFields.TIMESTAMP_COLUMN].diff() / np.timedelta64(1, 's')
        timesincelastevent.iloc[0] = 0.0

        case[DataFrameFields.TIMESTAMP_COLUMN] = timesincelastevent

        cases_times = pd.concat([cases_times, case])

    return cases_times


def load_data(data, divisor=None, max_len=None):
    if divisor is None:
        divisor = np.mean(data[DataFrameFields.TIMESTAMP_COLUMN])

    new_max_len = 0

    X_a = []
    X_t = []
    y_a = []
    y_t = []

    prefixes_acts = []
    prefixes_times = []
    cases = data.groupby(DataFrameFields.CASE_COLUMN)
    for _, case in cases:
        case = case.reset_index(drop=True)

        for i in range(1, len(case)):
            prefixes_acts.append(case[DataFrameFields.ACTIVITY_COLUMN][0:i].tolist())
            prefixes_times.append(np.log(case[DataFrameFields.TIMESTAMP_COLUMN][0:i] + 1).tolist())

            y_a.append(case[DataFrameFields.ACTIVITY_COLUMN][i])
            y_t.append(case[DataFrameFields.TIMESTAMP_COLUMN][i] / divisor)

            if len(prefixes_acts[-1]) > new_max_len:
                new_max_len = len(prefixes_acts[-1])

    X_a = np.array(prefixes_acts, dtype=object)
    X_t = np.array(prefixes_times, dtype=object)
    y_a = np.array(y_a)
    y_t = np.array(y_t)

    if max_len is None:
        max_len = new_max_len

    X_a = pad_sequences(X_a, maxlen=max_len, padding='pre', truncating='pre', dtype='float64')
    X_t = pad_sequences(X_t, maxlen=max_len, padding='pre', truncating='pre', dtype='float64')

    return (X_a, X_t), (y_a, y_t), divisor, max_len


def get_model(input_length: int, emb_size: int, n_classes: int, n_filters=3, n_modules=5, learning_rate=0.002):
    inputs = []
    for i in range(2):
        inputs.append(Input(shape=(input_length,)))

    inputs_ = []
    inputs_.append(Embedding(n_classes, emb_size, input_length=input_length)(inputs[0]))
    inputs_.append(Reshape((input_length, 1))(inputs[1]))

    filters_inputs = Concatenate(axis=2)(inputs_)

    for m in range(n_modules):
        filters = []
        for i in range(n_filters):
            filters.append(
                Conv1D(filters=32, strides=1, kernel_size=1 + i, activation='relu', padding='same')(filters_inputs))
        filters.append(MaxPooling1D(pool_size=3, strides=1, padding='same')(filters_inputs))
        filters_inputs = Concatenate(axis=2)(filters)

    pool = GlobalMaxPooling1D()(filters_inputs)

    out = Dense(n_classes, activation='softmax')(pool)

    optimizer = Adam(learning_rate=learning_rate)

    model = Model(inputs=inputs, outputs=out)
    model.compile(optimizer=optimizer, loss='mse', metrics=['acc'])
    # model.summary()

    return model


#####################################################################################################

if __name__ == "__main__":
    args = read_arguments()

    logger = NextActPredLogger(args.print_mode)

    if args.crossvalidation:
        accuracies = []
        f1_scores = []

        for i in range(Config.CV_FOLDS):
            eventlog = EventlogDataset(args.dataset, cv_fold=i, read_test=True)

            acc_fold, f1_fold = start(eventlog, args.author, logger)

            logger.log_console(f'{str(args.author)} TEST FOLD {i}:\n'
                               f'\tACCURACY: {acc_fold} \t | F1_SCORE: {f1_fold}')

            accuracies.append(acc_fold)
            f1_scores.append(f1_fold)
        logger.log_metrics_cv(Path(args.dataset).stem, str(args.author),
                              "ORIGINAL", 0, 0, accuracies, f1_scores)

    else:
        eventlog = EventlogDataset(args.dataset, read_test=True)

        accuracy, f1_score = start(eventlog, args.author, logger)
        logger.log_console(f'TEST ACCURACY: {accuracy} \t | F1_SCORE: {f1_score}')
        logger.log_metrics_holdout(Path(args.dataset).stem, str(args.author),
                                   "ORIGINAL", 0, 0, accuracy, f1_score)
