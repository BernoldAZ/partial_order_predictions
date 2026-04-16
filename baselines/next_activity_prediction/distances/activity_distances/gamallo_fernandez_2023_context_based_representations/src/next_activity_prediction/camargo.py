from pathlib import Path
from pandas_dataset import EventlogDataset
from utils import PrintMode, AuthorModel, Config, DataFrameFields, EmbType
from args_reader import NextActPredInputArgs
from logger import NextActPredLogger
from scipy.stats import pearsonr
import networkx as nx
import pandas as pd
import numpy as np
from operator import itemgetter
import itertools
from nltk.util import ngrams
from sklearn.metrics import f1_score
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


def execution(eventlog: EventlogDataset, embeddings_acts_list: dict,
              embeddings_rol_list: dict, logger: NextActPredLogger,
              emb_type: EmbType, emb_size: int, win_size: int) -> (float, float):
    """
    Generate, train and validate the Camargo model for next activity prediction
    :param eventlog: EventlogDataset with all data and information about
    the dataset
    :param embeddings_acts_list: List of activity embeddings
    :param embeddings_rol_list: List of rol embeddings
    :param logger: NextActPredLogger to print the output
    :param emb_type: Type of embeddings used
    :param emb_size: Size of the embeddings used
    :param win_size: Size of the context window used to generate the embeddings
    :return: the accuracy and the f1-score in the test set
    """

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
    unique_roles = list(log_df_resources[DataFrameFields.ROLE_COLUMN].unique())

    eventlog.df_train = prepare_data_camargo(eventlog.df_train)
    eventlog.df_val = prepare_data_camargo(eventlog.df_val)
    eventlog.df_test = prepare_data_camargo(eventlog.df_test)

    eventlog.df_train = add_calculated_times_camargo(eventlog.df_train)
    eventlog.df_val = add_calculated_times_camargo(eventlog.df_val)
    eventlog.df_test = add_calculated_times_camargo(eventlog.df_test)

    examples_train = vectorize_camargo(eventlog.df_train, eventlog.num_activities, len(unique_roles))
    examples_val = vectorize_camargo(eventlog.df_val, eventlog.num_activities, len(unique_roles))

    model, path = training_model_camargo(examples_train, np.array(list(embeddings_acts_list.values())),
                                         np.array(list(embeddings_rol_list.values())), examples_val, logger,
                                         eventlog.filename, emb_type, emb_size, win_size)
    model = load_model(path)

    examples_test = sample_next_event_camargo(eventlog.df_test, eventlog.num_activities, len(unique_roles))
    preds = predict_next_event_camargo(model, examples_test)

    acc = accuracy_score(examples_test['next_evt'][DataFrameFields.ACTIVITY_COLUMN], preds)
    f1 = f1_score(examples_test['next_evt'][DataFrameFields.ACTIVITY_COLUMN], preds, average="weighted")

    return acc, f1


def set_device(logger):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            logger.print_error('No GPU device found')


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
        records.append({DataFrameFields.ROLE_COLUMN: i,
                        'quantity': len(sub_graphs[i]),
                        'members': users_names})
    # Sort roles by number of resources
    records = sorted(records, key=itemgetter('quantity'), reverse=True)
    for i in range(0, len(records)):
        records[i]['role'] = i
    resource_table = list()
    for record in records:
        for member in record['members']:
            resource_table.append({DataFrameFields.ROLE_COLUMN: record['role'],
                                   'resource': member})

    return resource_table


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
            temp_event[DataFrameFields.ROLE_COLUMN] = new_event
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
        for x in [DataFrameFields.ACTIVITY_COLUMN, DataFrameFields.ROLE_COLUMN, 'dur_norm']:
            serie = list(ngrams(log[i][x], 5, pad_left=True, left_pad_symbol=0))
            y_serie = [x[-1] for x in serie]
            serie = serie[:-1]
            y_serie = y_serie[1:]
            vec['prefixes'][x] = vec['prefixes'][x] + serie if i > 0 else serie
            vec['next_evt'][x] = vec['next_evt'][x] + y_serie if i > 0 else y_serie
    # Transform task, dur and role prefixes in vectors
    for value in [DataFrameFields.ACTIVITY_COLUMN, DataFrameFields.ROLE_COLUMN, 'dur_norm']:
        vec['prefixes'][value] = np.array(vec['prefixes'][value])
        vec['next_evt'][value] = np.array(vec['next_evt'][value])
    # Reshape dur (prefixes, n-gram size, 1) i.e. time distribute
    vec['prefixes']['dur_norm'] = vec['prefixes']['dur_norm'].reshape(
        (vec['prefixes']['dur_norm'].shape[0],
         vec['prefixes']['dur_norm'].shape[1], 1))
    # one-hot encode target values
    vec['next_evt'][DataFrameFields.ACTIVITY_COLUMN] = ku.to_categorical(
        vec['next_evt'][DataFrameFields.ACTIVITY_COLUMN], num_classes=num_activities+2)
    vec['next_evt'][DataFrameFields.ROLE_COLUMN] = ku.to_categorical(
        vec['next_evt'][DataFrameFields.ROLE_COLUMN], num_classes=num_roles+2)

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


def training_model_camargo(vec, ac_weights, rl_weights, vec_val, logger,
                           filename, emb_type, emb_size, win_size):

    ac_emb_size = ac_weights.shape[1]
    ac_weights = np.append(ac_weights, [np.random.rand(ac_emb_size)], axis=0)  # Start event
    ac_weights = np.append(ac_weights, [np.random.rand(ac_emb_size)], axis=0)  # End event

    rl_emb_size = rl_weights.shape[1]
    rl_weights = np.append(rl_weights, [np.random.rand(rl_emb_size)], axis=0)  # Start event
    rl_weights = np.append(rl_weights, [np.random.rand(rl_emb_size)], axis=0)  # End event


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
    output_folder = Config.MODEL_PATH + '/' + filename + '/prediction/' + str(AuthorModel.CAMARGO)
    pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)

    name = str(emb_type) + 'embsize' + str(emb_size) + '_winsize' + str(win_size) + '.m'
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
        for x in [DataFrameFields.ACTIVITY_COLUMN, DataFrameFields.ROLE_COLUMN, 'dur_norm']:
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
