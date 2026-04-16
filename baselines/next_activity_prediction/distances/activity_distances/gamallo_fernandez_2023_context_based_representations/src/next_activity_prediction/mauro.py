import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
import pathlib
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Conv1D, GlobalAveragePooling1D, \
    GlobalMaxPooling1D, Reshape, MaxPooling1D, Flatten, Dense, Embedding, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, brier_score_loss
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, f1_score
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pandas_dataset import EventlogDataset
from logger import NextActPredLogger
from utils import DataFrameFields, Config, AuthorModel, EmbType, PrintMode

seed = 123
np.random.seed(seed)
tf.random.set_seed(seed)


def execution(eventlog: EventlogDataset, embeddings_list: dict,
              logger: NextActPredLogger, emb_type: EmbType,
              emb_size: int, win_size: int) -> (float, float):
    """
    Generate, train and validate the Mauro model for next activity prediction
    :param eventlog: EventlogDataset with all data and information about
    the dataset
    :param embeddings_list: List of embeddings
    :param logger: NextActPredLogger to print the output
    :param emb_type: Type of embeddings used
    :param emb_size: Size of the embeddings used
    :param win_size: Size of the context window used to generate the embeddings
    :return: The accuracy and the f1-score in the test set
    """

    set_device(logger)

    eventlog.df_train = get_cases_times(eventlog.df_train)
    eventlog.df_val = get_cases_times(eventlog.df_val)
    eventlog.df_test = get_cases_times(eventlog.df_test)

    (X_a_train, X_t_train), (y_a_train, y_t_train), divisor, max_len = load_data(eventlog.df_train,
                                                                                 embeddings_list)
    (X_a_val, X_t_val), (y_a_val, y_t_val), _, _ = load_data(eventlog.df_val, embeddings_list,
                                                             divisor, max_len)
    (X_a_test, X_t_test), (y_a_test, y_t_test), _, _ = load_data(eventlog.df_test, embeddings_list,
                                                                 divisor, max_len)

    y_a_train = to_categorical(y_a_train, num_classes=eventlog.num_activities)
    y_a_val = to_categorical(y_a_val, num_classes=eventlog.num_activities)
    y_a_test = to_categorical(y_a_test, num_classes=eventlog.num_activities)

    model = get_model(input_length=max_len, emb_size=emb_size,
                      n_classes=eventlog.num_activities, n_modules=3,
                      learning_rate=Config.LEARNING_RATE)
    early_stopping = EarlyStopping(monitor='val_loss', patience=20)
    path = Config.MODEL_PATH + '/' + eventlog.filename + '/prediction/' + str(AuthorModel.MAURO)
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    checkpoint_filepath = path + '/' + str(emb_type) + 'embsize' + str(emb_size) + '_winsize' + str(win_size) + '.m'
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


def set_device(logger: NextActPredLogger):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            logger.print_error('No GPU device found')


def get_cases_times(data: pd.DataFrame) -> pd.DataFrame:
    cases_times = pd.DataFrame()

    cases = data.groupby(DataFrameFields.CASE_COLUMN)
    for _, case in cases:
        case = case.reset_index(drop=True)

        timesincelastevent = case.loc[:, DataFrameFields.TIMESTAMP_COLUMN].diff() / np.timedelta64(1, 's')
        timesincelastevent.iloc[0] = 0.0

        case[DataFrameFields.TIMESTAMP_COLUMN] = timesincelastevent

        cases_times = pd.concat([cases_times, case])

    return cases_times


def load_data(data: pd.DataFrame, embeddings_list: dict,
              divisor: int = None, max_len: int = None) -> ((np.array, np.array),
                                                            (np.array, np.array),
                                                            int, int):
    """
    Generate prefixes from the cases and the next activity to be predicted
    :param data: Pandas DataFrame with the cases and events
    :param embeddings_list: Dictionary with the list of embeddings
    :param divisor: Value to normalize times
    :param max_len: Maximum size of the prefixes
    :return: Activity and time prefixes, next activities and next times,
    and the divisor to normalize the time
    """

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

    for prefix_acts, prefix_times in zip(prefixes_acts, prefixes_times):
        encoded_prefix_act = []
        encoded_prefix_time = []
        for act, time in zip(prefix_acts, prefix_times):
            emb_act = np.array(embeddings_list[act])
            encoded_prefix_act.append(emb_act)

            encoded_prefix_time.append([time])
        X_a.append(np.array(encoded_prefix_act))
        X_t.append(np.array(encoded_prefix_time))

    X_a = np.array(X_a, dtype=object)
    X_t = np.array(X_t, dtype=object)
    y_a = np.array(y_a)
    y_t = np.array(y_t)

    if max_len is None:
        max_len = new_max_len

    X_a = pad_sequences(X_a, maxlen=max_len, padding='pre', truncating='pre', dtype='float64')
    X_t = pad_sequences(X_t, maxlen=max_len, padding='pre', truncating='pre', dtype='float64')

    return (X_a, X_t), (y_a, y_t), divisor, max_len


def get_model(input_length: int, emb_size: int, n_classes: int, n_filters=3, n_modules=5, learning_rate=0.002):
    inputs = []
    inputs.append(Input(shape=(input_length, emb_size)))
    inputs.append(Input(shape=(input_length, 1)))

    filters_inputs = Concatenate(axis=2)(inputs)

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
