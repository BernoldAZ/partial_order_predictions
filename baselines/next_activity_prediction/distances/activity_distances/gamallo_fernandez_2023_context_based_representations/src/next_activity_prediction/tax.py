import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from pandas_dataset import EventlogDataset
from logger import NextActPredLogger
from utils import PrintMode, DataFrameFields, Config, AuthorModel, EmbType
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
import pathlib
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Conv1D, GlobalAveragePooling1D, \
    GlobalMaxPooling1D, Reshape, MaxPooling1D, Flatten, Dense, Embedding, Dropout, \
    LSTM, BatchNormalization
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import f1_score, accuracy_score
from tensorflow.keras.preprocessing.sequence import pad_sequences

tf.compat.v1.set_random_seed(42)
import random
random.seed(42)
np.random.seed(42)


def execution(eventlog: EventlogDataset, embeddings_list: dict,
              logger: NextActPredLogger, emb_type: EmbType,
              emb_size: int, win_size: int) -> (float, float):
    """
    Generate, train and validate the Tax model for next activity prediction
    :param eventlog: EventlogDataset with all data and information about
    the dataset
    :param embeddings_list: List of embeddings
    :param logger: NextActPredLogger to print the output
    :param emb_type: Type of embeddings used
    :param emb_size: Size of the embeddings used
    :param win_size: Size of the context window used to generate the embeddings
    :return: The accuracy and the f1-score in the test set
    """

    eventlog.df_train = create_tax_time_variables(eventlog.df_train)
    eventlog.df_val = create_tax_time_variables(eventlog.df_val)
    eventlog.df_test = create_tax_time_variables(eventlog.df_test)

    eventlog.df_train = add_eoc(eventlog.df_train, eventlog.num_activities)
    eventlog.df_val = add_eoc(eventlog.df_val, eventlog.num_activities)
    eventlog.df_test = add_eoc(eventlog.df_test, eventlog.num_activities)

    divisor = eventlog.df_train['times1'].mean()
    divisor2 = eventlog.df_train['times2'].mean()

    X_train, y_a_train, y_t_train = vectorize_fold_tax(eventlog.df_train, divisor, divisor2, embeddings_list,
                                                       eventlog.max_len_case, eventlog.num_activities, emb_size)
    X_val, y_a_val, y_t_val = vectorize_fold_tax(eventlog.df_val, divisor, divisor2, embeddings_list,
                                                 eventlog.max_len_case, eventlog.num_activities, emb_size)
    X_test, y_a_test, y_t_test = vectorize_fold_tax(eventlog.df_test, divisor, divisor2, embeddings_list,
                                                    eventlog.max_len_case, eventlog.num_activities, emb_size)

    main_input = Input(shape=(eventlog.max_len_case, emb_size + 5), name='main_input')

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

    path = Config.MODEL_PATH + '/' + eventlog.filename + '/prediction/' + str(AuthorModel.TAX)
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    name = str(emb_type) + 'embsize' + str(emb_size) + '_winsize' + str(win_size) + '.m'
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
    metrics = model.evaluate(X_test, {'act_output': y_a_test, 'time_output': y_t_test},
                             verbose=0, batch_size=eventlog.max_len_case)
    preds = model.predict([X_test])
    y_a_pred_probs = preds[0]
    y_t_pred_probs = preds[1]
    y_a_pred = np.argmax(y_a_pred_probs, axis=1)
    y_true = np.argmax(y_a_test, axis=1)

    acc = accuracy_score(y_true, y_a_pred)
    f1 = f1_score(y_true, y_a_pred, average="weighted")

    return acc, f1


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


def add_eoc(data, num_activities):
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


def vectorize_fold_tax(data, divisor, divisor2, embeddings_list, max_len, num_activities, emb_size):
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
    X = np.zeros((len(sentences), max_len, emb_size+5), dtype=np.float32)
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
            embedding = embeddings_list[act]
            for e, el in enumerate(embedding):
                X[i, t + leftpad, e] = el
            X[i, t + leftpad, emb_size] = t + 1
            X[i, t + leftpad, emb_size + 1] = sentence_t[t] / divisor
            X[i, t + leftpad, emb_size + 2] = sentence_t2[t] / divisor2
            X[i, t + leftpad, emb_size + 3] = sentence_t3[t] / 86400
            X[i, t + leftpad, emb_size + 4] = sentence_t4[t] / 7

        y_a[i, next_chars[i]] = 1
        y_t[i] = next_t / divisor

    return X, y_a, y_t

