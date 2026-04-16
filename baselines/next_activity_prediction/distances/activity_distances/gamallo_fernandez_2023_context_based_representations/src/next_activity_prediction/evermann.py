import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
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


#############################################################################
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

    seq_len = 20
    batch_size = 20
    rnn_units = 32
    maxGradNorm = 5
    lrDecay = 0.75
    lr = 1.0

    X_train_ids, X_train_embs = vectorize_fold_evermann(eventlog.df_train, eventlog.num_activities, embeddings_list)
    X_val_ids, X_val_embs = vectorize_fold_evermann(eventlog.df_val, eventlog.num_activities, embeddings_list)
    X_test_ids, X_test_embs = vectorize_fold_evermann(eventlog.df_test, eventlog.num_activities, embeddings_list)

    train_dataset = to_dataset_evermann(X_train_ids, X_train_embs, seq_len, batch_size)
    val_dataset = to_dataset_evermann(X_val_ids, X_val_embs, seq_len, batch_size)
    test_dataset = to_dataset_evermann(X_test_ids, X_test_embs, seq_len, batch_size)

    model = build_model_evermann(eventlog.num_activities + 1, emb_size, rnn_units, seq_len, batch_size)

    optimizer = tf.keras.optimizers.SGD(learning_rate=lr, decay=lrDecay, clipnorm=maxGradNorm, momentum=0.9,
                                        nesterov=True)

    path = Config.MODEL_PATH + '/' + eventlog.filename + '/prediction/' + str(AuthorModel.EVERMANN)
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    name = str(emb_type) + 'embsize' + str(emb_size) + '_winsize' + str(win_size) + '.m'
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
    model = build_model_evermann(eventlog.num_activities + 1, emb_size, rnn_units, seq_len, 1)
    model.load_weights(os.path.join(path, name))
    model.build(tf.TensorShape([1, None]))
    y_pred = []
    y_true = []
    for trace in X_test_ids:
        for i, event in enumerate(trace):
            model.reset_states()
            inp = trace[:i + 1]
            trace_embs = []
            for e in inp:
                trace_embs.append(embeddings_list[e])
            next_event = trace[i + 1]
            full_preds = model(tf.expand_dims(trace_embs, 0))
            probs = tf.nn.softmax(tf.squeeze(full_preds, 0).numpy()[-1])
            y_pred.append(probs)
            y_true.append(np.eye(eventlog.num_activities + 1)[next_event])
            if next_event == eventlog.num_activities:
                break

    y_pred_a = np.argmax(y_pred, axis=1)
    y_true_a = np.argmax(y_true, axis=1)

    acc = accuracy_score(y_true_a, y_pred_a)
    f1 = f1_score(y_true_a, y_pred_a, average="weighted")

    return acc, f1


def vectorize_fold_evermann(data, num_activities, embeddings_list):
    vectorized_log_ids = []
    vectorized_log_embs = []

    data_group = data.groupby(DataFrameFields.CASE_COLUMN)
    for name, gr in data_group:
        trace_ids = gr[DataFrameFields.ACTIVITY_COLUMN].tolist()
        # trace_embs = gr[DataFrameFields.ACTIVITY_COLUMN].tolist()
        trace_embs = []
        for id in trace_ids:
            trace_embs.append(embeddings_list[id])

        # Add end-Of-Case
        trace_ids.append(num_activities)
        # trace_embs.append(num_activities)
        emb_size = len(embeddings_list[0])
        trace_embs.append(list(np.zeros(emb_size)))

        vectorized_log_ids.append(trace_ids)
        vectorized_log_embs.append(trace_embs)
    return vectorized_log_ids, vectorized_log_embs


def split_input_target_evermann(chunk1, chunk2):
    input_text = chunk1[:-1]
    target_text = chunk2[1:]
    return input_text, target_text


def to_dataset_evermann(vectorized_log_ids, vectorized_log_embs, seq_len, batch_size):
    vectorized_log_ids = np.array(list(itertools.chain(*vectorized_log_ids)))
    vectorized_log_embs = np.array(list(itertools.chain(*vectorized_log_embs)))

    # List of ids
    char_dataset = tf.data.Dataset.from_tensor_slices((vectorized_log_embs, vectorized_log_ids))
    # Compact them on a list of traces of length seq_length+1
    sequences = char_dataset.batch(seq_len + 1, drop_remainder=True)

    dataset = sequences.map(split_input_target_evermann)

    dataset = dataset.shuffle(10000).batch(batch_size, drop_remainder=True)
    return dataset


def build_model_evermann(vocab_size, embedding_dim, rnn_units, seq_len, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(None, embedding_dim), batch_size=batch_size),
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
