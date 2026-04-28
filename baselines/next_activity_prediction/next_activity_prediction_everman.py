#!/usr/bin/env python

import os, sys, copy, random, time, re
#import ollama
from datetime import datetime
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from pathlib import Path
from pm4py.objects.log.importer.xes import importer as xes_importer
from definitions import ROOT_DIR
import pandas as pd
from distances.activity_distances.pmi.pmi import \
    get_activity_context_frequency_matrix_pmi, get_activity_activity_frequency_matrix_pmi
from distances.activity_distances.bose_2009_context_aware_trace_clustering.algorithm import \
    get_substitution_and_insertion_scores
from distances.activity_distances.de_koninck_2018_act2vec.algorithm import get_act2vec_distance_matrix
from distances.activity_distances.de_koninck_2018_act2vec.our_hyperparas import get_act2vec_distance_matrix_our
from distances.activity_distances.activity_activity_co_occurence.activity_activity_co_occurrence import \
    get_activity_activity_co_occurence_matrix
from distances.activity_distances.activity_context_frequency.activity_contex_frequency import \
    get_activity_context_frequency_matrix
from distances.activity_distances.chiorrini_2022_embedding_process_structure.embedding_process_structure import \
    get_embedding_process_structure_distance_matrix
from distances.activity_distances.gamallo_fernandez_2023_context_based_representations.src.embeddings_generator.main_new import \
    get_context_based_distance_matrix

# ----------------------- Configuration Parameters -----------------------
# Set these parameters as needed.
#DATASET_FILE = "path/to/your/dataset.xes.gz"  # Path to the main dataset file
TRAIN_FLAG = True  # Set to True to train the model.
TEST_FLAG = True  # Set to True to evaluate/test the model.
TEST_SUFFIX_FLAG = False  # Set to True if you want to run suffix evaluation.
TEST_SUFFIX_CALC_FLAG = False  # Set to True to calculate suffix metrics.

# Directories (assumed to be organized as in your project structure)
NA_DIR = os.path.join(ROOT_DIR, "evaluation")
RAW_DATASETS_DIR = os.path.join(NA_DIR, "raw_datasets_that_are_not_evaluated")
SPLIT_DATASETS_DIR = os.path.join(NA_DIR, "split_datasets")
RESULTS_DIR = os.path.join(NA_DIR, "results_everman")
MODELS_DIR = os.path.join(NA_DIR, "models_everman")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

embedding_source = "train_val"  # Options: "train", "validation", "train_val"
print("Embedding source:", embedding_source)

# Training parameters
seq_length = 20
BATCH_SIZE = 20
BUFFER_SIZE = 10000
embedding_dim = 32
rnn_units = 32
dropout = 0.2
maxGradNorm = 5
lrDecay = 0.75
lr = 1.0
EPOCHS = 100

tf.compat.v1.set_random_seed(42)
random.seed(42)
np.random.seed(42)

# Pre-defined file name for saving models/results
#file_name = Path(DATASET_FILE).stem
#model_file_name = file_name + ".h5"

# ----------------------- Vectorization Functions -----------------------
def vectorize_log(log, idx, current_idx, readonly=False, unk_id=None):
    """
    Convert a log (list of traces) into a list of lists of event IDs.
    Updates idx and appends a special “[EOC]” token at the end of each trace.
    If readonly=True, unseen activities are mapped to unk_id instead of extending idx.
    """
    vectorized_log = []
    for trace in log:
        trace_ids = []
        for event in trace:
            act = event["concept:name"]
            if act not in idx:
                if readonly:
                    trace_ids.append(unk_id)
                    continue
                idx[act] = current_idx
                current_idx += 1
            trace_ids.append(idx[act])
        if "[EOC]" not in idx:
            idx["[EOC]"] = current_idx
            current_idx += 1
        trace_ids.append(idx["[EOC]"])
        vectorized_log.append(trace_ids)
    return vectorized_log, idx, current_idx


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


def to_dataset(vectorized_log):
    vectorized_log = np.array(list(itertools.chain(*vectorized_log)))
    # List of ids
    char_dataset = tf.data.Dataset.from_tensor_slices(vectorized_log)
    # Compact them on a list of traces of length seq_length+1
    sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

    dataset = sequences.map(split_input_target)

    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    return dataset


# ----------------------- Data Loading -----------------------
""" 
parent = Path(DATASET_FILE).parent
filename = os.path.basename(DATASET_FILE)

# Load full, train, validation, and test logs.
log_full = xes_import_factory.apply(DATASET_FILE,
                                    parameters={"timestamp_sort": True, "timestamp_key": "time:timestamp"})
train_log = xes_import_factory.apply(os.path.join(parent, "train_" + filename),
                                     parameters={"timestamp_sort": True, "timestamp_key": "time:timestamp"})
val_log = xes_import_factory.apply(os.path.join(parent, "val_" + filename),
                                   parameters={"timestamp_sort": True, "timestamp_key": "time:timestamp"})
test_log = xes_import_factory.apply(os.path.join(parent, "test_" + filename),
                                    parameters={"timestamp_sort": True, "timestamp_key": "time:timestamp"})

vectorized_full, _ = vectorize_log(log_full)
X_train, _ = vectorize_log(train_log)
X_val, _ = vectorize_log(val_log)
X_test, _ = vectorize_log(test_log)

print("Full log events:", sum(len(x) for x in vectorized_full))
print("Train events:", sum(len(x) for x in X_train))
print("Validation events:", sum(len(x) for x in X_val))
print("Test events:", sum(len(x) for x in X_test))

train_dataset = to_dataset(X_train)
val_dataset = to_dataset(X_val)
test_dataset = to_dataset(X_test)

vocab_size = len(idx)
"""

# ----------------------- Model Building -----------------------
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, batch_input_shape=[batch_size, None]),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.LSTM(rnn_units, return_sequences=True, stateful=True,
                             recurrent_initializer='glorot_uniform', dropout=dropout),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model


def build_model_pretrained(vocab_size, chosen_emb_dim, rnn_units, batch_size, embedding_matrix):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=chosen_emb_dim, batch_input_shape=[batch_size, None],
                                  weights=[embedding_matrix], trainable=False),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.LSTM(rnn_units, return_sequences=True, stateful=True,
                             recurrent_initializer='glorot_uniform', dropout=dropout),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model


# ----------------------- Loss and Optimizer -----------------------
def loss_fn(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)



# ----------------------- Intrinsic Embedding Methods -----------------------
# Supported encoding methods (names should match your intrinsic functions)
import itertools

embedding_methods = [
    'one_hot',
    "Bose 2009 Substitution Scores",
    "De Koninck 2018 act2vec CBOW",
    "De Koninck 2018 act2vec skip-gram",
    "Activity-Activitiy Co Occurrence Bag Of Words",
    "Activity-Activitiy Co Occurrence N-Gram",
    "Activity-Activitiy Co Occurrence Bag Of Words PMI",
    "Activity-Activitiy Co Occurrence N-Gram PMI",
    "Activity-Activitiy Co Occurrence Bag Of Words PPMI",
    "Activity-Activitiy Co Occurrence N-Gram PPMI",
    "Activity-Context Bag Of Words",
    "Activity-Context N-Grams",
    "Activity-Context Bag Of Words PMI",
    "Activity-Context N-Grams PMI",
    "Activity-Context Bag Of Words PPMI",
    "Activity-Context N-Grams PPMI",
    "Chiorrini 2022 Embedding Process Structure"]

from evaluation.data_util.util_activity_distances_intrinsic import add_window_size_evaluation

window_size_list = [3, 5, 9]

encoding_methods = add_window_size_evaluation(embedding_methods, window_size_list)
encoding_methods.append("Gamallo Fernandez 2023 Context Based w_3")


print("Encoding methods to evaluate:")
print(encoding_methods)


def extract_window_size(s):
    match = re.search(r"w_(\d+)", s)
    return int(match.group(1)) if match else 3





def get_embeddings_for_method(method, embedding_input, idx):
    # Remove termination tokens.

    log_input = [trace[:-1] for trace in embedding_input]
    log_input = [[str(num) for num in sublist] for sublist in log_input]

    alphabet = sorted(set(token for trace in log_input for token in trace))
    win_size = extract_window_size(method)
    if method == "one_hot":
        emb = {activity: np.eye(len(alphabet))[i] for i, activity in enumerate(alphabet)}
        return emb, len(alphabet)
    elif method == "Uniform Zero Embedding":
        # Each activity gets a zero vector of dimension equal to number of unique activities.
        emb = {activity: np.zeros(len(alphabet)) for activity in alphabet}
        return emb, len(alphabet)
    elif method == "Random Uniform Embedding":
        # Each activity gets a random vector with values in the range [-10, 10]
        emb = {activity: np.random.uniform(-10, 10, size=(len(alphabet),)) for activity in alphabet}
        return emb, len(alphabet)
    elif method.startswith("Unit Distance"):
        emb = {activity: np.eye(len(alphabet))[i] for i, activity in enumerate(alphabet)}
        return emb, len(alphabet)
    elif method.startswith("Bose 2009 Substitution Scores"):
        _, emb = get_substitution_and_insertion_scores(log_input, alphabet, win_size)
        return emb, len(alphabet)
    elif method.startswith("De Koninck 2018 act2vec"):
        sg = 0 if "CBOW" in method else 1
        _, emb = get_act2vec_distance_matrix(log_input, alphabet, sg, win_size)
        return emb, len(next(iter(emb.values())))
    elif method.startswith("Our act2vec"):
        emb = get_act2vec_distance_matrix_our(log_input, alphabet, win_size)
        return emb, len(next(iter(emb.values())))
    elif method.startswith("Activity-Activitiy Co Occurrence"):
        bag = True if "Bag Of Words" in method else False
        _, emb, activity_freq_dict, activity_index = get_activity_activity_co_occurence_matrix(log_input, alphabet,
                                                                                               win_size,
                                                                                               bag_of_words=bag)
        if "PPMI" in method:
            _, emb = get_activity_activity_frequency_matrix_pmi(emb, activity_freq_dict, activity_index, 1)
            return emb, len(next(iter(emb.values())))
        elif "PMI" in method:
            _, emb = get_activity_activity_frequency_matrix_pmi(emb, activity_freq_dict, activity_index, 0)
        return emb, len(next(iter(emb.values())))
    elif method.startswith("Activity-Context"):
        if "Bag Of Words" in method and "N-Grams" not in method:
            bag_mode = 2
        elif "N-Grams" in method:
            bag_mode = 0
        else:
            bag_mode = 2

        _, emb, activity_freq_dict, context_freq_dict, context_index = get_activity_context_frequency_matrix(log_input,
                                                                                                             alphabet,
                                                                                                             win_size,
                                                                                                             bag_of_words=bag_mode)

        if "PPMI" in method:
            _, emb = get_activity_context_frequency_matrix_pmi(emb, activity_freq_dict, context_freq_dict,
                                                               context_index, 1)
            return emb, len(next(iter(emb.values())))
        elif "PMI" in method:
            _, emb = get_activity_context_frequency_matrix_pmi(emb, activity_freq_dict, context_freq_dict,
                                                               context_index, 0)
        return emb, len(next(iter(emb.values())))
    elif method.startswith("Chiorrini 2022 Embedding Process Structure"):
        _, emb = get_embedding_process_structure_distance_matrix(log_input, alphabet, False)
        return emb, len(next(iter(emb.values())))
    elif method.startswith("Gamallo Fernandez 2023 Context Based"):
        _, emb = get_context_based_distance_matrix(log_input, win_size)
        return emb, len(next(iter(emb.values())))
    else:
        raise ValueError("Unknown encoding method: " + method)

    """
    elif method.startswith("nomic"):
        # Resulting dict mapping int IDs to embeddings
        emb = {}
        for text, id in idx.items():
            response = ollama.embeddings(model='nomic-embed-text', prompt=text)
            embedding = response['embedding']  # Extracting the embedding
            # Convert the embedding list into a numpy array
            emb[id] = np.array(embedding)
        return emb, len(next(iter(emb.values())))
    """

def create_embedding_matrix(idx, activity_embeddings, emb_dim):
    activity_embeddings = {int(key): value for key, value in activity_embeddings .items()}
    matrix = np.zeros((len(idx), emb_dim))
    for token, i in idx.items():
        if i in activity_embeddings:
            matrix[i] = activity_embeddings[i].reshape(-1)
        else:
            matrix[i] = np.zeros(emb_dim)
    return matrix

# ----------------------- Single-log endpoint -----------------------
def run_everman_for_log(
    log_name,
    method,
    split_dir=None,
    results_dir=None,
    models_dir=None,
    embedding_source='train_val',
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    random_seed=42,
):
    """Train and evaluate the Everman LSTM model for one event log and one method.

    Designed to be called from a Jupyter notebook::

        from next_activity_prediction_everman import run_everman_for_log

        result = run_everman_for_log(
            log_name='bpic2019',
            method='one_hot',
            epochs=100,
        )

    Parameters
    ----------
    log_name : str
        Base name of the event log (without path or extension).
    method : str
        Encoding method to use.  Must be one of the values in
        ``encoding_methods``.
    split_dir : str or None
        Directory containing pre-split XES files named
        ``train_<log_name>.xes.gz``, ``val_<log_name>.xes.gz``, and
        ``test_<log_name>.xes.gz``.  Defaults to ``split_datasets``.
    results_dir : str or None
        Directory where result files are written.  Defaults to
        ``results_everman``.
    models_dir : str or None
        Directory where model checkpoints are saved.  Defaults to
        ``models_everman``.
    embedding_source : {'train_val', 'train', 'validation'}
        Which split is used to compute activity embeddings.
    batch_size : int
        Mini-batch size.  Default taken from module-level ``BATCH_SIZE``.
    epochs : int
        Maximum training epochs.  Default taken from module-level ``EPOCHS``.
    random_seed : int
        Seed applied to Python ``random``, NumPy, and TensorFlow.

    Returns
    -------
    dict
        Dictionary with keys ``log``, ``method``, ``accuracy``, ``mcc``,
        ``brier``, ``recall``, ``precision``, ``f1``.
    """
    from sklearn.metrics import (accuracy_score, matthews_corrcoef, f1_score,
                                  precision_score, recall_score)

    # ── Seeds ────────────────────────────────────────────────────────────────
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.compat.v1.set_random_seed(random_seed)

    # ── Directories ──────────────────────────────────────────────────────────
    _split_dir   = split_dir   or SPLIT_DATASETS_DIR
    _results_dir = results_dir or RESULTS_DIR
    _models_dir  = models_dir  or MODELS_DIR
    os.makedirs(_results_dir, exist_ok=True)
    os.makedirs(_models_dir,  exist_ok=True)

    # ── Load splits ──────────────────────────────────────────────────────────
    train_path = os.path.join(_split_dir, f"train_{log_name}.xes.gz")
    val_path   = os.path.join(_split_dir, f"val_{log_name}.xes.gz")
    test_path  = os.path.join(_split_dir, f"test_{log_name}.xes.gz")
    print(f"\n{'='*60}")
    print(f"Processing log: {log_name}  |  method: {method}")
    print(f"  train : {train_path}")
    print(f"  val   : {val_path}")
    print(f"  test  : {test_path}")
    print(f"{'='*60}")

    train_log = xes_importer.apply(train_path,
                                   parameters={"timestamp_sort": True, "timestamp_key": "time:timestamp"})
    val_log   = xes_importer.apply(val_path,
                                   parameters={"timestamp_sort": True, "timestamp_key": "time:timestamp"})
    test_log  = xes_importer.apply(test_path,
                                   parameters={"timestamp_sort": True, "timestamp_key": "time:timestamp"})

    # ── Vectorise ────────────────────────────────────────────────────────────
    idx = {}
    current_idx = 0
    X_train, idx, current_idx = vectorize_log(train_log, idx, current_idx)
    X_val,   idx, current_idx = vectorize_log(val_log,   idx, current_idx)
    if "[UNK]" not in idx:
        idx["[UNK]"] = current_idx
        current_idx += 1
    unk_id     = idx["[UNK]"]
    vocab_size = current_idx
    X_test, _, _ = vectorize_log(test_log, idx, current_idx, readonly=True, unk_id=unk_id)

    print(f"Train events : {sum(len(x) for x in X_train)}")
    print(f"Val events   : {sum(len(x) for x in X_val)}")
    print(f"Test events  : {sum(len(x) for x in X_test)}")

    train_dataset = to_dataset(X_train)
    val_dataset   = to_dataset(X_val)

    # ── Embeddings ───────────────────────────────────────────────────────────
    print(f"\n--- Encoding method: {method} ---")
    if method == "one_hot":
        use_one_hot_flag = True
        chosen_emb_dim   = embedding_dim
    else:
        use_one_hot_flag = False
        if embedding_source == "train":
            embedding_input = X_train
        elif embedding_source == "validation":
            embedding_input = X_val
        elif embedding_source == "train_val":
            embedding_input = X_train + X_val
        else:
            raise ValueError(f"Unknown embedding_source: {embedding_source}")
        activity_embeddings, chosen_emb_dim = get_embeddings_for_method(method, embedding_input, idx)

    print(f"Encoding dimension: {chosen_emb_dim}")

    if not use_one_hot_flag:
        embedding_matrix = create_embedding_matrix(idx, activity_embeddings, chosen_emb_dim)

    # ── Build model ──────────────────────────────────────────────────────────
    if use_one_hot_flag:
        model = build_model(vocab_size, embedding_dim, rnn_units, batch_size)
    else:
        model = build_model_pretrained(vocab_size, chosen_emb_dim, rnn_units, batch_size, embedding_matrix)
    model.summary()

    optimizer = tf.keras.optimizers.legacy.SGD(
        learning_rate=lr, decay=lrDecay, clipnorm=maxGradNorm, momentum=0.9, nesterov=True)
    model_directory = os.path.join(_models_dir, method, f"{log_name}.h5")
    os.makedirs(os.path.dirname(model_directory), exist_ok=True)
    model.compile(loss=loss_fn, optimizer=optimizer,
                  metrics=[tf.keras.metrics.sparse_categorical_accuracy])
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        model_directory, monitor="val_loss",
        save_weights_only=True, save_best_only=True, verbose=1)

    # ── Train ────────────────────────────────────────────────────────────────
    print(f"Training ...")
    model.fit(
        train_dataset,
        epochs=epochs,
        callbacks=[checkpoint_callback],
        validation_data=val_dataset,
    )
    if not os.path.exists(model_directory):
        print("No best checkpoint saved — saving final weights as backup.")
        model.save_weights(model_directory)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    print(f"Evaluating ...")
    if use_one_hot_flag:
        test_model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
        test_model.build(tf.TensorShape([1, None]))
        test_model.load_weights(model_directory)
    else:
        test_model = build_model_pretrained(vocab_size, chosen_emb_dim, rnn_units, batch_size=1,
                                            embedding_matrix=embedding_matrix)
        dummy_input = tf.zeros([1, 10], dtype=tf.int32)
        _ = test_model(dummy_input)
        test_model.load_weights(model_directory)

    y_pred = []
    y_true = []
    last_case_id = idx["[EOC]"]
    result_dir = os.path.join(_results_dir, log_name, method)
    os.makedirs(result_dir, exist_ok=True)

    raw_result_file = os.path.join(result_dir, f"raw_{log_name}.csv")
    with open(raw_result_file, "w") as f:
        f.write("prefix_length;ground_truth;predicted;prediction_probs\n")
        for trace in X_test:
            for i, _ in enumerate(trace):
                if i + 1 >= len(trace):
                    break
                test_model.reset_states()
                inp        = trace[:i + 1]
                next_event = trace[i + 1]
                full_preds = test_model(tf.expand_dims(inp, 0), training=False)
                probs      = tf.nn.softmax(tf.squeeze(full_preds, 0).numpy()[-1])
                y_pred.append(probs)
                y_true.append(np.eye(vocab_size)[next_event])
                f.write(str(len(inp)) + ";" + str(next_event) + ";" +
                        str(np.argmax(probs)) + ";" +
                        np.array2string(probs.numpy(), separator=",", max_line_width=99999) + "\n")
                if next_event == last_case_id:
                    break

    y_pred_a = np.argmax(y_pred, axis=1)
    y_true_a = np.argmax(y_true, axis=1)

    def _brier(y_p, y_t):
        return np.mean(np.sum((np.array(y_t) - np.array(y_p)) ** 2, axis=1))

    acc       = accuracy_score(y_true_a, y_pred_a)
    mcc       = matthews_corrcoef(y_true_a, y_pred_a)
    brier     = _brier(y_pred, y_true)
    w_recall  = recall_score(y_true_a, y_pred_a, average="weighted")
    w_prec    = precision_score(y_true_a, y_pred_a, average="weighted")
    w_f1      = f1_score(y_true_a, y_pred_a, average="weighted")

    result_file = os.path.join(result_dir, f"{log_name}.txt")
    with open(result_file, "w") as f:
        f.write(f"Accuracy: {acc}\n")
        f.write(f"MCC: {mcc}\n")
        f.write(f"Brier score: {brier}\n")
        f.write(f"Weighted recall: {w_recall}\n")
        f.write(f"Weighted precision: {w_prec}\n")
        f.write(f"Weighted f1: {w_f1}\n")

    print(f"  Accuracy={acc:.4f}  MCC={mcc:.4f}  F1={w_f1:.4f}")

    # Delete raw CSV after metrics are computed
    if os.path.exists(raw_result_file):
        os.remove(raw_result_file)

    tf.keras.backend.clear_session()

    result = {
        "log": log_name, "method": method,
        "accuracy": acc, "mcc": mcc, "brier": brier,
        "recall": w_recall, "precision": w_prec, "f1": w_f1,
    }
    pd.DataFrame([result]).to_csv(
        os.path.join(result_dir, f"{log_name}_{method}_results.csv"), index=False)
    print(f"Results saved to '{result_dir}'")
    return result


if __name__ == "__main__":
    # ----------------------- Training and Evaluation -----------------------
    results_summary = []

    # Loop over each raw log in RAW_DATASETS_DIR.
    raw_logs = [f for f in os.listdir(RAW_DATASETS_DIR) if f.endswith(".xes.gz")]
    for raw_log in raw_logs:
        log_name = os.path.splitext(os.path.splitext(raw_log)[0])[0]
        train_path = os.path.join(SPLIT_DATASETS_DIR, f"train_{log_name}.xes.gz")
        val_path = os.path.join(SPLIT_DATASETS_DIR, f"val_{log_name}.xes.gz")
        test_path = os.path.join(SPLIT_DATASETS_DIR, f"test_{log_name}.xes.gz")
        print("\n========== Processing log:", log_name, "==========")
        print("Train:", train_path)
        print("Val:", val_path)
        print("Test:", test_path)

        train_log = xes_importer.apply(train_path,
                                            parameters={"timestamp_sort": True, "timestamp_key": "time:timestamp"})
        val_log = xes_importer.apply(val_path, parameters={"timestamp_sort": True, "timestamp_key": "time:timestamp"})
        test_log = xes_importer.apply(test_path,
                                            parameters={"timestamp_sort": True, "timestamp_key": "time:timestamp"})

        idx = {}
        current_idx = 0
        X_train, idx, current_idx = vectorize_log(train_log, idx, current_idx)
        X_val, idx, current_idx = vectorize_log(val_log, idx, current_idx)
        if "[UNK]" not in idx:
            idx["[UNK]"] = current_idx
            current_idx += 1
        unk_id = idx["[UNK]"]
        vocab_size = current_idx  # freeze vocab before test
        X_test, _, _ = vectorize_log(test_log, idx, current_idx, readonly=True, unk_id=unk_id)

        print("Train events:", sum(len(x) for x in X_train))
        print("Test events:", sum(len(x) for x in X_test))

        train_dataset = to_dataset(X_train)
        val_dataset = to_dataset(X_val)
        test_dataset = to_dataset(X_test)

        for method in encoding_methods:
            print("\n--- Encoding method:", method, "for log:", log_name, "---")
            if method == "one_hot":
                use_one_hot_flag = True
                chosen_emb_dim = embedding_dim
            else:
                use_one_hot_flag = False
                if method.startswith("Unit Distance"):
                    activity_embeddings = {activity: np.eye(vocab_size)[i] for activity, i in idx.items() if
                                        activity != "[EOC]"}
                    chosen_emb_dim = vocab_size
                else:
                    if embedding_source == "train":
                        embedding_input = X_train
                    elif embedding_source == "validation":
                        embedding_input = X_val
                    elif embedding_source == "train_val":
                        embedding_input = X_train + X_val
                    else:
                        raise ValueError("Unknown embedding_source")
                    try:
                        activity_embeddings, chosen_emb_dim = get_embeddings_for_method(method, embedding_input, idx)
                    except Exception as e:
                        print(f"Error computing embeddings for method {method} on log {log_name}: {e}")
            print("Final encoding dimension:", chosen_emb_dim)

            if not use_one_hot_flag:
                embedding_matrix = create_embedding_matrix(idx, activity_embeddings, chosen_emb_dim)

            # Build the model.
            if use_one_hot_flag:
                model = build_model(vocab_size, embedding_dim, rnn_units, BATCH_SIZE)
            else:
                model = build_model_pretrained(vocab_size, chosen_emb_dim, rnn_units, BATCH_SIZE, embedding_matrix)
            model.summary()
            optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=lr, decay=lrDecay, clipnorm=maxGradNorm, momentum=0.9,
                                                    nesterov=True)

            model_file_name = log_name + ".h5"
            model_directory = os.path.join(MODELS_DIR, method, model_file_name)

            model.compile(loss=loss_fn, optimizer=optimizer,
                        metrics=[tf.keras.metrics.sparse_categorical_accuracy])
            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                model_directory,
                monitor="val_loss", save_weights_only=True, save_best_only=True, verbose=1)

            history = model.fit(
                train_dataset,
                epochs=EPOCHS,
                callbacks=[checkpoint_callback],
                validation_data=val_dataset
            )
            # After training, check if a best checkpoint was saved.
            if not os.path.exists(model_directory):
                print(
                    "No best checkpoint was saved (validation loss never improved), so saving final model weights as backup.")
                model.save_weights(model_directory)

            if TEST_FLAG:
                print("Start testing for log", log_name, "with method", method)
                if use_one_hot_flag:
                    # For the baseline one_hot model, build the model with batch_size=1
                    test_model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
                    # This call sets the input shape (batch=1, variable sequence length)
                    test_model.build(tf.TensorShape([1, None]))
                    test_model.load_weights(model_directory)
                else:
                    # For pretrained embedding models, build with the same batch size
                    test_model = build_model_pretrained(vocab_size, chosen_emb_dim, rnn_units, batch_size=1,
                                                        embedding_matrix=embedding_matrix)
                    # Instead of calling build() (which would reinitialize weights), call the model on a dummy input.
                    dummy_input = tf.zeros([1, 10], dtype=tf.int32)  # dummy sequence of 10 tokens
                    _ = test_model(dummy_input)
                    test_model.load_weights(model_directory)
                y_pred = []
                y_true = []
                last_case_id = idx["[EOC]"]
                d = os.path.join(RESULTS_DIR, log_name, method)
                os.makedirs(d, exist_ok=True)

                raw_result_file = "raw_" + log_name + ".csv"
                with open(os.path.join(d, raw_result_file), "w") as f:
                    f.write("prefix_length;ground_truth;predicted;prediction_probs\n")
                    for trace in X_test:
                        for i, event in enumerate(trace):
                            test_model.reset_states()
                            inp = trace[:i + 1]
                            next_event = trace[i + 1]
                            full_preds = test_model(tf.expand_dims(inp, 0), training=False)
                            probs = tf.nn.softmax(tf.squeeze(full_preds, 0).numpy()[-1])
                            y_pred.append(probs)
                            y_true.append(np.eye(vocab_size)[next_event])
                            f.write(str(len(inp)) + ";" + str(next_event) + ";" +
                                    str(np.argmax(probs)) + ";" +
                                    np.array2string(probs.numpy(), separator=",", max_line_width=99999) + "\n")
                            if next_event == last_case_id:
                                break

                from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, precision_score, recall_score

                y_pred_a = np.argmax(y_pred, axis=1)
                y_true_a = np.argmax(y_true, axis=1)
                result_file = log_name + ".txt"
                with open(os.path.join(d, result_file), "w") as f:
                    f.write("Accuracy: " + str(accuracy_score(y_true_a, y_pred_a)))
                    f.write("\nMCC: " + str(matthews_corrcoef(y_true_a, y_pred_a)))


                    def calculate_brier_score(y_pred, y_true):
                        return np.mean(np.sum((y_true - y_pred) ** 2, axis=1))


                    f.write("\nBrier score: " + str(calculate_brier_score(np.array(y_true), np.array(y_pred))))
                    f.write("\nWeighted recall: " + str(recall_score(y_true_a, y_pred_a, average="weighted")))
                    f.write("\nWeighted precision: " + str(precision_score(y_true_a, y_pred_a, average="weighted")))
                    f.write("\nWeighted f1: " + str(f1_score(y_true_a, y_pred_a, average="weighted")))
                print(f"Results for log {log_name} with method {method} saved.")

                # Delete the CSV after it's no longer needed
                if os.path.exists(os.path.join(d, raw_result_file)):
                    os.remove(os.path.join(d, raw_result_file))

            tf.keras.backend.clear_session()
            # For demonstration, append dummy results.
            from sklearn.metrics import accuracy_score, f1_score

            preds = model.predict(test_dataset)
            y_act_pred_probs = preds[0]
            y_act_pred = np.argmax(y_act_pred_probs, axis=1)
            result_dict = {"log": log_name, "method": method, "accuracy": accuracy_score(y_true_a, y_pred_a), "f1": f1_score(y_true_a, y_pred_a, average="weighted")}
            results_summary.append(result_dict)
    df_results = pd.DataFrame(results_summary)
    df_results.to_csv(os.path.join(RESULTS_DIR, "all_logs_results.csv"), index=False)
    print("\nOverall results:")
    print(df_results)
    avg_results = df_results.groupby("method").agg({"accuracy": "mean", "f1": "mean"}).reset_index()
    print("\nAverage results per method:")
    print(avg_results)
    avg_results.to_csv(os.path.join(RESULTS_DIR, "average_results_per_method.csv"), index=False)

    print("Pipeline evaluation finished.")

