#!/usr/bin/env python
"""
Next Activity Prediction using LSTM with Various Activity Embedding Methods

This script trains and evaluates LSTM models for next activity prediction in process mining,
comparing different activity embedding approaches including one-hot encoding and various
intrinsic embedding methods.

UPDATED: Follows best practices from "Creating Unbiased Public Benchmark Datasets 
with Data Leakage Prevention for Predictive Process Monitoring" paper.
"""

import os
import sys
import copy
import random
import time
import re
import csv
import itertools
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

# TensorFlow configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, BatchNormalization, Input
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Process mining imports
from pm4py.objects.log.importer.xes import importer as xes_importer

# Project-specific imports
from definitions import ROOT_DIR
from distances.activity_distances.pmi.pmi import (
    get_activity_context_frequency_matrix_pmi,
    get_activity_activity_frequency_matrix_pmi
)
from distances.activity_distances.bose_2009_context_aware_trace_clustering.algorithm import (
    get_substitution_and_insertion_scores
)
from distances.activity_distances.de_koninck_2018_act2vec.algorithm import (
    get_act2vec_distance_matrix
)
from distances.activity_distances.de_koninck_2018_act2vec.our_hyperparas import (
    get_act2vec_distance_matrix_our
)
from distances.activity_distances.activity_activity_co_occurence.activity_activity_co_occurrence import (
    get_activity_activity_co_occurence_matrix
)
from distances.activity_distances.activity_context_frequency.activity_contex_frequency import (
    get_activity_context_frequency_matrix
)
from distances.activity_distances.chiorrini_2022_embedding_process_structure.embedding_process_structure import (
    get_embedding_process_structure_distance_matrix
)
from distances.activity_distances.gamallo_fernandez_2023_context_based_representations.src.embeddings_generator.main_new import (
    get_context_based_distance_matrix
)
from evaluation.data_util.util_activity_distances_intrinsic import (
    add_window_size_evaluation
)

# =====================================================================
# CONFIGURATION PARAMETERS
# =====================================================================

# Random seeds for reproducibility
RANDOM_SEED = 42
tf.compat.v1.set_random_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Dataset configuration
TRAIN_FLAG = True
TEST_FLAG = True
TEST_SUFFIX_FLAG = False
TEST_SUFFIX_CALC_FLAG = False

# Directory structure
NA_DIR = os.path.join(ROOT_DIR, "evaluation", "evaluation_of_activity_distances", "next_activity_prediction")
RESULTS_DIR = os.path.join(NA_DIR, "results_everman")
MODELS_DIR = os.path.join(NA_DIR, "models_everman")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Training hyperparameters
SEQ_LENGTH = 20
BATCH_SIZE = 20
BUFFER_SIZE = 10000
EMBEDDING_DIM = 32
RNN_UNITS = 32
DROPOUT = 0.2
MAX_GRAD_NORM = 5
LR_DECAY = 0.75
LEARNING_RATE = 1.0
EPOCHS = 2

# Embedding configuration - ONLY use training data for embeddings
EMBEDDING_SOURCE = "train"  # Changed from "train_val" to prevent leakage
print(f"Embedding source: {EMBEDDING_SOURCE}")

# Window sizes for context-based embeddings
WINDOW_SIZE_LIST = [3, 5, 9]

EMBEDDING_METHODS = [
    'one_hot', # Evermann
    'tax_one_hot'  # Tax 
    #"Bose 2009 Substitution Scores",
    #"De Koninck 2018 act2vec CBOW",
    #"De Koninck 2018 act2vec skip-gram",
    #"Activity-Activitiy Co Occurrence Bag Of Words",
    #"Activity-Activitiy Co Occurrence N-Gram",
    #"Activity-Activitiy Co Occurrence Bag Of Words PMI",
    #"Activity-Activitiy Co Occurrence N-Gram PMI",
    #"Activity-Activitiy Co Occurrence Bag Of Words PPMI",
    #"Activity-Activitiy Co Occurrence N-Gram PPMI",
    #"Activity-Context Bag Of Words",
    #"Activity-Context N-Grams",
    #"Activity-Context Bag Of Words PMI",
    #"Activity-Context N-Grams PMI",
    #"Activity-Context Bag Of Words PPMI",
    #"Activity-Context N-Grams PPMI",
    #"Chiorrini 2022 Embedding Process Structure"
]

ENCODING_METHODS = add_window_size_evaluation(EMBEDDING_METHODS, WINDOW_SIZE_LIST)
ENCODING_METHODS.append("Gamallo Fernandez 2023 Context Based w_3")



# =====================================================================
# DATA PREPROCESSING FUNCTIONS
# =====================================================================

def vectorize_log(log, idx, current_idx):
    """
    Convert a log (list of traces) into a list of lists of event IDs.
    
    Args:
        log: List of traces (each trace is a list of events)
        idx: Dictionary mapping activity names to integer IDs
        current_idx: Next available integer ID
        
    Returns:
        tuple: (vectorized_log, updated_idx, updated_current_idx)
    """
    vectorized_log = []
    
    for trace in log:
        trace_ids = []
        for event in trace:
            act = event["concept:name"]
            if act not in idx:
                idx[act] = current_idx
                current_idx += 1
            trace_ids.append(idx[act])
        
        # Add end-of-case token
        if "[EOC]" not in idx:
            idx["[EOC]"] = current_idx
            current_idx += 1
        trace_ids.append(idx["[EOC]"])
        
        vectorized_log.append(trace_ids)
    
    return vectorized_log, idx, current_idx


def vectorize_log_with_unk(log, frozen_idx):
    """
    Convert a log to event IDs using a frozen vocabulary.
    Unknown activities are mapped to [UNK] token.
    
    Args:
        log: List of traces (each trace is a list of events)
        frozen_idx: Frozen vocabulary dictionary (no modifications allowed)
        
    Returns:
        tuple: (vectorized_log, frozen_idx, vocab_size)
    """
    vectorized_log = []
    unk_id = frozen_idx["[UNK]"]
    eoc_id = frozen_idx["[EOC]"]
    
    for trace in log:
        trace_ids = []
        for event in trace:
            act = event["concept:name"]
            # Map to [UNK] if activity not in training vocabulary
            event_id = frozen_idx.get(act, unk_id)
            trace_ids.append(event_id)
        
        # Add end-of-case token
        trace_ids.append(eoc_id)
        vectorized_log.append(trace_ids)
    
    return vectorized_log, frozen_idx, len(frozen_idx)


def split_input_target(chunk):
    """Split a sequence into input and target for next-event prediction."""
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


def to_dataset(vectorized_log):
    """
    Convert vectorized log to TensorFlow dataset for training.
    
    Args:
        vectorized_log: List of lists of event IDs
        
    Returns:
        tf.data.Dataset: Batched and shuffled dataset
    """
    # Flatten all traces into a single sequence
    vectorized_log = np.array(list(itertools.chain(*vectorized_log)))
    
    # Create dataset from tensor slices
    char_dataset = tf.data.Dataset.from_tensor_slices(vectorized_log)
    
    # Batch into sequences of length SEQ_LENGTH + 1
    sequences = char_dataset.batch(SEQ_LENGTH + 1, drop_remainder=True)
    
    # Split into input and target
    dataset = sequences.map(split_input_target)
    
    # Shuffle and batch
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    
    return dataset


def extract_log(full_path):
    """
    Load and split event log into train, validation, and test sets.
    Follows best practices from "Creating Unbiased Public Benchmark Datasets 
    with Data Leakage Prevention for Predictive Process Monitoring" paper.
    
    Key principles:
    - Vocabulary built ONLY from training data
    - Unknown activities in val/test mapped to [UNK] token
    - Temporal ordering preserved
    - No information leakage from future splits
    
    Args:
        full_path: Path to XES event log file
        
    Returns:
        dict: Dictionary containing datasets and metadata
    """
    # Import event log with temporal sorting
    log_full = xes_importer.apply(
        full_path,
        parameters={
            "timestamp_sort": True,
            "timestamp_key": "time:timestamp"
        }
    )
    
    # Split FIRST (before any vectorization): 65% train, 15% validation, 20% test
    n_traces = len(log_full)
    n_train_traces = int(0.65 * n_traces)
    n_val_traces = int(0.15 * n_traces)
    
    log_train = log_full[:n_train_traces]
    log_val = log_full[n_train_traces:n_train_traces + n_val_traces]
    log_test = log_full[n_train_traces + n_val_traces:]
    
    print(f"Split sizes - Train: {len(log_train)}, Val: {len(log_val)}, Test: {len(log_test)}")
    
    # Build vocabulary ONLY from training data
    idx = {}
    current_idx = 0
    
    # Reserve index 0 for unknown token
    idx["[UNK]"] = current_idx
    current_idx += 1
    
    # Vectorize training data (builds vocabulary)
    vectorized_train, idx, current_idx = vectorize_log(log_train, idx, current_idx)
    
    # Freeze vocabulary - create immutable copy
    train_vocab_size = current_idx
    frozen_idx = idx.copy()
    
    print(f"Training vocabulary size: {train_vocab_size} (including [UNK] and [EOC])")
    
    # Vectorize validation data using ONLY training vocabulary
    vectorized_val, _, _ = vectorize_log_with_unk(log_val, frozen_idx)
    
    # Vectorize test data using ONLY training vocabulary  
    vectorized_test, _, _ = vectorize_log_with_unk(log_test, frozen_idx)
    
    # Create TensorFlow datasets
    train_dataset = to_dataset(vectorized_train)
    val_dataset = to_dataset(vectorized_val)
    test_dataset = to_dataset(vectorized_test)
    
    # Count unknown activities in val/test for reporting
    unk_id = frozen_idx["[UNK]"]
    val_unk_count = sum(1 for trace in vectorized_val for event_id in trace if event_id == unk_id)
    test_unk_count = sum(1 for trace in vectorized_test for event_id in trace if event_id == unk_id)
    
    if val_unk_count > 0:
        print(f"Warning: {val_unk_count} unknown activities in validation set")
    if test_unk_count > 0:
        print(f"Warning: {test_unk_count} unknown activities in test set")
    
    return {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "X_train": vectorized_train,
        "X_val": vectorized_val,
        "X_test": vectorized_test,
        "idx": frozen_idx,
        "vocab_size": train_vocab_size,
        "log_train": log_train,  # Original traces for embedding generation
        "log_val": log_val,
        "log_test": log_test
    }


# =====================================================================
# MODEL BUILDING FUNCTIONS
# =====================================================================

def build_model(vocab_size, embedding_dim, rnn_units, batch_size=None):
    """
    Build LSTM model with learnable embeddings (for one-hot encoding).
    
    Args:
        vocab_size: Size of vocabulary
        embedding_dim: Dimension of embeddings
        rnn_units: Number of LSTM units
        batch_size: Batch size (None for flexible)
        
    Returns:
        tf.keras.Model: Compiled model
    """
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(None,)),
        tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim
        ),
        tf.keras.layers.Dropout(DROPOUT),
        tf.keras.layers.LSTM(
            rnn_units,
            return_sequences=True,
            stateful=False,
            recurrent_initializer='glorot_uniform',
            dropout=DROPOUT
        ),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model


def build_model_pretrained(vocab_size, chosen_emb_dim, rnn_units, 
                          batch_size=None, embedding_matrix=None):
    """
    Build LSTM model with pre-trained frozen embeddings.
    
    Args:
        vocab_size: Size of vocabulary
        chosen_emb_dim: Dimension of pre-trained embeddings
        rnn_units: Number of LSTM units
        batch_size: Batch size (None for flexible)
        embedding_matrix: Pre-computed embedding matrix
        
    Returns:
        tf.keras.Model: Compiled model
    """
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(None,)),
        tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=chosen_emb_dim,
            weights=[embedding_matrix] if embedding_matrix is not None else None,
            trainable=False
        ),
        tf.keras.layers.Dropout(DROPOUT),
        tf.keras.layers.LSTM(
            rnn_units,
            return_sequences=True,
            stateful=False,
            recurrent_initializer='glorot_uniform',
            dropout=DROPOUT
        ),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model


def loss_fn(labels, logits):
    """Sparse categorical crossentropy loss."""
    return tf.keras.losses.sparse_categorical_crossentropy(
        labels, logits, from_logits=True
    )


# =====================================================================
# EMBEDDING GENERATION FUNCTIONS
# =====================================================================

def extract_window_size(method_name):
    """Extract window size from method name (e.g., 'w_3' -> 3)."""
    match = re.search(r"w_(\d+)", method_name)
    return int(match.group(1)) if match else 3


def get_embeddings_for_method(method, embedding_input, idx):
    """
    Generate activity embeddings using the specified method.
    Uses ONLY training data to prevent leakage.
    
    Args:
        method: Name of the embedding method
        embedding_input: Vectorized log for computing embeddings (TRAINING DATA ONLY)
        idx: Activity to ID mapping (built from training data only)
        
    Returns:
        tuple: (embeddings_dict, embedding_dimension)
    """
    # Remove end-of-case tokens
    log_input = [trace[:-1] for trace in embedding_input]
    log_input = [[str(num) for num in sublist] for sublist in log_input]
    
    # Build alphabet from training data only
    alphabet = sorted(set(token for trace in log_input for token in trace))
    win_size = extract_window_size(method)
    
    print(f"  Generating embeddings using {method} on {len(log_input)} training traces")
    print(f"  Training alphabet size: {len(alphabet)}")
    
    # One-hot encoding
    if method == "one_hot":
        emb = {activity: np.eye(len(alphabet))[i] for i, activity in enumerate(alphabet)}
        return emb, len(alphabet)
    
    # Uniform zero embedding
    elif method == "Uniform Zero Embedding":
        emb = {activity: np.zeros(len(alphabet)) for activity in alphabet}
        return emb, len(alphabet)
    
    # Random uniform embedding
    elif method == "Random Uniform Embedding":
        emb = {activity: np.random.uniform(-10, 10, size=(len(alphabet),)) 
               for activity in alphabet}
        return emb, len(alphabet)
    
    # Unit distance
    elif method.startswith("Unit Distance"):
        emb = {activity: np.eye(len(alphabet))[i] for i, activity in enumerate(alphabet)}
        return emb, len(alphabet)
    
    # Bose 2009 substitution scores
    elif method.startswith("Bose 2009 Substitution Scores"):
        _, emb = get_substitution_and_insertion_scores(log_input, alphabet, win_size)
        return emb, len(alphabet)
    
    # De Koninck 2018 act2vec
    elif method.startswith("De Koninck 2018 act2vec"):
        sg = 0 if "CBOW" in method else 1
        _, emb = get_act2vec_distance_matrix(log_input, alphabet, sg, win_size)
        return emb, len(next(iter(emb.values())))
    
    # Our act2vec implementation
    elif method.startswith("Our act2vec"):
        emb = get_act2vec_distance_matrix_our(log_input, alphabet, win_size)
        return emb, len(next(iter(emb.values())))
    
    # Activity-Activity co-occurrence
    elif method.startswith("Activity-Activitiy Co Occurrence"):
        bag = True if "Bag Of Words" in method else False
        _, emb, activity_freq_dict, activity_index = get_activity_activity_co_occurence_matrix(
            log_input, alphabet, win_size, bag_of_words=bag
        )
        
        if "PPMI" in method:
            _, emb = get_activity_activity_frequency_matrix_pmi(
                emb, activity_freq_dict, activity_index, 1
            )
        elif "PMI" in method:
            _, emb = get_activity_activity_frequency_matrix_pmi(
                emb, activity_freq_dict, activity_index, 0
            )
        
        return emb, len(next(iter(emb.values())))
    
    # Activity-Context embeddings
    elif method.startswith("Activity-Context"):
        if "Bag Of Words" in method and "N-Grams" not in method:
            bag_mode = 2
        elif "N-Grams" in method:
            bag_mode = 0
        else:
            bag_mode = 2
        
        _, emb, activity_freq_dict, context_freq_dict, context_index = \
            get_activity_context_frequency_matrix(
                log_input, alphabet, win_size, bag_of_words=bag_mode
            )
        
        if "PPMI" in method:
            _, emb = get_activity_context_frequency_matrix_pmi(
                emb, activity_freq_dict, context_freq_dict, context_index, 1
            )
        elif "PMI" in method:
            _, emb = get_activity_context_frequency_matrix_pmi(
                emb, activity_freq_dict, context_freq_dict, context_index, 0
            )
        
        return emb, len(next(iter(emb.values())))
    
    # Chiorrini 2022 embedding process structure
    elif method.startswith("Chiorrini 2022 Embedding Process Structure"):
        _, emb = get_embedding_process_structure_distance_matrix(log_input, alphabet, False)
        return emb, len(next(iter(emb.values())))
    
    # Gamallo Fernandez 2023 context-based
    elif method.startswith("Gamallo Fernandez 2023 Context Based"):
        _, emb = get_context_based_distance_matrix(log_input, win_size)
        return emb, len(next(iter(emb.values())))
    
    else:
        raise ValueError(f"Unknown encoding method: {method}")


def create_embedding_matrix(idx, activity_embeddings, emb_dim):
    """
    Create embedding matrix from activity embeddings dictionary.
    Handles unknown token ([UNK]) by assigning zero embedding.
    
    Args:
        idx: Activity to ID mapping (includes [UNK] and [EOC])
        activity_embeddings: Dictionary of embeddings (from training data only)
        emb_dim: Embedding dimension
        
    Returns:
        np.ndarray: Embedding matrix of shape (vocab_size, emb_dim)
    """
    # Ensure keys are integers
    activity_embeddings = {int(key): value for key, value in activity_embeddings.items()}
    
    # Initialize matrix with zeros
    matrix = np.zeros((len(idx), emb_dim))
    
    for token, i in idx.items():
        if i in activity_embeddings:
            matrix[i] = activity_embeddings[i].reshape(-1)
        else:
            # [UNK] and [EOC] get zero embeddings
            # Activities not in training vocab won't appear here since idx is frozen
            matrix[i] = np.zeros(emb_dim)
    
    return matrix


# =====================================================================
# TAX APPROACH FUNCTIONS
# =====================================================================

def build_vocab_tax(lines):
    """
    Build vocabulary for tax approach.
    Should ONLY be called with training data.
    
    Args:
        lines: List of traces (each trace is a list of activity names) - TRAINING ONLY
        
    Returns:
        tuple: (vocab, target_tokens, target_token_indices, indices_token)
    """
    vocab = set()
    for trace in lines:
        for token in trace:
            vocab.add(token)
    vocab = sorted(list(vocab))
    
    target_tokens = vocab.copy()
    target_token_indices = {token: i for i, token in enumerate(target_tokens)}
    indices_token = {i: token for i, token in enumerate(vocab)}
    
    return vocab, target_tokens, target_token_indices, indices_token


def vectorize_fold_tax(sent, sent_ts, sent_ts2, sent_ts3, sent_ts4, divisor, divisor2, 
                       encoding_dim, use_one_hot=False, one_hot_indices=None, 
                       activity_embeddings=None, maxlen=99):
    """
    Vectorize traces for tax approach with temporal features.
    
    Args:
        sent: List of traces (activity sequences)
        sent_ts: Time since previous event
        sent_ts2: Time since case start
        sent_ts3: Time since midnight
        sent_ts4: Day of week
        divisor: Normalization factor for sent_ts
        divisor2: Normalization factor for sent_ts2
        encoding_dim: Dimension of activity encoding
        use_one_hot: Whether to use one-hot encoding
        one_hot_indices: Mapping of activities to indices (for one-hot)
        activity_embeddings: Pre-computed embeddings (for embedding mode)
        maxlen: Maximum sequence length
        
    Returns:
        tuple: (X, y_act, y_time) - inputs and targets
    """
    # Get target tokens from training vocabulary
    vocab = set()
    for trace in sent:
        for token in trace:
            if use_one_hot and one_hot_indices is not None:
                # Only use activities in training vocabulary
                if token in one_hot_indices:
                    vocab.add(token)
            elif activity_embeddings is not None:
                # Only use activities in training embeddings
                if token in activity_embeddings:
                    vocab.add(token)
            else:
                vocab.add(token)
    
    vocab = sorted(list(vocab))
    target_tokens = vocab.copy()
    target_token_indices = {token: i for i, token in enumerate(target_tokens)}
    
    total_features = encoding_dim + 5
    
    X = np.zeros((len(sent), maxlen, total_features), dtype=np.float32)
    y_act = np.zeros((len(sent), len(target_tokens)), dtype=np.float32)
    y_time = np.zeros((len(sent),), dtype=np.float32)
    
    for i, trace in enumerate(sent):
        left_pad_size = maxlen - len(trace)
        
        for j, token in enumerate(trace):
            pos = left_pad_size + j
            
            # Activity encoding - skip unknown activities
            if use_one_hot:
                if token in one_hot_indices:
                    X[i, pos, one_hot_indices[token]] = 1
                # else: unknown activity, leave as zero
            else:
                if token in activity_embeddings:
                    emb_vec = activity_embeddings[token]
                    X[i, pos, :encoding_dim] = emb_vec
                # else: unknown activity, leave as zero
            
            # Temporal features (always available)
            X[i, pos, encoding_dim] = sent_ts[i][j] / divisor
            X[i, pos, encoding_dim + 1] = sent_ts2[i][j] / divisor2
            X[i, pos, encoding_dim + 2] = sent_ts3[i][j] / 86400.0
            X[i, pos, encoding_dim + 3] = sent_ts4[i][j] / 7.0
            X[i, pos, encoding_dim + 4] = j / maxlen
        
        # Set target (last activity in trace)
        if len(trace) > 0:
            target = trace[-1]
            if target in target_token_indices:
                y_act[i, target_token_indices[target]] = 1
            # else: unknown target, leave as zero (will be ignored in loss)
            
        y_time[i] = sent_ts[i][-1] / divisor if len(sent_ts[i]) > 0 else 0
    
    return X, y_act, y_time


def build_tax_model(maxlen, total_features, num_activities):
    """
    Build the tax LSTM model architecture.
    
    Args:
        maxlen: Maximum sequence length
        total_features: Total number of input features per timestep
        num_activities: Number of unique activities (output classes)
        
    Returns:
        tf.keras.Model: Compiled model
    """
    main_input = Input(shape=(maxlen, total_features), name='main_input')
    
    # First LSTM layer
    l1 = LSTM(100, kernel_initializer='glorot_uniform', return_sequences=True, 
              dropout=0.2, unroll=True)(main_input)
    b1 = BatchNormalization()(l1)
    
    # Second LSTM layers (parallel for activity and time prediction)
    l2_1 = LSTM(100, kernel_initializer='glorot_uniform', return_sequences=False, 
                dropout=0.2, unroll=True)(b1)
    b2_1 = BatchNormalization()(l2_1)
    
    l2_2 = LSTM(100, kernel_initializer='glorot_uniform', return_sequences=False, 
                dropout=0.2, unroll=True)(b1)
    b2_2 = BatchNormalization()(l2_2)
    
    # Output layers
    act_output = Dense(num_activities, activation='softmax', 
                       kernel_initializer='glorot_uniform', name='act_output')(b2_1)
    time_output = Dense(1, kernel_initializer='glorot_uniform', name='time_output')(b2_2)
    
    model = Model(inputs=[main_input], outputs=[act_output, time_output])
    
    # Compile model
    opt = Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipvalue=3)
    model.compile(
        loss={'act_output': 'categorical_crossentropy', 'time_output': 'mae'},
        optimizer=opt,
        metrics={"act_output": "acc", "time_output": "mae"}
    )
    
    return model


# =====================================================================
# TRAINING FUNCTIONS
# =====================================================================

def train_model(data_dict, method, log_name):
    """Dispatcher for training different model types."""
    if method == "tax_one_hot":
        model, model_path, maxlen, emb_dim, vocab_info = train_tax_model(
            data_dict, method, log_name
        )
        return {
            "model": model,
            "model_path": model_path,
            "maxlen": maxlen,
            "emb_dim": emb_dim,
            "vocab_info": vocab_info
        }
    else:
        model, model_path, use_one_hot, emb_dim = train_evermann(
            data_dict, method, log_name
        )
        return {
            "model": model,
            "model_path": model_path,
            "use_one_hot": use_one_hot,
            "emb_dim": emb_dim
        }


def train_evermann(data_dict, method, log_name):
    """
    Train LSTM model with specified embedding method.
    Uses ONLY training data for embeddings to prevent leakage.
    
    Args:
        data_dict: Dictionary containing datasets and metadata
        method: Embedding method name
        log_name: Name of the event log
        
    Returns:
        tuple: (model, model_path, use_one_hot_flag, embedding_dim)
    """
    X_train = data_dict["X_train"]
    X_val = data_dict["X_val"]
    idx = data_dict["idx"]
    vocab_size = data_dict["vocab_size"]
    train_dataset = data_dict["train_dataset"]
    val_dataset = data_dict["val_dataset"]
    
    # Determine encoding approach
    if method == "one_hot":
        use_one_hot_flag = True
        chosen_emb_dim = EMBEDDING_DIM
        embedding_matrix = None
    else:
        use_one_hot_flag = False
        
        if method.startswith("Unit Distance"):
            activity_embeddings = {
                activity: np.eye(vocab_size)[i]
                for activity, i in idx.items() if activity not in ["[EOC]", "[UNK]"]
            }
            chosen_emb_dim = vocab_size
        else:
            # CRITICAL: Only use training data for embeddings
            embedding_input = X_train
            activity_embeddings, chosen_emb_dim = get_embeddings_for_method(
                method, embedding_input, idx
            )
        
        embedding_matrix = create_embedding_matrix(idx, activity_embeddings, chosen_emb_dim)
    
    # Build model
    if use_one_hot_flag:
        model = build_model(vocab_size, EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)
    else:
        model = build_model_pretrained(
            vocab_size, chosen_emb_dim, RNN_UNITS, BATCH_SIZE, embedding_matrix
        )
    
    # Configure optimizer
    optimizer = tf.keras.optimizers.legacy.SGD(
        learning_rate=LEARNING_RATE,
        decay=LR_DECAY,
        clipnorm=MAX_GRAD_NORM,
        momentum=0.9,
        nesterov=True
    )
    
    # Compile model
    model.compile(
        loss=loss_fn,
        optimizer=optimizer,
        metrics=[tf.keras.metrics.sparse_categorical_accuracy]
    )
    
    # Setup model checkpoint
    model_directory = os.path.join(MODELS_DIR, method, log_name + ".h5")
    os.makedirs(os.path.dirname(model_directory), exist_ok=True)
    
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        model_directory,
        monitor="val_loss",
        save_weights_only=True,
        save_best_only=True,
        verbose=1
    )
    
    # Early stopping callback
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=32,
        verbose=1,
        restore_best_weights=True
    )
    
    # Train model
    print(f"\nTraining model with {method} embeddings...")
    model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
        callbacks=[checkpoint_callback, early_stopping_callback]
    )
    
    # Save weights if not saved by checkpoint
    if not os.path.exists(model_directory):
        model.save_weights(model_directory)
    
    return model, model_directory, use_one_hot_flag, chosen_emb_dim


def train_tax_model(data_dict, method, log_name):
    """
    Train tax LSTM model with dual-head architecture.
    FIXED: Uses only training data for vocabulary and maxlen calculation.
    
    Args:
        data_dict: Dictionary containing datasets and metadata
        method: Should be 'tax_one_hot'
        log_name: Name of the event log
        
    Returns:
        tuple: (model, model_path, maxlen, encoding_dim, vocab_info)
    """
    # Extract traces from vectorized log
    X_train = data_dict["X_train"]
    X_val = data_dict["X_val"]
    idx = data_dict["idx"]
    vocab_size = data_dict["vocab_size"]
    
    # Convert vectorized traces back to activity names for tax approach
    idx_reverse = {v: k for k, v in idx.items()}
    
    # Remove [EOC] tokens and convert to activity names
    lines_train = [[idx_reverse[event_id] for event_id in trace if event_id != idx["[EOC]"]] 
                   for trace in X_train]
    lines_val = [[idx_reverse[event_id] for event_id in trace if event_id != idx["[EOC]"]] 
                 for trace in X_val]
    
    # FIXED: Build vocabulary ONLY from training data
    vocab, target_tokens, target_token_indices, indices_token = build_vocab_tax(lines_train)
    
    print(f"Tax vocabulary size (training only): {len(vocab)}")
    
    # FIXED: Calculate maxlen ONLY from training data
    maxlen = max(len(trace) for trace in lines_train)
    print(f"Max sequence length (training only): {maxlen}")
    
    # Create dummy temporal features for compatibility
    # In a real scenario, these would come from the actual event log timestamps
    ts_train = [[1.0] * len(trace) for trace in lines_train]
    ts_val = [[1.0] * len(trace) for trace in lines_val]
    ts2_train = [[float(i)] * len(trace) for i, trace in enumerate(lines_train)]
    ts2_val = [[float(i)] * len(trace) for i, trace in enumerate(lines_val)]
    ts3_train = [[0.0] * len(trace) for trace in lines_train]
    ts3_val = [[0.0] * len(trace) for trace in lines_val]
    ts4_train = [[1.0] * len(trace) for trace in lines_train]
    ts4_val = [[1.0] * len(trace) for trace in lines_val]
    
    # FIXED: Calculate divisors ONLY from training data
    all_times = [t for trace in ts_train for t in trace]
    divisor = np.mean(all_times) if all_times else 1.0
    all_times2 = [t for trace in ts2_train for t in trace]
    divisor2 = np.mean(all_times2) if all_times2 else 1.0
    
    # Vectorize data
    encoding_dim = len(vocab)
    one_hot_indices = {token: i for i, token in enumerate(vocab)}
    
    X_train_tax, y_act_train, y_time_train = vectorize_fold_tax(
        lines_train, ts_train, ts2_train, ts3_train, ts4_train,
        divisor, divisor2, encoding_dim,
        use_one_hot=True, one_hot_indices=one_hot_indices, maxlen=maxlen
    )
    
    # Validation data may have unknown activities - they will be handled
    X_val_tax, y_act_val, y_time_val = vectorize_fold_tax(
        lines_val, ts_val, ts2_val, ts3_val, ts4_val,
        divisor, divisor2, encoding_dim,
        use_one_hot=True, one_hot_indices=one_hot_indices, maxlen=maxlen
    )
    
    # Build model
    total_features = encoding_dim + 5
    model = build_tax_model(maxlen, total_features, len(target_tokens))
    
    print(f"\nTraining tax model with one-hot embeddings...")
    model.summary()
    
    # Setup callbacks
    model_directory = os.path.join(MODELS_DIR, method, log_name + ".h5")
    os.makedirs(os.path.dirname(model_directory), exist_ok=True)
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=32)
    model_checkpoint = ModelCheckpoint(
        model_directory, monitor='val_loss', verbose=1,
        save_best_only=True, save_weights_only=False
    )
    lr_reducer = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=10, 
        verbose=1, min_delta=0.0001
    )
    
    # Train model
    model.fit(
        X_train_tax,
        {'act_output': y_act_train, 'time_output': y_time_train},
        validation_data=(X_val_tax, {"act_output": y_act_val, "time_output": y_time_val}),
        verbose=1,
        callbacks=[early_stopping, model_checkpoint, lr_reducer],
        batch_size=32,
        epochs=EPOCHS,
        shuffle=False
    )
    
    # Store vocab info for evaluation
    vocab_info = {
        'vocab': vocab,
        'target_tokens': target_tokens,
        'target_token_indices': target_token_indices,
        'one_hot_indices': one_hot_indices,
        'divisor': divisor,
        'divisor2': divisor2
    }
    
    return model, model_directory, maxlen, encoding_dim, vocab_info


# =====================================================================
# EVALUATION FUNCTIONS
# =====================================================================

def evaluate_model(data_dict, train_output, method, log_name):
    """Dispatcher for evaluating different model types."""
    if method == "tax_one_hot":
        return evaluate_tax_model(
            data_dict,
            train_output["model_path"],
            method,
            log_name,
            train_output["maxlen"],
            train_output["emb_dim"],
            train_output["vocab_info"]
        )
    else:
        return evaluate_evermann_model(
            data_dict,
            train_output["model_path"],
            method,
            log_name,
            train_output["use_one_hot"],
            train_output["emb_dim"]
        )


def evaluate_evermann_model(data_dict, model_directory, method, log_name, 
                            use_one_hot_flag, chosen_emb_dim):
    """
    Evaluate trained model on test set.
    
    Args:
        data_dict: Dictionary containing datasets and metadata
        model_directory: Path to saved model weights
        method: Embedding method name
        log_name: Name of the event log
        use_one_hot_flag: Whether one-hot encoding was used
        chosen_emb_dim: Dimension of embeddings
        
    Returns:
        dict: Evaluation results
    """
    X_test = data_dict["X_test"]
    idx = data_dict["idx"]
    vocab_size = data_dict["vocab_size"]
    
    # Build model (stateless for faster inference)
    if use_one_hot_flag:
        test_model = build_model(vocab_size, EMBEDDING_DIM, RNN_UNITS, batch_size=None)
    else:
        test_model = build_model_pretrained(
            vocab_size, chosen_emb_dim, RNN_UNITS,
            batch_size=None, embedding_matrix=None
        )
    
    # Load trained weights
    test_model.load_weights(model_directory)
    
    y_pred = []
    y_true = []
    
    last_case_id = idx["[EOC]"]
    unk_id = idx["[UNK]"]
    
    # Process traces
    print(f"\nEvaluating {method} on {log_name}...")
    unknown_predictions = 0
    total_predictions = 0
    
    for trace in tqdm(X_test, desc="Processing traces"):
        if len(trace) < 2:
            continue
        
        # Input = full trace except last event
        inp = np.array(trace[:-1])[None, :]  # shape (1, T)
        targets = trace[1:]  # shape (T)
        
        # Predict entire sequence at once
        preds = test_model(inp, training=False).numpy()[0]  # (T, vocab)
        probs = tf.nn.softmax(preds, axis=-1).numpy()
        
        # Collect predictions and ground truth
        for i, next_event in enumerate(targets):
            # Track if target is unknown
            if next_event == unk_id:
                unknown_predictions += 1
            
            total_predictions += 1
            y_pred.append(probs[i])
            y_true.append(np.eye(vocab_size)[next_event])
            
            # Stop at end-of-case token
            if next_event == last_case_id:
                break
    
    # Calculate metrics
    y_pred_a = np.argmax(y_pred, axis=1)
    y_true_a = np.argmax(y_true, axis=1)
    
    results = {
        "log": log_name,
        "method": method,
        "accuracy": accuracy_score(y_true_a, y_pred_a),
        "f1": f1_score(y_true_a, y_pred_a, average="weighted"),
        "unknown_predictions": unknown_predictions,
        "total_predictions": total_predictions,
        "unknown_ratio": unknown_predictions / total_predictions if total_predictions > 0 else 0
    }
    
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
    if unknown_predictions > 0:
        print(f"Unknown predictions: {unknown_predictions}/{total_predictions} ({results['unknown_ratio']:.2%})")
    
    return results


def evaluate_tax_model(data_dict, model_directory, method, log_name, 
                       maxlen, encoding_dim, vocab_info):
    """
    Evaluate trained tax model on test set.
    
    Args:
        data_dict: Dictionary containing datasets and metadata
        model_directory: Path to saved model
        method: Should be 'tax_one_hot'
        log_name: Name of the event log
        maxlen: Maximum sequence length (from training)
        encoding_dim: Dimension of activity encoding
        vocab_info: Vocabulary information from training
        
    Returns:
        dict: Evaluation results
    """
    X_test = data_dict["X_test"]
    idx = data_dict["idx"]
    
    # Convert vectorized traces back to activity names
    idx_reverse = {v: k for k, v in idx.items()}
    lines_test = [[idx_reverse[event_id] for event_id in trace if event_id != idx["[EOC]"]] 
                  for trace in X_test]
    
    # Create dummy temporal features
    ts_test = [[1.0] * len(trace) for trace in lines_test]
    ts2_test = [[float(i)] * len(trace) for i, trace in enumerate(lines_test)]
    ts3_test = [[0.0] * len(trace) for trace in lines_test]
    ts4_test = [[1.0] * len(trace) for trace in lines_test]
    
    # Vectorize test data using training vocabulary
    X_test_tax, y_act_test, y_time_test = vectorize_fold_tax(
        lines_test, ts_test, ts2_test, ts3_test, ts4_test,
        vocab_info['divisor'], vocab_info['divisor2'], encoding_dim,
        use_one_hot=True, one_hot_indices=vocab_info['one_hot_indices'],
        maxlen=maxlen
    )
    
    # Load model
    total_features = encoding_dim + 5
    model = build_tax_model(maxlen, total_features, len(vocab_info['target_tokens']))
    model.load_weights(model_directory)
    
    # Recompile model
    opt = Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipvalue=3)
    model.compile(
        loss={'act_output': 'categorical_crossentropy', 'time_output': 'mae'},
        optimizer=opt,
        metrics={"act_output": "acc", "time_output": "mae"}
    )
    
    # Predict
    print(f"\nEvaluating {method} on {log_name}...")
    preds = model.predict(X_test_tax, verbose=1, batch_size=32)
    y_act_pred_probs = preds[0]
    y_act_pred = np.argmax(y_act_pred_probs, axis=1)
    y_act_true = np.argmax(y_act_test, axis=1)
    
    # Calculate metrics
    results = {
        "log": log_name,
        "method": method,
        "accuracy": accuracy_score(y_act_true, y_act_pred),
        "f1": f1_score(y_act_true, y_act_pred, average="weighted")
    }
    
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
    
    # Clear session
    tf.keras.backend.clear_session()
    
    return results


# =====================================================================
# MAIN EXECUTION
# =====================================================================

def run_kirchmann_experiments(full_path, log_name, csv_file):
    """
    Run experiments following data leakage prevention principles.
    
    Args:
        full_path: Path to XES event log
        log_name: Name of the log for identification
        csv_file: Path to CSV file for results
        
    Returns:
        Path to CSV file with results
    """
    fieldnames = [
        "log",
        "method",
        "accuracy",
        "f1",
        "training_time_seconds",
        "testing_time_seconds",
        "unknown_predictions",
        "total_predictions",
        "unknown_ratio"
    ]

    try:
        print(f"\n{'='*70}")
        print(f"Processing log: {log_name}")
        print(f"{'='*70}")
        
        # Extract and split data (with leakage prevention)
        data_dict = extract_log(full_path)

        # Create CSV only if it does NOT exist
        file_exists = os.path.isfile(csv_file)

        with open(csv_file, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            # Write header only once
            if not file_exists:
                writer.writeheader()

        for method in ENCODING_METHODS:
            print(f"\n{'-'*70}")
            print(f"Method: {method}")
            print(f"{'-'*70}")
            
            start_time = time.time()

            train_output = train_model(data_dict, method, log_name)
            train_end_time = time.time()

            results = evaluate_model(
                data_dict, train_output, method, log_name
            )
            test_end_time = time.time()

            results["training_time_seconds"] = train_end_time - start_time
            results["testing_time_seconds"] = test_end_time - train_end_time
            
            # Add unknown tracking if not present
            if "unknown_predictions" not in results:
                results["unknown_predictions"] = 0
                results["total_predictions"] = 0
                results["unknown_ratio"] = 0.0

            # Append result directly to CSV
            with open(csv_file, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(results)
            
            print(f"\nResults saved for method: {method}")
            print(f"Training time: {results['training_time_seconds']:.2f}s")
            print(f"Testing time: {results['testing_time_seconds']:.2f}s")
            
            # Clear session to free memory
            tf.keras.backend.clear_session()

        print(f"\n{'='*70}")
        print(f"All experiments completed for {log_name}")
        print(f"Results saved to: {csv_file}")
        print(f"{'='*70}\n")
        
        return csv_file

    except Exception as e:
        print(f"Error with experiments: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
