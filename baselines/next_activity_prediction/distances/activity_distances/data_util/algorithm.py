import os
from math import floor, ceil
from typing import List
from definitions import ROOT_DIR
import csv
from datetime import datetime, timedelta
from collections import defaultdict
from typing import List, Tuple, Dict
import numpy as np
import scipy

#Given Pm4py Event Log, return List of Lists of Activities (List of Traces)
def give_log_padding(log, ngram_size):
    log_list = list()
    for trace in log:
        log_list.append(list())
    i = 0
    padding_left_size =  floor((ngram_size - 1) / 2)
    padding_right_size = ceil((ngram_size - 1) / 2)
    padding_left = ["."]*(padding_left_size)
    padding_right = ["."]*(padding_right_size)
    for trace in log:
        #adjust for different ngram size
        log_list[i].extend(padding_left)
        log_list[i].extend(trace)
        log_list[i].extend(padding_right)
        i += 1
    return log_list


def transform_control_flow_lists_to_csv(control_flow_lists):
    # Convert event log to CSV format
    start_time = datetime(1970, 1, 1)
    time_delta = timedelta(hours=1)

    events = []
    for case_id, activities in enumerate(control_flow_lists):
        for event_index, activity in enumerate(activities):
            timestamp = start_time + event_index * time_delta
            events.append([case_id, activity, timestamp.isoformat() + "+00:00"])

    process_id = os.getpid()
    # Output CSV file name
    output_file = f"event_log_{process_id}.csv"


    # Write to CSV
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["CaseID", "Activity", "Timestamp"])
        writer.writerows(events)

    return output_file


def get_ngrams_dict(log: List[List[str]], ngram_size: int) -> Dict[Tuple[str, ...], int]:
    ngrams_dict = defaultdict(int)  # Using defaultdict to handle counting
    for sublist in log:
        for i in range(len(sublist) - ngram_size + 1):
            ngram = tuple(sublist[i:i + ngram_size])  # Convert the n-gram to a tuple to use as a dictionary key
            ngrams_dict[ngram] += 1  # Increment the count for this n-gram

    return dict(ngrams_dict)  # Convert back to a regular dictionary if desired


def get_context_dict(ngrams_dict: Dict[Tuple[str, ...], int]) -> Dict[str, Dict[Tuple[str, ...], int]]:
    context_dict = defaultdict(lambda: defaultdict(int))  # Nested defaultdict for context frequencies

    for ngram, count in ngrams_dict.items():
        middle_index = len(ngram) // 2
        middle_gram = ngram[middle_index]
        # Create context by removing the middle element
        context_before = ngram[:middle_index]
        context_after = ngram[middle_index + 1:]
        surrounding_grams = context_before + context_after

        context_dict[middle_gram][surrounding_grams] += count

    return {k: dict(v) for k, v in context_dict.items()}  # Convert inner defaultdicts to regular

def get_cosine_distance_dict(embeddings):
    # Normalize embeddings
    """
    embeddings = {
        activity: embedding / np.linalg.norm(embedding)
        for activity, embedding in embeddings.items()
    }
    """

    # Compute distances
    distances = {}
    for activity1 in embeddings.keys():
        for activity2 in embeddings.keys():
            distance = scipy.spatial.distance.cosine(embeddings[activity1], embeddings[activity2])
            #distance = cosine_distance(embeddings[activity1], embeddings[activity2])

            distances[(activity1, activity2)] = distance
    return distances

def cosine_distance(array1, array2):
    # Compute the dot product and magnitudes
    dot_product = np.dot(array1, array2)
    magnitude1 = np.linalg.norm(array1)
    magnitude2 = np.linalg.norm(array2)

    # Compute cosine similarity
    if magnitude1 == 0 or magnitude2 == 0:  # Handle zero vectors
        return 1.0  # Maximum cosine distance for orthogonal vectors
    cosine_similarity = dot_product / (magnitude1 * magnitude2)

    # Compute cosine distance
    return 1 - cosine_similarity
