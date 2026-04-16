"""
Lightweight subset of intrinsic evaluation metrics and helpers.

Why this exists
--------------
`evaluation/data_util/util_activity_distances_intrinsic.py` imports other modules
that transitively depend on tensorflow. For the uncertain intrinsic benchmark we
want a minimal dependency surface (no tensorflow required).

This module copies the metric-related parts (and the activity-selection helper)
needed by the intrinsic evaluation script:
  - get_activities_to_replace
  - get_knn_dict, get_precision_at_k
  - get_diameter, get_triplet
and their internal helper functions.

Behavior is intended to match the deterministic benchmark.
"""

from __future__ import annotations

import math
import random
import sys
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Tuple


def get_activities_to_replace(alphabet: List[str], different_activities_to_replace_count: int, sample_size: int):
    activity_dict = {i: activity for i, activity in enumerate(alphabet)}
    alphabet_len = len(alphabet)
    alphabet_len_minus_one = alphabet_len - 1
    sampled_combinations = set()
    comb_number = math.comb(alphabet_len, different_activities_to_replace_count)

    while len(sampled_combinations) < sample_size and len(sampled_combinations) < comb_number:
        activity_index_set = set()
        while len(activity_index_set) < different_activities_to_replace_count:
            activity_index_set.add(random.randint(0, alphabet_len_minus_one))
        activity_index_list = list(activity_index_set)
        activity_index_list.sort()
        sampled_combinations.add(frozenset(activity_index_list))

    sampled_combinations_id_list = [tuple(sampled_combination) for sampled_combination in sampled_combinations]
    sampled_combinations_list = [tuple(activity_dict[i] for i in tpl) for tpl in sampled_combinations_id_list]
    return sampled_combinations_list


def get_nested_dict(activity_distance_matrix: Dict[Tuple[str, str], float]) -> Dict[str, Dict[str, float]]:
    nested_dict = defaultdict(dict)
    for (outer_key, inner_key), value in activity_distance_matrix.items():
        nested_dict[outer_key][inner_key] = value
    return nested_dict


def get_n_nearest_neighbors(n, replaced_activities, similarity_scores_of_activities, activities_to_replace_with_count, reverse):
    neighbors = dict()
    similarity_scores_of_activities = get_nested_dict(similarity_scores_of_activities)
    for replaced_activity in replaced_activities:
        for i in range(activities_to_replace_with_count):
            replaced_activity_i = replaced_activity + ":" + str(i)
            distances = []
            if replaced_activity_i in similarity_scores_of_activities:
                for activity in similarity_scores_of_activities[replaced_activity_i]:
                    if activity != replaced_activity_i:
                        distances.append((activity, similarity_scores_of_activities[replaced_activity_i][activity]))
                if len(distances) > 0:
                    distances.sort(key=lambda x: x[1], reverse=reverse)
                    nearest_neighbors = [activity for activity, _ in distances[:n]]
                    neighbors[replaced_activity_i] = nearest_neighbors
    return neighbors


def get_knn_dict(activity_distance_matrix_dict, activities_to_replace_with_count, reverse, knn_count):
    knn_dict = defaultdict(lambda: defaultdict())
    for activity_distance_function in activity_distance_matrix_dict:
        for replaced_activities in activity_distance_matrix_dict[activity_distance_function].keys():
            nearest_neighbors = get_n_nearest_neighbors(
                knn_count,
                replaced_activities,
                activity_distance_matrix_dict[activity_distance_function][replaced_activities],
                activities_to_replace_with_count,
                reverse,
            )
            knn_dict[activity_distance_function][replaced_activities] = nearest_neighbors
    return dict(knn_dict)


def get_precision_at_k(knn_dict, activity_distances):
    precision_at_k_dict = defaultdict(float)
    for activity_distance in activity_distances:
        precision_replaced_activity_at_k = 0
        contributing_runs = 0
        for replaced_activities in knn_dict[activity_distance].keys():
            # If no replaced activities are present at this uncertainty level, skip.
            if not knn_dict[activity_distance][replaced_activities]:
                continue
            precision_at_k_sum = 0
            denom_replaced_with = 0
            for replaced_activity in replaced_activities:
                for replaced_activity_with in knn_dict[activity_distance][replaced_activities].keys():
                    replaced_activity_with_split = replaced_activity_with.split(":")
                    if not replaced_activity_with_split[1].isdigit():
                        sys.exit("Naming Error")
                    if replaced_activity_with_split[0] == replaced_activity:
                        precision_sum = 0
                        for activity in knn_dict[activity_distance][replaced_activities][replaced_activity_with]:
                            activity_split = activity.split(":")
                            if activity_split[0] == replaced_activity_with_split[0]:
                                precision_sum += 1
                        a = len(knn_dict[activity_distance][replaced_activities][replaced_activity_with])
                        if a > 0:
                            precision_at_k_sum += precision_sum / a
                            denom_replaced_with += 1
            if denom_replaced_with > 0:
                precision_replaced_activity_at_k += precision_at_k_sum / denom_replaced_with
                contributing_runs += 1
        precision_at_k_dict[activity_distance] = (
            precision_replaced_activity_at_k / contributing_runs if contributing_runs > 0 else 0.0
        )
    return dict(precision_at_k_dict)


def get_replaced_activities_dict(replaced_activities, activity_distance_matrix_nested):
    replaced_activities_dict = {key: list() for key in replaced_activities}
    for activity in activity_distance_matrix_nested:
        replaced_activity_with_split = activity.split(":")
        if len(replaced_activity_with_split) > 1:
            if not replaced_activity_with_split[1].isdigit():
                sys.exit("Naming Error")
            replaced_activities_dict[replaced_activity_with_split[0]].append(activity)
    return replaced_activities_dict


def get_out_of_class_activities(activity_distance_matrix_nested, replaced_activity):
    out_of_class_activities = list()
    for activity in activity_distance_matrix_nested.keys():
        if activity.split(":")[0] != replaced_activity:
            out_of_class_activities.append(activity)
    return out_of_class_activities


def get_in_class_combinations(replaced_activities_of_class):
    in_class_combinations = list(combinations(replaced_activities_of_class, 2))
    result = []
    for comb in in_class_combinations:
        result.append(comb)
        result.append(comb[::-1])
    return result


def get_normalized_similarity_scores_of_activities(similarity_scores_of_activities):
    min_value = min(similarity_scores_of_activities.values())
    max_value = max(similarity_scores_of_activities.values())
    if max_value == min_value:
        return {k: 0.0 for k in similarity_scores_of_activities.keys()}
    return {key: (value - min_value) / (max_value - min_value) for key, value in similarity_scores_of_activities.items()}


def get_avg_diameter_value(replaced_activities, similarity_scores_of_activities, activities_to_replace_with_count, reverse, alphabet):
    if reverse:
        similarity_scores_of_activities = {key: -value for key, value in similarity_scores_of_activities.items()}
    normalized = get_normalized_similarity_scores_of_activities(similarity_scores_of_activities)
    normalized = get_nested_dict(normalized)
    replaced_activities_dict = get_replaced_activities_dict(replaced_activities, normalized)
    diameter_per_class = []
    for replaced_activity in replaced_activities:
        in_class_combinations = get_in_class_combinations(replaced_activities_dict[replaced_activity])
        diameter_in_class = []
        for a, b in in_class_combinations:
            diameter_in_class.append(normalized[a][b])
        if len(in_class_combinations) != 0:
            diameter_per_class.append(sum(diameter_in_class) / len(diameter_in_class))
        else:
            diameter_per_class.append(0)
    return sum(diameter_per_class) / len(diameter_per_class)


def get_diameter(activity_distance_matrix_dict, activities_to_replace_with_count, reverse, alphabet):
    diameter_list = []
    for activity_distance_function in activity_distance_matrix_dict:
        for replaced_activities in activity_distance_matrix_dict[activity_distance_function].keys():
            diameter_list.append(
                get_avg_diameter_value(
                    replaced_activities,
                    activity_distance_matrix_dict[activity_distance_function][replaced_activities],
                    activities_to_replace_with_count,
                    reverse,
                    alphabet,
                )
            )
    return sum(diameter_list) / len(diameter_list)


def get_avg_triplet_value(replaced_activities, similarity_scores_of_activities, activities_to_replace_with_count, reverse, alphabet):
    # high bose similarity means low distance -> flip if needed
    if reverse:
        similarity_scores_of_activities = {key: -value for key, value in similarity_scores_of_activities.items()}

    normalized = get_normalized_similarity_scores_of_activities(similarity_scores_of_activities)
    normalized = get_nested_dict(normalized)

    replaced_activities_dict = get_replaced_activities_dict(replaced_activities, normalized)
    triplet_scores = []

    for replaced_activity in replaced_activities:
        in_class = replaced_activities_dict[replaced_activity]
        out_class = get_out_of_class_activities(normalized, replaced_activity)
        if not in_class or not out_class:
            continue

        # For each ordered pair of in-class activities, check how often it beats out-of-class
        in_pairs = get_in_class_combinations(in_class)
        for anchor, positive in in_pairs:
            ap = normalized[anchor][positive]
            better = 0
            for negative in out_class:
                if ap < normalized[anchor][negative]:
                    better += 1
            triplet_scores.append(better / len(out_class))

    if not triplet_scores:
        return 0.0
    return sum(triplet_scores) / len(triplet_scores)


def get_triplet(activity_distance_matrix_dict, activities_to_replace_with_count, reverse, alphabet):
    triplet_list = []
    for activity_distance_function in activity_distance_matrix_dict:
        for replaced_activities in activity_distance_matrix_dict[activity_distance_function].keys():
            triplet_list.append(
                get_avg_triplet_value(
                    replaced_activities,
                    activity_distance_matrix_dict[activity_distance_function][replaced_activities],
                    activities_to_replace_with_count,
                    reverse,
                    alphabet,
                )
            )
    return sum(triplet_list) / len(triplet_list) if triplet_list else 0.0


