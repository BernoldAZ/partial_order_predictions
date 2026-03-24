# =============================================================================
# Based on:
# Chiorrini, Andrea, et al. "Embedding Process Structure in Activities for
# Process Mapping and Comparison." European Conference on Advances in
# Databases and Information Systems. Cham: Springer International Publishing, 2022.
# https://doi.org/10.1007/978-3-031-15743-1_12
# =============================================================================
import re
from pm4py.objects.process_tree.obj import Operator


def make_visible(data):
    def replacement(match):
        if match.group(6) != '$invisible$':
            return match.group(0)
        orig_ident = match.group(2)
        orig_name = match.group(4)
        orig_activity = match.group(6)
        ident = orig_ident
        name = f'{orig_ident} {orig_name}'.replace(' ',
                                                   '-')  # matches regular expression '^n[0-9]+-tau-' or contains '-tau-'
        activity = name
        return match.group(1) + ident + match.group(3) + name + match.group(5) + activity + match.group(7)

    return re.compile(
        '(<transition id=")([^"]*)(">.*?<name><text>)(.*?)(</text></name>.*?activity=")([^"]*)(".*?</transition>)').sub(
        replacement, data)

def merge_dictionaries(dictionaries):
    return {key: value for dictionary in dictionaries for (key, value) in dictionary.items()}

def depth_first(aggregate, leaf):
    def inner(tree):
        return aggregate(tree, map(inner, tree.children)) if tree.children else leaf(tree)
    return inner

def maximum_degree_of_parallelism_aggregate(tree, subresults):
    subresults = list(subresults)
    if tree.operator == Operator.PARALLEL:
        subdegrees = [max(subresult.values()) for subresult in subresults]
        total_degree = sum(subdegrees)
        return {node: total_degree - subdegree + degree for (subresult, subdegree) in zip(subresults, subdegrees) for (node, degree) in subresult.items()}
    return merge_dictionaries(subresults)

def degree_of_choice_aggregate(tree, subresults):
    if tree.operator == Operator.XOR:
        return {node: degree + len(tree.children) - 1 for subresult in subresults for (node, degree) in subresult.items()}
    return merge_dictionaries(subresults)

def strictly_loopable_aggregate(tree, subresults):
    if tree.operator == Operator.LOOP and all(child.operator is None for child in tree.children):
        return {node: True for subresult in subresults for node in subresult}
    return merge_dictionaries(subresults)

def loopable_aggregate(tree, subresults):
    if tree.operator == Operator.LOOP:
        return {node: True for subresult in subresults for node in subresult}
    return merge_dictionaries(subresults)

def collate_dictionaries(dictionaries):
    keys = {key for dictionary in dictionaries for key in dictionary.keys()}
    return {key: tuple(dictionary[key] for dictionary in dictionaries) for key in keys}

def feature_map(tree):
    return {key: (maximum_degree_of_parallelism, degree_of_choice, int(strictly_loopable), int(loopable and not strictly_loopable)) for (key, (maximum_degree_of_parallelism, degree_of_choice, loopable, strictly_loopable)) in collate_dictionaries((depth_first(maximum_degree_of_parallelism_aggregate, lambda node: {node: 1})(tree), depth_first(degree_of_choice_aggregate, lambda node: {node: 1})(tree), depth_first(loopable_aggregate, lambda node: {node: False})(tree), depth_first(strictly_loopable_aggregate, lambda node: {node: False})(tree))).items()}