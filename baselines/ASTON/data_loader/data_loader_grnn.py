from copy import copy
import os
import concurrent
from scipy import sparse as sp

from pm4py.algo.conformance.tokenreplay.variants.token_replay import apply_hidden_trans
# pm4py >= 2.3 moved these modules (pm4py.objects.petri -> pm4py.objects.petri_net)
from pm4py.objects.petri_net.utils.petri_utils import get_places_shortest_path_by_hidden
from pm4py.objects.petri_net.obj import PetriNet

import numpy as np
from data_loader.data_loader import __load_dataset_dict_simple_resource_fold
import networkx as nx
import pandas as pd
import pm4py

from pm4py.objects.petri_net.utils import networkx_graph
from pm4py.objects.petri_net import semantics


class GraphVectorizer:

    def __init__(self, net, initial_marking, final_marking):
        self.net = net
        self.initial_marking = initial_marking
        self.final_marking = final_marking
        self.nx_graph = networkx_graph.create_networkx_directed_graph(net)
        self.adjacency_matrix, self.N, self.nodes, self.places, self.transitions, self.arcs = self.get_features_from_graph(self.nx_graph, net, net.transitions)
        self.delete_transitions(self.nx_graph)
        self.max_transition_len = self.calculate_max_transitions()
        self.k = 5
        self.F = self.k  # Ignore transitions for now
        self.activity_to_transition = {}
        self.places_shortest_path = get_places_shortest_path_by_hidden(self.net, 50)
        for transition in self.net.transitions:
            self.activity_to_transition[transition.label] = transition


    def get_features_from_graph(self, nx_graph, net, transitions):
        # nx.to_numpy_matrix was removed in networkx 3.0
        adjacency_matrix = nx.to_numpy_array(nx_graph[0])
        number_of_nodes = adjacency_matrix.shape[0]
        nodes = {}
        for k, v in nx_graph[1].items():
            # If it is a transition
            if isinstance(v, PetriNet.Transition):
                if v.label is not None:
                    nodes[v.label] = k
                else:
                    nodes[v.name] = k
            else:
                nodes[v] = k
        #nodes = {v: k for k, v in nx_graph[1].items()}

        places = [p.name for p in net.places]
        places = sorted(places)
        transitions = [t.label if t.label is not None else t.name for t in net.transitions]
        transitions = sorted(transitions)
        arcs = net.arcs
        return adjacency_matrix, number_of_nodes, nodes, places, transitions, arcs


    def delete_transitions(self, nx_graph):
        # Correspondence between the adjacency matrix and the feature matrix using only places
        place_assignment = {}
        self.transition_assignment = {}

        # Start on 1 since the 0 is the padding value
        for i, place in enumerate(self.places, start=1):
            place_assignment[place] = i

        for i, transition in enumerate(self.transitions, start=1):
            self.transition_assignment[str(transition)] = i

        items = self.nodes.items()
        nodes = {str(key) : value for key, value in items}
        reverse_nodes = {value : str(key) for key, value in items}
        self.reverse_nodes = reverse_nodes
        self.nodes = place_assignment

        #print("Nodes: ", nodes)

        self.adjacency_matrix = np.zeros(shape=(len(self.places), len(self.places)))
        self.N = len(self.places)
        #print("Transition list: ", self.transitions)
        #print("Nodes: ", nodes)
        for transition in self.transitions:
            # Get the id of the transition in the graph
            transition_id = nodes[transition]
            in_edges = self.nx_graph[0].in_edges(transition_id)
            out_edges = self.nx_graph[0].out_edges(transition_id)
            #print("Transition id: ", transition_id)
            #print("IN EDGES: ", in_edges)
            #print("OUT EDGES: ", out_edges)
            for in_edge in in_edges:
                for out_edge in out_edges:
                    in_place = reverse_nodes[in_edge[0]]
                    out_place = reverse_nodes[out_edge[1]]
                    i = place_assignment[in_place]
                    j = place_assignment[out_place]
                    # Substract 1 since the ids start in 1
                    self.adjacency_matrix[i - 1, j - 1] = 1

    def calculate_max_transitions(self):
        items = self.nodes.items()
        nodes = {str(key) : value for key, value in items}
        max_len = 0
        for place in self.net.places:
            in_edges = self.nx_graph[0].in_edges(nodes[str(place)])
            if max_len < len(in_edges):
                max_len = len(in_edges)

        return max_len

    def vectorize(self, trace):
        trace_events = []
        F_event = np.zeros(shape=(self.N, self.F), dtype="float32")


        marking = self.initial_marking
        node_activations = {}
        for n in self.nodes:
            node_activations[n] = 0

        #for place_raw in marking:
            #place = str(place_raw)

            # Substract 1 in the matrix so as to get a correct reference on the node matrix (the id are 1..+ and the matrix is 1..+-1)
            #F_event[self.nodes[place] - 1][0] = self.N + 1
            #for i in range(self.max_transition_len):
                #F_event[self.nodes[place] - 1][1 + i] = len(self.transitions) + 1

        accumulated_markings = []

        for curr_event, event in enumerate(trace, 1):
            all_markings, marking, fired_transitions = self.replay_event(event, marking, node_activations)
            #for place_list in all_markings:
            #    for place_raw in place_list:

            accumulated_markings.append(all_markings)

            if len(accumulated_markings) > 1:
                if len(accumulated_markings) <= self.k:
                    for i in range(len(accumulated_markings) - 1):
                        prev_marking = accumulated_markings[-(i+2)]
                        for place_list in prev_marking:
                            for place_raw in place_list:
                                place = str(place_raw)
                                F_event[self.nodes[place] - 1][i] = 1
                else:
                    for i in range(self.k):
                        prev_marking = accumulated_markings[-(i+2)]
                        for place_list in prev_marking:
                            for place_raw in place_list:
                                place = str(place_raw)
                                #print("F event shape: ", F_event.shape)
                                #print("Accessing: (", self.nodes[place] - 1, ", ", i + 1, ")")
                                F_event[self.nodes[place] - 1][i] = 1

            trace_events.append(np.expand_dims(F_event.copy(), axis=0))

        return np.concatenate(trace_events, axis=0)

    def replay_event(self, event, marking, node_activations=None):
        # Calculate the marking for the partial trace
        n_regular_firings = 0
        fired_transitions = []
        activated_places = []
        all_markings = []
        # This comprobation checks whether the event is contemplated in the model
        m_marking = copy(marking)
        if event["task"] in self.activity_to_transition:
            transition_to_activate = self.activity_to_transition[event["task"]]
            # If the transition we have to activate is not yet activated, try to activate every hidden transition
            # possible

            # TODO: añadir el marking anterior o no?
            #for m in m_marking:
            #print("Appending initial_marking: ", str(m))
            #all_markings.append(m)

            if not semantics.is_enabled(transition_to_activate, self.net, m_marking):
                # pm4py 2.7 added `exhaustive_invisible_exploration`; False (pm4py's
                # token-replay default) keeps the original shortest-path behaviour
                _, _, act_trans, _ = apply_hidden_trans(transition_to_activate, self.net, copy(m_marking),
                                                        self.places_shortest_path, [], 0, set(), [copy(m_marking)],
                                                        False)
                tmp_firing = []
                tmp_marking = []

                for act_tran in act_trans:
                    tmp_firing.append(act_tran)
                    for arc in act_tran.out_arcs:
                        activated_places.append(arc.target)
                    m_marking = semantics.execute(act_tran, self.net, m_marking)
                    for m in m_marking:
                        #print("Firing hidden transition: ", str(act_tran) + " for place ", str(m))
                        tmp_marking.append(m)

                all_markings.append(tmp_marking)
                fired_transitions.append(tmp_firing)

            if not semantics.is_enabled(transition_to_activate, self.net, m_marking):
                for arc in transition_to_activate.in_arcs:
                    if arc.source not in m_marking:
                        #print("M marking failed: ", arc.source)
                        m_marking[arc.source] += 1

            for arc in transition_to_activate.out_arcs:
                activated_places.append(arc.target)

            m_marking = semantics.execute(transition_to_activate, self.net, m_marking)
            tmp_firing = []
            tmp_marking = []
            for m in m_marking:
                #print("Firing transition ", str(transition_to_activate), " for place ", str(m))
                tmp_marking.append(m)
            tmp_firing.append(transition_to_activate)
            fired_transitions.append(tmp_firing)
            all_markings.append(tmp_marking)


            # Calculate the number of node activations
            for m in all_markings:
                for k in m:
                    if node_activations is not None:
                        node_activations[str(k)] += 1
        return all_markings, m_marking, fired_transitions

    @staticmethod
    def localpooling_filter(A, symmetric=True):
        r"""
        Computes the graph filter described in
        [Kipf & Welling (2017)](https://arxiv.org/abs/1609.02907).
        :param A: array or sparse matrix with rank 2 or 3;
        :param symmetric: boolean, whether to normalize the matrix as
        \(\D^{-\frac{1}{2}}\A\D^{-\frac{1}{2}}\) or as \(\D^{-1}\A\);
        :return: array or sparse matrix with rank 2 or 3, same as A;
        """
        fltr = A.copy()
        if sp.issparse(A):
            I = sp.eye(A.shape[-1], dtype=A.dtype)
        else:
            I = np.eye(A.shape[-1], dtype=A.dtype)
        if A.ndim == 3:
            for i in range(A.shape[0]):
                # TODO: disable self loops for now
                A_tilde = A[i] + I
                #A_tilde = A[i]
                fltr[i] = GraphVectorizer.normalized_adjacency(A_tilde, symmetric=symmetric)
        else:
            A_tilde = A + I
            #A_tilde = A
            fltr = GraphVectorizer.normalized_adjacency(A_tilde, symmetric=symmetric)

        if sp.issparse(fltr):
            fltr.sort_indices()
        return fltr

    @staticmethod
    def normalized_adjacency(A, symmetric=True):
        r"""
        Normalizes the given adjacency matrix using the degree matrix as either
        \(\D^{-1}\A\) or \(\D^{-1/2}\A\D^{-1/2}\) (symmetric normalization).
        :param A: rank 2 array or sparse matrix;
        :param symmetric: boolean, compute symmetric normalization;
        :return: the normalized adjacency matrix.
        """
        if symmetric:
            normalized_D = GraphVectorizer.degree_power(A, -0.5)
            output = normalized_D.dot(A).dot(normalized_D)
        else:
            normalized_D = GraphVectorizer.degree_power(A, -1.)
            output = normalized_D.dot(A)
        return output

    @staticmethod
    def degree_power(A, k):
        r"""
        Computes \(\D^{k}\) from the given adjacency matrix. Useful for computing
        normalised Laplacian.
        :param A: rank 2 array or sparse matrix.
        :param k: exponent to which elevate the degree matrix.
        :return: if A is a dense array, a dense array; if A is sparse, a sparse
        matrix in DIA format.
        """
        degrees = np.power(np.array(A.sum(1)), k).flatten()
        degrees[np.isinf(degrees)] = 0.
        if sp.issparse(A):
            D = sp.diags(degrees)
        else:
            D = np.diag(degrees)
        return D


def vectorize_trace(params):
    log, trace_index, vectorizer = params
    trace = log[trace_index]
    vectorized_trace = [vectorizer.vectorize(trace[:i + 1]) for i in range(len(trace))]
    return vectorized_trace

def _load_dataset_grnn(path, net, initial_marking, final_marking):
    df = pd.read_csv(path, header=0, delimiter=',', keep_default_na=False, dtype=object)
    log = pm4py.format_dataframe(df, case_id="caseid", activity_key="task", timestamp_key="end_timestamp")
    log = pm4py.convert_to_event_log(log)
    vectorizer = GraphVectorizer(net, initial_marking, final_marking)

    vectorized_log = []
    for trace in log:
        for i in range(len(trace)):
            vectorized_log.append(vectorizer.vectorize(trace[:i + 1]))

    """
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        params = [(log, i, vectorizer) for i in range(len(log))]
        results = executor.map(vectorize_trace, params)
    """

    #vectorized_log = [item for sublist in results for item in sublist]



    return vectorized_log, vectorizer



def load_grnn_dataset_fold(dataset, num_fold, net, initial_marking, final_marking):
    dataset_config = __load_dataset_dict_simple_resource_fold(dataset)
    # Datasets with different splits

    fold_train, vectorizer = _load_dataset_grnn('data/train_fold' + str(num_fold) + '_variation0_' + dataset_config['eventlog'], net, initial_marking, final_marking)
    fold_val, _ = _load_dataset_grnn('data/val_fold' + str(num_fold) + '_variation0_' + dataset_config['eventlog'], net, initial_marking, final_marking)
    fold_test, _ = _load_dataset_grnn('data/test_fold' + str(num_fold) + '_variation0_' + dataset_config['eventlog'], net, initial_marking, final_marking)

    adjacency_matrix = GraphVectorizer.normalized_adjacency(vectorizer.adjacency_matrix, symmetric=False)

    return fold_train, fold_val, fold_test, vectorizer.N, vectorizer.F, adjacency_matrix, vectorizer
