import pandas as pd
from pm4py.objects.log.importer.xes import importer as xes_importer
from datetime import datetime

import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt

import torch
from torch_geometric.data import Data, Batch
from collections import defaultdict
from tqdm import tqdm

from torch_geometric.loader import DataLoader

###################################################
# Extract trace for event log file (.xes)
###################################################

def extract_traces(log_path):
    """
    Load an event log and extract all traces + unique activities.

    Parameters:
        log_path (str): Path to the event log file

    Returns:
        tuple:
            - list: traces (with full event data)
            - list: unique activities in the log
    """
    
    log = xes_importer.apply(log_path)

    result = []
    activities = set()  # to store unique activity names

    for trace in log:
        trace_info = {
            "trace_attributes": dict(trace.attributes),
            "events": []
        }

        for event in trace:
            event_dict = dict(event)
            trace_info["events"].append(event_dict)

            # Collect activity name
            if "concept:name" in event_dict:
                activities.add(event_dict["concept:name"])

        result.append(trace_info)

    return result, activities

###################################################
# Truncade activities timestamps
###################################################

def truncate_datetime(dt, level):
    """
    Truncate a datetime object to a specified level.

    level:
    "year", "month", "day", "hour", "minute", "second"
    """

    levels = ["year", "month", "day", "hour", "minute", "second"]

    if level not in levels:
        raise ValueError(f"Invalid level. Choose from {levels}")

    # Default values for missing components
    values = {
        "year": dt.year,
        "month": 1,
        "day": 1,
        "hour": 0,
        "minute": 0,
        "second": 0
    }

    # Fill values up to desired level
    for l in levels:
        values[l] = getattr(dt, l)
        if l == level:
            break

    return datetime(
        values["year"],
        values["month"],
        values["day"],
        values["hour"],
        values["minute"],
        values["second"],
        tzinfo=dt.tzinfo
    )

def truncate_trace_timestamps(trace, level):
    """
    Apply datetime truncation to all events in a trace.

    Parameters:
        trace (dict): Trace with 'trace_attributes' and 'events'
        level (str): Truncation level (year, month, day, hour, minute, second, none)

    Returns:
        dict: New trace with truncated timestamps
    """

    if level == "none":
        return trace

    # Copy trace structure (avoid mutating original)
    new_trace = {
        "trace_attributes": dict(trace["trace_attributes"]),
        "events": []
    }

    for event in trace["events"]:
        new_event = dict(event)

        if "time:timestamp" in new_event:
            new_event["time:timestamp"] = truncate_datetime(
                new_event["time:timestamp"], level
            )

        new_trace["events"].append(new_event)

    return new_trace

###################################################
# Trace visualization (Partial order visualization)
###################################################

def trace_to_graph(trace):
    """
    Convert a trace into a DAG based on timestamp ordering.
    Events with identical timestamps are treated as a "block":
        - each event has its own node
        - all events in the block connect from same previous layer nodes
        - all events connect to same next layer nodes
    """
    
    G = nx.DiGraph()
    
    # Step 1: group events by timestamp
    time_groups = defaultdict(list)
    for event in trace["events"]:
        ts = event.get("time:timestamp")
        if ts is not None:
            time_groups[ts].append(event)
    
    # Step 2: sort timestamps
    sorted_times = sorted(time_groups.keys())
    
    # Keep track of nodes in the previous layer
    previous_nodes = []
    
    # Step 3: create nodes and connect edges layer by layer
    for ts in sorted_times:
        events = time_groups[ts]
        current_nodes = []
        
        # Create a node for each event
        for i, event in enumerate(events):
            # Node ID = timestamp index + event index
            node_id = f"{ts.isoformat()}_{i}"
            G.add_node(node_id, timestamp=ts, event=event, activity=event.get("concept:name"))
            current_nodes.append(node_id)
        
        # Connect previous layer nodes → all current nodes
        for prev in previous_nodes:
            for curr in current_nodes:
                G.add_edge(prev, curr)
        
        # Update previous_nodes for next iteration
        previous_nodes = current_nodes
    
    return G


def visualize_block(G):
    # Layer nodes by timestamp
    layers = sorted(set(nx.get_node_attributes(G, "timestamp").values()))
    pos = {}
    
    for layer_index, ts in enumerate(layers):
        # nodes in this layer
        nodes = [n for n, d in G.nodes(data=True) if d["timestamp"] == ts]
        for i, node in enumerate(nodes):
            pos[node] = (layer_index, -i)  # horizontal timeline, stack concurrent nodes vertically
    
    labels = {n: G.nodes[n]["activity"] for n in G.nodes}
    
    plt.figure(figsize=(20, 4))
    nx.draw(G, pos, with_labels=False, node_size=2000, node_color="lightblue")
    nx.draw_networkx_labels(G, pos, labels)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="->", width=2)
    plt.title("Trace Graph (Block-Style for Same Timestamp)")
    plt.axis("off")
    plt.show()



###################################################
# Traces to pytorch geometric dataloaders
###################################################
    
# ---------------------------
# 1. Prefix graph generator
# ---------------------------
def trace_to_pyg_prefixes(trace, activity_to_idx):

    dataset = []

    # Group by timestamp
    time_groups = defaultdict(list)
    for event in trace["events"]:
        ts = event.get("time:timestamp")
        if ts is not None:
            time_groups[ts].append(event)

    sorted_times = sorted(time_groups.keys())

    # Global storage (grows with prefixes)
    node_activities = []
    node_timestamps = []
    edge_list = []
    edge_attr_list = []

    previous_node_indices = []

    for t_idx in range(len(sorted_times) - 1):
        ts = sorted_times[t_idx]
        next_ts = sorted_times[t_idx + 1]

        current_node_indices = []

        # --- Add nodes ---
        for event in time_groups[ts]:
            node_idx = len(node_activities)

            node_activities.append(event.get("concept:name"))
            node_timestamps.append(ts)

            current_node_indices.append(node_idx)

        # --- Add edges ---
        for prev in previous_node_indices:
            for curr in current_node_indices:
                delta = (
                    node_timestamps[curr] - node_timestamps[prev]
                ).total_seconds()

                edge_list.append((prev, curr))
                edge_attr_list.append(delta)

        # --- Build PyG graph for this prefix ---
        indices = torch.tensor(
            [activity_to_idx[a] for a in node_activities]
        )

        x = torch.eye(len(activity_to_idx))[indices].float()

        if len(edge_list) > 0:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr_list, dtype=torch.float).unsqueeze(1)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1))

        # --- Target ---
        y = torch.zeros(len(activity_to_idx))
        for event in time_groups[next_ts]:
            act = event.get("concept:name")
            y[activity_to_idx[act]] = 1.0

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y.unsqueeze(0)
        )

        dataset.append(data)

        previous_node_indices = current_node_indices

    return dataset

# ---------------------------
# 2. Full pipeline
# ---------------------------
def traces_to_pyg_loaders(traces, activities, truncation_level):
    """
    Convert traces into prefix-based PyG dataset with targets
    Also returns trace-to-graph index mapping
    """

    activity_to_idx = {act: i for i, act in enumerate(activities)}

    all_graphs = []
    trace_graph_ranges = []

    for trace in tqdm(traces, desc="Processing traces"):
        trace_name = trace.get("trace_attributes", {}).get("concept:name", "unknown")

        start_idx = len(all_graphs)

        truncated_trace = truncate_trace_timestamps(trace, truncation_level)
        prefix_graphs = trace_to_pyg_prefixes(truncated_trace, activity_to_idx)

        all_graphs.extend(prefix_graphs)

        end_idx = len(all_graphs) - 1

        if prefix_graphs:  # avoid empty traces
            trace_graph_ranges.append({
                "concept:name": trace_name,
                "start": start_idx,
                "end": end_idx
            })

    # Split based on number of traces. 65% train, 15% validation, 20% test
    n_traces = len(trace_graph_ranges)
    n_train_traces = int(0.65 * n_traces)
    n_val_traces = int(0.15 * n_traces)
    n_test_traces = n_traces - n_train_traces - n_val_traces # Remaining traces go to test

    # Get graph indices using "end" index of the last trace
    train_end_idx = trace_graph_ranges[n_train_traces - 1]["end"] + 1
    val_end_idx = trace_graph_ranges[n_train_traces + n_val_traces - 1]["end"] + 1

    # Split the dataset
    train_data = all_graphs[:train_end_idx]
    val_data = all_graphs[train_end_idx:val_end_idx]
    test_data = all_graphs[val_end_idx:]

    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    print("Total traces ", len(trace_graph_ranges))
    print("Total inputs ", len(all_graphs))
    print("Training ", len(train_data))
    print("Validation ", len(val_data))
    print("Test ", len(test_data))

    return train_loader, val_loader, test_loader, activity_to_idx, trace_graph_ranges