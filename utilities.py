import os
import multiprocessing
import statistics
from datetime import datetime
from collections import defaultdict, Counter

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer


###################################################
# Extract traces from an event log file (.xes)
###################################################

def extract_traces(log_path):
    """
    Load an event log and extract all traces + unique activities.

    Parameters:
        log_path (str): Path to the event log file

    Returns:
        tuple:
            - list: traces (with full event data)
            - set: unique activities in the log
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
# Truncate activity timestamps
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
    n_test_traces = n_traces - n_train_traces - n_val_traces  # Remaining traces go to test

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


def make_concurrent_trace_iterator(traces):
    """
    Returns a callable. Each call returns (trace_index, trace) for the
    next trace in `traces` that has >=2 events sharing the same timestamp
    (i.e., concurrent activities). Cycles through the list; skips traces
    without concurrency.
    """
    state = {"idx": 0}

    def has_concurrency(trace):
        timestamps = [e["time:timestamp"] for e in trace["events"]]
        return len(timestamps) != len(set(timestamps))

    def next_concurrent_trace():
        n = len(traces)
        for _ in range(n):
            i = state["idx"]
            trace = traces[i]
            state["idx"] = (i + 1) % n
            if has_concurrency(trace):
                return i, trace
        return None  # no trace in the list has concurrent activities

    return next_concurrent_trace


###################################################
# Log-level statistics: duplicates, block sizes, timestamp granularity
###################################################

def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs):
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


# Ordered finest -> coarsest. Used to pick the finest granularity observed in a log.
GRANULARITY_RANK = {
    "microseconds": 0,
    "milliseconds": 1,
    "seconds": 2,
    "minutes": 3,
    "hours": 4,
    "days": 5,
}


def _timestamp_granularity(ts):
    """Return the finest non-zero time unit present in a single timestamp."""
    if ts.microsecond != 0:
        return "microseconds" if ts.microsecond % 1000 != 0 else "milliseconds"
    if ts.second != 0:
        return "seconds"
    if ts.minute != 0:
        return "minutes"
    if ts.hour != 0:
        return "hours"
    return "days"


def _process_single_file_wrapper(file_path):
    """
    Load an event log and compute duplicate-timestamp / block-size /
    timestamp-granularity / variant statistics for it.
    A "block" is the set of events in a trace sharing the same timestamp.
    """
    try:
        log = xes_importer.apply(file_path, parameters={"show_progress_bar": False})

        activities = set()
        trace_lengths, trace_pcts = [], []
        granularity_counts = Counter()

        # Block sizes = size of the group of events sharing the same timestamp
        # within a trace. "all_block_sizes" includes singleton blocks (size 1);
        # "concurrent_block_sizes" only includes blocks with >= 2 activities.
        all_block_sizes = []
        concurrent_block_sizes = []

        # Number of concurrent blocks (blocks with >= 2 activities) found in
        # each trace, one entry per trace (0 if the trace has none).
        concurrent_blocks_per_trace = []

        for trace in log:
            timestamp_counts = defaultdict(int)
            for e in trace:
                if "concept:name" in e:
                    activities.add(e["concept:name"])
                if "time:timestamp" in e:
                    ts = e["time:timestamp"]
                    timestamp_counts[ts] += 1
                    granularity_counts[_timestamp_granularity(ts)] += 1

            trace_length = len(trace)
            trace_lengths.append(trace_length)

            # Every distinct timestamp in the trace defines one block.
            block_sizes_this_trace = list(timestamp_counts.values())
            all_block_sizes.extend(block_sizes_this_trace)

            duplicates = [c for c in block_sizes_this_trace if c > 1]
            concurrent_block_sizes.extend(duplicates)
            concurrent_blocks_per_trace.append(len(duplicates))

            num_concurrent = sum(duplicates)
            trace_pcts.append(num_concurrent / trace_length * 100 if trace_length else 0.0)

        total_traces = len(log)
        traces_with_duplicates = sum(1 for c in concurrent_blocks_per_trace if c)
        total_activities = sum(trace_lengths)
        num_concurrent_activities = sum(concurrent_block_sizes)

        total_timestamps = sum(granularity_counts.values())
        finest_granularity = (
            min(granularity_counts, key=lambda g: GRANULARITY_RANK[g])
            if granularity_counts else "unknown"
        )
        granularity_pcts = {
            g: (count / total_timestamps * 100 if total_timestamps else 0.0)
            for g, count in granularity_counts.items()
        }

        return {
            "file": os.path.basename(file_path),
            "total_traces": total_traces,
            "num_variants": len(pm4py.get_variants(log)),
            "multiple_ts_%": round(traces_with_duplicates / total_traces, 2) if total_traces else 0.0,
            "%_conc_act": round(num_concurrent_activities / total_activities, 2) if total_activities else 0.0,
            "traces_with_duplicates": traces_with_duplicates,
            "num_activities": len(activities),
            "total_activities": total_activities,
            "num_concurrent_activities": num_concurrent_activities,

            # --- Block size stats: ALL blocks (including size-1 blocks) ---
            "num_blocks_all": len(all_block_sizes),
            "mean_block_size_all": _mean(all_block_sizes),
            "std_block_size_all": _std(all_block_sizes),
            "max_block_size_all": max(all_block_sizes) if all_block_sizes else 0,
            "min_block_size_all": min(all_block_sizes) if all_block_sizes else 0,

            # --- Block size stats: only blocks with >= 2 activities ---
            "num_blocks_concurrent": len(concurrent_block_sizes),
            "mean_block_size_concurrent": _mean(concurrent_block_sizes),
            "std_block_size_concurrent": _std(concurrent_block_sizes),
            "max_block_size_concurrent": max(concurrent_block_sizes) if concurrent_block_sizes else 0,
            "min_block_size_concurrent": min(concurrent_block_sizes) if concurrent_block_sizes else 0,

            # --- Concurrent blocks per trace (count, not size) ---
            # Averaged over ALL traces in the log, including traces with 0
            # concurrent blocks, so it reflects the log as a whole.
            "mean_concurrent_blocks_per_trace": _mean(concurrent_blocks_per_trace),
            "std_concurrent_blocks_per_trace": _std(concurrent_blocks_per_trace),
            "max_concurrent_blocks_per_trace": max(concurrent_blocks_per_trace) if concurrent_blocks_per_trace else 0,

            "mean_pct_concurrent_activities_per_trace": _mean(trace_pcts),
            "std_pct_concurrent_activities_per_trace": _std(trace_pcts),
            "mean_trace_size": _mean(trace_lengths),
            "std_trace_size": _std(trace_lengths),
            "max_trace_size": max(trace_lengths) if trace_lengths else 0,
            "min_trace_size": min(trace_lengths) if trace_lengths else 0,
            "finest_timestamp_granularity": finest_granularity,
            "timestamp_granularity_pcts": granularity_pcts,
        }
    except Exception as e:
        return {"file": os.path.basename(file_path), "error": str(e)}


def analyze_xes_folder_parallel(folder_path, max_workers=None):
    xes_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(folder_path)
        for f in files
        if f.lower().endswith((".xes", ".xes.gz"))
    ]

    with multiprocessing.Pool(max_workers or os.cpu_count() or 1) as pool:
        return list(tqdm(
            pool.imap(_process_single_file_wrapper, xes_files),
            total=len(xes_files), desc="Processing XES files", unit="file"
        ))


def summarize_duplicates_text(results):
    valid = [r for r in results if "error" not in r]
    if not valid:
        print("No valid results to summarize.")
        return

    files_with_duplicates = 0
    for r in valid:
        pct = (r["traces_with_duplicates"] / r["total_traces"] * 100) if r["total_traces"] else 0
        if r["traces_with_duplicates"] > 0:
            files_with_duplicates += 1

        granularity_breakdown = ", ".join(
            f"{g}: {p:.1f}%" for g, p in sorted(
                r["timestamp_granularity_pcts"].items(),
                key=lambda kv: GRANULARITY_RANK[kv[0]],
            )
        )

        print(
            f"Event Log: {r['file']}",
            f"Granularity breakdown: [{granularity_breakdown}]",
            f"Avg block size (all blocks): {r['mean_block_size_all']:.2f} "
            f"(n={r['num_blocks_all']})",
            f"Avg block size (blocks with >=2 activities only): "
            f"{r['mean_block_size_concurrent']:.2f} (n={r['num_blocks_concurrent']})",
            f"Avg concurrent blocks per trace: "
            f"{r['mean_concurrent_blocks_per_trace']:.2f} "
            f"(std={r['std_concurrent_blocks_per_trace']:.2f}, "
            f"max={r['max_concurrent_blocks_per_trace']})",
            f"\n"
        )

    print(f"\n{files_with_duplicates} out of {len(valid)} event logs have trace with duplicated timestamps.")