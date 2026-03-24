# =============================================================================
# Autoencoder:
# Neural embeddings via context reconstruction using an autoencoder architecture
#
# Based on:
# Gamallo-Fernandez, P., Vidal, J.C., and Lama, M.
# "Learning Context-Based Representations of Events in Complex Processes."
# 2023 IEEE International Conference on Web Services (ICWS), Chicago, IL, USA, 2023,
# pp. 214–223. https://doi.org/10.1109/ICWS60048.2023.00041
#
# Source code: https://gitlab.citius.gal/pedro.gamallo/PM_Embeddings/-/blob/master/src/embeddings_gener
# =============================================================================

import math
from pathlib import Path
import pandas as pd
from operator import itemgetter
import numpy as np
from scipy.stats import pearsonr
import networkx as nx
from distances.activity_distances.gamallo_fernandez_2023_context_based_representations.src.embeddings_generator.args_reader import read_input_args
from distances.activity_distances.gamallo_fernandez_2023_context_based_representations.src.embeddings_generator.logger import EmbGeneratorLogger
from distances.activity_distances.gamallo_fernandez_2023_context_based_representations.src.embeddings_generator.pandas_dataset import EventlogDataset
from distances.activity_distances.gamallo_fernandez_2023_context_based_representations.src.embeddings_generator.utils import EmbType, Config, DataFrameFields
import sys
import shutil
import random
import pm4py
from distances.activity_distances.data_util.algorithm import give_log_padding, get_ngrams_dict, get_context_dict, get_cosine_distance_dict

from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.util.xes_constants import DEFAULT_NAME_KEY

from pm4py.objects.log.exporter.xes import exporter as xes_exporter
import csv
import os
from datetime import datetime, timedelta
from distances.activity_distances.gamallo_fernandez_2023_context_based_representations.src.data_processor.main import write_csvs
import pytz
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.util.xes_constants import DEFAULT_NAME_KEY, DEFAULT_TRACEID_KEY, DEFAULT_TIMESTAMP_KEY
from distances.activity_distances.gamallo_fernandez_2023_context_based_representations.src.next_activity_prediction.utils import get_embeddings
from itertools import product
from scipy.spatial.distance import cosine

from distances.activity_distances.gamallo_fernandez_2023_context_based_representations.src.embeddings_generator import aerac
from definitions import ROOT_DIR
import time
import math
from itertools import product
from scipy.spatial.distance import cosine
from pathlib import Path

def get_context_based_distance_matrix(control_flow_lists, window_size, take_time = None):
    # Convert the control flow lists to an XES file and then to CSV.
    xes_filename, xes_log_path = transform_control_flow_lists_to_xes(control_flow_lists)
    csv_path, config_csv_path, attr_dict = write_csvs(xes_filename, xes_log_path)
    csv_filename = xes_filename.replace(".xes", ".csv")

    if take_time:
        start_time = time.time()
    # Setup sys.argv for holdout mode (train–val only).         "--win_size", str(math.floor((window_size - 1) / 2)),
    sys.argv = [
        "script.py",  # Fake script name (needed for argparse)
        "--dataset", "data/" + csv_filename,
        "--holdout",  # Use holdout mode instead of crossvalidation
        "--aerac",
        "--activity",
        "--emb_size", "128",
        "--win_size", str(math.floor((window_size - 1) / 2)),
        "--print_console_file"
    ]

    args = read_input_args()
    logger = EmbGeneratorLogger(args.print_mode)

    # Run the model in holdout mode (no test set).
    eventlog = EventlogDataset(args.dataset, cv_fold=None, read_test=False)
    loss, list_embeddings, list_embeddings_2, acc = start(eventlog, args, logger)

    logger.log_console(f'{str(args.emb_type)} VALIDATION LOSS: {loss:.6f}')

    if list_embeddings_2 is not None:
        Config.ATTR_TO_EMB = DataFrameFields.ACTIVITY_COLUMN
        logger.save_embeddings_holdout(list_embeddings, eventlog.filename, str(args.emb_type),
                                       args.emb_size, args.win_size)
        Config.ATTR_TO_EMB = DataFrameFields.RESOURCE_COLUMN
        logger.save_embeddings_holdout(list_embeddings_2, eventlog.filename, str(args.emb_type),
                                       args.emb_size, args.win_size)
    else:
        logger.save_embeddings_holdout(list_embeddings, eventlog.filename, str(args.emb_type),
                                       args.emb_size, args.win_size)

    logger.log_loss_holdout(Path(args.dataset).stem, str(args.emb_type),
                            args.emb_size, args.win_size, loss)

    # Load embeddings from the current run (no multiple folds to choose from)
    #embeddings_dict = get_embeddings(eventlog.filename, Config.ATTR_TO_EMB, str(args.emb_type),
    #                                 args.emb_size, args.win_size, logger)
    attr_to_embedding_dict = {attr: list_embeddings[id] for attr, id in attr_dict.items()}

    print()

    # Calculate pairwise cosine distances between attribute embeddings.
    pairwise_cosine_distances = {
        (key1, key2): float(cosine(attr_to_embedding_dict[key1], attr_to_embedding_dict[key2]))
        for key1, key2 in product(attr_to_embedding_dict.keys(), repeat=2)
    }

    pairwise_cosine_distances = get_cosine_distance_dict(attr_to_embedding_dict)
    if take_time:
        run_time = time.time() - start_time
    delete_temporary_files(xes_log_path, csv_path, config_csv_path, eventlog.filename)

    if take_time:
        return run_time
    else:
        return pairwise_cosine_distances, attr_to_embedding_dict






""" 
def get_context_based_distance_matrix(control_flow_lists, window_size):

    xes_filename, xes_log_path = transform_control_flow_lists_to_xes(control_flow_lists)

    csv_path, config_csv_path, attr_dict = write_csvs(xes_filename, xes_log_path)

    csv_filename = xes_filename.replace(".xes", ".csv")

    sys.argv = [
        "script.py",  # Fake script name (needed for argparse)
        "--dataset", "data/" + csv_filename,
        "--holdout",  # Use holdout mode instead of crossvalidation
        "--aerac",
        "--activity",
        "--emb_size", "200",
        "--win_size", str(math.floor((window_size - 1) / 2)),
        "--print_console_file"
    ]



    args = read_input_args()

    logger = EmbGeneratorLogger(args.print_mode)

    #list_of_list_embeddings = []

    if args.crossvalidation:
        losses = []
        accuracies = []
        for i in range(Config.CV_FOLDS):
            eventlog = EventlogDataset(args.dataset, cv_fold=i, read_test=True)

            loss_fold, list_embeddings, list_embeddings_2, acc = start(eventlog, args, logger)

            #list_of_list_embeddings.append(list_embeddings)

            logger.log_console(f'{str(args.emb_type)} TEST LOSS FOLD {i}: {loss_fold:.6f}')
            print(f'TEST ACCURACY FOLD {i}: {acc:.6f}')
            accuracies.append(acc)

            if list_embeddings_2 is not None:
                Config.ATTR_TO_EMB = DataFrameFields.ACTIVITY_COLUMN
                logger.save_embeddings_cv(list_embeddings, eventlog.filename, str(args.emb_type),
                                          args.emb_size, args.win_size, i)
                Config.ATTR_TO_EMB = DataFrameFields.RESOURCE_COLUMN
                logger.save_embeddings_cv(list_embeddings_2, eventlog.filename, str(args.emb_type),
                                          args.emb_size, args.win_size, i)
            else:
                logger.save_embeddings_cv(list_embeddings, eventlog.filename, str(args.emb_type),
                                          args.emb_size, args.win_size, i)

            losses.append(loss_fold)
        logger.log_loss_cv(Path(args.dataset).stem, str(args.emb_type),
                           args.emb_size, args.win_size, losses)
        print(f'MEAN ACCURACY: {np.mean(np.array(accuracies))}')

    else:
        eventlog = EventlogDataset(args.dataset, read_test=True)

        loss, list_embeddings, list_embeddings_2, acc = start(eventlog, args, logger)

        logger.log_console(f'{str(args.emb_type)} TEST LOSS: {loss:.6f}')

        if list_embeddings_2 is not None:
            Config.ATTR_TO_EMB = DataFrameFields.ACTIVITY_COLUMN
            logger.save_embeddings_holdout(list_embeddings, eventlog.filename, str(args.emb_type),
                                           args.emb_size, args.win_size)
            Config.ATTR_TO_EMB = DataFrameFields.RESOURCE_COLUMN
            logger.save_embeddings_holdout(list_embeddings_2, eventlog.filename, str(args.emb_type),
                                           args.emb_size, args.win_size)
        else:
            logger.save_embeddings_holdout(list_embeddings, eventlog.filename, str(args.emb_type),
                                           args.emb_size, args.win_size)
        logger.log_loss_holdout(Path(args.dataset).stem, str(args.emb_type),
                                args.emb_size, args.win_size, loss)

    best_embedding_id = losses.index(min(losses))

    embeddings_dict = get_embeddings(eventlog.filename, Config.ATTR_TO_EMB, str(args.emb_type), args.emb_size, args.win_size,
                   logger, fold=best_embedding_id)

    attr_to_embedding_dict = {attr: embeddings_dict[id] for attr, id in attr_dict.items()}

    #print(embeddings_dict)

    #log_file_path = Config.EMBS_PATH + '/crossvalidation/' + filename + '/' + name


    pairwise_cosine_distances = {
        (key1, key2): float(cosine(attr_to_embedding_dict[key1], attr_to_embedding_dict[key2]))
        for key1, key2 in product(attr_to_embedding_dict.keys(), repeat=2)
    }

    delete_temporary_files(xes_log_path, csv_path, config_csv_path)

    #print(pairwise_cosine_distances)

    return pairwise_cosine_distances, attr_to_embedding_dict
    """



def start(eventlog, args, logger) -> (float, list):
    """
    Executes the training of a specific embedding model
    and returns the loss and the embeddings
    :param eventlog: EventlogDataset object with all the data and information
    about the dataset used
    :param args: EmbGeneratorInputArgs with all the input arguments
    :param logger: EmbGeneratorLogger to print the outputs
    :return: The loss obtained in the test set and the list of embeddings
    """
    list_embeddings_2 = None
    acc = None

    if args.emb_size is None:
        args.emb_size = get_emb_size_power_of_two(eventlog, DataFrameFields.ACTIVITY_COLUMN)

    if Config.ATTR_TO_EMB == DataFrameFields.ROLE_COLUMN:
        log_df_resources = pd.DataFrame.from_records(get_resource_table(eventlog))
        log_df_resources = log_df_resources.rename(index=str, columns={"resource": DataFrameFields.RESOURCE_COLUMN})

        eventlog.df_train = eventlog.df_train.merge(log_df_resources,
                                                    on=[DataFrameFields.RESOURCE_COLUMN], how='left')
        eventlog.df_val = eventlog.df_val.merge(log_df_resources,
                                                on=[DataFrameFields.RESOURCE_COLUMN], how='left')
        eventlog.df_test = eventlog.df_test.merge(log_df_resources,
                                                  on=[DataFrameFields.RESOURCE_COLUMN], how='left')
        unique_roles = list(log_df_resources[DataFrameFields.ROLE_COLUMN].unique())
        eventlog.num_resources = len(unique_roles)

    """ 
    if args.emb_type == EmbType.ACOV:
        loss, list_embeddings = acov.execution(eventlog, args.emb_size, args.win_size, logger)
    elif args.emb_type == EmbType.MOVC:
        loss, list_embeddings = movc.execution(eventlog, args.emb_size, args.win_size, logger)
    elif args.emb_type == EmbType.GLOVE:
        loss, list_embeddings = glove.execution(eventlog, args.emb_size, args.win_size, logger)
    elif args.emb_type == EmbType.CAPE:
        loss, list_embeddings = cape.execution(eventlog, args.emb_size, args.win_size, logger)
    elif args.emb_type == EmbType.NEG_SAMPLING:
        loss, list_embeddings = negative_sampling.execution(eventlog, args.emb_size,
                                                            args.win_size, logger)
    elif args.emb_type == EmbType.DWC:
        loss, list_embeddings = dwc.execution(eventlog, args.emb_size, args.win_size, logger)
    elif args.emb_type == EmbType.DWC_RES:
        loss, list_embeddings = dwc_resources.execution(eventlog, args.emb_size, args.win_size, logger)
    elif args.emb_type == EmbType.DWC_T2V:
        loss, list_embeddings = dwc_t2v.execution(eventlog, args.emb_size, args.win_size, logger)
    """
    #args.emb_type == EmbType.AERAC:
        #loss, list_embeddings, acc = aerac.execution(eventlog, args.emb_size, args.win_size, logger)
    loss, list_embeddings, acc = aerac.run_AErac_model(eventlog, args.emb_size, args.win_size, logger)
    """ 
    elif args.emb_type == EmbType.GAEME:
        loss, list_embeddings, acc = gaeme.execution(eventlog, args.emb_size, args.win_size, logger)
    elif args.emb_type == EmbType.CAMARGO_NS:
        loss, list_embeddings, list_embeddings_2 = camargo_ns.execution(eventlog, args.emb_size,
                                                                        args.win_size, logger)
    """
    """ 
    else:
        loss = None
        list_embeddings = None
        logger.print_error('Not correct embedding type selected')
        exit(-1)
    """

    return loss, list_embeddings, list_embeddings_2, acc


def get_emb_size_power_of_two(eventlog: EventlogDataset, column: str) -> int:
    num_categories = None
    if column == DataFrameFields.ACTIVITY_COLUMN:
        num_categories = eventlog.num_activities
    elif column == DataFrameFields.RESOURCE_COLUMN:
        num_categories = eventlog.num_resources

    exp = int(math.log(num_categories, 2))

    return 2**exp


def get_resource_table(eventlog, sim_threshold=0.7):
    data = pd.concat([eventlog.df_train[[DataFrameFields.ACTIVITY_COLUMN, DataFrameFields.RESOURCE_COLUMN]],
                      eventlog.df_val[[DataFrameFields.ACTIVITY_COLUMN, DataFrameFields.RESOURCE_COLUMN]],
                      eventlog.df_test[[DataFrameFields.ACTIVITY_COLUMN, DataFrameFields.RESOURCE_COLUMN]]])

    tasks = {val: i for i, val in enumerate(data[DataFrameFields.ACTIVITY_COLUMN].unique())}
    users = {val: i for i, val in enumerate(data[DataFrameFields.RESOURCE_COLUMN].unique())}
    associations = lambda x: (tasks[x[DataFrameFields.ACTIVITY_COLUMN]],
                              users[x[DataFrameFields.RESOURCE_COLUMN]])
    data['ac_rl'] = data.apply(associations, axis=1)

    freq_matrix = (data.groupby(by='ac_rl')[DataFrameFields.ACTIVITY_COLUMN]
                   .count().reset_index()
                   .rename(columns={DataFrameFields.ACTIVITY_COLUMN: 'freq'}))
    freq_matrix = {x['ac_rl']: x['freq'] for x in freq_matrix.to_dict('records')}

    profiles = list()
    for user, idx in users.items():
        profile = [0, ] * len(tasks)
        for ac_rl, freq in freq_matrix.items():
            if idx == ac_rl[1]:
                profile[ac_rl[0]] = freq
        profiles.append({'user': idx, 'profile': profile})

    correl_matrix = list()
    for profile_x in profiles:
        for profile_y in profiles:
            x = np.array(profile_x['profile'])
            y = np.array(profile_y['profile'])
            r_row, p_value = pearsonr(x, y)
            correl_matrix.append(({'x': profile_x['user'],
                                   'y': profile_y['user'],
                                   'distance': r_row}))

    g = nx.Graph()
    for user in users.values():
        g.add_node(user)
    for rel in correl_matrix:
        # creation of edges between nodes excluding the same elements
        # and those below the similarity threshold
        if rel['distance'] > sim_threshold and rel['x'] != rel['y']:
            g.add_edge(rel['x'],
                       rel['y'],
                       weight=rel['distance'])

    sub_graphs = list(nx.connected_components(g))

    user_index = {v: k for k, v in users.items()}
    records = list()
    for i in range(0, len(sub_graphs)):
        users_names = [user_index[x] for x in sub_graphs[i]]
        records.append({DataFrameFields.ROLE_COLUMN: i,
                        'quantity': len(sub_graphs[i]),
                        'members': users_names})
    # Sort roles by number of resources
    records = sorted(records, key=itemgetter('quantity'), reverse=True)
    for i in range(0, len(records)):
        records[i]['role'] = i
    resource_table = list()
    for record in records:
        for member in record['members']:
            resource_table.append({DataFrameFields.ROLE_COLUMN: record['role'],
                                   'resource': member})

    return resource_table

"""
def transform_control_flow_lists_to_xes(control_flow_lists):
    event_log = EventLog()
    # Transform the list of traces into an EventLog object
    for trace_id, trace in enumerate(control_flow_lists):
        pm4py_trace = Trace()
        for event_id, activity in enumerate(trace):
            # Create an event with attributes
            event = Event({
                DEFAULT_NAME_KEY: activity,  # 'concept:name' for activity name
                "trace_id": trace_id,  # Custom trace attribute
                "event_index": event_id  # Index of the event in the trace
            })
            pm4py_trace.append(event)  # Add event to the trace
        event_log.append(pm4py_trace)  # Add trace to the event log

    pm4py.write_xes(event_log, )
    df = pm4py.convert_to_dataframe(event_log)
    #df.to_csv("data")
    
    # Create a mapping from activities to unique integer IDs
    activity_set = sorted(set(activity for seq in control_flow_lists for activity in seq))
    activity_to_id = {activity: idx for idx, activity in enumerate(activity_set)}

    # Convert event log to CSV format
    start_time = datetime(1970, 1, 1)
    time_delta = timedelta(hours=1)
    events = []

    for case_id, activities in enumerate(control_flow_lists):
        for event_index, activity in enumerate(activities):
            timestamp = start_time + event_index * time_delta
            events.append([case_id, activity_to_id[activity], timestamp.isoformat() + "+00:00"])

    # Generate output filename
    process_id = os.getpid()
    output_file = f"event_log_{process_id}.csv"

    # Write to CSV
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["CaseID", "Activity", "Timestamp"])
        writer.writerows(events)
    """

def create_csv_data(eventlog, log_name):
    """Generate sliding window samples from activity sequences."""
    unique_activities = sorted(set(a for seq in eventlog for a in seq))
    activity_to_idx = {act: idx for idx, act in enumerate(unique_activities)}
    idx_to_activity = {idx: act for act, idx in activity_to_idx.items()}


def transform_control_flow_lists_to_xes(control_flow_lists):
    event_log = EventLog()
    start_time = datetime(1970, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)  # Epoch start time

    global_event_counter = 0  # Ensures unique timestamps across traces

    for trace_id, trace in enumerate(control_flow_lists):
        pm4py_trace = Trace()

        for event_id, activity in enumerate(trace):
            event_time = start_time + timedelta(hours=global_event_counter)

            event = Event({
                DEFAULT_NAME_KEY: activity,  # 'concept:name' for activity name
                'case:concept:name': trace_id,  # Custom trace attribute
                "event_index": event_id,  # Index of the event in the trace
                DEFAULT_TIMESTAMP_KEY: event_time.strftime("%Y-%m-%d %H:%M:%S+00:00")  # Format as requested
            })
            pm4py_trace.append(event)
            global_event_counter += 1  # Increment global counter to maintain hour-based timestamps
        event_log.append(pm4py_trace)

    # Generate output filename
    script_dir = os.path.dirname(os.path.abspath(__file__))
    process_id = os.getpid()
    output_file_name = f"event_log_{process_id + random.randint(0, 1000000000)}.xes"
    output_file = os.path.join(script_dir, output_file_name)

    pm4py.write_xes(event_log, output_file)

    return output_file_name, output_file

def return_log(output_file):
    file_name = os.path.basename(output_file)
    file_name = "C:\\Users\\henri\\PycharmProjects\\PM_Embeddings\\data\\raw_datasets\\event_log_11188.xes"

    print(f"Processing {file_name}")
    if os.path.exists(file_name):
        try:
            # Try to open the file in read mode
            with open(file_name, 'r') as file:
                print("File is readable and opened successfully.")
                # You can read or process the file here
        except IOError:
            print("File exists but cannot be opened (perhaps due to permission issues).")
    else:
        print("File does not exist.")


def delete_temporary_files(xes_path, csv_path, config_csv_path, eventlog_name):
    log_csv_name  = os.path.basename(csv_path)
    gamallo_dict = os.path.join(ROOT_DIR, "distances", "activity_distances", "gamallo_fernandez_2023_context_based_representations")
    gamallo_data_dict = os.path.join(gamallo_dict, "data")
    os.remove(csv_path)
    os.remove(os.path.join(gamallo_data_dict, "holdout", f"train_{log_csv_name}"))
    os.remove(os.path.join(gamallo_data_dict, "holdout", f"val_{log_csv_name}"))

    gamallo_emb_dict = os.path.join(gamallo_dict, "embeddings")
    shutil.rmtree(os.path.join(gamallo_emb_dict, "holdout", eventlog_name))

    gamallo_logs_dict = os.path.join(gamallo_dict, "logs")
    os.remove(os.path.join(gamallo_logs_dict, "holdout", eventlog_name + "_Activity_embeddings.csv"))

    os.remove(xes_path)

    #if os.path.exists(os.path.join(ROOT_DIR, "evaluation", "evaluation_of_activity_distances", "intrinsic_evaluation", "models", eventlog_name)):
    #    shutil.rmtree(os.path.join(ROOT_DIR, "evaluation", "evaluation_of_activity_distances", "intrinsic_evaluation", "models", eventlog_name))


    #os.path.join(config_csv_path, "holdout", f"tain_{config_csv_path}")
    #os.path.join(config_csv_path, "holdout", f"val_{config_csv_path}")

    #os.remove(    os.path.join(config_csv_path, "holdout", f"tain_{config_csv_path}"))
    #os.remove(    os.path.join(config_csv_path, "holdout", f"val_{config_csv_path}"))

    #print("a")
    """ for cross fold cleanup
    os.remove(xes_path)
    os.remove(csv_path)

    filename = Path(csv_path).stem + ".csv"
    if config_csv_path:
        folder = config_csv_path
    else:
        folder = str(Path(csv_path).parent)
    cv_path = folder + "/crossvalidation/"
    for i in range(Config.CV_FOLDS):
        os.remove(cv_path + "fold" + str(i) + "_train_" + filename)
        os.remove(cv_path + "fold" + str(i) + "_val_" + filename)
        os.remove(cv_path + "fold" + str(i) + "_test_" + filename)
    """
    return
