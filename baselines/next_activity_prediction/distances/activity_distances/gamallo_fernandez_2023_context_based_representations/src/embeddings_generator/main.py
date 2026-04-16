"""
EMBEDDING GENERATOR:
Executes the embedding model training and stores the embeddings list.
Four types of embeddings: ACOV, MOVC, GloVe, CAPE.
"""
import math
from pathlib import Path
import pandas as pd
from operator import itemgetter
import numpy as np
from scipy.stats import pearsonr
import networkx as nx
from args_reader import read_input_args
from logger import EmbGeneratorLogger
from pandas_dataset import EventlogDataset
from utils import EmbType, Config, DataFrameFields
import acov
import movc
import glove
import cape
import negative_sampling
import dwc
import dwc_resources
import dwc_t2v
import aerac
import gaeme
import camargo_ns
import sys


def main():
    args = read_input_args()

    logger = EmbGeneratorLogger(args.print_mode)

    if args.crossvalidation:
        losses = []
        accuracies = []
        for i in range(Config.CV_FOLDS):
            eventlog = EventlogDataset(args.dataset, cv_fold=i, read_test=True)

            loss_fold, list_embeddings, list_embeddings_2, acc = start(eventlog, args, logger)

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
    elif args.emb_type == EmbType.AERAC:
        #loss, list_embeddings, acc = aerac.execution(eventlog, args.emb_size, args.win_size, logger)
        loss, list_embeddings, acc = aerac.run_AErac_model(eventlog, args.emb_size, args.win_size, logger)
    elif args.emb_type == EmbType.GAEME:
        loss, list_embeddings, acc = gaeme.execution(eventlog, args.emb_size, args.win_size, logger)
    elif args.emb_type == EmbType.CAMARGO_NS:
        loss, list_embeddings, list_embeddings_2 = camargo_ns.execution(eventlog, args.emb_size,
                                                                        args.win_size, logger)
    else:
        loss = None
        list_embeddings = None
        logger.print_error('Not correct embedding type selected')
        exit(-1)

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


if __name__ == "__main__":
    sys.argv = [
        "script.py",  # Fake script name (needed for argparse)
        "--dataset", "data/Helpdesk.csv",
        "--crossvalidation",
        "--aerac",
        "--activity",
        "--emb_size", "8",
        "--win_size", "4",
        "--print_console_file"
    ]

    main()  # Now calling main() will behave as if it was called from the command line


