"""
NEXT ACTIVITY PREDICTION:
Executes the next activity prediction model training and stores the results.
Models: Tax, Camargo, Evermann, Mauro
"""
import math
from pathlib import Path
from args_reader import read_input_args
from logger import NextActPredLogger
from pandas_dataset import EventlogDataset
from utils import AuthorModel, Config, get_embeddings, DataFrameFields
import tax
import camargo
import evermann
import mauro
import sys

def main():

    sys.argv = [
        "script.py",  # Fake script name (needed for argparse)
        "--dataset", "data/BPIC15_2.csv",
        "--crossvalidation",
        "--camargo",
        "--gaeme",
        "--emb_size", "256",
        "--win_size", "4",
        "--print_console_file"
    ]

    args = read_input_args()

    logger = NextActPredLogger(args.print_mode)

    if args.crossvalidation:
        accuracies = []
        f1_scores = []

        for i in range(Config.CV_FOLDS):
            eventlog = EventlogDataset(args.dataset, cv_fold=i, read_test=True)

            if args.emb_size is None:
                args.emb_size = get_emb_size_power_of_two(eventlog, DataFrameFields.ACTIVITY_COLUMN)

            act_embeddings_list = get_embeddings(eventlog.filename, DataFrameFields.ACTIVITY_COLUMN,
                                                 args.emb_type, args.emb_size, args.win_size, logger, i)
            res_embeddings_list = get_embeddings(eventlog.filename, DataFrameFields.RESOURCE_COLUMN,
                                                 args.emb_type, args.emb_size, args.win_size, logger, i)
            rol_embeddings_list = get_embeddings(eventlog.filename, DataFrameFields.ROLE_COLUMN,
                                                 args.emb_type, args.emb_size, args.win_size, logger, i)

            acc_fold, f1_fold = start(eventlog, args.author, act_embeddings_list,
                                      res_embeddings_list, rol_embeddings_list, logger, args)

            logger.log_console(f'{str(args.author)} TEST FOLD {i}:\n'
                               f'\tACCURACY: {acc_fold} \t | F1_SCORE: {f1_fold}')

            accuracies.append(acc_fold)
            f1_scores.append(f1_fold)
        logger.log_metrics_cv(Path(args.dataset).stem, str(args.author),
                              str(args.emb_type), args.emb_size, args.win_size,
                              accuracies, f1_scores)

    else:
        eventlog = EventlogDataset(args.dataset, read_test=True)

        if args.emb_size is None:
            args.emb_size = get_emb_size_power_of_two(eventlog, DataFrameFields.ACTIVITY_COLUMN)

        act_embeddings_list = get_embeddings(eventlog.filename, DataFrameFields.ACTIVITY_COLUMN,
                                             args.emb_type, args.emb_size, args.win_size, logger)
        res_embeddings_list = get_embeddings(eventlog.filename, DataFrameFields.RESOURCE_COLUMN,
                                             args.emb_type, args.emb_size, args.win_size, logger)
        rol_embeddings_list = get_embeddings(eventlog.filename, DataFrameFields.ROLE_COLUMN,
                                             args.emb_type, args.emb_size, args.win_size, logger)

        accuracy, f1_score = start(eventlog, args.author, act_embeddings_list,
                                   res_embeddings_list, rol_embeddings_list, logger, args)

        logger.log_console(f'TEST ACCURACY: {accuracy} \t | F1_SCORE: {f1_score}')
        logger.log_metrics_holdout(Path(args.dataset).stem, str(args.author),
                                   str(args.emb_type), args.emb_size, args.win_size,
                                   accuracy, f1_score)


def start(eventlog, author, act_embeddings_list, res_embeddings_list,
          rol_embeddings_list, logger, args) -> (float, list):
    """
    Executes the training of a specific prediction model
    and returns the accuracy and f1_score
    :param eventlog: EventlogDataset object with all the data and information
    about the dataset used
    :param author: AuthorModel object indicating the model
    :param act_embeddings_list: List of activity embeddings
    :param res_embeddings_list: List of resource embeddings
    :param logger: NextActPredLogger to print the outputs
    :param args: Input arguments
    :return: The accuracy and the f1-score in the test set
    """

    if author == AuthorModel.TAX:
        accuracy, f1_score = tax.execution(eventlog, act_embeddings_list, logger,
                                           args.emb_type, args.emb_size, args.win_size)
    elif author == AuthorModel.CAMARGO:
        accuracy, f1_score = camargo.execution(eventlog, act_embeddings_list, rol_embeddings_list,
                                               logger, args.emb_type, args.emb_size, args.win_size)
    elif author == AuthorModel.EVERMANN:
        accuracy, f1_score = evermann.execution(eventlog, act_embeddings_list, logger,
                                                args.emb_type, args.emb_size, args.win_size)
    elif author == AuthorModel.MAURO:
        accuracy, f1_score = mauro.execution(eventlog, act_embeddings_list, logger,
                                             args.emb_type, args.emb_size, args.win_size)
    else:
        accuracy = None
        f1_score = None
        logger.print_error('Not correct author selected')
        exit(-1)

    return accuracy, f1_score


def get_emb_size_power_of_two(eventlog: EventlogDataset, column: str) -> int:
    num_categories = None
    if column == DataFrameFields.ACTIVITY_COLUMN:
        num_categories = eventlog.num_activities
    elif column == DataFrameFields.RESOURCE_COLUMN:
        num_categories = eventlog.num_resources

    exp = int(math.log(num_categories, 2))

    return 2**exp


if __name__ == "__main__":
    main()
