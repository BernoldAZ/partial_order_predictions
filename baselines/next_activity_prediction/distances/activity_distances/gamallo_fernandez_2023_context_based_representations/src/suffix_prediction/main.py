"""
SUFFIX PREDICTION:
Executes the suffix prediction model training and stores the results.
Models: Tax, Camargo, Evermann
"""
import math
from pathlib import Path
from args_reader import read_input_args
from logger import SuffixPredLogger
from pandas_dataset import EventlogDataset
from utils import AuthorModel, Config, get_embeddings, DataFrameFields
import tax
# import camargo
# import evermann


def main():
    args = read_input_args()

    logger = SuffixPredLogger(args.print_mode)

    if args.crossvalidation:
        dl_scores = []

        for i in range(Config.CV_FOLDS):
            eventlog = EventlogDataset(args.dataset, cv_fold=i, read_test=True)

            if args.emb_size is None:
                args.emb_size = get_emb_size_power_of_two(eventlog, DataFrameFields.ACTIVITY_COLUMN)
            if args.win_size is None:
                args.win_size = 2

            act_embeddings_list = get_embeddings(eventlog.filename, DataFrameFields.ACTIVITY_COLUMN,
                                                 args.emb_type, args.emb_size, args.win_size, logger, i)
            res_embeddings_list = get_embeddings(eventlog.filename, DataFrameFields.RESOURCE_COLUMN,
                                                 args.emb_type, args.emb_size, args.win_size, logger, i)
            rol_embeddings_list = get_embeddings(eventlog.filename, DataFrameFields.ROLE_COLUMN,
                                                 args.emb_type, args.emb_size, args.win_size, logger, i)

            dl_fold = start(eventlog, args.author, act_embeddings_list,
                            res_embeddings_list, rol_embeddings_list, logger, args)

            logger.log_console(f'{str(args.author)} TEST FOLD {i}:\n'
                               f'Damerau-Levenshtain Distance: {dl_fold}')

            dl_scores.append(dl_fold)
        logger.log_metrics_cv(Path(args.dataset).stem, str(args.author),
                              str(args.emb_type), args.emb_size, args.win_size, dl_scores)

    else:
        eventlog = EventlogDataset(args.dataset, read_test=True)

        if args.emb_size is None:
            args.emb_size = get_emb_size_power_of_two(eventlog, DataFrameFields.ACTIVITY_COLUMN)
        if args.win_size is None:
            args.win_size = 2

        act_embeddings_list = get_embeddings(eventlog.filename, DataFrameFields.ACTIVITY_COLUMN,
                                             args.emb_type, args.emb_size, args.win_size, logger)
        res_embeddings_list = get_embeddings(eventlog.filename, DataFrameFields.RESOURCE_COLUMN,
                                             args.emb_type, args.emb_size, args.win_size, logger)
        rol_embeddings_list = get_embeddings(eventlog.filename, DataFrameFields.ROLE_COLUMN,
                                             args.emb_type, args.emb_size, args.win_size, logger)

        dl_score = start(eventlog, args.author, act_embeddings_list, res_embeddings_list,
                         rol_embeddings_list, logger, args)

        logger.log_console(f'TEST DL_SCORE: {dl_score}')
        logger.log_metrics_holdout(Path(args.dataset).stem, str(args.author),
                                   str(args.emb_type), args.emb_size, args.win_size,
                                   dl_score)


def start(eventlog, author, act_embeddings_list, res_embeddings_list,
          rol_embeddings_list, logger, args) -> (float, list):
    """
    Executes the training of a specific suffix prediction model
    and returns the Damerau-Levenshtein Distance
    :param eventlog: EventlogDataset object with all the data and information
    about the dataset used
    :param author: AuthorModel object indicating the model
    :param act_embeddings_list: List of activity embeddings
    :param res_embeddings_list: List of resource embeddings
    :param logger: SuffixPredLogger to print the outputs
    :param args: Input arguments
    :return: The DL-score in the test set
    """

    if author == AuthorModel.TAX:
        dl_score = tax.execution(eventlog, act_embeddings_list, logger,
                                 args.emb_type, args.emb_size, args.win_size)
        pass

    elif author == AuthorModel.CAMARGO:
        # dl_score = camargo.execution()
        pass

    elif author == AuthorModel.EVERMANN:
        # dl_score = evermann.execution()
        pass

    else:
        dl_score = None
        logger.print_error('Not correct author selected')
        exit(-1)

    return dl_score


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