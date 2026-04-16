import os
from pathlib import Path
import pandas as pd


class PrintMode:
    NONE = 0
    CONSOLE = 1
    TO_FILE = 2
    CONSOLE_AND_FILE = 3


class AuthorModel:
    TAX = "TaxLSTM"
    CAMARGO = "CamargoLSTM"
    EVERMANN = "EvermannLSTM"


class EmbType:
    ACOV = "ACOV"
    MOVC = "MOVC"
    GLOVE = "GloVe"
    CAPE = "CAPE"
    NEG_SAMPLING = "NegativeSampling"
    DWC = "DWC"
    DWC_RES = "DWC_Resources"
    DWC_T2V = "DWC_T2V"


class DataFrameFields:
    CASE_COLUMN = "CaseID"
    ACTIVITY_COLUMN = "Activity"
    TIMESTAMP_COLUMN = "Timestamp"
    RESOURCE_COLUMN = "Resource"
    ROLE_COLUMN = "role"


class Config:
    current_path = Path(__file__).parent
    MODEL_PATH = os.path.abspath(os.path.join(current_path, '../../models/suffix'))
    RESULTS_PATH = os.path.abspath(os.path.join(current_path, '../../results/suffix'))
    EMBS_PATH = os.path.abspath(os.path.join(current_path, '../../embeddings'))
    CV_FOLDS = 5
    ATTR_TO_EMB = DataFrameFields.ACTIVITY_COLUMN
    BATCH_SIZE = 32
    LEARNING_RATE = 0.002
    EPOCHS = 200
    EARLY_STOPPING_EPOCHS = 15


def get_embeddings(filename, attr_to_ebm, emb_type, emb_size, win_size,
                   logger, fold=None) -> dict:
    """
    Get the embeddings list from the file where its stored
    :param filename: Name of the dataset
    :param attr_to_ebm: Column to be converted to embedding
    :param emb_type: Type of embedding to use: ACOV, MOVC...
    :param emb_size: Size of the embeddings
    :param win_size: Size of the window context
    :param logger: NextActPredLogger to print output
    :param fold: If cross-validation is used, the number of the fold
    :return: A dictionary with the list of embeddings and their index
    """
    path = Config.EMBS_PATH
    if fold is None:
        path = path + '/holdout/' + filename + '/' + attr_to_ebm + '_' + str(emb_type)
    else:
        path = path + '/crossvalidation/' + filename + '/' + attr_to_ebm + '_' + str(emb_type) + '_fold' + str(fold)
    path = path + '_winsize' + str(win_size) + '_embsize' + str(emb_size) + '.csv'

    if os.path.exists(path):
        df_embeddings = pd.read_csv(path, header=None, index_col=False)

        embeddings_dict = dict()
        for idx, emb in df_embeddings.iterrows():
            embeddings_dict.update({
                idx: emb.tolist()
            })
    else:
        embeddings_dict = None
        # logger.print_error("The embeddings file doesn't exist...")
        # exit(-1)

    return embeddings_dict
