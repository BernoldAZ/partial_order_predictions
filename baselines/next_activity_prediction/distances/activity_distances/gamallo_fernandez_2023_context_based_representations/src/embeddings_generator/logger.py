import os
import numpy as np
import pandas as pd
from distances.activity_distances.gamallo_fernandez_2023_context_based_representations.src.embeddings_generator.utils import PrintMode, Config


class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class EmbGeneratorLogger:
    print_mode: PrintMode

    def __init__(self, print_mode: PrintMode):
        self.print_mode = print_mode

    def log_console(self, text: str):
        if self.print_mode == PrintMode.CONSOLE or \
                self.print_mode == PrintMode.CONSOLE_AND_FILE:
            print(text)

    def log_loss_cv(self, filename: str, emb_type: str, emb_size: int,
                    win_size: int, losses: list, path: str = None):
        if self.print_mode == PrintMode.CONSOLE_AND_FILE or \
                self.print_mode == PrintMode.TO_FILE:
            if path:
                if not os.path.exists(path):
                    os.makedirs(path)
                log_file_path = path + '/' + filename + '_' + Config.ATTR_TO_EMB + '_embeddings.csv'
            else:
                if not os.path.exists(Config.LOG_PATH + '/crossvalidation'):
                    os.makedirs(Config.LOG_PATH + '/crossvalidation')
                log_file_path = Config.LOG_PATH + '/crossvalidation/' + filename + '_' + \
                                Config.ATTR_TO_EMB + '_embeddings.csv'

            new_data = {'EMB_TYPE': emb_type,
                        'WIN_SIZE': win_size,
                        'EMB_SIZE': emb_size,
                        'LOSS': np.mean(np.array(losses)),
                        'fold_0': losses[0],
                        'fold_1': losses[1],
                        'fold_2': losses[2],
                        'fold_3': losses[3],
                        'fold_4': losses[4]}

            if os.path.exists(log_file_path):
                log_results = pd.read_csv(log_file_path)

                log_results = pd.concat([log_results,
                                         pd.DataFrame.from_records([new_data])],
                                        ignore_index=True)

            else:
                log_results = pd.Series(new_data).to_frame().T

            log_results.to_csv(log_file_path, index=False)

    def log_loss_holdout(self, filename: str, emb_type: str, emb_size: int,
                    win_size: int, loss: float, path: str = None):
        if self.print_mode == PrintMode.CONSOLE_AND_FILE or \
                self.print_mode == PrintMode.TO_FILE:
            if path:
                if not os.path.exists(path):
                    os.makedirs(path)
                log_file_path = path + '/' + filename + '_' + Config.ATTR_TO_EMB + '_embeddings.csv'
            else:
                if not os.path.exists(Config.LOG_PATH + '/holdout'):
                    os.makedirs(Config.LOG_PATH + '/holdout')
                log_file_path = Config.LOG_PATH + '/holdout/' + filename + '_' + Config.ATTR_TO_EMB +\
                                '_embeddings.csv'

            new_data = {'EMB_TYPE': emb_type,
                        'WIN_SIZE': win_size,
                        'EMB_SIZE': emb_size,
                        'LOSS': loss}

            if os.path.exists(log_file_path):
                log_results = pd.read_csv(log_file_path)

                log_results = pd.concat([log_results,
                                         pd.DataFrame.from_records([new_data])],
                                        ignore_index=True)

            else:
                log_results = pd.Series(new_data).to_frame().T

            log_results.to_csv(log_file_path, index=False)

    def print_error(self, text: str):
        if self.print_mode == PrintMode.CONSOLE or \
                self.print_mode == PrintMode.CONSOLE_AND_FILE:
            print(f'{Colors.FAIL}ERROR: {text}{Colors.ENDC}')

    def save_embeddings_cv(self, list_embeddings: dict, filename: str, emb_type: str,
                           emb_size: int, win_size: int, fold: int, path: str = None):
        if self.print_mode == PrintMode.CONSOLE_AND_FILE or \
                self.print_mode == PrintMode.TO_FILE:
            name = Config.ATTR_TO_EMB + '_' + emb_type + '_fold' + str(fold) + '_winsize' + str(win_size) + '_embsize' + str(emb_size) + '.csv'
            if path:
                if not os.path.exists(path):
                    os.makedirs(path)
                log_file_path = path + '/' + name
            else:
                if not os.path.exists(Config.EMBS_PATH + '/crossvalidation/' + filename):
                    os.makedirs(Config.EMBS_PATH + '/crossvalidation/' + filename)
                log_file_path = Config.EMBS_PATH + '/crossvalidation/' + filename + '/' + name

            pd_embeddings = pd.DataFrame(list_embeddings).T

            pd_embeddings.to_csv(log_file_path, header=False, index=False)

    def save_embeddings_holdout(self, list_embeddings: dict, filename: str, emb_type: str,
                           emb_size: int, win_size: int, path: str = None):
        if self.print_mode == PrintMode.CONSOLE_AND_FILE or \
                self.print_mode == PrintMode.TO_FILE:
            name = Config.ATTR_TO_EMB + '_' + emb_type + '_winsize' + str(win_size) + '_embsize' + str(emb_size) + '.csv'
            if path:
                if not os.path.exists(path):
                    os.makedirs(path)
                log_file_path = path + '/' + name
            else:
                if not os.path.exists(Config.EMBS_PATH + '/holdout/' + filename):
                    os.makedirs(Config.EMBS_PATH + '/holdout/' + filename)
                log_file_path = Config.EMBS_PATH + '/holdout/' + filename + '/' + name

            pd_embeddings = pd.DataFrame(list_embeddings).T

            pd_embeddings.to_csv(log_file_path, header=False, index=False)
