import os
import numpy as np
import pandas as pd
from utils import PrintMode, Config


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


class NextActPredLogger:
    print_mode: PrintMode

    def __init__(self, print_mode: PrintMode):
        self.print_mode = print_mode

    def log_console(self, text: str):
        if self.print_mode == PrintMode.CONSOLE or \
                self.print_mode == PrintMode.CONSOLE_AND_FILE:
            print(text)

    def print_error(self, text: str):
        if self.print_mode == PrintMode.CONSOLE or \
                self.print_mode == PrintMode.CONSOLE_AND_FILE:
            print(f'{Colors.FAIL}ERROR: {text}{Colors.ENDC}')

    def log_metrics_cv(self, filename: str, author: str, emb_type: str,
                       emb_size: int, win_size: int, accuracies: list,
                       f1_scores: list):
        if self.print_mode == PrintMode.CONSOLE_AND_FILE or \
                self.print_mode == PrintMode.TO_FILE:
            if not os.path.exists(Config.RESULTS_PATH + '/crossvalidation/' + author):
                os.makedirs(Config.RESULTS_PATH + '/crossvalidation/' + author)
            log_file_path = Config.RESULTS_PATH + '/crossvalidation/' + author + '/' + filename + '_predictions.csv'

            new_data = {
                'EMB_TYPE': emb_type,
                'WIN_SIZE': win_size,
                'EMB_SIZE': emb_size,
                'ACCURACY': np.mean(np.array(accuracies)),
                'F1-SCORE': np.mean(np.array(f1_scores)),
                'acc_fold_0': accuracies[0],
                'acc_fold_1': accuracies[1],
                'acc_fold_2': accuracies[2],
                'acc_fold_3': accuracies[3],
                'acc_fold_4': accuracies[4],
                'f1_fold_0': f1_scores[0],
                'f1_fold_1': f1_scores[1],
                'f1_fold_2': f1_scores[2],
                'f1_fold_3': f1_scores[3],
                'f1_fold_4': f1_scores[4]
            }

            if os.path.exists(log_file_path):
                log_results = pd.read_csv(log_file_path)

                log_results = pd.concat([log_results,
                                         pd.DataFrame.from_records([new_data])],
                                        ignore_index=True)

            else:
                log_results = pd.Series(new_data).to_frame().T

            log_results.to_csv(log_file_path, index=False)

    def log_metrics_holdout(self, filename: str, author: str, emb_type: str,
                            emb_size: int, win_size: int, accuracy: float,
                            f1_score: float):
        if self.print_mode == PrintMode.CONSOLE_AND_FILE or \
                self.print_mode == PrintMode.TO_FILE:
            if not os.path.exists(Config.RESULTS_PATH + '/holdout/' + author):
                os.makedirs(Config.RESULTS_PATH + '/holdout/' + author)
            log_file_path = Config.RESULTS_PATH + '/holdout/' + author + '/' + filename + '_predictions.csv'

            new_data = {
                'EMB_TYPE': emb_type,
                'WIN_SIZE': win_size,
                'EMB_SIZE': emb_size,
                'ACCURACY': accuracy,
                'F1-SCORE': f1_score
            }

            if os.path.exists(log_file_path):
                log_results = pd.read_csv(log_file_path)

                log_results = pd.concat([log_results,
                                         pd.DataFrame.from_records([new_data])],
                                        ignore_index=True)

            else:
                log_results = pd.Series(new_data).to_frame().T

            log_results.to_csv(log_file_path, index=False)
