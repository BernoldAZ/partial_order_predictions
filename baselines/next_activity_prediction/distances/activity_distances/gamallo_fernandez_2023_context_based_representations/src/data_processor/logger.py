import os
import pandas as pd
from distances.activity_distances.gamallo_fernandez_2023_context_based_representations.src.data_processor.utils import PrintMode, Config


class DataProcessorLogger:
    print_mode: PrintMode

    def __init__(self, print_mode: PrintMode):
        self.print_mode = print_mode

    def log_console(self, text: str):
        if self.print_mode == PrintMode.CONSOLE or \
                self.print_mode == PrintMode.CONSOLE_AND_FILE:
            print(text)

    def log_metrics(self, stats: list, path: str = None):
        if self.print_mode == PrintMode.TO_FILE or \
                self.print_mode == PrintMode.CONSOLE_AND_FILE:
            if path:
                if not os.path.exists(path):
                    os.makedirs(path)
                log_file_path = path + '/' + Config.STATS_FILE
            else:
                if not os.path.exists(Config.LOG_PATH):
                    os.makedirs(Config.LOG_PATH)
                log_file_path = Config.LOG_PATH + '/' + Config.STATS_FILE

            new_data = {'FILENAME': stats[0],
                        'NUM_EVENTS': stats[1],
                        'NUM_ACTIVITIES': stats[2],
                        'NUM_CASES': stats[3],
                        'AVG_CASE_LEN': stats[4],
                        'MAX_CASE_LEN': stats[5],
                        'NUM_VARIANTS': stats[6]}

            if os.path.exists(log_file_path):
                stats_data = pd.read_csv(log_file_path)

                if stats_data[stats_data.FILENAME == stats[0]].empty:
                    # Add new row
                    new_data = pd.DataFrame([new_data])
                    stats_data = pd.concat([stats_data, new_data],
                                           ignore_index=True)
                else:
                    # Overwrite the existing row
                    stats_data.loc[stats_data['FILENAME'] == stats[0]] = \
                        [pd.Series(new_data)]

            else:
                stats_data = pd.Series(new_data).to_frame().T

            stats_data.to_csv(log_file_path, index=False)
