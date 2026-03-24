import os
from definitions import ROOT_DIR
EVENT_LOGS_DIR = os.path.join(ROOT_DIR, "event_logs")

def rename_files(folder_path):
    mappings = {
        "BPI_2020": "BPIC20",
        "BPI_Challenge_2012": "BPIC12",
        "BPI_Challenge_2013": "BPIC13"
    }

    for filename in os.listdir(folder_path):
        for old_prefix, new_prefix in mappings.items():
            if filename.startswith(old_prefix):
                new_filename = filename.replace(old_prefix, new_prefix, 1)
                old_path = os.path.join(folder_path, filename)
                new_path = os.path.join(folder_path, new_filename)
                os.rename(old_path, new_path)
                print(f'Renamed: {filename} -> {new_filename}')

rename_files(EVENT_LOGS_DIR)