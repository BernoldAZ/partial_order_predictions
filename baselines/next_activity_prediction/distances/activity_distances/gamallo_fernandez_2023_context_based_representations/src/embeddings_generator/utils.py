import os
from pathlib import Path


class PrintMode:
    NONE = 0
    CONSOLE = 1
    TO_FILE = 2
    CONSOLE_AND_FILE = 3


class EmbType:
    ACOV = "ACOV"
    MOVC = "MOVC"
    GLOVE = "GloVe"
    CAPE = "CAPE"
    NEG_SAMPLING = "NegativeSampling"
    DWC = "DWC"
    DWC_RES = "DWC_Resources"
    DWC_T2V = "DWC_T2V"
    AERAC = "AErac"
    GAEME = "GAEME"
    CAMARGO_NS = "Camargo_NegSampling"


class DataFrameFields:
    CASE_COLUMN = "CaseID"
    ACTIVITY_COLUMN = "Activity"
    TIMESTAMP_COLUMN = "Timestamp"
    RESOURCE_COLUMN = "Resource"
    ROLE_COLUMN = "role"


class Config:
    current_path = Path(__file__).parent
    MODEL_PATH = os.path.abspath(os.path.join(current_path, '../../models'))
    LOG_PATH = os.path.abspath(os.path.join(current_path, '../../logs'))
    EMBS_PATH = os.path.abspath(os.path.join(current_path, '../../embeddings'))
    CV_FOLDS = 5
    ATTR_TO_EMB = DataFrameFields.ACTIVITY_COLUMN
    BATCH_SIZE = 32
    LEARNING_RATE = 0.002
    EPOCHS = 100
    EARLY_STOPPING_EPOCHS = 10
