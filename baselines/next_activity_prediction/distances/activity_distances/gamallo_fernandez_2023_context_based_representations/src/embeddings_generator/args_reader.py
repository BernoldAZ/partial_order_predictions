import argparse
from dataclasses import dataclass
from distances.activity_distances.gamallo_fernandez_2023_context_based_representations.src.embeddings_generator.utils import PrintMode, EmbType, Config, DataFrameFields


@dataclass
class EmbGeneratorInputArgs:
    dataset: str
    crossvalidation: bool
    emb_type: EmbType
    emb_size: int
    win_size: int
    print_mode: PrintMode


def read_input_args() -> EmbGeneratorInputArgs:
    """
    Read the user input arguments
    :return: EmbGeneratorInputArgs object with the input arguments
    """
    parser = argparse.ArgumentParser(description="Execute the training of the "
                                                 "embedding model")
    parser.add_argument("-d", "--dataset", required=True, type=str,
                        help="Full path to the dataset")
    partition_mode = parser.add_mutually_exclusive_group(required=True)
    partition_mode.add_argument("--holdout", help="Split in train/validation/test",
                                action='store_true')
    partition_mode.add_argument("--crossvalidation", help="5-fold cross validation",
                                action='store_true')
    embedding_type = parser.add_mutually_exclusive_group(required=True)
    embedding_type.add_argument("--acov", help="Execute All-Context-in-One-Vector "
                                               "embedding model", action='store_true')
    embedding_type.add_argument("--movc", help="Execute Multi-Onehot-Vector-Context "
                                               "embedding model", action='store_true')
    embedding_type.add_argument("--glove", help="Execute GloVe embedding model",
                                action='store_true')
    embedding_type.add_argument("--negativesampling", help="Execute Negative Sampling embedding "
                                                           "model", action='store_true')
    embedding_type.add_argument("--cape", help="Execute Context-Ahead-Prediction-Embedding "
                                               "model", action='store_true')
    embedding_type.add_argument("--dwc", help="Execute Distance-Weighted Context "
                                              "embedding model", action='store_true')
    embedding_type.add_argument("--dwc_resources", help="Execute Distance-Weighted Context "
                                                        "embedding model with reosurces", action='store_true')
    embedding_type.add_argument("--dwc_t2v", help="Execute Distance-Weighted Context "
                                                  "embedding model with times", action='store_true')
    embedding_type.add_argument("--aerac", help="Execute AutoEncoder to reconstruct activity context "
                                                "embedding model", action='store_true')
    embedding_type.add_argument("--gaeme", help="Execute Generative-AutoEncoder-Model-Embeddings",
                                action="store_true")
    embedding_type.add_argument("--camargo_ns", help="Execute Negative Sampling Camargo embedding "
                                                     "model", action='store_true')
    attr_to_emb = parser.add_mutually_exclusive_group(required=False)
    attr_to_emb.add_argument("--activity", help="Generate embeddings for activities",
                             action='store_true')
    attr_to_emb.add_argument("--resource", help="Generate embeddings for resources",
                             action='store_true')
    attr_to_emb.add_argument("--role", help="Generate embeddings for roles",
                             action='store_true')
    parser.add_argument("--emb_size", type=int)
    parser.add_argument("--win_size", type=int)
    print_mode = parser.add_mutually_exclusive_group(required=False)
    print_mode.add_argument("--print_console", help="Print output to the console", action='store_true')
    print_mode.add_argument("--print_file", help="Print output to a file", action='store_true')
    print_mode.add_argument("--print_console_file", help="Print output to the console and file", action='store_true')
    args = parser.parse_args()

    # Full path to the dataset
    dataset = args.dataset

    # Split mode
    if args.holdout:
        crossvalidation = False
    else:
        crossvalidation = True

    # Embedding type
    if args.acov:
        emb_type = EmbType.ACOV
    elif args.movc:
        emb_type = EmbType.MOVC
    elif args.glove:
        emb_type = EmbType.GLOVE
    elif args.cape:
        emb_type = EmbType.CAPE
    elif args.negativesampling:
        emb_type = EmbType.NEG_SAMPLING
    elif args.dwc:
        emb_type = EmbType.DWC
    elif args.dwc_resources:
        emb_type = EmbType.DWC_RES
    elif args.dwc_t2v:
        emb_type = EmbType.DWC_T2V
    elif args.aerac:
        emb_type = EmbType.AERAC
    elif args.gaeme:
        emb_type = EmbType.GAEME
    elif args.camargo_ns:
        emb_type = EmbType.CAMARGO_NS
    else:
        emb_type = None

    # Attribute to embed
    if args.activity:
        Config.ATTR_TO_EMB = DataFrameFields.ACTIVITY_COLUMN
    elif args.resource:
        Config.ATTR_TO_EMB = DataFrameFields.RESOURCE_COLUMN
    elif args.role:
        Config.ATTR_TO_EMB = 'role'

    # Embedding size
    emb_size = args.emb_size

    # Window size
    win_size = args.win_size

    # Check the print mode
    if args.print_console:
        print_mode = PrintMode.CONSOLE
    elif args.print_file:
        print_mode = PrintMode.TO_FILE
    elif args.print_console_file:
        print_mode = PrintMode.CONSOLE_AND_FILE
    else:
        print_mode = PrintMode.NONE

    return EmbGeneratorInputArgs(dataset, crossvalidation, emb_type,
                                 emb_size, win_size, print_mode)

