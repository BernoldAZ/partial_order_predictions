import os
import shutil
import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping
from torch.utils.data import DataLoader

from logger import NextActPredLogger
from pandas_dataset import EventlogDataset
from utils import DataFrameFields, Config, EmbType, PrintMode, get_embeddings

# To ensure reproducibility
pl.seed_everything(123)


class NextActPredDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: pd.DataFrame, max_prefix_len: int,
                 num_activities: int, embeddings_dict: dict):
        self.dataset = dataset.reset_index(drop=True)
        self.max_prefix_len = max_prefix_len
        self.num_activities = num_activities
        self.embeddings_dict = embeddings_dict

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        event = self.dataset.iloc[idx]
        trace = self.dataset[self.dataset[DataFrameFields.CASE_COLUMN] == event[DataFrameFields.CASE_COLUMN]]
        prefix = trace.loc[:idx]

        try:
            next_event = trace.loc[idx+1]
            target = torch.tensor(next_event[DataFrameFields.ACTIVITY_COLUMN].item(), dtype=torch.long)
        except KeyError:
            target = torch.tensor(self.num_activities, dtype=torch.long)  # End-of-case

        vectorized_prefix = prefix[DataFrameFields.ACTIVITY_COLUMN].tolist()
        vectorized_prefix = torch.tensor(vectorized_prefix, dtype=torch.long)
        if vectorized_prefix.shape[0] < self.max_prefix_len:
            pad_size = self.max_prefix_len - vectorized_prefix.shape[0]
            vectorized_prefix = F.pad(vectorized_prefix, (pad_size, 0), 'constant', self.num_activities)

        return vectorized_prefix, target


class BasicLSTM(pl.LightningModule):
    def __init__(self, input_type: str, num_categories: int,
                 emb_size: int, embeddings_dict: dict):
        """
        :param input_type: 'onehot', 'emblayer' or 'embeddings'
        :param num_categories: Number of unique activities to predict
        :param emb_size: Size of the embeddings. Only needed with input_type in ['emblayer', 'embeddings']
        :param embeddings_dict: Dictionary with activity embeddings
        """
        super().__init__()

        self.num_categories = num_categories
        self.input_type = input_type
        self.embeddings_dict = embeddings_dict
        self.hidden_size = 100

        self.loss_function = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_categories+1)
        self.f1_score = torchmetrics.classification.MulticlassF1Score(num_classes=num_categories+1,
                                                                      average='macro')

        if self.input_type == 'onehot':
            self.lstm = nn.LSTM(input_size=self.num_categories+1,
                                hidden_size=self.hidden_size,
                                batch_first=True, num_layers=1)
        elif self.input_type == 'emblayer':
            self.emb_size = emb_size

            self.embeddings = nn.Embedding(num_embeddings=self.num_categories+1,
                                           # +1 = Embedding for padding representation
                                           embedding_dim=self.emb_size,
                                           padding_idx=num_categories)
            self.lstm = nn.LSTM(input_size=self.emb_size,
                                hidden_size=self.hidden_size,
                                batch_first=True, num_layers=1)
        elif self.input_type == 'embeddings':
            self.emb_size = emb_size

            self.lstm = nn.LSTM(input_size=self.emb_size,
                                hidden_size=self.hidden_size,
                                batch_first=True, num_layers=1)

        self.act_output = nn.Linear(in_features=self.hidden_size,
                                    out_features=self.num_categories+1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.input_type == 'onehot':
            inputs = F.one_hot(inputs[:, :], self.num_categories+1).type(torch.float)
        elif self.input_type == 'emblayer':
            inputs = self.embeddings(inputs)
        elif self.input_type == 'embeddings':
            emb_inputs = []
            for prefix in inputs[:, :]:
                embs = []
                for event in prefix:
                    if event.item() == self.num_categories:
                        emb = np.zeros(self.emb_size).tolist()  # 0-padding
                    else:
                        emb = self.embeddings_dict[event.item()]
                    embs.append(emb)
                emb_inputs.append(embs)
            inputs = torch.tensor(emb_inputs).to(self.device)

        outputs, (hidden, cell) = self.lstm(inputs)
        act_output = self.act_output(hidden)

        return torch.squeeze(act_output, 0)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        preds = self.forward(inputs)

        loss = self.loss_function(preds, targets)
        self.log("train_loss", loss.item(), on_step=True)

        accuracy = self.accuracy(preds, targets)
        self.log("train_acc", accuracy, on_step=True)

        f1_score = self.f1_score(preds, targets)
        self.log("train_f1", f1_score, on_step=True)

        return {"loss": loss, "accuracy": accuracy, "f1_score": f1_score}

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        preds = self.forward(inputs)

        loss = self.loss_function(preds, targets)
        self.log("val_loss", loss.item(), on_step=True)

        accuracy = self.accuracy(preds, targets)
        self.log("val_acc", accuracy, on_step=True)

        f1_score = self.f1_score(preds, targets)
        self.log("val_f1", f1_score, on_step=True)

        return {"loss": loss, "accuracy": accuracy, "f1_score": f1_score}

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        preds = self.forward(inputs)

        loss = self.loss_function(preds, targets)
        self.log("test_loss", loss.item(), on_step=True)

        accuracy = self.accuracy(preds, targets)
        self.log("test_acc", accuracy, on_step=True)

        f1_score = self.f1_score(preds, targets)
        self.log("test_f1", f1_score, on_step=True)

        return {"loss": loss, "accuracy": accuracy, "f1_score": f1_score}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=Config.LEARNING_RATE)


def run_basicLSTM_model(eventlog: EventlogDataset, input_type: str, emb_size: int,
                        embeddings_dict: dict) -> (float, float):
    torch.set_float32_matmul_precision('medium')

    # Create the specific dataset for each split
    train_dataset = NextActPredDataset(eventlog.df_train, eventlog.max_len_case,
                                       eventlog.num_activities, embeddings_dict)
    val_dataset = NextActPredDataset(eventlog.df_val, eventlog.max_len_case,
                                     eventlog.num_activities, embeddings_dict)
    test_dataset = NextActPredDataset(eventlog.df_test, eventlog.max_len_case,
                                      eventlog.num_activities, embeddings_dict)

    # Create dataloader for each split
    num_cores = os.cpu_count()
    train_dataloader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE,
                                  shuffle=True, num_workers=num_cores)
    val_dataloader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE,
                                shuffle=True, num_workers=num_cores)
    test_dataset = DataLoader(test_dataset, batch_size=1,
                              num_workers=num_cores)

    model = BasicLSTM(input_type, eventlog.num_activities, emb_size, embeddings_dict)

    path = "./models/" + eventlog.filename + "/basicLSTM/checkpoints/"
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        dirpath=path,
        filename="best_model",
        save_top_k=1,
        mode="max"
    )

    early_stopping = EarlyStopping(monitor="val_acc", mode="max", patience=15)

    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=Config.EPOCHS,
                         callbacks=[TQDMProgressBar(refresh_rate=20), checkpoint_callback, early_stopping])

    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass
    trainer.fit(model, train_dataloader, val_dataloader)

    best_model = BasicLSTM.load_from_checkpoint(
        path + "best_model.ckpt",
        input_type=input_type,
        num_categories=eventlog.num_activities,
        emb_size=emb_size,
        embeddings_dict=embeddings_dict
    )

    print("Testing...")
    dict_metrics = trainer.test(best_model, test_dataset)

    return dict_metrics[0]["test_acc_epoch"], dict_metrics[0]["test_f1_epoch"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute the training of the Basic LSTM")

    parser.add_argument("-d", "--dataset", required=True, type=str,
                        help="Full path to the dataset")
    partition_mode = parser.add_mutually_exclusive_group(required=True)
    partition_mode.add_argument("--holdout", help="Split in train/validation/test",
                                action='store_true')
    partition_mode.add_argument("--crossvalidation", help="5-fold cross validation",
                                action='store_true')
    parser.add_argument("--input_type", help="One in [onehot, emblayer, embeddings]", type=str)
    embedding_type = parser.add_mutually_exclusive_group(required=False)
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
    embedding_type.add_argument("--dwc", help="Execute Distance-Weighted Context embedding"
                                              "model", action='store_true')
    embedding_type.add_argument("--dwc_resources", help="Execute Distance-Weighted Context embedding"
                                                        "model", action='store_true')
    embedding_type.add_argument("--dwc_t2v", help="Execute Distance-Weighted Context embedding"
                                                  "model", action='store_true')
    embedding_type.add_argument("--aerac", help="Execute AutoEncoder to reconstruct activity context "
                                                "embedding model", action='store_true')
    embedding_type.add_argument("--gaeme", help="Execute Generative-AutoEncoder-Model-Embeddings",
                                action="store_true")
    parser.add_argument("--emb_size", type=int)
    parser.add_argument("--win_size", type=int)
    print_mode = parser.add_mutually_exclusive_group(required=False)
    print_mode.add_argument("--print_console", help="Print output to the console", action='store_true')
    print_mode.add_argument("--print_file", help="Print output to a file", action='store_true')
    print_mode.add_argument("--print_console_file", help="Print output to the console and file", action='store_true')
    args = parser.parse_args()

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
    else:
        emb_type = None

    # Check the print mode
    if args.print_console:
        print_mode = PrintMode.CONSOLE
    elif args.print_file:
        print_mode = PrintMode.TO_FILE
    elif args.print_console_file:
        print_mode = PrintMode.CONSOLE_AND_FILE
    else:
        print_mode = PrintMode.NONE

    logger = NextActPredLogger(print_mode)

    logger.log_console(f'TESTING BasicLSTM MODEL...')
    logger.log_console(f'Params:\n'
                       f'\tFilename: {Path(args.dataset).stem}\n'
                       f'\tInput type: {args.input_type}\n'
                       f'\tEmbedding type: {emb_type}\n'
                       f'\tContext window size: {args.win_size}\n'
                       f'\tEmbedding size: {args.emb_size}\n')

    if args.holdout:
        eventlog = EventlogDataset(args.dataset, read_test=True)

        accuracy = 0
        f1_score = 0
        if args.input_type == 'onehot':
            accuracy, f1_score = run_basicLSTM_model(eventlog, args.input_type, 0, dict())
        elif args.input_type == 'emblayer':
            accuracy, f1_score = run_basicLSTM_model(eventlog, args.input_type, args.emb_size, dict())
        elif args.input_type == 'embeddings':
            embeddings_dict = get_embeddings(eventlog.filename, DataFrameFields.ACTIVITY_COLUMN,
                                             str(emb_type), args.emb_size, args.win_size, logger)
            accuracy, f1_score = run_basicLSTM_model(eventlog, args.input_type, args.emb_size, embeddings_dict)
        else:
            logger.print_error('Incorrect input type. Select one in [onehot, emblayer, embeddings]')
            sys.exit(-1)

        logger.log_console(f'{eventlog.filename} ACCURACY: {accuracy:.6f}')
        logger.log_console(f'{eventlog.filename} F1-Score: {f1_score:.6f}')

        logger.log_metrics_holdout(Path(args.dataset).stem, 'BasicLSTM',
                                   str(emb_type) if emb_type else args.input_type,
                                   args.emb_size if args.emb_size else 0,
                                   args.win_size if args.win_size else 0,
                                   accuracy, f1_score)

    elif args.crossvalidation:
        accuracies = []
        f1_scores = []

        for i in range(Config.CV_FOLDS):
            eventlog = EventlogDataset(args.dataset, read_test=True)

            if args.input_type == 'onehot':
                accuracy, f1_score = run_basicLSTM_model(eventlog, args.input_type, 0, dict())
            elif args.input_type == 'emblayer':
                accuracy, f1_score = run_basicLSTM_model(eventlog, args.input_type, args.emb_size, dict())
            elif args.input_type == 'embeddings':
                embeddings_dict = get_embeddings(eventlog.filename, DataFrameFields.ACTIVITY_COLUMN,
                                                 str(emb_type), args.emb_size, args.win_size, logger, fold=i)
                accuracy, f1_score = run_basicLSTM_model(eventlog, args.input_type, args.emb_size, embeddings_dict)
            else:
                logger.print_error('Incorrect input type. Select one in [onehot, emblayer, embeddings]')
                sys.exit(-1)

            logger.log_console(f'FOLD {i}:\n'
                               f'\tAccuracy: {accuracy}\n'
                               f'\tF1_Score: {f1_score}')

            accuracies.append(accuracy)
            f1_scores.append(f1_score)

        logger.log_console(f'FINAL RESULTS:\n'
                           f'\tACCURACY: {np.mean(np.array(accuracies))}\n'
                           f'\tF1_SCORE: {np.mean(np.array(f1_scores))}')

        logger.log_metrics_cv(Path(args.dataset).stem, 'BasicLSTM',
                              str(emb_type) if emb_type else args.input_type,
                              args.emb_size if args.emb_size else 0,
                              args.win_size if args.win_size else 0,
                              accuracies, f1_scores)

    else:
        logger.print_error("Incorrect partition mode. Select one in [holdout, crossvalidation]")
