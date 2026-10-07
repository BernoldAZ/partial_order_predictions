import torch
from torch.utils.data import Dataset
import numpy as np

class TracesDatasetSeqTime(Dataset):
    def __init__(self, X, X_valid_len, Y, Y_valid_len, grnn_fold, N, F, max_len):
        self._X = X
        self._X_valid_len = X_valid_len
        self._Y = Y
        self._Y_valid_len = Y_valid_len
        self.grnn_fold = grnn_fold
        self.N = N
        self.F = F
        self.max_len = max_len

    def __len__(self):
        return len(self._X)

    def __getitem__(self, idx):
        #X, X_valid_len, Y, Y_valid_len
        trace_grnn = self.grnn_fold[idx]
        zero_pad = np.zeros((self.max_len - len(trace_grnn), self.N, self.F))
        pad_trace_grnn = np.concatenate([zero_pad, trace_grnn], axis=0)

        return_tuple = (torch.tensor(self._X[idx]).permute(1,0),
                torch.tensor(self._X_valid_len[idx]),
                torch.tensor(self._Y[idx]),
                torch.tensor(self._Y_valid_len[idx]),
                torch.tensor(pad_trace_grnn)
                )

        return return_tuple