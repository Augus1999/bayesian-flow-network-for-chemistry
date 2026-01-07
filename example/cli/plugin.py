# -*- coding: utf-8 -*-
"""
Plugin script example.
"""
import random
import torch
import pandas as pd
from bayesianflow_for_chem.data import collate, CSVData

num_workers = 0
shuffle = False
max_sequence_length = 125


def collate_fn(x):
    random.shuffle(x)
    return collate(x)


class CustomData(CSVData):
    def __init__(self, file, chunksize: int = 100000):
        super().__init__(file)
        self.file = file
        self.chunksize = chunksize
        self.data_iterator = pd.read_csv(file, chunksize=chunksize)
        self.current_chunk = next(self.data_iterator)
        self.chunk_index = 0

    def __len__(self):
        return 40000000

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        chunk_idx = idx // self.chunksize
        row_idx = idx % self.chunksize
        if chunk_idx != self.chunk_index:
            if chunk_idx < self.chunk_index:
                self.data_iterator = pd.read_csv(self.file, chunksize=self.chunksize)
                self.current_chunk = next(self.data_iterator)
                self.chunk_index = 0
            while self.chunk_index < chunk_idx:
                self.current_chunk = next(self.data_iterator)
                self.chunk_index += 1

        row = self.current_chunk.iloc[row_idx]
        # You can add "mask": torch.tensor(...) to enable the masked training.
        return self.mapping({"safe": [row.safe]})


__all__ = ["collate_fn", "num_workers", "shuffle", "max_sequence_length", "CustomData"]
