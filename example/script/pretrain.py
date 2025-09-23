# -*- coding: utf-8 -*-
# author: Nianze A. TAO (SUENO Omozawa)
"""
pretraining.

e.g.,
$ python pretrain.py --nepoch=15 --datafile="./dataset/train.csv"
"""
import os
import argparse
from pathlib import Path
import lightning as L
from torch.utils.data import DataLoader
from lightning.pytorch import loggers
from lightning.pytorch.callbacks import ModelCheckpoint
from bayesianflow_for_chem import ChemBFN
from bayesianflow_for_chem.train import Model
from bayesianflow_for_chem.data import smiles2token, collate, CSVData, VOCAB_COUNT


cwd = Path(__file__).parent

parser = argparse.ArgumentParser()
parser.add_argument("--datafile", default="./train.csv", type=str, help="dataset file")
parser.add_argument("--nepoch", default=15, type=int, help="number of epochs")
args = parser.parse_args()

workdir = cwd / "pretrain"
logdir = cwd / "log"


model = Model(ChemBFN(VOCAB_COUNT))
checkpoint_callback = ModelCheckpoint(dirpath=workdir, every_n_train_steps=1000)
logger = loggers.TensorBoardLogger(logdir, "pretrain")
trainer = L.Trainer(
    max_epochs=args.nepoch,
    log_every_n_steps=500,
    logger=logger,
    accelerator="gpu",
    callbacks=[checkpoint_callback],
    enable_progress_bar=False,
)


if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
    dataset = CSVData(args.datafile)
    dataset.map(lambda x: {"token": smiles2token(".".join(x["smiles"]))})
    data = DataLoader(dataset, 512, True, collate_fn=collate)
    trainer.fit(model, data)
    model.export_model(workdir)
