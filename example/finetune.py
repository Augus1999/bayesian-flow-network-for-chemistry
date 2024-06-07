# -*- coding: utf-8 -*-
# author: Nianze A. TAO (SUENO Omozawa)
"""
Fine-tuning.

e.g.,
$ python fintune.py --name=esol --nepoch=300 --datadir="./dataset/moleculenet" --ckpt="./ckpt/zinc15_40m.pt" --mode="regression"
"""
import os
import argparse
from pathlib import Path
import lightning as L
from torch.utils.data import DataLoader
from lightning.pytorch import loggers
from lightning.pytorch.callbacks import ModelCheckpoint
from bayesianflow_for_chem import ChemBFN, MLP
from bayesianflow_for_chem.tool import test
from bayesianflow_for_chem.train import Regressor
from bayesianflow_for_chem.data import collate, CSVData


cwd = Path(__file__).parent

parser = argparse.ArgumentParser()
parser.add_argument(
    "--datadir", default="./moleculenet", type=str, help="dataset folder"
)
parser.add_argument(
    "--ckpt", default="./ckpt/zinc15_40m.pt", type=str, help="ckpt file"
)
parser.add_argument("--name", default="esol", type=str, help="dataset name")
parser.add_argument("--nepoch", default=100, type=int, help="number of epochs")
parser.add_argument(
    "--mode", default="regression", type=str, help="regression or classification"
)
args = parser.parse_args()

workdir = cwd / args.name
logdir = cwd / "log"
datadir = Path(args.datadir)

l_hparam = {
    "mode": args.mode,
    "lr_scheduler_factor": 0.8,
    "lr_scheduler_patience": 20,
    "lr_warmup_step": 1000,
    "max_lr": 1e-4,
    "freeze": False,
}

model = ChemBFN.from_checkpoint(args.ckpt)
mlp = MLP([512, 256, 1])
regressor = Regressor(model, mlp, l_hparam)

checkpoint_callback = ModelCheckpoint(dirpath=workdir, monitor="val_loss")
logger = loggers.TensorBoardLogger(logdir, args.name)
trainer = L.Trainer(
    max_epochs=args.nepoch,
    log_every_n_steps=5,
    logger=logger,
    accelerator="gpu",
    callbacks=[checkpoint_callback],
    enable_progress_bar=False,
)

traindata = DataLoader(
    CSVData(datadir / f"{args.name}_train.csv"), 32, True, collate_fn=collate
)
valdata = DataLoader(CSVData(datadir / f"{args.name}_val.csv"), 32, collate_fn=collate)
testdata = DataLoader(
    CSVData(datadir / f"{args.name}_test.csv"), 32, collate_fn=collate
)

if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
    trainer.fit(regressor, traindata, valdata)
    regressor = Regressor.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path, model=model, mlp=mlp
    )
    regressor.export_model(workdir)
    result = test(regressor.model, regressor.mlp, testdata, l_hparam["mode"])
    print(result)