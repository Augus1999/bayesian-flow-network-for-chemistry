# -*- coding: utf-8 -*-
# author: Nianze A. TAO (SUENO Omozawa)
"""
Fine-tuning.

e.g.,
$ python fintune.py --name=esol --nepoch=100 --datadir="./dataset/moleculenet" --ckpt="./ckpt/zinc15_40m.pt" --mode="regression" --dropout=0.0
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
# in most cases, --ntask=2 when --mode=classification and --ntask=1 when --mode=regression
parser.add_argument("--ntask", default=1, type=int, help="number of tasks")
parser.add_argument(
    "--mode", default="regression", type=str, help="regression or classification"
)
parser.add_argument("--dropout", default=0.5, type=float, help="dropout rate")
args = parser.parse_args()

workdir = cwd / args.name
logdir = cwd / "log"
datadir = Path(args.datadir)

l_hparam = {
    "mode": args.mode,
    "lr_scheduler_factor": 0.8,
    "lr_scheduler_patience": 20,
    "lr_warmup_step": 1000 if args.mode == "regression" else 100,
    "max_lr": 1e-4,
    "freeze": False,
}

model = ChemBFN.from_checkpoint(args.ckpt)
mlp = MLP([512, 256, args.ntask], dropout=args.dropout)
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
    regressor.export_model(workdir)
    result = test(model, regressor.mlp, testdata, l_hparam["mode"])
    print("last:", result)
    regressor = Regressor.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path, model=model, mlp=mlp
    )
    result = test(regressor.model, regressor.mlp, testdata, l_hparam["mode"])
    print("best:", result)
