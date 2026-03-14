# -*- coding: utf-8 -*-
# author: Nianze A. TAO (SUENO Omozawa)
"""
An example of training a MLFF.
"""
import os
import datetime
from pathlib import Path
import torch
import lightning as L
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from bayesianflow_for_chem.mlff import PAINN
from bayesianflow_for_chem.train import GNN
from bayesianflow_for_chem.data import graph_collate, XYZData

cwd = Path(__file__).parent
workdir = cwd / "ckpt/coll"
logdir = cwd / "logs"
max_epochs = 10000
lightning_model_hparam = {
    "model_unit": "eV",
    "lr_scheduler_factor": 0.5,
    "lr_scheduler_patience": 50,
    "lr_scheduler_interval": "epoch",  # can also be "step"
    "lr_scheduler_frequency": 1,
    "lr_warmup_step": 10000,
    "max_lr": 1e-3,
    "ema_alpha": 0.1,  # EMA alpha value
}

model = PAINN(cutoff_radius=5.0, max_neighbour=15)
gnn = GNN(model, lightning_model_hparam)
gnn.compile()

checkpoint_callback = ModelCheckpoint(dirpath=workdir, monitor="val_loss")
earlystop_callback = EarlyStopping(monitor="val_loss", patience=500)
logger = TensorBoardLogger(
    logdir,
    f"mlff",
    datetime.datetime.now().strftime("%Y%m%d%H%M%S"),
)
trainer = L.Trainer(
    max_epochs=max_epochs,
    log_every_n_steps=1,
    logger=logger,
    accelerator="gpu",
    callbacks=[checkpoint_callback, earlystop_callback],
    enable_progress_bar=True,
    inference_mode=False,
    gradient_clip_val=4,  # recommend to use gradient clip
)

if __name__ == "__main__":
    os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:64"
    torch.set_float32_matmul_precision("high")
    # Say you have the COLL dataset: https://figshare.com/articles/dataset/COLL_Dataset_v1_2/13289165
    trainset = XYZData(cwd / "dataset/coll/coll_v1.2_AE_train.xyz")
    traindata = DataLoader(trainset, 64, True, collate_fn=graph_collate, num_workers=1)
    valset = XYZData(cwd / "dataset/coll/coll_v1.2_AE_val.xyz")
    valdata = DataLoader(valset, 64, False, collate_fn=graph_collate, num_workers=1)
    trainer.fit(gnn, traindata, valdata)
    testset = XYZData(cwd / "dataset/coll/coll_v1.2_AE_test.xyz")
    testdata = DataLoader(testset, 64, False, collate_fn=graph_collate, num_workers=1)
    trainer.test(gnn, testdata)
    gnn.export_model(workdir / "last")
    gnn = GNN.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path, model=model
    )
    trainer.test(gnn, testdata)
    gnn.export_model(workdir / "best")
