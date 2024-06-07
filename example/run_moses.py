# -*- coding: utf-8 -*-
# author: Nianze A. TAO (SUENO Omozawa)
"""
Training, sampling, and testing on MOSES dataset.

e.g.,
$ python run_moses.py --version=smiles --samplestep=100 --datadir="./dataset/moses"
"""
import os
import json
import argparse
from pathlib import Path
import moses
import torch
import lightning as L
from torch.utils.data import DataLoader
from lightning.pytorch import loggers
from lightning.pytorch.callbacks import ModelCheckpoint
from bayesianflow_for_chem import ChemBFN
from bayesianflow_for_chem.tool import sample
from bayesianflow_for_chem.train import Model
from bayesianflow_for_chem.data import (
    VOCAB_KEYS,
    VOCAB_COUNT,
    collate,
    load_vocab,
    split_selfies,
    CSVData,
    BaseCSVDataClass,
)


cwd = Path(__file__).parent

parser = argparse.ArgumentParser()
parser.add_argument("--datadir", default="./moses", type=str, help="dataset folder")
parser.add_argument("--version", default="smiles", type=str, help="SMIlES or SELFIES")
parser.add_argument("--samplestep", default=100, type=int, help="sample steps")
args = parser.parse_args()

assert args.version.lower() in ("smiles", "selfies")

workdir = cwd / f"moses_{args.version}"
logdir = cwd / "log"

if args.version.lower() == "smiles":
    pad_len = 59  # 57 + 2
    num_vocab = VOCAB_COUNT
    vocab_keys = VOCAB_KEYS
    dataset_file = args.datadir + "/train.csv"
    train_data = CSVData(dataset_file)
else:
    import selfies

    pad_len = 57  # 55 + 2
    dataset_file = args.datadir + "/trian_selfies.csv"
    vocab_file = cwd / "moses_selfies_vocab.txt"
    if not os.path.exists(dataset_file):
        with open(args.datadir + "/train.csv", "r") as f:
            smiles_data = f.readlines()[1:]
        selfies_list = [selfies.encoder(i.split(",")[0]) for i in smiles_data]
        if not os.path.exists(vocab_file):
            vocab = []
            for i in selfies_list:
                vocab += split_selfies(i)
            vocab = ["<pad>", "<start>", "<end>"] + list(set(vocab))
            with open(vocab_file, "w") as f:
                f.write("\n".join(vocab))
        with open(dataset_file, "w") as f:
            f.write("\n".join(["selfies"] + selfies_list))
    vocab_data = load_vocab(vocab_file)
    num_vocab = vocab_data["vocab_count"]
    vocab_dict = vocab_data["vocab_dict"]
    vocab_keys = vocab_data["vocab_keys"]

    def selfies2token(s):
        return torch.tensor(
            [1] + [vocab_dict[i] for i in split_selfies(s)] + [2], dtype=torch.long
        )

    class SELData(BaseCSVDataClass):
        def __getitem__(self, idx) -> None:
            if torch.is_tensor(idx):
                idx = idx.tolist()
            d = self.data[idx + 1].replace("\n", "").split(",")
            s = ".".join([d[i] for i in self.selfies_idx if d[i] != ""])
            token = selfies2token(s)
            return {"token": token}

    train_data = SELData(dataset_file)

model = Model(ChemBFN(num_vocab))
checkpoint_callback = ModelCheckpoint(dirpath=workdir, every_n_train_steps=1000)
logger = loggers.TensorBoardLogger(logdir, f"moses_{args.version}")
trainer = L.Trainer(
    max_epochs=100,  # you can run it longer
    log_every_n_steps=50,
    logger=logger,
    accelerator="gpu",
    callbacks=[checkpoint_callback],
    enable_progress_bar=False,
)


if __name__ == "__main__":
    os.environ["MAX_PADDING_LENGTH"] = f"{pad_len}"  # set the global padding length
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=120,  # reduce batch-size if your GPU has less than 5GB of VRAM
        shuffle=True,
        collate_fn=collate,
        num_workers=2,
    )
    trainer.fit(model, train_dataloader)
    model.export_model(workdir)
    metrics = []
    result = {
        "name": "MOSES",
        "version": args.version,
        "sample step": args.samplestep,
        "metrics": {},
        "samples": {},
    }
    for k in [1, 2, 3]:
        smiles_list = []
        for _ in range(10):
            smiles_list += sample(
                model.model, 3000, pad_len, args.samplestep, vocab_keys=vocab_keys
            )
        if args.version.lower() == "selfies":
            smiles_list = [selfies.decoder(i) for i in smiles_list]
        result["samples"][f"run {k}"] = smiles_list
        m = moses.get_all_metrics(smiles_list)
        metrics.append(m)
        result["metrics"][f"run {k}"] = m
    mean, std = {}, {}
    for key in metrics[0]:
        mean[key] = torch.tensor([i[key] for i in metrics]).mean().item()
        std[key] = torch.tensor([i[key] for i in metrics]).std().item()
    result["metrics"]["mean"] = mean
    result["metrics"]["std"] = std
    with open(
        cwd / f"moses_{args.version}_samplestep_{args.samplestep}_results.json", "w"
    ) as f:
        json.dump(result, f, indent=4, separators=(",", ": "))
