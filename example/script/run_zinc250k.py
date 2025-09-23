# -*- coding: utf-8 -*-
# author: Nianze A. TAO (SUENO Omozawa)
"""
Training and sampling on ZINC250k dataset.

e.g.,
$ python run_zinc250k.py --version=smiles --train_mode=sar --target=fa7 --samplestep=1000 --datadir="./dataset/zinc250k"
"""
import os
import json
import argparse
from pathlib import Path
import lightning as L
import torch
from torch.utils.data import DataLoader
from lightning.pytorch import loggers
from lightning.pytorch.callbacks import ModelCheckpoint
from bayesianflow_for_chem import ChemBFN, MLP
from bayesianflow_for_chem.train import Model
from bayesianflow_for_chem.tool import sample
from bayesianflow_for_chem.data import (
    VOCAB_COUNT,
    VOCAB_KEYS,
    CSVData,
    collate,
    load_vocab,
    smiles2token,
    split_selfies,
)

parser = argparse.ArgumentParser()
parser.add_argument("--datadir", default="./zinc250k", type=str, help="dataset folder")
parser.add_argument("--version", default="smiles", type=str, help="SMIlES or SELFIES")
parser.add_argument("--target", default="parp1", type=str, help="target protein")
parser.add_argument("--train_mode", default="normal", type=str, help="normal or sar")
parser.add_argument("--samplestep", default=1000, type=int, help="sample steps")
args = parser.parse_args()

cwd = Path(__file__).parent
targets = "parp1,fa7,5ht1b,braf,jak2".split(",")
assert args.target in targets
dataset_file = f"{args.datadir}/zinc250k.csv"
workdir = cwd / f"zinc250k_{args.train_mode}/{args.target}_{args.version}"
logdir = cwd / "log"
max_epochs = 100
l_hparam = {"lr": 5e-5, "lr_warmup_step": 1000, "uncond_prob": 0.2}

if args.version.lower() == "smiles":

    def encode(x):
        smiles = x["smiles"][0]
        value = [x["qed"][0], x["sa"][0], x[args.target][0]]
        value = [float(i) for i in value]
        return {"token": smiles2token(smiles), "value": torch.tensor(value)}

    pad_len = 111
    num_vocab = VOCAB_COUNT
    vocab_keys = VOCAB_KEYS
    train_data = CSVData(dataset_file)
    train_data.map(encode)
else:
    import selfies

    pad_len = 74
    dataset_file = dataset_file.replace(".csv", "_selfies.csv")
    vocab_file = cwd / "zinc250k_selfies_vocab.txt"
    if not os.path.exists(dataset_file):
        with open(dataset_file.replace("_selfies.csv", "csv"), "r") as f:
            _data = f.readlines()
        selfies_list = []
        line0 = _data[0].split(",")
        line0[0] = "selfies"
        _data[0] = ",".join(line0)
        for j, line in enumerate(_data[1:]):
            _info = line.split(",")
            s = selfies.encoder(_info[0])
            _info[0] = s
            _data[j + 1] = ",".join(_info)
            selfies_list.append(s)
        if not os.path.exists(vocab_file):
            vocab = []
            for i in selfies_list:
                vocab += split_selfies(i)
            vocab = ["<pad>", "<start>", "<end>"] + list(set(vocab))
            with open(vocab_file, "w") as f:
                f.write("\n".join(vocab))
        with open(dataset_file, "w", newline="") as f:
            f.write("".join(_data))
    vocab_data = load_vocab(vocab_file)
    num_vocab = vocab_data["vocab_count"]
    vocab_dict = vocab_data["vocab_dict"]
    vocab_keys = vocab_data["vocab_keys"]

    def selfies2token(s):
        return torch.tensor(
            [1] + [vocab_dict[i] for i in split_selfies(s)] + [2], dtype=torch.long
        )

    def encode(x):
        s = x["selfies"][0]
        value = [x["qed"][0], x["sa"][0], x[args.target][0]]
        value = [float(i) for i in value]
        return {"token": selfies2token(s), "value": torch.tensor(value)}

    train_data = CSVData(dataset_file)
    train_data.map(encode)

bfn = ChemBFN(num_vocab)
mlp = MLP([3, 256, 512])
model = Model(bfn, mlp, hparam=l_hparam)
if args.train_mode == "normal":
    model.model.semi_autoregressive = False
elif args.train_mode == "sar":
    model.model.semi_autoregressive = True
else:
    raise NotImplementedError

checkpoint_callback = ModelCheckpoint(dirpath=workdir, every_n_train_steps=1000)
logger = loggers.TensorBoardLogger(logdir, f"zinc250k_{args.version}")
trainer = L.Trainer(
    max_epochs=max_epochs,
    log_every_n_steps=500,
    logger=logger,
    accelerator="gpu",
    callbacks=[checkpoint_callback],
    enable_progress_bar=False,
)

if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    os.environ["MAX_PADDING_LENGTH"] = f"{pad_len}"
    torch.set_float32_matmul_precision("medium")
    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=128,
        shuffle=True,
        collate_fn=collate,
        num_workers=2,
    )
    trainer.fit(model, train_dataloader)
    model.export_model(workdir)

    model = ChemBFN.from_checkpoint(workdir / "model.pt")
    mlp = MLP.from_checkpoint(workdir / "mlp.pt")
    # note that the objective values in the dataset
    # have been normalised as (QED, (10 - SA) / 9, -DS, ...)
    y = mlp(torch.tensor([[0.8, 0.8, 12.0]])).repeat(3000, 1)[:, None, :]
    norm_sam, sar_sam = {}, {}
    model.semi_autoregressive = False
    for i in range(5):
        _sample = sample(
            model, 3000, pad_len, args.samplestep, y, 0.5, vocab_keys=vocab_keys
        )
        norm_sam[f"sample_{i+1}"] = [
            selfies.decoder(i) if args.version.lower() == "selfies" else i
            for i in _sample
        ]
    model.semi_autoregressive = True
    for i in range(5):
        _sample = sample(
            model, 3000, pad_len, args.samplestep, y, 0.5, vocab_keys=vocab_keys
        )
        sar_sam[f"sample_{i+1}"] = [
            selfies.decoder(i) if args.version.lower() == "selfies" else i
            for i in _sample
        ]
    with open(
        cwd / f"zinc250k_{args.target}_{args.train_mode}_{args.version}.json", "w"
    ) as f:
        json.dump(
            {"normal_sample": norm_sam, "sar_sample": sar_sam},
            f,
            indent=4,
            separators=(",", ": "),
        )
