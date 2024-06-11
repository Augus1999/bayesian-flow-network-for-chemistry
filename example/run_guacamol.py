# -*- coding: utf-8 -*-
# author: Nianze A. TAO (SUENO Omozawa)
"""
Training, sampling, and testing on GuacaMol dataset.

e.g.,
$ python run_guacamol.py --version=smiles --samplestep=100 --datadir="./dataset/guacamol"
"""
import os
import argparse
from pathlib import Path
import torch
import numpy as np
import lightning as L
from torch.utils.data import DataLoader, Dataset
from lightning.pytorch import loggers
from lightning.pytorch.callbacks import ModelCheckpoint
from guacamol.distribution_matching_generator import DistributionMatchingGenerator
from guacamol.assess_distribution_learning import assess_distribution_learning
from bayesianflow_for_chem import ChemBFN
from bayesianflow_for_chem.tool import sample
from bayesianflow_for_chem.train import Model
from bayesianflow_for_chem.data import (
    VOCAB_KEYS,
    VOCAB_COUNT,
    collate,
    load_vocab,
    smiles2token,
    split_selfies,
)


cwd = Path(__file__).parent

parser = argparse.ArgumentParser()
parser.add_argument("--datadir", default="./guacamol", type=str, help="dataset folder")
parser.add_argument("--version", default="smiles", type=str, help="SMIlES or SELFIES")
parser.add_argument("--samplestep", default=100, type=int, help="sample steps")
args = parser.parse_args()

assert args.version.lower() in ("smiles", "selfies")

workdir = cwd / f"guacamol_{args.version}"
logdir = cwd / "log"

if args.version.lower() == "smiles":
    pad_len = 103  # 101 + 2
    num_vocab = VOCAB_COUNT
    vocab_keys = VOCAB_KEYS
    dataset_file = args.datadir + "/guacamol_v1_train.smiles"

    class SMIData(Dataset):
        def __init__(self, file: str) -> None:
            super().__init__()
            with open(file, "r") as f:
                self.data = f.readlines()

        def __len__(self) -> int:
            return len(self.data)

        def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()
            d: str = self.data[idx]
            s = d.replace("\n", "")
            token = smiles2token(s)
            return {"token": token}

    train_data = SMIData(dataset_file)
else:
    import selfies

    pad_len = 111  # 109 + 2
    dataset_file = args.datadir + "/guacamol_v1_train.selfies"
    vocab_file = cwd / "guacamol_selfies_vocab.txt"
    if not os.path.exists(dataset_file):
        with open(args.datadir + "/guacamol_v1_train.smiles", "r") as f:
            smiles_data = f.readlines()
        selfies_list = [
            selfies.encoder(i.replace("\n", ""), False) for i in smiles_data
        ]
        if not os.path.exists(vocab_file):
            vocab = []
            for i in selfies_list:
                vocab += split_selfies(i)
            vocab = ["<pad>", "<start>", "<end>"] + list(set(vocab))
            with open(vocab_file, "w") as f:
                f.write("\n".join(vocab))
        with open(dataset_file, "w") as f:
            f.write("\n".join(selfies_list))
    vocab_data = load_vocab(vocab_file)
    num_vocab = vocab_data["vocab_count"]
    vocab_dict = vocab_data["vocab_dict"]
    vocab_keys = vocab_data["vocab_keys"]

    def selfies2token(s):
        return torch.tensor(
            [1] + [vocab_dict[i] for i in split_selfies(s)] + [2], dtype=torch.long
        )

    class SELData(Dataset):
        def __init__(self, file: str) -> None:
            super().__init__()
            with open(file, "r") as f:
                self.data = f.readlines()

        def __len__(self) -> int:
            return len(self.data)

        def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()
            d: str = self.data[idx]
            s = d.replace("\n", "")
            token = selfies2token(s)
            return {"token": token}

    train_data = SELData(dataset_file)

model = Model(ChemBFN(num_vocab))
checkpoint_callback = ModelCheckpoint(dirpath=workdir, every_n_train_steps=1000)
logger = loggers.TensorBoardLogger(logdir, f"guacamol_{args.version}")
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
        batch_size=120,  # reduce batch-size if your GPU has less than 10GB of VRAM
        shuffle=True,
        collate_fn=collate,
        num_workers=2,
    )
    trainer.fit(model, train_dataloader)
    model.export_model(workdir)
    smiles_list = []
    for _ in range(30):
        smiles_list += sample(
            model.model, 1000, pad_len, args.samplestep, vocab_keys=vocab_keys
        )
    if args.version.lower() == "selfies":
        smiles_list = [selfies.decoder(i) for i in smiles_list]
    with open(
        cwd / f"guacamol_{args.version}_sample_samplestep_{args.samplestep}.csv", "w"
    ) as f:
        f.write("\n".join(smiles_list))

    class Sampler(DistributionMatchingGenerator):
        """
        Generator that samples SMILES strings from a predefined list.
        """

        def __init__(self, data: list) -> None:
            self.data = data

        def generate(self, number_samples: int):
            return list(np.random.choice(self.data, size=number_samples))

    for i in [1, 2, 3]:
        generator = Sampler(smiles_list)
        assess_distribution_learning(
            generator,
            chembl_training_file=args.datadir + "/guacamol_v1_train.smiles",
            json_output_file=cwd
            / f"guacamol_{args.version}_sample_{i}_metrics_samplestep_{args.samplestep}.json",
            benchmark_version="v2",
        )
