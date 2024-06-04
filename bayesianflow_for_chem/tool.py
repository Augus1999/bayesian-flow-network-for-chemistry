# -*- coding: utf-8 -*-
# Author: Nianze A. TAO (Omozawa SUENO)
"""
Tools.
"""
import csv
import random
from typing import List, Dict, Union, Optional
import torch
from torch import cuda, Tensor, softmax
from torch.utils.data import DataLoader
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
from .data import VOCAB_KEYS
from .model import ChemBFN, MLP


def _find_device() -> torch.device:
    if cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def test(
    model: ChemBFN,
    mlp: MLP,
    data: DataLoader,
    mode: str = "regression",
    device: Union[str, torch.device, None] = None,
) -> Dict[str, float]:
    """
    Test the trained network.

    :param model: pretrained ChemBFN model
    :param mlp: trained MLP model for testing
    :param data: DataLoader instance
    :param mode: testing mode chosen from `'regression'` and `'classification'`
    :param device: hardware accelerator
    :return: MAE & RMSE & MAPE / ROC-AUC & PRC-AUC
    """
    if device is None:
        device = _find_device()
    model.to(device).eval()
    mlp.to(device).eval()
    tae, tse, tape, scale = 0, 0, 0, 0
    predict_cls, label_cls = [], []
    for d in data:
        x, y = d["token"].to(device), d["value"]
        if mode == "regression":
            y = y.to(device)
            label_mask = (y != torch.inf).float()
            y_mask = y.masked_fill(y == torch.inf, 0)
            y_hat = model.inference(x, mlp) * label_mask
            l1 = (y_hat - y_mask).abs().sum(0).detach()
            l2 = (y_hat - y_mask).pow(2).sum(0).detach()
            l3 = ((y_hat - y_mask).abs() / y).sum(0).detach()
            tae += l1.to("cpu").numpy()
            tse += l2.to("cpu").numpy()
            tape += l3.to("cpu").numpy()
            scale += label_mask.sum(0).detach().to("cpu").numpy()
        if mode == "classification":
            y_hat = softmax(model.inference(x, mlp), -1)
            predict_cls.append(y_hat.detach().to("cpu"))
            label_cls.append(y.flatten())
    if mode == "regression":
        MAE = list(tae / scale)
        RMSE = list((tse / scale) ** 0.5)
        MAPE = list(100 * tape / scale)
        return {"MAE": MAE, "RMSE": RMSE, "MAPE": MAPE}
    if mode == "classification":
        predict_cls = torch.cat(predict_cls, 0).numpy()
        label_cls = torch.cat(label_cls, -1).numpy()
        roc_auc = roc_auc_score(label_cls, predict_cls[:, 1])
        precision, recall, _ = precision_recall_curve(label_cls, predict_cls[:, 1])
        prc_auc = auc(recall, precision)
        return {"ROC-AUC": roc_auc, "PRC-AUC": prc_auc}


def split_dataset(
    file: str, split_ratio: List[int] = [8, 1, 1], method: str = "random"
) -> None:
    """
    Split a dataset.

    :param file: dataset file <file>
    :param split_ratio: traing-testing-validation ratio
    :param method: chosen from `'random'` and `'scaffold'`
    :return: None
    """
    assert file.endswith(".csv")
    assert len(split_ratio) == 3
    assert method in ("random", "scaffold")
    with open(file, "r") as f:
        data = list(csv.reader(f))
    header = data[0]
    raw_data = data[1:]
    smiles_idx = []  # only first index will be used
    for key, h in enumerate(header):
        if h == "smiles":
            smiles_idx.append(key)
    assert len(smiles_idx) > 0
    data_len = len(raw_data)
    train_ratio = split_ratio[0] / sum(split_ratio)
    test_ratio = sum(split_ratio[:2]) / sum(split_ratio)
    train_idx, test_idx = int(data_len * train_ratio), int(data_len * test_ratio)
    if method == "random":
        random.shuffle(raw_data)
        train_set = raw_data[:train_idx]
        test_set = raw_data[train_idx:test_idx]
        val_set = raw_data[test_idx:]
    if method == "scaffold":
        scaffolds: Dict[str, List] = {}
        for key, d in enumerate(raw_data):
            # compute Bemis-Murcko scaffold
            scaffold = MurckoScaffoldSmiles(d[smiles_idx[0]])
            if scaffold in scaffolds:
                scaffolds[scaffold].append(key)
            else:
                scaffolds[scaffold] = [key]
        scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
        train_set, test_set, val_set = [], [], []
        for idxs in scaffolds.values():
            if len(train_set) + len(idxs) > train_idx:
                if len(train_set) + len(test_set) + len(idxs) > test_idx:
                    val_set += [raw_data[i] for i in idxs]
                else:
                    test_set += [raw_data[i] for i in idxs]
            else:
                train_set += [raw_data[i] for i in idxs]
    with open(file.replace(".csv", "_train.csv"), "w", newline="") as ftr:
        writer = csv.writer(ftr)
        writer.writerows([header] + train_set)
    with open(file.replace(".csv", "_test.csv"), "w", newline="") as fte:
        writer = csv.writer(fte)
        writer.writerows([header] + test_set)
    with open(file.replace(".csv", "_val.csv"), "w", newline="") as fva:
        writer = csv.writer(fva)
        writer.writerows([header] + val_set)


def sample(
    model: ChemBFN,
    batch_size: int,
    sequence_size: int,
    sample_step: int = 1000,
    y: Optional[Tensor] = None,
    device: Union[str, torch.device, None] = None,
    vocab_keys: List[str] = VOCAB_KEYS,
) -> List[str]:
    """
    Sampling.

    :param model: trained ChemBFN model
    :param batch_size: batch size
    :param sequence_size: max sequence length
    :param sample_step: number of sampling steps
    :param y: conditioning vector;      shape: (n_b, 1, n_f)
    :param device: hardware accelerator
    :param vocab_keys: a list of (ordered) vocabulary
    :return: a list of generated SMILES strings
    """
    if device is None:
        device = _find_device()
    model.to(device).eval()
    tokens = model.sample(batch_size, sequence_size, y, sample_step)
    smiles = [
        "".join([vocab_keys[i] for i in j])
        .split("<start>")[-1]
        .split("<end>")[0]
        .replace("<pad>", "")
        for j in tokens
    ]
    return smiles
