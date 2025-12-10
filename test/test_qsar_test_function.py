# -*- coding: utf-8 -*-
# Author: Nianze A. Tao (Omozawa Sueno)
"""
`bayesianflow_for_chem.tool.test` should work with user provided metrics.
"""
from functools import partial
import torch
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import f1_score as _f1_score
from torch.utils.data import DataLoader, Dataset
from bayesianflow_for_chem import ChemBFN, MLP
from bayesianflow_for_chem.data import VOCAB_COUNT, smiles2token, collate
from bayesianflow_for_chem.tool import test as _test


class DummyMLP(MLP):
    def forward(self, x):
        y = super().forward(x)
        if y.shape[-1] == 3:
            return y.new_tensor(
                [[0.01, -0.1, 0.0], [-0.1, 0.0, -0.01], [0.5, 0.0, -0.1]]
            )
        return y.new_tensor(
            [
                [0.01, -0.1, 0.0, 0.3, 0.03, 0.1],
                [-0.1, 0.0, -0.01, 0.8, 0.01, -0.2],
                [0.5, 0.0, -0.1, 0.2, 0.3, -0.09],
            ]
        )


model = ChemBFN(VOCAB_COUNT)
mlp1 = DummyMLP([512, 256, 3], dropout=0.5)
mlp2 = DummyMLP([512, 256, 6], dropout=0.5)

smi = ["CCCO", "c1ccccc1", "N#C.O"]
x = [smiles2token(i) for i in smi]
y_true1 = [
    torch.tensor([0.02, 0.1, -0.1]),
    torch.tensor([0.0, -0.1, 0.3]),
    torch.tensor([1.2, 0.0, -0.1]),
]
y_true2 = [
    torch.tensor([1], dtype=torch.long),
    torch.tensor([2], dtype=torch.long),
    torch.tensor([0], dtype=torch.long),
]
y_true3 = [
    torch.tensor([0, 0, 1], dtype=torch.long),
    torch.tensor([1, 1, 0], dtype=torch.long),
    torch.tensor([0, 1, 0], dtype=torch.long),
]


def acc(y_true, y_pred):
    return np.mean((y_true == y_pred.argmax(-1, keepdim=True)).float().numpy())


def r_score(y_true, y_pred):
    return pearsonr(y_true, y_pred).statistic


def f1_score(y_true, y_pred, **karg):
    return _f1_score(y_true.flatten(), y_pred.argmax(-1).flatten(), **karg)


class DummyData(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return {"token": self.x[idx], "value": self.y[idx]}


loader1 = DataLoader(DummyData(x, y_true1), 3, False, collate_fn=collate)
loader2 = DataLoader(DummyData(x, y_true2), 3, False, collate_fn=collate)
loader3 = DataLoader(DummyData(x, y_true3), 3, False, collate_fn=collate)

res1 = {
    "MAE": [0.27000001072883606, 0.10000000149011612, 0.1366666704416275],
    "RMSE": [0.4082891345024109, 0.12909944355487823, 0.18806026875972748],
    "R^2": [0.4703826308250427, -1.5, 0.005312561988830566],
    "R": [0.9874663352966309, -0.866025447845459, 0.419313907623291],
}
res2 = {
    "ROC-AUC": [0.6666666666666666],
    "PRC-AUC": [],
    "accuracy": [0.3333333432674408],
    "F1": [0.2222222222222222],
}
res3 = {
    "ROC-AUC": [1.0, 0.75, 1.0],
    "PRC-AUC": [1.0, 0.9166666666666666, 1.0],
    "accuracy": [1.0, 0.6666666865348816, 1.0],
    "F1": [1.0, 0.8, 1.0],
}


def test():
    result1 = _test(model, mlp1, loader1, "regression", other_metrics={"R": r_score})
    result2 = _test(
        model,
        mlp1,
        loader2,
        "classification",
        other_metrics={"accuracy": acc, "F1": partial(f1_score, average="weighted")},
    )
    result3 = _test(
        model,
        mlp2,
        loader3,
        "classification",
        other_metrics={"accuracy": acc, "F1": f1_score},
    )
    assert result1 == res1
    assert result2 == res2
    assert result3 == res3
