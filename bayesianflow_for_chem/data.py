# -*- coding: utf-8 -*-
# Author: Nianze A. TAO (Omozawa SUENO)
"""
Tokenise SMILES strings.
"""
import os
import re
from pathlib import Path
from typing import List, Dict, Union, Optional
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

__filedir__ = Path(__file__).parent

SMI_REGEX_PATTERN = (
    r"(\[|\]|H[e,f,g,s,o]?|"
    r"L[i,v,a,r,u]|"
    r"B[e,r,a,i,h,k]?|"
    r"C[l,a,r,o,u,d,s,n,e,m,f]?|"
    r"N[e,a,i,b,h,d,o,p]?|"
    r"O[s,g]?|S[i,c,e,r,n,m,b,g]?|"
    r"K[r]?|T[i,c,e,a,l,b,h,m,s]|"
    r"G[a,e,d]|R[b,u,h,e,n,a,f,g]|"
    r"Yb?|Z[n,r]|P[t,o,d,r,a,u,b,m]?|"
    r"F[e,r,l,m]?|M[g,n,o,t,c,d]|"
    r"A[l,r,s,g,u,t,c,m]|I[n,r]?|"
    r"W|X[e]|E[u,r,s]|U|D[b,s,y]|"
    r"b|c|n|o|s|p|"
    r"\(|\)|\.|=|#|-|\+|\\|\/|:|"
    r"~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"
)
SEL_REGEX_PATTERN = r"(\[[^\]]+]|\.)"
smi_regex = re.compile(SMI_REGEX_PATTERN)
sel_regex = re.compile(SEL_REGEX_PATTERN)


def load_vocab(
    vocab_file: Union[str, Path]
) -> Dict[str, Union[int, List[str], Dict[str, int]]]:
    """
    Load vocabulary from source file.

    :param vocab_file: file that contains vocabulary
    :return: {"vocab_keys": vocab_keys, "vocab_count": vocab_count, "vocab_dict": vocab_dict}
    """
    with open(vocab_file, "r", encoding="utf-8") as f:
        lines = f.read().strip()
    vocab_keys = lines.split("\n")
    vocab_count = len(vocab_keys)
    vocab_dict = dict(zip(vocab_keys, range(vocab_count)))
    return {
        "vocab_keys": vocab_keys,
        "vocab_count": vocab_count,
        "vocab_dict": vocab_dict,
    }


_DEFUALT_VOCAB = load_vocab(__filedir__ / "vocab.txt")
VOCAB_KEYS = _DEFUALT_VOCAB["vocab_keys"]
VOCAB_DICT = _DEFUALT_VOCAB["vocab_dict"]
VOCAB_COUNT = _DEFUALT_VOCAB["vocab_count"]


def smiles2vec(smiles: str) -> List[int]:
    """
    SMILES tokenisation using a dataset-independent regex pattern.

    :param smiles: SMILES string
    :return: tokens w/o <start> and <end>
    """
    tokens = [token for token in smi_regex.findall(smiles)]
    return [VOCAB_DICT[token] for token in tokens]


def split_selfies(selfies: str) -> List[str]:
    """
    SELFIES tokenisation.

    :param selfies: SELFIES string
    :return: SELFIES vocab
    """
    return [token for token in sel_regex.findall(selfies)]


def smiles2token(smiles: str) -> Tensor:
    # start token: <start> = 1; end token: <esc> = 2
    return torch.tensor([1] + smiles2vec(smiles) + [2], dtype=torch.long)


def collate(batch: List) -> Dict[str, Tensor]:
    """
    Padding the data in one batch into the same size.\n
    Should be passed to `~torch.utils.data.DataLoader` as `DataLoader(collate_fn=collate, ...)`.

    :param batch: a list of data (one batch)
    :return: batched {"token": token} or {"token": token, "value": value}
    """
    token = [i["token"] for i in batch]
    if hasattr(os.environ, "MAX_PADDING_LENGTH"):
        lmax = int(os.environ["MAX_PADDING_LENGTH"])
    else:
        lmax = max([len(w) for w in token])
    token = torch.cat(
        [F.pad(i, (0, lmax - len(i)), value=0)[None, :] for i in token], 0
    )
    out_dict = {"token": token}
    if "value" in batch[0]:
        out_dict["value"] = torch.cat([i["value"][None, :] for i in batch], 0)
    if "mask" in batch[0]:
        mask = [i["mask"] for i in batch]
        out_dict["mask"] = torch.cat(
            [F.pad(i, (0, lmax - len(i)), value=0)[None, :] for i in mask], 0
        )
    return out_dict


class BaseCSVDataClass(Dataset):
    def __init__(
        self,
        file: str,
        limit: Optional[int] = None,
        label_idx: Optional[List[int]] = None,
    ) -> None:
        """
        Define dataset stored in CSV file.\n
        This is the base class that should not be accessed directly.

        :param file: dataset file name <file>
        :param limit: item limit
        :param label_idx: a list of indices indicating which value to be input;
                          use `'None'` for inputting all values
        """
        super().__init__()
        self.data = []
        with open(file, "r") as db:
            self.data = db.readlines()
        self.label_idx = label_idx
        self.smiles_idx, self.selfies_idx, self.value_idx = [], [], []
        for key, i in enumerate(self.data[0].replace("\n", "").split(",")):
            i = i.lower()
            if i == "smiles" or i == "safe":
                self.smiles_idx.append(key)
            if i == "selfies":
                self.selfies_idx.append(key)
            if i == "value":
                self.value_idx.append(key)
        if limit:
            self.data = self.data[: limit + 1]

    def __len__(self) -> int:
        return len(self.data) - 1

    def __getitem__(self, idx: Union[int, Tensor]) -> None:
        """
        You need to overwrite this method in the inherited class.
        See `~bayesianflow_for_chem.data.CSVData` as an example.
        """
        return super().__getitem__(idx)


class CSVData(BaseCSVDataClass):
    def __getitem__(self, idx: Union[int, Tensor]) -> Dict[str, Dict[str, Tensor]]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # valid `idx` should start from 1 instead of 0
        d: List[str] = self.data[idx + 1].replace("\n", "").split(",")
        smiles = ".".join([d[i] for i in self.smiles_idx if d[i] != ""])
        values = [
            float(d[i]) if d[i].strip() != "" else torch.inf for i in self.value_idx
        ]
        if self.label_idx:
            values = [values[i] for i in self.label_idx]
        token = smiles2token(smiles)
        out_dict = {"token": token}
        if len(values) != 0:
            out_dict["value"] = torch.tensor(values, dtype=torch.float32)
        return out_dict


if __name__ == "__main__":
    ...
