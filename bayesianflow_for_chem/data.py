# -*- coding: utf-8 -*-
# Author: Nianze A. TAO (Omozawa SUENO)
"""
Tokenise SMILES/SAFE/SELFIES/FASTA strings.
"""
import os
import re
from pathlib import Path
from typing import Any, List, Dict, Union, Callable
import torch
import torch.nn.functional as F
from ase import Atoms
from ase.io import read
from torch import Tensor
from torch.utils.data import Dataset

__filedir__ = Path(__file__).parent

_SMI_REGEX_PATTERN = (
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
    r"~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9]|"
    r"<pad>)"
)
_SEL_REGEX_PATTERN = r"(\[[^\]]+]|\.|<pad>)"
_FAS_REGEX_PATTERN = (
    r"(A|B|C|D|E|F|G|H|I|J|K|L|M|N|O|P|Q|R|S|T|U|V|W|X|Y|Z|-|\*|\.|<pad>)"
)
_smi_regex = re.compile(_SMI_REGEX_PATTERN)
_sel_regex = re.compile(_SEL_REGEX_PATTERN)
_fas_regex = re.compile(_FAS_REGEX_PATTERN)


def load_vocab(
    vocab_file: Union[str, Path],
) -> Dict[str, Union[int, List[str], Dict[str, int]]]:
    """
    Load vocabulary from source file.

    :param vocab_file: file that contains vocabulary
    :type vocab_file: str | pathlib.Path
    :return: {"vocab_keys": vocab_keys, "vocab_count": vocab_count, "vocab_dict": vocab_dict}
    :rtype: dict
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


_DEFUALT_VOCAB = load_vocab(__filedir__ / "_data/vocab.txt")
VOCAB_KEYS: List[str] = _DEFUALT_VOCAB["vocab_keys"]
VOCAB_DICT: Dict[str, int] = _DEFUALT_VOCAB["vocab_dict"]
VOCAB_COUNT: int = _DEFUALT_VOCAB["vocab_count"]
FASTA_VOCAB_KEYS = (
    VOCAB_KEYS[0:3]
    + "A B C D E F G H I K L M N P Q R S T V W Y Z - . J O U X *".split()
)
FASTA_VOCAB_COUNT = len(FASTA_VOCAB_KEYS)
FASTA_VOCAB_DICT = dict(zip(FASTA_VOCAB_KEYS, range(FASTA_VOCAB_COUNT)))


def smiles2vec(smiles: str) -> List[int]:
    """
    SMILES tokenisation using a dataset-independent regex pattern.

    :param smiles: SMILES string
    :type smiles: str
    :return: tokens w/o `<start>` and `<end>`
    :rtype: list
    """
    # tokens = [token for token in _smi_regex.findall(smiles)]
    tokens = list(_smi_regex.findall(smiles))
    return [VOCAB_DICT[token] for token in tokens]


def fasta2vec(fasta: str) -> List[int]:
    """
    FASTA sequence tokenisation using a dataset-independent regex pattern.

    :param fasta: protein (amino acid) sequence
    :type fasta: str
    :return: tokens w/o `<start>` and `<end>`
    :rtype: list
    """
    # tokens = [token for token in _fas_regex.findall(fasta)]
    tokens = list(_fas_regex.findall(fasta))
    return [FASTA_VOCAB_DICT[token] for token in tokens]


def split_selfies(selfies: str) -> List[str]:
    """
    SELFIES tokenisation.

    :param selfies: SELFIES string
    :type selfies: str
    :return: SELFIES vocab
    :rtype: list
    """
    # return [token for token in _sel_regex.findall(selfies)]
    return list(_sel_regex.findall(selfies))


def smiles2token(smiles: str) -> Tensor:
    """
    SMILES string -> token tensor.

    :param smiles: SMILES string
    :type smiles: str
    :return: token tensor;  shape: (n_token)
    :rtype: torch.Tensor
    """
    # start token: <start> = 1; end token: <esc> = 2
    return torch.tensor([1] + smiles2vec(smiles) + [2], dtype=torch.long)


def fasta2token(fasta: str) -> Tensor:
    """
    FASTA string -> token tensor.

    :param fasta: FASTA string
    :type fasta: str
    :return: token tensor;  shape: (n_token)
    :rtype: torch.Tensor
    """
    # start token: <start> = 1; end token: <end> = 2
    return torch.tensor([1] + fasta2vec(fasta) + [2], dtype=torch.long)


def collate(batch: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    """
    Padding the data in one batch into the same size.\n
    Should be passed to `~torch.utils.data.DataLoader` as `DataLoader(collate_fn=collate, ...)`.

    :param batch: a list of data (one batch)
    :type batch: list
    :return: batched {"token": token} or {"token": token, "value": value}
    :rtype: dict
    """
    token = [i["token"] for i in batch]
    if "MAX_PADDING_LENGTH" in os.environ:
        lmax = int(os.environ["MAX_PADDING_LENGTH"])
    else:
        lmax = max(len(w) for w in token)
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


def graph_collate(batch: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    """
    Collating the graph data into one batch.\n
    Should be passed to `~torch.utils.data.DataLoader`
    as `DataLoader(collate_fn=graph_collate, ...)`.

    :param batch: a list of data (one batch)
    :type batch: list
    :return: batched {
                        "Z": atmoic numbers,
                        "R": nuclear coordinates,
                        "batch": batching mask,
                        "lattice": unit cell vectors (optional)
                        "scalar": energies (optional),
                        "vector": atomic forces (optional),
                        }
    :rtype: dict
    """
    scalars, vectors = [], []
    charges, positions, mask, lattice = [], [], [], []
    for item in batch:
        charges.append(item["Z"])
        positions.append(item["R"])
        if "scalar" in item:
            scalars.append(item["scalar"].unsqueeze(0))
        if "vector" in item:
            vectors.append(item["vector"])
        if "lattice" in item:
            lattice.append(item["lattice"][None, ...])
    charges = torch.cat(charges, 0)[None, ...]
    positions = torch.cat(positions, 0)[None, ...]
    n_total = charges.shape[1]
    i = 0
    for item in batch:
        n = item["Z"].shape[0]
        batch = torch.ones(1, n)
        p1, p2 = torch.zeros(1, i), torch.zeros(1, n_total - n - i)
        mask.append(torch.cat([p1, batch, p2], -1))
        i += n
    mask = torch.cat(mask, 0)[..., None]
    out_dict = {"Z": charges, "R": positions, "batch": mask}
    if lattice:
        out_dict["lattice"] = torch.cat(lattice, 0)
    if scalars:
        out_dict["scalar"] = torch.cat(scalars, 0)
    if vectors:
        out_dict["vector"] = torch.cat(vectors, 0).unsqueeze(0)
    return out_dict


class CSVData(Dataset):
    """
    Customisable CSV dataset class.
    """

    def __init__(self, file: Union[str, Path]) -> None:
        """
        Define dataset stored in CSV file.

        :param file: dataset file name <file>
        :type file: str | pathlib.Path
        """
        super().__init__()
        with open(file, "r", encoding="utf-8") as db:
            self.data = db.readlines()
        self.header_idx_dict: Dict[str, List[int]] = {}
        for key, i in enumerate(self.data[0].replace("\n", "").split(",")):
            if i in self.header_idx_dict:
                self.header_idx_dict[i].append(key)
            else:
                self.header_idx_dict[i] = [key]
        self.mapping = lambda x: x

    def __len__(self) -> int:
        return len(self.data) - 1

    def __getitem__(self, idx: Union[int, Tensor]) -> Dict[str, Tensor]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # valid `idx` should start from 1 instead of 0
        data: List[str] = self.data[idx + 1].replace("\n", "").split(",")
        data_dict: Dict[str, List[str]] = {}
        for key, item in self.header_idx_dict.items():
            data_dict[key] = [data[i] for i in item]
        return self.mapping(data_dict)

    def map(self, mapping: Callable[[Dict[str, List[str]]], Any]) -> None:
        """
        Pass a customised mapping function to transform the data entities to tensors.

        e.g.
        ```python
        import torch
        from bayesianflow_for_chem.data import smiles2token, CSVData


        def encode(x):
            return {
                "token": smiles2token(".".join(x["smiles"])),
                "value": torch.tensor([float(i) if i != "" else torch.inf for i in x["value"]]),
            }

        dataset = CSVData(...)
        dataset.map(encode)
        ```

        :param mapping: customised mapping function
        :type mapping: callable
        :return:
        :rtype: None
        """
        self.mapping = mapping


class XYZData(Dataset):
    """
    XYZ dataset class.
    """

    def __init__(self, file: Union[str, Path], use_pbc: bool = False) -> None:
        """
        Define dataset stored in extended-XYZ file.

        :param file: dataset file name <file>
        :param use_pbc: whether to use PBC
        :type file: str | pathlib.Path
        :type use_pbc: bool
        """
        super().__init__()
        self.data = read(file, index=":")
        self.use_pbc = use_pbc

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: Union[int, Tensor]) -> Dict[str, Tensor]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        d = self.data[idx]
        z = torch.tensor(d.numbers, dtype=torch.long)
        r = torch.tensor(d.positions, dtype=torch.float32)
        data_dict = {"Z": z, "R": r}
        lattice = torch.tensor(d.cell.tolist(), dtype=torch.float32)
        if lattice.abs().sum() > 0:
            pbc = torch.tensor(d.pbc, dtype=torch.float32)
            if pbc.sum() > 0 and self.use_pbc:
                # mask the non-periodic direction(s)
                data_dict["lattice"] = lattice * pbc[:, None]
        data_dict.update(self._mapping(d))
        return data_dict

    @staticmethod
    def _mapping(atoms: Atoms) -> Dict[str, Tensor]:
        label: Dict[str, Tensor] = {}
        results = getattr(atoms.calc, "results", {})
        if "energy" in results:
            energy = torch.tensor([atoms.get_total_energy()], dtype=torch.float32)
            label["scalar"] = energy  # shape: (1,)
        if "forces" in results:
            forces = torch.tensor(atoms.get_forces(), dtype=torch.float32)
            label["vector"] = forces  # shape: (n_a, 3)
        return label


if __name__ == "__main__":
    ...
