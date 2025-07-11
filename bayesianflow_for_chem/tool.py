# -*- coding: utf-8 -*-
# Author: Nianze A. TAO (Omozawa SUENO)
"""
Essential tools.
"""
import csv
import random
import warnings
from copy import deepcopy
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional
import torch
import numpy as np
import torch.nn as nn
from torch import cuda, Tensor, softmax
from torch.ao import quantization
from torch.utils.data import DataLoader
from typing_extensions import Self
from rdkit.Chem.rdchem import Mol, Bond
from rdkit.Chem import (
    rdDetermineBonds,
    MolFromXYZBlock,
    MolFromSmiles,
    MolToSmiles,
    CanonSmiles,
    AllChem,
    AddHs,
)
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles  # type: ignore
from sklearn.metrics import (
    roc_auc_score,
    auc,
    precision_recall_curve,
    r2_score,
    mean_absolute_error,
    root_mean_squared_error,
)
from .data import VOCAB_KEYS
from .model import ChemBFN, MLP, Linear, EnsembleChemBFN


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
    :type model: bayesianflow_for_chem.model.ChemBFN
    :type mlp: bayesianflow_for_chem.model.MLP
    :type data: torch.utils.data.DataLoader
    :type mode: str
    :type device: str | torch.device | None
    :return: MAE & RMSE & R^2 / ROC-AUC & PRC-AUC
    :rtype: dict
    """
    if device is None:
        device = _find_device()
    model.to(device).eval()
    mlp.to(device).eval()
    predict_y, label_y = [], []
    for d in data:
        x, y = d["token"].to(device), d["value"]
        label_y.append(y)
        if mode == "regression":
            y_hat = model.inference(x, mlp)
        if mode == "classification":
            n_b, n_y = y.shape
            y_hat = softmax(model.inference(x, mlp).reshape(n_b * n_y, -1), -1)
            y_hat = y_hat.reshape(n_b, -1)
        predict_y.append(y_hat.detach().to("cpu"))
    predict_y, label_y = torch.cat(predict_y, 0), torch.cat(label_y, 0).split(1, -1)
    if mode == "regression":
        predict_y = [
            predict[label_y[i] != torch.inf]
            for (i, predict) in enumerate(predict_y.split(1, -1))
        ]
        label_y = [label[label != torch.inf] for label in label_y]
        y_zipped = list(zip(label_y, predict_y))
        mae = [mean_absolute_error(label, predict) for (label, predict) in y_zipped]
        rmse = [
            root_mean_squared_error(label, predict) for (label, predict) in y_zipped
        ]
        r2 = [r2_score(label, predict) for (label, predict) in y_zipped]
        return {"MAE": mae, "RMSE": rmse, "R^2": r2}
    if mode == "classification":
        n_c = len(label_y)
        predict_y = predict_y.chunk(n_c, -1)
        y_zipped = list(zip(label_y, predict_y))
        roc_auc = [
            roc_auc_score(
                label.flatten(),
                predict[:, 1] if predict.shape[-1] == 2 else predict,
                multi_class="raise" if predict.shape[-1] == 2 else "ovo",
                labels=None if predict.shape[-1] == 2 else range(predict.shape[-1]),
            )
            for (label, predict) in y_zipped
        ]
        try:
            prc = [
                precision_recall_curve(label.flatten(), predict[:, 1])[:2]
                for (label, predict) in y_zipped
            ]
            prc_auc = [auc(recall, precision) for (precision, recall) in prc]
        except ValueError:
            prc_auc = []
        return {"ROC-AUC": roc_auc, "PRC-AUC": prc_auc}


def split_dataset(
    file: Union[str, Path], split_ratio: List[int] = [8, 1, 1], method: str = "random"
) -> None:
    """
    Split a dataset.

    :param file: dataset file <file>
    :param split_ratio: traing-testing-validation ratio
    :param method: chosen from `'random'` and `'scaffold'`
    :type file: str | pathlib.Path
    :type split_ratio: list
    :type method: str
    :return:
    :rtype: None
    """
    if isinstance(file, Path):
        file = file.__str__()
    assert file.endswith(".csv")
    assert len(split_ratio) == 3
    assert method in ("random", "scaffold")
    with open(file, "r") as f:
        data = list(csv.reader(f))
    header = data[0]
    raw_data = data[1:]
    smiles_idx = []  # only first index will be used
    for key, h in enumerate(header):
        if "smiles" in h.lower():
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
            if len(smiles_idx) > 1:
                warnings.warn(
                    "\033[32;1m"
                    f"We found {len(smiles_idx)} SMILES strings in a row!"
                    " Only the first SMILES will be used to compute the molecular scaffold."
                    "\033[0m",
                    stacklevel=2,
                )
            try:
                scaffold = MurckoScaffoldSmiles(d[smiles_idx[0]])
                if scaffold in scaffolds:
                    scaffolds[scaffold].append(key)
                else:
                    scaffolds[scaffold] = [key]
            except ValueError:  # do nothing when SMILES is not valid
                ...
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


@torch.no_grad()
def sample(
    model: Union[ChemBFN, EnsembleChemBFN],
    batch_size: int,
    sequence_size: int,
    sample_step: int = 100,
    y: Optional[Union[Tensor, Dict[str, Tensor], List[Tensor]]] = None,
    guidance_strength: float = 4.0,
    device: Union[str, torch.device, None] = None,
    vocab_keys: List[str] = VOCAB_KEYS,
    seperator: str = "",
    method: str = "BFN",
    allowed_tokens: Union[str, List[str]] = "all",
    sort: bool = False,
) -> List[str]:
    """
    Sampling.

    :param model: trained ChemBFN model
    :param batch_size: batch size
    :param sequence_size: max sequence length
    :param sample_step: number of sampling steps
    :param y: conditioning vector;             shape: (n_b, 1, n_f) or (n_b, n_f) \n
              or a list/`dict` of conditions;  shape: (n_b, n_c) * n_h

    :param guidance_strength: strength of conditional generation. It is not used if y is null.
    :param device: hardware accelerator
    :param vocab_keys: a list of (ordered) vocabulary
    :param separator: token separator; default is `""`
    :param method: sampling method chosen from `"ODE:x"` or `"BFN"` where `x` is the value of sampling temperature; default is `"BFN"`
    :param allowed_tokens: a list of allowed tokens; default is `"all"`
    :param sort: whether to sort the samples according to entropy values; default is `False`
    :type model: bayesianflow_for_chem.model.ChemBFN | bayesianflow_for_chem.model.EnsembleChemBFN
    :type batch_size: int
    :type sequence_size: int
    :type sample_step: int
    :type y: torch.Tensor | list | dict | None
    :type guidance_strength: float
    :type device: str | torch.device | None
    :type vocab_keys: list
    :type separator: str
    :type method: str
    :type allowed_tokens: str | list
    :type sort: bool
    :return: a list of generated molecular strings
    :rtype: list
    """
    assert method.split(":")[0].lower() in ("ode", "bfn")
    if isinstance(model, EnsembleChemBFN):
        assert y is not None, "conditioning is required while using an ensemble model."
        assert isinstance(y, list) or isinstance(y, dict)
    else:
        assert isinstance(y, Tensor) or y is None
    if device is None:
        device = _find_device()
    model.to(device).eval()
    if y is not None:
        if isinstance(y, Tensor):
            y = y.to(device)
        elif isinstance(y, list):
            y = [i.to(device) for i in y]
        elif isinstance(y, dict):
            y = {k: v.to(device) for k, v in y.items()}
        else:
            raise NotImplementedError
    if isinstance(allowed_tokens, list):
        token_mask = [0 if i in allowed_tokens else 1 for i in vocab_keys]
        token_mask = torch.tensor([[token_mask]], dtype=torch.bool).to(device)
    else:
        token_mask = None
    if "ode" in method.lower():
        tp = float(method.split(":")[-1])
        assert tp > 0, "Sampling temperature should be higher than 0."
        tokens, entropy = model.ode_sample(
            batch_size, sequence_size, y, sample_step, guidance_strength, token_mask, tp
        )
    else:
        tokens, entropy = model.sample(
            batch_size, sequence_size, y, sample_step, guidance_strength, token_mask
        )
    if sort:
        sorted_idx = entropy.argsort(stable=True)
        tokens = tokens[sorted_idx]
    return [
        seperator.join([vocab_keys[i] for i in j])
        .split("<start>" + seperator)[-1]
        .split(seperator + "<end>")[0]
        .replace("<pad>", "")
        for j in tokens
    ]


@torch.no_grad()
def inpaint(
    model: Union[ChemBFN, EnsembleChemBFN],
    x: Tensor,
    sample_step: int = 100,
    y: Optional[Union[Tensor, Dict[str, Tensor], List[Tensor]]] = None,
    guidance_strength: float = 4.0,
    device: Union[str, torch.device, None] = None,
    vocab_keys: List[str] = VOCAB_KEYS,
    separator: str = "",
    method: str = "BFN",
    allowed_tokens: Union[str, List[str]] = "all",
    sort: bool = False,
) -> List[str]:
    """
    Inpaint (context guided) sampling.

    :param model: trained ChemBFN model
    :param x: categorical indices of scaffold;  shape: (n_b, n_t)
    :param sample_step: number of sampling steps
    :param y: conditioning vector;              shape: (n_b, 1, n_f) or (n_b, n_f) \n
              or a list/`dict` of conditions;   shape: (n_b, n_c) * n_h

    :param guidance_strength: strength of conditional generation. It is not used if y is null.
    :param device: hardware accelerator
    :param vocab_keys: a list of (ordered) vocabulary
    :param separator: token separator; default is `""`
    :param method: sampling method chosen from `"ODE:x"` or `"BFN"` where `x` is the value of sampling temperature; default is `"BFN"`
    :param allowed_tokens: a list of allowed tokens; default is `"all"`
    :param sort: whether to sort the samples according to entropy values; default is `False`
    :type model: bayesianflow_for_chem.model.ChemBFN | bayesianflow_for_chem.model.EnsembleChemBFN
    :type x: torch.Tensor
    :type sample_step: int
    :type y: torch.Tensor | list | dict | None
    :type guidance_strength: float
    :type device: str | torch.device | None
    :type vocab_keys: list
    :type separator: str
    :type method: str
    :type allowed_tokens: str | list
    :type sort: bool
    :return: a list of generated molecular strings
    :rtype: list
    """
    assert method.split(":")[0].lower() in ("ode", "bfn")
    if isinstance(model, EnsembleChemBFN):
        assert y is not None, "conditioning is required while using an ensemble model."
        assert isinstance(y, list) or isinstance(y, dict)
    else:
        assert isinstance(y, Tensor) or y is None
    if device is None:
        device = _find_device()
    model.to(device).eval()
    x = x.to(device)
    if y is not None:
        if isinstance(y, Tensor):
            y = y.to(device)
        elif isinstance(y, list):
            y = [i.to(device) for i in y]
        elif isinstance(y, dict):
            y = {k: v.to(device) for k, v in y.items()}
        else:
            raise NotImplementedError
    if isinstance(allowed_tokens, list):
        token_mask = [0 if i in allowed_tokens else 1 for i in vocab_keys]
        token_mask = torch.tensor([[token_mask]], dtype=torch.bool).to(device)
    else:
        token_mask = None
    if "ode" in method.lower():
        tp = float(method.split(":")[-1])
        assert tp > 0, "Sampling temperature should be higher than 0."
        tokens, entropy = model.ode_inpaint(
            x, y, sample_step, guidance_strength, token_mask, tp
        )
    else:
        tokens, entropy = model.inpaint(
            x, y, sample_step, guidance_strength, token_mask
        )
    if sort:
        sorted_idx = entropy.argsort(stable=True)
        tokens = tokens[sorted_idx]
    return [
        separator.join([vocab_keys[i] for i in j])
        .split("<start>" + separator)[-1]
        .split(separator + "<end>")[0]
        .replace("<pad>", "")
        for j in tokens
    ]


def quantise_model(model: ChemBFN) -> nn.Module:
    """
    Dynamic quantisation of the trained model to `torch.qint8` data type.

    :param model: trained ChemBFN model
    :type model: bayesianflow_for_chem.model.ChemBFN
    :return: quantised model
    :rtype: torch.nn.Module
    """
    from torch.ao.nn.quantized import dynamic
    from torch.ao.nn.quantized.modules.utils import _quantize_weight
    from torch.ao.quantization.qconfig import default_dynamic_qconfig

    class QuantisedLinear(dynamic.Linear):
        # Modified from https://github.com/pytorch/pytorch/blob/main/torch/ao/nn/quantized/dynamic/modules/linear.py
        # We made it compatible with our LoRA linear layer.
        # LoRA parameters will not be quantised.
        def __init__(
            self,
            in_features: int,
            out_features: int,
            bias_: bool = True,
            dtype: torch.dtype = torch.qint8,
        ) -> None:
            super().__init__(in_features, out_features, bias_, dtype=dtype)
            self.version = self._version
            self.lora_enabled: bool = False
            self.lora_A: Optional[nn.Parameter] = None
            self.lora_B: Optional[nn.Parameter] = None
            self.scaling: Optional[float] = None
            self.lora_dropout: Optional[float] = None

        def _get_name(self) -> str:
            return "DynamicQuantizedLoRALinear"

        def enable_lora(
            self, r: int = 8, lora_alpha: int = 1, lora_dropout: float = 0.0
        ) -> None:
            assert r > 0, "Rank should be larger than 0."
            device = self._weight_bias()[0].device
            self.lora_A = nn.Parameter(
                torch.zeros((r, self.in_features), device=device)
            )
            self.lora_B = nn.Parameter(
                torch.zeros((self.out_features, r), device=device)
            )
            self.scaling = lora_alpha / r
            self.lora_dropout = lora_dropout
            self.lora_enabled = True
            nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
            nn.init.zeros_(self.lora_B)
            self._packed_params.requires_grad_(False)

        def forward(self, x: Tensor) -> Tensor:
            if self._packed_params.dtype == torch.qint8:
                if self.version is None or self.version < 4:
                    Y = torch.ops.quantized.linear_dynamic(
                        x, self._packed_params._packed_params
                    )
                else:
                    Y = torch.ops.quantized.linear_dynamic(
                        x, self._packed_params._packed_params, reduce_range=True
                    )
            elif self._packed_params.dtype == torch.float16:
                Y = torch.ops.quantized.linear_dynamic_fp16(
                    x, self._packed_params._packed_params
                )
            else:
                raise RuntimeError("Unsupported dtype on dynamic quantized linear!")
            result = Y.to(x.dtype)
            if self.lora_enabled and isinstance(self.lora_dropout, float):
                result += (
                    nn.functional.dropout(x, self.lora_dropout, self.training)
                    @ self.lora_A.transpose(0, 1)
                    @ self.lora_B.transpose(0, 1)
                ) * self.scaling
            return result

        @classmethod
        def from_float(
            cls, mod: Linear, use_precomputed_fake_quant: bool = False
        ) -> Self:
            assert hasattr(
                mod, "qconfig"
            ), "Input float module must have qconfig defined"
            if use_precomputed_fake_quant:
                warnings.warn("Fake quantize operator is not implemented.")
            if mod.qconfig is not None and mod.qconfig.weight is not None:
                weight_observer = mod.qconfig.weight()
            else:
                weight_observer = default_dynamic_qconfig.weight()
            dtype = weight_observer.dtype
            assert dtype in [torch.qint8, torch.float16], (
                "The only supported dtypes for "
                f"dynamic quantized linear are qint8 and float16 got: {dtype}"
            )
            weight_observer(mod.weight)
            if dtype == torch.qint8:
                qweight = _quantize_weight(mod.weight.float(), weight_observer)
            elif dtype == torch.float16:
                qweight = mod.weight.float()
            else:
                raise RuntimeError(
                    "Unsupported dtype specified for dynamic quantized Linear!"
                )
            qlinear = cls(mod.in_features, mod.out_features, dtype=dtype)
            qlinear.set_weight_bias(qweight, mod.bias)
            if mod.lora_enabled:
                qlinear.lora_enabled = True
                qlinear.lora_A = nn.Parameter(mod.lora_A.clone().detach_())
                qlinear.lora_B = nn.Parameter(mod.lora_B.clone().detach_())
                qlinear.scaling = deepcopy(mod.scaling)
                qlinear.lora_dropout = deepcopy(mod.lora_dropout)
            return qlinear

        @classmethod
        def from_reference(cls, ref_qlinear: Self) -> Self:
            qlinear = cls(
                ref_qlinear.in_features,
                ref_qlinear.out_features,
                dtype=ref_qlinear.weight_dtype,
            )
            qweight = ref_qlinear.get_quantized_weight()
            bias = ref_qlinear.bias
            qlinear.set_weight_bias(qweight, bias)
            if ref_qlinear.lora_enabled:
                qlinear.lora_enabled = True
                qlinear.lora_A = nn.Parameter(ref_qlinear.lora_A.clone().detach_())
                qlinear.lora_B = nn.Parameter(ref_qlinear.lora_B.clone().detach_())
                qlinear.scaling = deepcopy(ref_qlinear.scaling)
                qlinear.lora_dropout = deepcopy(ref_qlinear.lora_dropout)
            return qlinear

    mapping = deepcopy(quantization.DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS)
    mapping[Linear] = QuantisedLinear
    quantised_model = quantization.quantize_dynamic(
        model, {nn.Linear, Linear}, torch.qint8, mapping
    )
    return quantised_model


class GeometryConverter:
    """
    Converting between different 2D/3D molecular representations.
    """

    @staticmethod
    def _xyz2mol(symbols: List[str], coordinates: np.ndarray) -> Mol:
        xyz_block = [str(len(symbols)), ""]
        r = coordinates
        for i, atom in enumerate(symbols):
            xyz_block.append(f"{atom} {r[i][0]:.10f} {r[i][1]:.10f} {r[i][2]:.10f}")
        return MolFromXYZBlock("\n".join(xyz_block))

    @staticmethod
    def _bond_pair_idx(bonds: Bond) -> List[List[int]]:
        return [[i.GetBeginAtomIdx(), i.GetEndAtomIdx()] for i in bonds]

    @staticmethod
    def smiles2cartesian(
        smiles: str, num_conformers: int = 50, random_seed: int = 42
    ) -> Tuple[List[str], np.ndarray]:
        """
        Guess the 3D geometry from SMILES string via MMFF conformer search.

        :param smiles: a valid SMILES string
        :param num_conformers: number of initial conformers
        :param random_seed: random seed used to generate conformers
        :type smiles: str
        :type num_conformers: int
        :type random_seed: int
        :return: atomic symbols \n
                 cartesian coordinates;  shape: (n_a, 3)
        :rtype: tuple
        """
        mol = MolFromSmiles(smiles)
        mol = AddHs(mol)
        AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers, randomSeed=random_seed)
        symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        energies = []
        for conf_id in range(num_conformers):
            ff = AllChem.MMFFGetMoleculeForceField(
                mol, AllChem.MMFFGetMoleculeProperties(mol), confId=conf_id
            )
            energy = ff.CalcEnergy()
            energies.append((conf_id, energy))
        lowest_energy_conf = min(energies, key=lambda x: x[1])
        coordinates = mol.GetConformer(id=lowest_energy_conf[0]).GetPositions()
        return symbols, coordinates

    def cartesian2smiles(
        self,
        symbols: List[str],
        coordinates: np.ndarray,
        charge: int = 0,
        canonical: bool = True,
    ) -> str:
        """
        Transform (guess out) molecular geometry to SMILES string.

        :param symbols: a list of atomic symbols
        :param coordinates: Cartesian coordinates;  shape: (n_a, 3)
        :param charge: net charge
        :param canonical: whether to canonicalise the SMILES
        :type symbols: list
        :type coordinates: numpy.ndarray
        :type charge: int
        :type canonical: bool
        :return: SMILES string
        :rtype: str
        """
        mol = self._xyz2mol(symbols, coordinates)
        rdDetermineBonds.DetermineBonds(mol, charge=charge)
        smiles = MolToSmiles(mol)
        if canonical:
            smiles = CanonSmiles(smiles)
        return smiles
