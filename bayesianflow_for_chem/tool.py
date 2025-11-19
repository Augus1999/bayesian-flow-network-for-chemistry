# -*- coding: utf-8 -*-
# Author: Nianze A. TAO (Omozawa SUENO)
"""
Essential tools.
"""
import csv
import random
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional, Literal
import torch
import numpy as np
from torch import cuda, Tensor, softmax
from torch.utils.data import DataLoader
from rdkit.Chem import (
    rdDetermineBonds,
    GetFormalCharge,
    MolFromXYZBlock,
    MolFromSmiles,
    MolToSmiles,
    CanonSmiles,
    AllChem,
    AddHs,
    Mol,
)
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from .data import VOCAB_KEYS
from .model import ChemBFN, MLP, EnsembleChemBFN


def _find_device() -> torch.device:
    if cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _parse_and_assert_param(
    model: Union[ChemBFN, EnsembleChemBFN],
    y: Optional[Union[Tensor, Dict[str, Tensor], List[Tensor]]],
    method: str,
) -> Optional[float]:
    assert method.split(":")[0].lower() in ("ode", "bfn")
    if isinstance(model, EnsembleChemBFN):
        assert y is not None, "conditioning is required while using an ensemble model."
        assert isinstance(y, list) or isinstance(y, dict)
    else:
        assert isinstance(y, Tensor) or (y is None)
    if "ode" in method.lower():
        tp = float(method.split(":")[-1])
        assert tp > 0, "Sampling temperature should be higher than 0."
        return tp
    return None


def _map_model_to_device(
    model: Union[ChemBFN, EnsembleChemBFN, MLP, torch.fx.GraphModule],
    device: Union[str, torch.device],
):
    if isinstance(model, torch.fx.GraphModule):
        return model.to(device)
    else:
        return model.to(device).eval()


def _map_value_to_device(
    y: Optional[Union[Tensor, Dict[str, Tensor], List[Tensor]]],
    device: Union[str, torch.device],
) -> Optional[Union[Tensor, Dict[str, Tensor], List[Tensor]]]:
    if y is not None:
        if isinstance(y, Tensor):
            y = y.to(device)
        elif isinstance(y, list):
            y = [i.to(device) for i in y]
        elif isinstance(y, dict):
            y = {k: v.to(device) for k, v in y.items()}
        else:
            raise NotImplementedError
    return y


def _build_token_mask(
    allowed_tokens: Union[str, List[str]],
    vocab_keys: List[str],
    device: Union[str, torch.tensor],
) -> Optional[Tensor]:
    if isinstance(allowed_tokens, list):
        token_mask = [0 if i in allowed_tokens else 1 for i in vocab_keys]
        token_mask = torch.tensor([[token_mask]], dtype=torch.bool).to(device)
    else:
        token_mask = None
    return token_mask


def _token_to_seq(
    tokens: Tensor, entropy: Tensor, vocab_keys: List[str], separator: str, sort: bool
) -> List[str]:
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


def _inference(
    model: Union[ChemBFN, torch.fx.GraphModule],
    mlp: Union[MLP, torch.fx.GraphModule],
    x: Tensor,
) -> Tensor:
    if isinstance(model, torch.fx.GraphModule):
        import os

        graphmodule_sar_flag = os.environ.get("GRAPHMODULE_SAR_FLAG", "0") != "0"
        t = torch.ones((x.shape[0], 1, 1), device=x.device)
        mask = (x != 0).float()[..., None]
        theta = torch.nn.functional.one_hot(x, model.embedding.weight.shape[-1])
        z = model.forward(2 * theta.float() - 1, t, mask, None)
        mb = z[x == 2].view(z.shape[0], -1) if graphmodule_sar_flag else z[::, 0]
        return mlp.forward(mb)
    return model.inference(x, mlp)


@torch.no_grad()
def test(
    model: ChemBFN,
    mlp: MLP,
    data: DataLoader,
    mode: Literal["regression", "classification"] = "regression",
    device: Union[str, torch.device, None] = None,
) -> Dict[str, float]:
    """
    Test the trained network. \n
    Note: If your model is a `~torch.fx.GraphModule` instance exported via `torch.export.export(...)`,
    set environment variable GRAPHMODULE_SAR_FLAG="1" to enable semi-autoregressive behaviour.

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
    model = _map_model_to_device(model, device)
    mlp = _map_model_to_device(mlp, device)
    predict_y, label_y = [], []
    for d in data:
        x, y = d["token"].to(device), d["value"]
        label_y.append(y)
        if mode == "regression":
            # y_hat = model.inference(x, mlp)
            y_hat = _inference(model, mlp, x)
        if mode == "classification":
            n_b, n_y = y.shape
            # y_hat = softmax(model.inference(x, mlp).reshape(n_b * n_y, -1), -1)
            y_hat = softmax(_inference(model, mlp, x).reshape(n_b * n_y, -1), -1)
            y_hat = y_hat.reshape(n_b, -1)
        predict_y.append(y_hat.detach().to("cpu"))
    predict_y, label_y = torch.cat(predict_y, 0), torch.cat(label_y, 0).split(1, -1)
    if mode == "regression":
        from sklearn.metrics import (
            r2_score,
            mean_absolute_error,
            root_mean_squared_error,
        )

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
        from sklearn.metrics import roc_auc_score, auc, precision_recall_curve

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
    file: Union[str, Path],
    split_ratio: List[int] = [8, 1, 1],
    method: Literal["random", "scaffold"] = "random",
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
                    f"We found {len(smiles_idx)} SMILES strings in a row!"
                    " Only the first SMILES will be used to compute the molecular scaffold.",
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
    if val_set:
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
    Sampling molecules.

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
    tp = _parse_and_assert_param(model, y, method)
    device = _find_device() if device is None else device
    model = _map_model_to_device(model, device)
    y = _map_value_to_device(y, device)
    token_mask = _build_token_mask(allowed_tokens, vocab_keys, device)
    if tp:
        tokens, entropy = model.ode_sample(
            batch_size, sequence_size, y, sample_step, guidance_strength, token_mask, tp
        )
    else:
        tokens, entropy = model.sample(
            batch_size, sequence_size, y, sample_step, guidance_strength, token_mask
        )
    return _token_to_seq(tokens, entropy, vocab_keys, seperator, sort)


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
    tp = _parse_and_assert_param(model, y, method)
    device = _find_device() if device is None else device
    model = _map_model_to_device(model, device)
    x = _map_value_to_device(x, device)
    y = _map_value_to_device(y, device)
    token_mask = _build_token_mask(allowed_tokens, vocab_keys, device)
    if tp:
        tokens, entropy = model.ode_inpaint(
            x, y, sample_step, guidance_strength, token_mask, tp
        )
    else:
        tokens, entropy = model.inpaint(
            x, y, sample_step, guidance_strength, token_mask
        )
    return _token_to_seq(tokens, entropy, vocab_keys, separator, sort)


@torch.no_grad()
def optimise(
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
    Optimising template molecules (mol2mol).

    :param model: trained ChemBFN model
    :param x: categorical indices of template;  shape: (n_b, n_t)
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
    tp = _parse_and_assert_param(model, y, method)
    device = _find_device() if device is None else device
    model = _map_model_to_device(model, device)
    x = _map_value_to_device(x, device)
    y = _map_value_to_device(y, device)
    token_mask = _build_token_mask(allowed_tokens, vocab_keys, device)
    if tp:
        tokens, entropy = model.ode_optimise(
            x, y, sample_step, guidance_strength, token_mask, tp
        )
    else:
        tokens, entropy = model.optimise(
            x, y, sample_step, guidance_strength, token_mask
        )
    return _token_to_seq(tokens, entropy, vocab_keys, separator, sort)


def quantise_model_(model: ChemBFN) -> None:
    """
    In-place dynamic quantisation of the trained model to `int8` data type. \n
    Due to some limitations of `torchao` module, not all layers will be quantised.

    :param model: trained ChemBFN model
    :type model: bayesianflow_for_chem.model.ChemBFN
    :return:
    :rtype: None
    """
    from torchao.quantization.quant_api import (
        quantize_,
        Int8DynamicActivationInt8WeightConfig,
    )

    quantize_(model, Int8DynamicActivationInt8WeightConfig())


def adjust_lora_(model: ChemBFN, lora_scale: float = 1.0) -> None:
    """
    In-place adjust LoRA scaling parameter.

    :param model: trained ChemBFN model
    :param lora_scale: LoRA scaling multiplier; setting a value smaller than 1 to decrease LoRA control
    :type model: bayesianflow_for_chem.model.ChemBFN
    :type lora_scale: float
    :return:
    :rtype: None
    """
    if not model.lora_enabled:
        return
    for module in model.modules():
        if hasattr(module, "lora_A"):
            module.scaling = module.scaling * lora_scale


def merge_lora_(model: ChemBFN) -> None:
    """
    In-place merge LoRA parameters into base-model. \n
    This function does not work on a quantised model.

    :param model: trained ChemBFN model
    :type model: bayesianflow_for_chem.model.ChemBFN
    :return:
    :rtype: None
    """
    if not model.lora_enabled:
        return
    for module in model.modules():
        if hasattr(module, "lora_A"):
            try:
                module.weight.data += (module.lora_B @ module.lora_A) * module.scaling
                module.lora_enabled = False
                module.lora_A = None
                module.lora_B = None
                module.scaling = None
                module.lora_dropout = None
            except NotImplementedError:
                warnings.warn("Cannot merge LoRA parameters into quantised model.")
                return
    model.lora_enabled = False


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
    def smiles2cartesian(
        smiles: str,
        num_conformers: int = 250,
        rdkit_ff_type: Literal["MMFF", "UFF"] = "MMFF",
        refine_with_crest: bool = False,
        spin: float = 0.0,
    ) -> Tuple[List[str], np.ndarray]:
        """
        Guess the 3D geometry from SMILES string via conformer search.

        :param smiles: a valid SMILES string
        :param num_conformers: number of initial conformers
        :param rdkit_ff_type: force field type chosen in `'MMFF'` and `'UFF'`
        :param refine_with_crest: find the best conformer via CREST
        :param spin: total spin; only required when `refine_with_crest=True`
        :type smiles: str
        :type num_conformers: int
        :type rdkit_ff_type: str
        :type refine_with_crest: bool
        :type spin: float
        :return: atomic symbols \n
                 cartesian coordinates;  shape: (n_a, 3)
        :rtype: tuple
        """
        assert rdkit_ff_type.lower() in ("mmff", "uff")
        if refine_with_crest:
            from tempfile import TemporaryDirectory
            from subprocess import run

            # We need both CREST and xTB installed.
            if run("crest --version", shell=True).returncode != 0:
                raise RuntimeError(
                    "`CREST` is not found! Make sure it is installed and added into the PATH."
                )
            if run("xtb --version", shell=True).returncode != 0:
                raise RuntimeError(
                    "`xTB` is not found! Make sure it is installed and added into the PATH."
                )
        mol = MolFromSmiles(smiles)
        mol = AddHs(mol)
        AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers, params=AllChem.ETKDG())
        symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        energies = []
        for conf_id in range(num_conformers):
            if rdkit_ff_type.lower() == "mmff":
                ff = AllChem.MMFFGetMoleculeForceField(
                    mol, AllChem.MMFFGetMoleculeProperties(mol), confId=conf_id
                )
            else:  # UFF
                ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
            energy = ff.CalcEnergy()
            energies.append((conf_id, energy))
        lowest_energy_conf = min(energies, key=lambda x: x[1])
        coordinates = mol.GetConformer(id=lowest_energy_conf[0]).GetPositions()
        if refine_with_crest:
            xyz = f"{len(symbols)}\n\n" + "\n".join(
                f"{s} {coordinates[i][0]:.10f} {coordinates[i][1]:.10f} {coordinates[i][2]:.10f}"
                for i, s in enumerate(symbols)
            )
            chrg = GetFormalCharge(mol)
            uhf = int(spin * 2)
            with TemporaryDirectory(dir=Path.cwd()) as temp_dir:
                with open(Path(temp_dir) / "mol.xyz", "w", encoding="utf-8") as f:
                    f.write(xyz)
                s = run(
                    f"crest mol.xyz -gfn2 -quick -prop ohess{f' --chrg {chrg}' if chrg != 0 else ''}{f' --uhf {uhf}' if uhf != 0 else ''}",
                    shell=True,
                    cwd=temp_dir,
                )
                if s.returncode == 0:
                    with open(Path(temp_dir) / "crest_property.xyz", "r") as f:
                        xyz = f.readlines()
                    xyz_data = []
                    for i in xyz[2:]:
                        if i == xyz[0]:
                            break
                        xyz_data.append(i.strip().split())
                    xyz_data = np.array(xyz_data)
                    symbols, coordinates = np.split(xyz_data, [1], axis=-1)
                    symbols = symbols.flatten().tolist()
                    coordinates = coordinates.astype(np.float64)
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
