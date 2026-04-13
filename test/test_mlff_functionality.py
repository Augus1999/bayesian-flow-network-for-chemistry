# -*- coding: utf-8 -*-
# Author: Nianze A. Tao (Omozawa Sueno)
"""
MLFF model should output a scalar value and a vector.
"""
import torch
import pytest
from bayesianflow_for_chem.data import graph_collate
from bayesianflow_for_chem.mlff import PAINN, BUILTIN_MLFF
from bayesianflow_for_chem.tool import GeometryConverter

torch.manual_seed(8964)
model = PAINN()
gcr = GeometryConverter()


@pytest.mark.parametrize(
    "size",
    [(12, 1), (17, 3, 23, 20), (25, 9, 2, 1, 4), (128, 24, 4), (3, 24, 12, 5, 17)],
)
def test_shape(size):
    batch = []
    energy, forces = [], []
    for _size in size:
        z = torch.randint(1, 120, (_size,))
        r = torch.randn((_size, 3))
        _mol = {"Z": z, "R": r}
        batch.append(_mol)
        _batch = graph_collate([_mol])
        _s, _v = model(_batch["Z"], _batch["R"], _batch["batch"])
        energy.append(_s)
        forces.append(_v)
    batch = graph_collate(batch)
    s, v = model(batch["Z"], batch["R"], batch["batch"])
    energy, forces = torch.cat(energy, 0), torch.cat(forces, 1)
    assert s.shape == (len(size), 1)
    assert v.shape == (1, sum(size), 3)
    # comparing individually computed results against batched results
    torch.testing.assert_close(energy, s)
    torch.testing.assert_close(forces, v)


@pytest.mark.parametrize("smi", ["CCO", "c1ccccc1OCCN", "CCOC1NNC(C)C1F"])
def test_relax(smi):
    _model = BUILTIN_MLFF["COLL-v1.2"]
    z, r = gcr.smiles2cartesian2(
        smi, _model["file"], 1000, 1.0, energy_unit=_model["unit"]
    )
    assert len(z) == len(r)
