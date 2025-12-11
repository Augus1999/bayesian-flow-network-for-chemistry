# -*- coding: utf-8 -*-
# Author: Nianze A. Tao (Omozawa Sueno)
"""
`bayesianflow_for_chem.tool.split_dataset` should work both for random split and scaffold split.
"""
import os
import shutil
from pathlib import Path
from bayesianflow_for_chem.tool import split_dataset


data_str = r"""smiles,c
c1ccccc1OC,1
CC(Cc1cc(O)ccc1)NC(C#N)O,0
CCCCOCCCCC,2
NCc1ccccc1,0
O,3
NCCCCCCO,0
N,1
c1cccnc1CCC,2
CCCCNCCCC,3
C1CCC(c1ccccc1-c2ccc(CCO)cc2)OCC1N,0
"""

cwd = Path(__file__).parent
data_path = cwd / "dataset/dummy_data.csv"
if not os.path.exists(data_path):
    os.makedirs(data_path.parent)
with open(data_path, "w") as f:
    f.write(data_str)


def test():
    split_dataset(data_path, [8, 1, 1], "random")
    with open(cwd / "dataset/dummy_data_train.csv", "r") as f:
        d = f.readlines()
    assert len(d) == 9
    assert d[0] == "smiles,c\n"
    with open(cwd / "dataset/dummy_data_test.csv", "r") as f:
        d = f.readlines()
    assert len(d) == 2
    assert d[0] == "smiles,c\n"
    with open(cwd / "dataset/dummy_data_val.csv", "r") as f:
        d = f.readlines()
    assert len(d) == 2
    assert d[0] == "smiles,c\n"
    split_dataset(data_path, [8, 1, 1], "scaffold")
    with open(cwd / "dataset/dummy_data_train.csv", "r") as f:
        d = f.readlines()
    assert len(d) == 9
    assert d[0] == "smiles,c\n"
    with open(cwd / "dataset/dummy_data_test.csv", "r") as f:
        d = f.readlines()
    assert len(d) == 2
    assert d[0] == "smiles,c\n"
    with open(cwd / "dataset/dummy_data_val.csv", "r") as f:
        d = f.readlines()
    assert len(d) == 2
    assert d[0] == "smiles,c\n"
    shutil.rmtree(data_path.parent)
