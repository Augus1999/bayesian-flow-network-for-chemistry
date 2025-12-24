# -*- coding: utf-8 -*-
# Author: Nianze A. Tao (Omozawa Sueno)
"""
CLI tool should be able to read a provided plugin python script.
"""
import os
import shutil
from pathlib import Path
import pytest
from bayesianflow_for_chem.data import smiles2token
from bayesianflow_for_chem.cli import _load_plugin


script_str = r'''
# -*- coding: utf-8 -*-
"""
Plugin script example.
"""
import random
import pandas as pd
from bayesianflow_for_chem.data import collate

num_workers = 0
shuffle = False
max_sequence_length = 125


def collate_fn(x):
    random.shuffle(x)
    return collate(x)


__all__ = ["collate_fn", "num_workers", "shuffle", "max_sequence_length"]

'''
script_str2 = r'''
# -*- coding: utf-8 -*-
"""
Plugin script example.
"""
import random
from bayesianflow_for_chem.data import collate, CSVData

num_workers = 0
shuffle = False
max_sequence_length = 125


def collate_fn(x):
    random.shuffle(x)
    return collate(x)

class CustomData(CSVData):
    def __init__(self, fn):
        with open(fn, "w") as f:
            self.data = f.readlines()


__all__ = ["collate_fn", "num_workers", "shuffle", "max_sequence_length"]

'''

smiles = ["CCN", "C#N", "Cc1ccccc1"]

cwd = Path(__file__).parent
plugin_path = cwd / "plugin/test_plugin.py"
if not os.path.exists(plugin_path.parent):
    os.makedirs(plugin_path.parent)
with open(plugin_path, "w") as f:
    f.write(script_str)
plugin_path2 = cwd / "plugin/test_plugin2.py"
with open(plugin_path2, "w") as f:
    f.write(script_str2)


def test():
    x = [{"token": smiles2token(i)} for i in smiles]
    p = _load_plugin(plugin_path)
    assert p["collate_fn"](x)["token"].shape == (3, 11)
    assert p["shuffle"] == False
    assert p["num_workers"] == 0
    assert p["max_sequence_length"] == 125
    assert p["CustomData"] is None
    with pytest.raises(ValueError):
        _load_plugin(plugin_path2)
    shutil.rmtree(plugin_path.parent)
