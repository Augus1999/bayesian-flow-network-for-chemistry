# -*- coding: utf-8 -*-
# Author: Nianze A. Tao (Omozawa Sueno)
"""
CLI tool should read a provided plugin python script.
"""
import os
import shutil
from pathlib import Path
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

smiles = ["CCN", "C#N", "Cc1ccccc1"]

cwd = Path(__file__).parent
plugin_path = cwd / "plugin/test_plugin.py"
if not os.path.exists(plugin_path):
    os.makedirs(plugin_path.parent)
with open(plugin_path, "w") as f:
    f.write(script_str)

p = _load_plugin(plugin_path)


def test():
    x = [{"token": smiles2token(i)} for i in smiles]
    assert p["collate_fn"](x)["token"].shape == (3, 11)
    assert p["shuffle"] == False
    assert p["num_workers"] == 0
    assert p["max_sequence_length"] == 125
    assert p["CustomData"] is None
    shutil.rmtree(plugin_path.parent)
