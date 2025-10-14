# -*- coding: utf-8 -*-
# Author: Nianze A. Tao (Omozawa Sueno)
"""
Model should be compatible with TorchScript.
"""
import torch
from bayesianflow_for_chem import ChemBFN

model = ChemBFN(512)
model_method = [
    "sample",
    "ode_sample",
    "inpaint",
    "ode_inpaint",
    "optimise",
    "ode_optimise",
]


@torch.inference_mode()
def test():
    jit_model = torch.jit.script(model).eval()
    assert isinstance(jit_model, torch.jit.ScriptModule)
    for method in model_method:
        assert hasattr(jit_model, method)
    jit_model = torch.jit.freeze(jit_model, model_method)
    for method in model_method:
        assert hasattr(jit_model, method)
