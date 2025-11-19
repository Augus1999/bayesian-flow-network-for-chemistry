# -*- coding: utf-8 -*-
# Author: Nianze A. Tao (Omozawa Sueno)
"""
Model should be traceable.
"""
import torch
from bayesianflow_for_chem import ChemBFN

model = ChemBFN(246)

x = torch.softmax(torch.randn(32, 66, 246), -1)
t = torch.rand(32, 1, 1)
example_args = (x, t, None, None)
batch = torch.export.Dim("batch")
dynamic_shape = {"x": {0: batch}, "t": {0: batch}, "mask": None, "y": None}


@torch.inference_mode()
def test():
    x1 = model(*example_args)
    model_aot = torch.export.export(model, example_args, dynamic_shapes=dynamic_shape)
    model.compile()
    x2 = model.forward(*example_args)
    x3 = model_aot.module()(*example_args)
    assert (x2 != x1).float().sum() == 0
    assert (x3 != x1).float().sum() == 0
