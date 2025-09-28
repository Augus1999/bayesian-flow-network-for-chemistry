# -*- coding: utf-8 -*-
# Author: Nianze A. Tao (Omozawa Sueno)
"""
Model output should be almost identical before and after emerging LoRA parameters into base model.
"""
import torch
from bayesianflow_for_chem import ChemBFN, MLP
from bayesianflow_for_chem.tool import merge_lora_
from bayesianflow_for_chem.data import VOCAB_COUNT, smiles2token, collate

torch.manual_seed(8964)

model = ChemBFN(VOCAB_COUNT)
model.enable_lora(r=8)
model.eval()
mlp = MLP([512, 256, 3], dropout=0.7)
mlp.eval()
for module in model.modules():
    if hasattr(module, "lora_B"):
        torch.nn.init.kaiming_uniform_(module.lora_B, a=5**0.5)

x = collate(
    [{"token": smiles2token("c1ccccc1O")}, {"token": smiles2token("[NH4+]CCCCCC[O-]")}]
)["token"]


@torch.inference_mode()
def test():
    model.semi_autoregressive = False
    y1 = model.inference(x, mlp)
    model.semi_autoregressive = True
    y2 = model.inference(x, mlp)
    merge_lora_(model)
    model.semi_autoregressive = False
    y3 = model.inference(x, mlp)
    model.semi_autoregressive = True
    y4 = model.inference(x, mlp)
    assert not model.lora_enabled
    assert (y1 - y3).abs().mean() < 1e-6
    assert (y2 - y4).abs().mean() < 1e-6
