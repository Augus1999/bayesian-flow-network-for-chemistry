# -*- coding: utf-8 -*-
# Author: Nianze A. Tao (Omozawa Sueno)
"""
Molecular embedding vectors should not be affected by <pad> tokens.
"""
from functools import partial
import torch
from bayesianflow_for_chem import ChemBFN, MLP
from bayesianflow_for_chem.data import VOCAB_COUNT, smiles2token

torch.manual_seed(8964)

model = ChemBFN(VOCAB_COUNT)
model.eval()
mlp1 = MLP([512, 256, 3], dropout=0.7)
mlp1.eval()
mlp2 = MLP([1024, 512, 3], dropout=0.7)
mlp2.eval()

x = smiles2token("c1ccccc1O.[NH4+]CCCCCC[O-]")
x1 = x[None, ...]
x2 = torch.nn.functional.pad(x1, (0, 7, 0, 0))


def embed_fn(z, sar_flag, mask, x):
    mb0 = z[x == 2].view(z.shape[0], -1) if sar_flag else z[::, 0]
    mb1 = (z * mask[..., None]).sum(1) / (mask != 0).float().sum(1, True)
    return torch.cat([mb0, mb1], -1)


@torch.inference_mode()
def test():
    model.semi_autoregressive = False
    y1 = model.inference(x1, mlp1)
    y2 = model.inference(x2, mlp1)
    assert (y1 != y2).sum() == 0
    model.semi_autoregressive = True
    y1 = model.inference(x1, mlp1)
    y2 = model.inference(x2, mlp1)
    assert (y1 != y2).sum() == 0
    # ------- customised embedding extraction -------
    mask1 = torch.tensor([[0] + [0.7] * 9 + [0] + [0.3] * 16 + [0]])
    mask2 = torch.tensor([[0] + [0.7] * 9 + [0] + [0.3] * 16 + [0] * 8])
    model.semi_autoregressive = False
    y1 = model.inference(
        x1,
        mlp2,
        partial(embed_fn, sar_flag=model.semi_autoregressive, mask=mask1, x=x1),
    )
    y2 = model.inference(
        x2,
        mlp2,
        partial(embed_fn, sar_flag=model.semi_autoregressive, mask=mask2, x=x2),
    )
    assert (y1 != y2).sum() == 0
    model.semi_autoregressive = True
    y1 = model.inference(
        x1,
        mlp2,
        partial(embed_fn, sar_flag=model.semi_autoregressive, mask=mask1, x=x1),
    )
    y2 = model.inference(
        x2,
        mlp2,
        partial(embed_fn, sar_flag=model.semi_autoregressive, mask=mask2, x=x2),
    )
    assert (y1 != y2).sum() == 0
