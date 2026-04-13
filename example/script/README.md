## This folder contains example scripts.

* To run the example of MOSES benchmark, you should first install `molsets` package by following the instruction [here](https://github.com/molecularsets/moses/blob/master/README.md#manually), then excute the python script as:
```bash
$ python run_moses.py --datadir={YOUR_MOSES_DATASET_FOLDER} --samplestep=100
```

* To run the example of GuacaMol benchmark, you should install `guacamol` package first, then excute the python script as:
```bash
$ python run_guacamol.py --datadir={YOUR_GUACAMOL_DATASET_FOLDER} --samplestep=100
```

* To run the example of ZINC250k benchmark, you should first download the dataset [here](https://github.com/SeulLee05/MOOD/blob/main/data/zinc250k.csv), then excute the python script as :
```bash
$ python run_zinc250k.py --datadir={YOUR_ZINC250K_DATASET_FOLDER} --train_mode={normal,sar} --target={parp1,fa7,5ht1b,braf,jak2} --samplestep=1000
```

You can switch to the SELFIES version by using flag `--version=selfies`, but the package `selfies` is required.

* To train a MLFF, see [train_mlff.py](./train_mlff.py) as an example.


## JIT version _v.s._ AOT version

Since `torch.jit.script` is deprecated, we recommand to use `torch.compile(...)` instead.
```python
import torch
from bayesianflow_for_chem import ChemBFN
from bayesianflow_for_chem.data import smiles2vec
from bayesianflow_for_chem.tool import sample, inpaint

model = ChemBFN.from_checkpoint("YOUR_MODEL.pt").eval().to("cuda")
model.compile()
# ------- generate molecules -------
smiles = sample(model, 1, 60, 100, method="ODE:0.5")  # or `method="BFN"`
# ------- inpaint (sacffold extension) -------
scaffold = r"Cc1cc(OC5)cc(C6)c1."
x = torch.tensor([1] + smiles2vec(scaffold) + [0] * (84 - len(scaffold)), dtype=torch.long)
x = x[None, ...].repeat(5, 1).to("cuda")
smiles = inpaint(model, x, 100)
```

Our model can be fully traced and captured into a graph, however, exporting a `ChemBFN` or `EnsembleChemBFN` object via `torch.export.export(...)` does not work as we have few essential values and methods that are not directly used in `forward` path.

## SAR version?

Set `model.semi_autoregressive = True` before starting the training and/or sampling.

## Enable LoRA parameters

```python
from bayesianflow_for_chem import ChemBFN

model = ChemBFN.from_checkpoint("YOUR_MODEL.pt")
model.enable_lora(r=4, ...)  # or r=8, 16, ...
```

## Quantise thy trained model

```python
>>> from bayesianflow_for_chem.tool import quantise_model_

>>> quantise_model_(model)
```

Now `model` is your dyanmically quantised model that can be directly used.

## A note for customising tokenisation and vocabulary

Three special tokens (`<pad>`, `<start>`, and `<end>`) should be encluded and they need to have indices of **0**, **1**, and **2**, respectively.

## 3D conformation relaxation w/ MLFF

We have a built-in modified PAINN model to handle this task.
```python
from bayesianflow_for_chem.tool import GeometryConverter

gcr = GeometryConverter()
z, r = gcr.smiles2cartesian2("c1ccccc1OCCN", "YOUR/MODEL/FILE.pt", 1000, 0.05, energy_unit="Hartree")
# energy_unit should match your trained model!
xyz = f"{len(z)}\n\n"
for i, j in enumerate(z):
    xyz += f"{j} {r[i][0]} {r[i][1]} {r[i][2]}\n"
with open("result.xyz", "w") as f:
    f.write(xyz)
```

You can also access to our pretrained MLFF (we, however, do not guarantee the usability), e.g.,
```python
>>> from bayesianflow_for_chem.mlff import BUILTIN_MLFF

>>> model = BUILTIN_MLFF["COLL-v1.2"]
>>> z, r = gcr.smiles2cartesian2("c1ccccc1OCCN", model["file"], 1000, 0.5, energy_unit=model["unit"])
```
