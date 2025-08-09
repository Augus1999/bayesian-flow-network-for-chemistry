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


## JIT version?

Our implementation supports TorchScript.
```python
import torch
from bayesianflow_for_chem import ChemBFN
from bayesianflow_for_chem.data import smiles2vec
from bayesianflow_for_chem.tool import sample, inpaint

model = ChemBFN.from_checkpoint("YOUR_MODEL.pt").eval().to("cuda")
model = torch.jit.freeze(torch.jit.script(model), ["sample", "inpaint", "ode_sample", "ode_inpaint"])
# or model.compile()
# ------- generate molecules -------
smiles = sample(model, 1, 60, 100, method="ODE:0.5")  # or `method="BFN"`
# ------- inpaint (sacffold extension) -------
scaffold = r"Cc1cc(OC5)cc(C6)c1."
x = torch.tensor([1] + smiles2vec(scaffold) + [0] * (84 - len(scaffold)), dtype=torch.long)
x = x[None, ...].repeat(5, 1).to("cuda")
smiles = inpaint(model, x, 100)
```

## SAR version?

Set `model.semi_autoregressive = True` before starting the training and/or sampling.

## Enable LoRA parameters

```python
from bayesianflow_for_chem import ChemBFN

model = ChemBFN.from_checkpoint("YOUR_MODEL.pt")
model.enable_lora(r=4, ...)
```

## Quantise thy trained model

```python
>>> from bayesianflow_for_chem.tool import quantise_model_

>>> quantise_model_(model)
```

Now `model` is your dyanmically quantised model that can be directly used.
