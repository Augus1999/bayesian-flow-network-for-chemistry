## This folder contains example scripts.

* To run the example of MOSES dataset, you should first install `molsets` package by following the instruction [here](https://github.com/molecularsets/moses/blob/master/README.md#manually), then excute the python script as:
```bash
$ python run_moses.py --datadir={YOUR_MOSES_DATASET_FOLDER} --samplestep=100
```

* To run the example of GuacaMol dataset, you should install `guacamol` package first, then excute the python script as:
```bash
$ python run_guacamol.py --datadir={YOUR_GUACAMOL_DATASET_FOLDER} --samplestep=100
```

You can switch to the SELFIES version by using flag `--version=selfies`, but the package `selfies` is required.

* To finetune a model, you need to prepare the dataset in the same format as below:
```csv
smiles,value
CC(=O)C3(C)CCC4C2C=C(C)C1=CC(=O)CCC1(C)C2CCC34C,-5.27
CC(=O)OC3(CCC4C2C=C(C)C1=CC(=O)CCC1(C)C2CCC34C)C(C)=O,-5.35
CN(C(=O)COc1nc2ccccc2s1)c3ccccc3,-4.873
O=C(Nc1ccccc1)Nc2ccccc2,-3.15
Clc1ccc(CN(C2CCCC2)C(=O)Nc3ccccc3)cc1,-5.915
CC2(C)C1CCC(C)(C1)C2=O,-1.85
CC1(C)C2CCC1(C)C(=O)C2,-1.96

```

## JIT version?

Our implementation supports TorchScript.
```python
import torch
from bayesianflow_for_chem import ChemBFN
from bayesianflow_for_chem.data import smiles2vec
from bayesianflow_for_chem.tool import sample, inpaint

model = ChemBFN.from_checkpoint("YOUR_MODEL.pt").eval().to("cuda")
model = torch.jit.freeze(torch.jit.script(model), ["sample", "inpaint"])
# or model.compile()
# ------- generate molecules -------
smiles = sample(model, 1, 60, 100)
# ------- inpaint (sacffold extension) -------
scaffold = r"Cc1cc(OC5)cc(C6)c1."
x = torch.tensor([1] + smiles2vec(scaffold) + [0] * (84 - len(scaffold)), dtype=torch.long)
x = x[None, ...].repeat(5, 1).to("cuda")
smiles = inpaint(model, x, 100)
```