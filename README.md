# ChemBFN: Bayesian Flow Network for Chemistry

[![DOI](https://zenodo.org/badge/DOI/10.1021/acs.jcim.4c01792.svg)](https://doi.org/10.1021/acs.jcim.4c01792)
[![DOI](https://zenodo.org/badge/DOI/10.11546/cicsj.43.10.svg)](https://doi.org/10.11546/cicsj.43.10)
[![DOI](https://zenodo.org/badge/DOI/10.1186/s13321-026-01248-9.svg)](https://doi.org/10.1186/s13321-026-01248-9)

This is the repository of the PyTorch implementation of ChemBFN model.

### Build State

[![PyPI](https://img.shields.io/pypi/v/bayesianflow-for-chem?color=5d9bff)](https://pypi.org/project/bayesianflow-for-chem/)
![CI](https://github.com/Augus1999/bayesian-flow-network-for-chemistry/actions/workflows/pytest.yml/badge.svg)
[![document](https://github.com/Augus1999/bayesian-flow-network-for-chemistry/actions/workflows/pages/pages-build-deployment/badge.svg)](https://augus1999.github.io/bayesian-flow-network-for-chemistry/)

## Features

ChemBFN provides the state-of-the-art functionalities of
* SMILES or SELFIES-based *de novo* molecule generation
* Protein sequence *de novo* generation
* Template optimisation (mol2mol)
* Classifier-free guidance conditional generation (single or multi-objective optimisation)
* Context-guided conditional generation (inpaint)
* Outstanding out-of-distribution chemical space sampling
* Fast sampling via ODE solver
* Pseudo-continuous learning to learn from different datasets
* Molecular property and activity prediction finetuning
* Reaction yield prediction finetuning

in an all-in-one-model style. A built-in MLFF can further convert the generated SMILES strings to the corresponding 3D conformations.

## News

* [26/06/2026] Our second paper has been accepted by [J. Cheminform](https://link.springer.com/article/10.1186/s13321-026-01248-9).
* [26/12/2025] We were invited to submit a short report about ChemBFN for [CICSJ Bulletin](https://www.jstage.jst.go.jp/article/cicsj/43/1/43_10/_article/-char/ja).
* [09/10/2025] A web app [`chembfn_webui`](https://github.com/Augus1999/ChemBFN-WebUI) for hosting ChemBFN models is available on [PyPI](https://pypi.org/project/chembfn-webui/).
* [30/01/2025] The package `bayesianflow_for_chem` is available on [PyPI](https://pypi.org/project/bayesianflow-for-chem/).
* [21/01/2025] Our first paper has been accepted by [JCIM](https://pubs.acs.org/doi/10.1021/acs.jcim.4c01792).
* [17/12/2024] The second paper of out-of-distribution generation is available on [arxiv.org](https://arxiv.org/abs/2412.11439).
* [31/07/2024] Paper is available on [arxiv.org](https://arxiv.org/abs/2407.20294).
* [21/07/2024] Paper was submitted to arXiv.

## Install

```bash
$ pip install -U bayesianflow_for_chem
```

## Usage

You can find example scripts in [📁example](./example) folder.

## Pre-trained Model

You can find pretrained models (linked to pretraining datasets) on our [🤗Hugging Face model page](https://huggingface.co/suenoomozawa/ChemBFN).

## Dataset Handling

We provide a Python class [`CSVData`](./bayesianflow_for_chem/data.py) to handle data stored in CSV or similar format containing headers to identify the entities. The following is a quickstart.

1. Download your dataset file (e.g., ESOL from [MoleculeNet](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv)) and split the file:
```python
>>> from bayesianflow_for_chem.tool import split_data

>>> split_data("delaney-processed.csv", method="scaffold")
```

2. Load the split data:
```python
>>> from bayesianflow_for_chem.data import smiles2token, collate, CSVData

>>> dataset = CSVData("delaney-processed_train.csv")
>>> dataset[0]
{'Compound ID': ['Thiophene'], 
'ESOL predicted log solubility in mols per litre': ['-2.2319999999999998'], 
'Minimum Degree': ['2'], 
'Molecular Weight': ['84.14299999999999'], 
'Number of H-Bond Donors': ['0'], 
'Number of Rings': ['1'], 
'Number of Rotatable Bonds': ['0'], 
'Polar Surface Area': ['0.0'], 
'measured log solubility in mols per litre': ['-1.33'], 
'smiles': ['c1ccsc1']}
```

3. Create a mapping function to tokenise the dataset and select values:
```python
>>> import torch

>>> def encode(x):
...   smiles = x["smiles"][0]
...   value = [float(i) for i in x["measured log solubility in mols per litre"]]
...   return {"token": smiles2token(smiles), "value": torch.tensor(value)}

>>> dataset.map(encode)
>>> dataset[0]
{'token': tensor([  1, 151,  23, 151, 151, 154, 151,  23,   2]), 
'value': tensor([-1.3300])}
```

4. Wrap the dataset in <u>torch.utils.data.DataLoader</u>:
```python
>>> dataloader = torch.utils.data.DataLoader(dataset, 32, collate_fn=collate)
```

## Cite This Work

```bibtex
@article{2025chembfn,
    title={Bayesian Flow Network Framework for Chemistry Tasks},
    author={Tao, Nianze and Abe, Minori},
    journal={Journal of Chemical Information and Modeling},
    volume={65},
    number={3},
    pages={1178-1187},
    year={2025},
    doi={10.1021/acs.jcim.4c01792},
}
```
```bibtex
@article{2025chembfn_report,
    title={Molecular Structure Design via Bayesian Flow Network},
    author={Tao, Nianze and Nagai, Touma and Abe, Minori},
    journal={CICSJ Bulletin},
    volume={43},
    number={1},
    pages={10-14},
    year={2025},
    doi={10.11546/cicsj.43.10},
}
```

Out-of-distribution generation and fast sampling:
```bibtex
@article{2026chembfn_ood,
    title={Sampling Out-of-Distribution Chemical Spaces via Bayesian Flow}, 
    author={Tao, Nianze and Abe, Minori},
    journal={Journal of Cheminformatics},
    year={2026},
    doi={10.1186/s13321-026-01248-9}, 
}
```
