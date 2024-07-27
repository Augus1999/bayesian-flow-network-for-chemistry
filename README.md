# ChemBFN: Bayesian Flow Network for Chemistry

**Abstract**: In this work, we introduce ChemBFN, a language model that handles chemistry tasks based on
Bayesian flow networks working on discrete data. A new accuracy schedule is proposed to improve the sampling quality by significantly reducing the reconstruction loss. We show evidence that our method is appropriate for generating molecules with satisfied diversity even when a smaller number of sampling steps is used. A classifier-free guidance method is adapted to conditional generation. It is also worthwhile to point out that after generative training, our model can be fine-tuned on regression and classification tasks with the state-of-the-art performance, which opens the gate of building all-in-one models in a single module style.

## News

* [21/07/2024] Paper was submitted to arXiv.

## Usage

You can find example scripts in [📁example](./example) folder.

## Pre-trained Model

You can find pretrained models in [release](https://github.com/Augus1999/bayesian-flow-network-for-chemistry/releases).

## Dataset Format

We provide a Python class [`CSVData`](./bayesianflow_for_chem/data.py) to handle data stored in CSV or similar format containing headers with the following tags:
* __smiles__ or __safe__ or __selfies__ (_mandatory_): the entities under this tag should be molecule SMILES, SAFE or SELFIES strings. Multiple tags are acceptable.
* __value__ (_optional_): entities under this tag should be molecular properties or classes. Multiple tags are acceptable and in this case you can tell `CSVData` which value(s) should be loaded by specifying `label_idx=[...]`. If a property is not defined, leave it empty and the entity will be automatically masked to torch.inf telling the model that this property is unknown.

## Cite This Work

```bibtex
@misc{2024chembfn,
      title={Bayesian-Flow-Network-for-Chemistry Github repository}, 
      author={Nianze Tao and Minori Abe},
      year={2024},
}
```
