`Madmol` is a CLI tool to simplify the workflow of training a generative model on a given dataset and/or sampling molecules.

### 1. Get Version

```bash
madmol --version
```

### 2. Get Help

```bash
madmol --help
```

### 3. Check The Settings Before Running A Job

```bash
madmol [YOUR_CONFIG.toml] [YOUR_MODEL_CONFIG.toml] --dryrun
```

This command will give you hints, if any, of misconfigurations that will probably terminate your job, _e.g._, setting conflicts, missing files, _etc_.

> ⚠️ If you specified checkpoints, this process does not check the hyperparameters, weight shapes, _etc_, therefore the risk of encountering shape mismatch error during runtime is still there.

### 4. Run Your Job

```bash
madmol [YOUR_CONFIG.toml] [YOUR_MODEL_CONFIG.toml]
```

#### 4.1. Defining task

The first positional argument `[YOUR_CONFIG.toml]` should be an absolute path pointing to a TOML file defining the runtime configurations. The format should follow the example below.

```toml
device = "auto"   # <-- any device supportrd by PyTorch, e.g., "cpu", "cuda:0"
run_name = "qm9"  # <-- job name

[tokeniser]
name = "SMILES"    # <-- "SMILES", "SAFE", "FASTA" or "SELFIES"
vocab = "default"  # <-- it should be a vocabulary file name in absolute path iff name = "SELFIES"

[train]  # <-- remove this table if training is unnecessary
epoch = 100
batch_size = 512
semi_autoregressive = false
enable_lora = false
dynamic_padding = false                  # <-- only set to true when pretraining a model
restart = ""                             # <-- a checkpoint file in absolute path if necessary
dataset = "/home/user/project/dataset/qm9.csv"
molecule_tag = "smiles"                  # <-- the header tag under which the molecules are stored
objective_tag = ["homo", "lumo", "gap"]  # <-- the header tag(s) under which the objective values are stored; set to empty array [] if the model is unconditional
enforce_validity = true                  # <-- no effect if SMILES or SAFE is not used
logger_name = "wandb"                    # <-- "wandb", "csv" or "tensorboard"
logger_path = "/home/user/project/logs"
checkpoint_save_path = "/home/user/project/ckpt"
train_strategy = "auto"                  # <-- any strategy supported by Lightning, e.g., "ddp"
accumulate_grad_batches = 1
enable_progress_bar = false
plugin_script = ""                       # <-- define customised behaviours of dataset, datasetloader, etc in a python script

[inference]  # <-- Remove this table if inference is unnecessary
mini_batch_size = 50
sequence_length = "match dataset"           # <-- must be an integer in an inference-only job
sample_size = 1000                          # <-- the minimum number of samples you want
sample_step = 100
sample_method = "ODE:0.5"                   # <-- meaning ODE-solver with temperature of 0.5; another choice is "BFN"
semi_autoregressive = false
lora_scaling = 1.0                          # <-- adjusting the LoRA effectiveness if applied
guidance_objective = [-0.023, 0.09, 0.113]  # <-- for unconditional jobs set it to empty array []
guidance_objective_strength = 4.0           # <-- unnecessary if guidance_objective = []
guidance_scaffold = "c1ccccc1"              # <-- if no scaffold is used set it to empty string ""
sample_template = ""                        # <-- template for mol2mol task; leave it blank if scaffold is used
unwanted_token = []
exclude_invalid = true                      # <-- whether to only store valid samples
exclude_duplicate = true                    # <-- whether to only store unique samples
result_file = "/home/user/project/result/result.csv"
```

Important notes:

> `lora_scaling` and `sample_template` were added in version 2.1.0
>
> `plugin_script` was added in version 2.2.0
>
> Since version 2.4.3, you can use `"<pad>"` as a token, _e.g._ `"c1cc<pad><pad>"`, in `guidance_scaffold`

#### 4.2. Defining model architecture

The second positional argument `[YOUR_MODEL_CONFIG.toml]` should be an absolute path pointing to a TOML file defining the model hyperparameters. The following example shows the format.

```toml
[ChemBFN]
num_vocab = "match vocabulary size"  # <-- you can set to a specific integer
channel = 512
num_layer = 12
num_head = 8
dropout = 0.01
base_model = []                      # <-- specify a base model checkpoint file in absolute path when necessary; format ["basemodel.pt", "lora.pt" (optional)]

[MLP]  # <-- Reomve this table if MLP is not needed.
size = [3, 256, 512]                 # <-- dimension of the vector goes as 3 --> 256 --> 512
class_input = false                  # <-- set to true if the inputs are class indices
base_model = ""                      # <-- specify a base model checkpoint in absolute path when necessary
```

#### 4.3. Defining customised behaviours

Since version _**2.2.0**_, it is possible to pass a Python3 script to the program via `plugin_script={PATH/TO/YOUR/SCRIPT.py}` in `[YOUR_CONFIG.toml]` to control the behaviours of dataset loading and sequence padding. Recently, the accepted customised values are `collate_fn`, `num_workers`, `shuffle`, `max_sequence_length`, and `CustomData`.

For instance, to disable shuffling the batches

```python
shuffle = False
```
to change the number of workers (_default value is 4_) in the `~torch.utils.data.DataLoader` instance

```python
num_worker = 0
```

to define a padding length (_default is the maximum length in the dataset_)

```python
max_sequence_length = 125
```

to use a customised collating function

```python
import random
from bayesianflow_for_chem.data import collate

def collate_fn(x):
    # shufffle inside a mini-batch
    random.shuffle(x)
    return collate(x)
```

or to define your own dataset object (_e.g._, chunked dataset class)

```python
import torch
import pandas as pd
from bayesianflow_for_chem.data import CSVData

class CustomData(CSVData):
    def __init__(self, file, chunksize: int = 100000):
        super().__init__(file)
        ...  # your code

    def __len__(self):
        return ...  # your code
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        ...  # your code
        return self.mapping(...)
```

In order to tell the program which customised values should be used, it is necessary to encapsulate them in the `__all__` variable, _e.g._, `__all__ = ["collate_fn", "num_workers", "shuffle", "max_sequence_length", "CustomData"]`.

Note that

(1) if you define a dataset class not inherited from `CSVData`, make sure you include the `map(...)` method. If `map(...)` method is unnecessary for your `CustomData`, set it to a function equivalent to `lambda x: None`;

(2) if `max_sequence_length` is not provided, the program will always calculate this value even when `dynamic_padding = true` is set in `[YOUR_CONFIG.toml]`. To bypass this behaviour, set `max_sequence_length = "n.a."`;

(3) for safety reasons, we banned the use of `open(...)` inside the plugin script. Please use methods provided by `pandas`, `scipy`, _etc_. However, there is NO isolation sandbox used, so do not paste any code you do not understand into the script! If you are planning to deploy this program into a service, do remember to restrict the network and file system premissions for the users.

A detailed example can be found [here](https://github.com/Augus1999/bayesian-flow-network-for-chemistry/blob/main/example/cli/plugin.py).

Tips:
If you have a very large dataset, employing shuffle inside mini-batches rather than global shuffle will significantly accelerate the training process.

### 5. Get Example Config Files

Since version _**2.4.0**_, it became possible to obtain example configurations from the CLI, _i.e._,

```bash
madmol --example_config
```

This command will bring you a `config.toml` file and a `model_config.toml` file under current working directory with pre-defined configurations. You then can change the contents to fit thy own purpose.
