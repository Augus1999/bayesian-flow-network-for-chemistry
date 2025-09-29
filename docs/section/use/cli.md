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

This command will give you hints, if any, of misconfigurations that will probably terminate your job, e.g., setting conflicts, missing files, etc.

### 4. Run Your Job

```bash
madmol [YOUR_CONFIG.toml] [YOUR_MODEL_CONFIG.toml]
```

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
dataset = "home/user/project/dataset/qm9.csv"
molecule_tag = "smiles"                  # <-- the header tag under which the molecules are stored
objective_tag = ["homo", "lumo", "gap"]  # <-- the header tag(s) under which the objective values are stored; set to empty array [] if the model is unconditional
enforce_validity = true                  # <-- no effect if SMILES or SAFE is not used
logger_name = "wandb"                    # <-- "wandb", "csv" or "tensorboard"
logger_path = "home/user/project/logs"
checkpoint_save_path = "home/user/project/ckpt"
train_strategy = "auto"                  # <-- any strategy supported by Lightning, e.g., "ddp"
accumulate_grad_batches = 1
enable_progress_bar = false

[inference]  # <-- Remove this table if inference is unnecessary
mini_batch_size = 50
sequence_length = "match dataset"           # <-- must be an integer in an inference-only job
sample_size = 1000                          # <-- the minimum number of samples you want
sample_step = 100
sample_method = "ODE:0.5"                   # <-- meaning ODE-solver with temperature of 0.5; another choice is "BFN"
semi_autoregressive = false
guidance_objective = [-0.023, 0.09, 0.113]  # <-- for unconditional jobs set it to empty array []
guidance_objective_strength = 4.0           # <-- unnecessary if guidance_objective = []
guidance_scaffold = "c1ccccc1"              # <-- if no scaffold is used set it to empty string ""
unwanted_token = []
exclude_invalid = true                      # <-- whether to only store valid samples
exclude_duplicate = true                    # <-- whether to only store unique samples
result_file = "home/user/project/result/result.csv"
```

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