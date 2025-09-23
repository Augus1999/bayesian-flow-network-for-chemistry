## This folder contains example commands and configuration files.

We provide `madmol`, a CLI tool, to handle basic generative tasks (training-only, training-inference, inference-only). Note that advanced functionalities (e.g., quantisation, ensemble, etc.) and QSAR/QSPR route are currently not included in the CLI tool, which should be referred to Python API.

### 1. Get version

```bash
madmol --version
```

### 2. Get help

```bash
madmol --help
```

### 3. Dry-run to check settings

```bash
madmol config.toml model_config.toml --dryrun
```

### 4. Run a job

```bash
madmol config.toml model_config.toml
```

Examples of configurations are in [config.toml](./config.toml) and [model_config.toml](./model_config.toml).
