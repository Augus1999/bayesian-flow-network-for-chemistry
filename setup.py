# -*- coding: utf-8 -*-
# Author: Nianze A. TAO (Omozawa SUENO)
import os
import re
from pathlib import Path
from shutil import rmtree
from setuptools import setup, find_packages

source_path = Path("bayesianflow_for_chem")

with open(source_path / "__init__.py", mode="r", encoding="utf-8") as f:
    lines = f.readlines()
for line in lines:
    if "__version__" in line:
        version = re.findall(r"[0-9]+\.[0-9]+\.[0-9]+", line)
        if len(version) != 0:
            version = version[0]
            print("version:", version)
            break
with open(source_path / "data.py", mode="r", encoding="utf-8") as f:
    lines = f.readlines()
for i, line in enumerate(lines):
    if "class CSVData(Dataset):" in line:
        break

with open("README.md", mode="r", encoding="utf-8") as fh:
    long_description = fh.read()

long_description = long_description.replace(
    r"(./example)",
    r"(https://github.com/Augus1999/bayesian-flow-network-for-chemistry/tree/main/example)",
)
long_description = long_description.replace(
    r"(./bayesianflow_for_chem/data.py)",
    rf"(https://github.com/Augus1999/bayesian-flow-network-for-chemistry/blob/main/bayesianflow_for_chem/data.py#L{i + 1})",
)

setup(
    name="bayesianflow_for_chem",
    version=version,
    url="https://augus1999.github.io/bayesian-flow-network-for-chemistry/",
    description="Bayesian flow network framework for Chemistry",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="AGPL-3.0-or-later",
    license_files=["LICEN[CS]E*"],
    package_dir={"bayesianflow_for_chem": "bayesianflow_for_chem"},
    package_data={"bayesianflow_for_chem": ["./_data/*.txt"]},
    include_package_data=True,
    author="Nianze A. Tao",
    author_email="tao-nianze@hiroshima-u.ac.jp",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "rdkit>=2025.3.5",
        "torch>=2.8.0",
        "torchao>=0.12",
        "colorama>=0.4.6",
        "numpy>=2.3.2",
        "scipy>=1.16.1",
        "loralib>=0.1.2",
        "lightning>=2.5.3",
        "scikit-learn>=1.7.1",
    ],
    project_urls={
        "Source": "https://github.com/Augus1999/bayesian-flow-network-for-chemistry"
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=["Chemistry", "CLM", "ChemBFN"],
    entry_points={"console_scripts": ["madmol=bayesianflow_for_chem:main"]},
)

if os.path.exists("build"):
    rmtree("build")
if os.path.exists("bayesianflow_for_chem.egg-info"):
    rmtree("bayesianflow_for_chem.egg-info")
