# -*- coding: utf-8 -*-
# Author: Nianze A. TAO (Omozawa SUENO)
import os
import re
from pathlib import Path
from shutil import rmtree
from setuptools import setup, find_packages

init_file = Path("bayesianflow_for_chem") / "__init__.py"

with open(init_file, mode="r", encoding="utf-8") as f:
    lines = f.readlines()
    for line in lines:
        if "__version__" in line:
            version = re.findall(r"[0-9]+\.[0-9]+\.[0-9]+", line)
            if len(version) != 0:
                version = version[0]
                print("version:", version)
                break

with open("README.md", mode="r", encoding="utf-8") as fh:
    long_description = fh.read()

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
    package_data={"bayesianflow_for_chem": ["./*.txt", "./*.py"]},
    include_package_data=True,
    author="Nianze A. Tao",
    author_email="tao-nianze@hiroshima-u.ac.jp",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "rdkit>=2023.9.6",
        "torch>=2.3.1",
        "numpy>=1.26.4",
        "loralib>=0.1.2",
        "lightning>=2.2.0",
        "scikit-learn>=1.5.0",
        "typing_extensions>=4.8.0",
    ],
    project_urls={
        "Source": "https://github.com/Augus1999/bayesian-flow-network-for-chemistry"
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=["Chemistry", "CLM", "ChemBFN"],
)

if os.path.exists("build"):
    rmtree("build")
if os.path.exists("bayesianflow_for_chem.egg-info"):
    rmtree("bayesianflow_for_chem.egg-info")
