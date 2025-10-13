# -*- coding: utf-8 -*-
# Author: Nianze A. Tao (Omozawa Sueno)
"""
ChemBFN package.
"""
import colorama
from . import data, tool, train, scorer, spectra
from .model import ChemBFN, MLP, EnsembleChemBFN
from .cli import main_script

__all__ = [
    "data",
    "tool",
    "train",
    "scorer",
    "spectra",
    "ChemBFN",
    "MLP",
    "EnsembleChemBFN",
]
__version__ = "2.0.5"
__author__ = "Nianze A. Tao (Omozawa Sueno)"


def main() -> None:
    """
    CLI main function.

    :return:
    :rtype: None
    """
    colorama.just_fix_windows_console()
    main_script(__version__)
    colorama.deinit()
