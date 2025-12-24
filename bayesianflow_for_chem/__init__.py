# -*- coding: utf-8 -*-
# Author: Nianze A. Tao (Omozawa Sueno)
"""
ChemBFN package.
"""
from . import data, tool, train, scorer, spectra
from .model import ChemBFN, MLP, EnsembleChemBFN

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
__version__ = "2.3.5"
__author__ = "Nianze A. Tao (Omozawa Sueno)"


def main() -> None:
    """
    CLI main function.

    :return:
    :rtype: None
    """
    import platform
    from bayesianflow_for_chem.cli import main_script

    _is_windows = platform.system() == "Windows"
    if _is_windows:
        import colorama

        colorama.just_fix_windows_console()
    main_script(__version__)
    if _is_windows:
        colorama.deinit()
