# -*- coding: utf-8 -*-
# Author: Nianze A. Tao (Omozawa Sueno)
"""
ChemBFN package.
"""
import importlib as _importlib
from typing import TYPE_CHECKING, List, Any


_models = ["ChemBFN", "MLP", "EnsembleChemBFN"]
_submodules = ["data", "tool", "train", "scorer", "spectra"]

__all__ = [
    "MLP",
    "ChemBFN",
    "EnsembleChemBFN",
    "data",
    "tool",
    "train",
    "scorer",
    "spectra",
]
__version__ = "2.4.0"
__author__ = "Nianze A. Tao"


def __dir__() -> List[str]:
    return __all__


def __getattr__(name: str) -> Any:
    if name in _submodules:
        _importlib.import_module(f"bayesianflow_for_chem.{name}")
    elif name in _models:
        _imported_models = _importlib.import_module("bayesianflow_for_chem.model")
        return _imported_models.__dict__[name]
    else:
        try:
            return globals()[name]
        except KeyError as exc:
            raise AttributeError(
                f"Module 'bayesianflow_for_chem' has no attribute '{name}'"
            ) from exc


if TYPE_CHECKING:
    from . import data, tool, train, scorer, spectra
    from .model import ChemBFN, MLP, EnsembleChemBFN

assert set(_models + _submodules) == set(__all__)


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
