# -*- coding: utf-8 -*-
# Author: Nianze A. TAO (Omozawa SUENO)
"""
Build and analysis spectra.
"""
import numpy as np
from scipy.stats import wasserstein_distance


def build_uv_vis_spectrum(
    etoscs: np.ndarray, etenergies: np.ndarray, lambdas: np.ndarray
) -> np.ndarray:
    """
    Build UV/Vis spectrum from calculated electron transtion energies and oscillator strengths. \n
    This function follows the GaussView style: https://gaussian.com/uvvisplot/.

    :param etoscs: oscillator strengths
    :param etenergies: transtion energies
    :param lambdas: wavelengths
    :type etoscs: numpy.ndarray
    :type etenergies: numpy.ndarray
    :type lambdas: numpy.ndarray
    :return: absorption coefficient corrospending to the wavelengths
    :rtype: numpy.ndarray
    """
    return (
        etoscs[:, None]
        * np.exp(
            -np.pow((1 / lambdas[None, :] - etenergies[:, None] / 45.5634) * 3099.6, 2)
        )
    ).sum(0) * 40489.99421


def spectra_wasserstein_score(
    spectrum_u: np.ndarray, spectrum_v: np.ndarray, x_axis: np.ndarray
) -> float:
    """
    Return the Wasserstein distance (earth mover's distance) between two
    continuous spectra scaled by the area under the first spectrum curve `spectrum_u`.

    :param spectrum_u: the reference spectrum
    :param spectrum_v: the
    :param x_axis: the shared x-axis of the spectra
    :type spectrum_u: numpy.ndarray
    :type spectrum_v: numpy.ndarray
    :type x_axis: numpy.ndarray
    :return: spectra Wasserstein score
    :rtype: float
    """
    assert spectrum_u.size == spectrum_v.size, "Spectra sizes should be matched."
    a = np.sqrt(np.trapezoid(spectrum_u, x_axis))
    return (wasserstein_distance(spectrum_u, spectrum_v) / a).item()


if __name__ == "__main__":
    ...
