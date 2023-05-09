"""
Functions for computing noise-realised quantities from optimal SNRs, for the purposes of estimating signal/population detectability.
"""

import torch
from scipy.stats import ncx2
from typing import Union
import numpy as np

def matched_filter_snr_from_optimal_snr(optimal_snr: Union[np.ndarray, torch.tensor, float], number_of_detectors=1):
    """Computes detection probabilities from optimal snr values with respect to a detection threshold by sampling
    a non-central chi-square distribution.

    This function is not GPU-compatible and will therefore force synchronisation and movement of data between CPU and GPU. 
    The outputs will be on the same device as the inputs.

    Parameters
    ----------
    optimal_snr : Union[np.ndarray, torch.tensor, float]
        Optimal snr values to convert into matched filter snrs.
    number_of_detectors : int, optional
        The number of detectors in use, by default 1, by default 1

    Returns
    -------
    matched_filter_snrs: np.ndarray or torch.tensor
        Matched filter SNRs corresponding to the given optimal SNRs.
    """
    if isinstance(optimal_snr, float):
        optimal_snr = np.array([optimal_snr,])
    optimal_snr[optimal_snr < 0] = 0

    return_device = None
    if isinstance(optimal_snr, torch.Tensor):
        return_device = optimal_snr.device
        optimal_snr = optimal_snr.cpu().numpy()
    in_shape = optimal_snr.shape
    matched_filter_snrs = (ncx2(number_of_detectors,optimal_snr.flatten()**2).rvs(optimal_snr.size).reshape(in_shape))**0.5

    if return_device is not None:
        matched_filter_snrs = torch.as_tensor(matched_filter_snrs, device=return_device)
    return matched_filter_snrs

def detection_probability_from_optimal_snr(optimal_snr: Union[np.ndarray, torch.tensor, float], threshold: float, number_of_detectors=1):
    """Computes detection probabilities from optimal snr values with respect to a detection threshold using the survival function of
    a non-central chi-square distribution.

    This function is not GPU-compatible and will therefore force synchronisation and movement of data between CPU and GPU. 
    The outputs will be on the same device as the inputs.

    Parameters
    ----------
    optimal_snr : np.ndarray or torch.tensor
        Optimal snr values to convert into detection probabilities.
    threshold : float
        The detection threshold.
    number_of_detectors : int, optional
        The number of detectors in use, by default 1

    Returns
    -------
    detection_probabilities: np.ndarray or torch.tensor
        The resuling detection probablities for the given detection threshold.
    """
    if isinstance(optimal_snr, float):
        optimal_snr = np.array([optimal_snr,])
    optimal_snr[optimal_snr > threshold * 10] = threshold * 10

    return_device = None
    if isinstance(optimal_snr, torch.Tensor):
        return_device = optimal_snr.device
        optimal_snr = optimal_snr.cpu().numpy()
    in_shape = optimal_snr.shape
    probs = (1-ncx2(number_of_detectors,optimal_snr.flatten()**2).cdf(threshold**2)).reshape(in_shape)

    if return_device is not None:
        probs = torch.as_tensor(probs, device=return_device)
    return probs

def selection_function_from_optimal_snr(optimal_snr: Union[np.ndarray, torch.tensor], threshold: float, number_of_detectors=1):
    """Computes the selection function (i.e. the mean detection probability) from a set of optimal snr values with respect to a detection threshold using the survival function of
    a non-central chi-square distribution.

    This function is not GPU-compatible and will therefore force synchronisation and movement of data between CPU and GPU. 
    The outputs will be on the same device as the inputs.

    Parameters
    ----------
    optimal_snr : np.ndarray or torch.tensor
        Optimal snr values to convert into detection probabilities.
    threshold : float
        The detection threshold.
    number_of_detectors : int, optional
        The number of detectors in use, by default 1

    Returns
    -------
    selection function: np.ndarray or torch.tensor
        The resuling selection function for the given detection threshold.
    """
    return detection_probability_from_optimal_snr(optimal_snr, threshold, number_of_detectors).mean(axis=0)