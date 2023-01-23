import torch
from scipy.stats import ncx2
from typing import Union
import numpy as np

def interpolate(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """One-dimensional linear interpolation for monotonically increasing sample
    points.

    Returns the one-dimensional piecewise linear interpolant to a function with
    given discrete data points :math:`(xp, fp)`, evaluated at :math:`x`.

    From https://github.com/pytorch/pytorch/issues/50334#issuecomment-1247611276.
  
    Parameters
    ----------
    x : torch.Tensor
        the :math:`x`-coordinates at which to evaluate the interpolated values.
    xp : torch.Tensor
        the :math:`x`-coordinates of the data points, must be increasing.
    fp : torch.Tensor
        the :math:`y`-coordinates of the data points, same length as `xp`.

    Returns
    -------
    torch.Tensor
        the interpolated values, same size as `x`
    """
    m = (fp[:,1:] - fp[:,:-1]) / (xp[:,1:] - xp[:,:-1])  #slope
    b = fp[:, :-1] - (m.mul(xp[:, :-1]) )

    indicies = torch.sum(torch.ge(x[:, :, None], xp[:, None, :]), -1) - 1  #torch.ge:  x[i] >= xp[i] ? true: false
    indicies = torch.clamp(indicies, 0, m.shape[-1] - 1)

    line_idx = torch.linspace(0, indicies.shape[0], 1, device=indicies.device).to(torch.long)
    line_idx = line_idx.expand(indicies.shape)
    # idx = torch.cat([line_idx, indicies] , 0)
    return m[line_idx, indicies].mul(x) + b[line_idx, indicies]

def detection_probabilty_from_optimal_snr(optimal_snr: Union[np.ndarray, torch.tensor, float], threshold: float, number_of_detectors=1):
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
    optimal_snr[optimal_snr < 0] = 0

    return_device = None
    if isinstance(optimal_snr, torch.tensor):
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
    return detection_probabilty_from_optimal_snr(optimal_snr, threshold, number_of_detectors).mean(axis=0)