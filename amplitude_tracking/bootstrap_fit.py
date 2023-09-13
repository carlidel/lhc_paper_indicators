"""This file contains the code for a bootstrap fit.
The idea is to fit the data multiple times, each time with a different
random sample of the data and random redundancies.
"""
import joblib
import lmfit
import numpy as np
from numba import njit


def resample_data(
    data_x: np.ndarray, data_y: np.ndarray, fraction: float, redundancies: float
):
    """Resample the data.

    Parameters
    ----------
    data_x : np.ndarray
        The data to resample.
    data_y : np.ndarray
        The data to resample.
    fraction : float
        The fraction of the data to use.
    redundancies : float
        The fraction of the data to use as redundancies.

    Returns
    -------
    np.ndarray, np.ndarray
        The resampled data.
    """
    if fraction < 0 or fraction > 1:
        raise ValueError("fraction must be between 0 and 1")
    if redundancies < 0:
        raise ValueError("redundancies must be positive")
    rng = np.random.default_rng()
    n = len(data_x)
    n_samples = int(n * fraction)
    n_redundancies = int(n * redundancies)
    data = np.stack((data_x, data_y), axis=1)
    sample = rng.choice(data, n_samples, replace=False)
    redundancies = rng.choice(sample, n_redundancies, replace=True)
    combo = np.concatenate((sample, redundancies))
    return combo[:, 0], combo[:, 1]


def single_bootstrap_fit(
    data_x: np.ndarray,
    data_y: np.ndarray,
    func: callable,
    params: lmfit.Parameters,
    fraction=0.8,
    redunancies=0.2,
    lmfit_options=None,
):
    neo_data_x, neo_data_y = resample_data(data_x, data_y, fraction, redunancies)
    if lmfit_options is None:
        lmfit_options = {}
    result = lmfit.minimize(
        func, params, args=(neo_data_x, neo_data_y), **lmfit_options
    )
    return result, neo_data_x, neo_data_y


def bootstrap_fit(
    data_x: np.ndarray,
    data_y: np.ndarray,
    func: callable,
    params: lmfit.Parameters,
    n_samples=1000,
    fraction=0.8,
    redunancies=0.2,
    n_jobs=-1,
    lmfit_options=None,
) -> list:
    """Fit the data multiple times, each time with a different random sample of
    the data and random redundancies.

    Parameters
    ----------
    data : np.ndarray
        The data to fit.
    func : callable
        The residual function to fit the data to.
    params : lmfit.Parameters
        The parameters to fit.
    n_samples : int, optional
        The number of samples to take, by default 1000
    fraction : float, optional
        The fraction of the data to use for each sample, by default 0.8
    redunancies : float, optional
        The fraction of the data to use as redundancies, by default 0.2
    n_jobs : int, optional
        The number of jobs to use, by default -1
    lmfit_options : dict, optional
        The options to pass to lmfit.minimize, by default None
    """
    if n_jobs == -1:
        n_jobs = joblib.cpu_count()
    result_list = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(single_bootstrap_fit)(
            data_x, data_y, func, params, fraction, redunancies, lmfit_options
        )
        for _ in range(n_samples)
    )
    return result_list
