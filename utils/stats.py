import numpy as np
from numba import njit


@njit
def pseudo_log_likelihood(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    """

    :param x: [n, 1]
    :param mean: [n, 1]
    :param cov: [n, n]
    :return: log likelihood w/out the constant term
    """
    residual = x - mean
    log_likelihood = -0.5 * residual.T @ np.linalg.inv(cov) @ residual
    return log_likelihood.item()
