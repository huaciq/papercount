"""Chen et al. III-A: analytic Gaussian and gamma-normalized second derivative."""
import numpy as np
from scipy.ndimage import convolve1d


def validate_sigma(sigma):
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('sigma must be finite and positive')


def gaussian(x, sigma):
    validate_sigma(sigma)
    x = np.asarray(x, dtype=np.float64)
    return np.exp(-x * x / (2 * sigma * sigma)) / (np.sqrt(2 * np.pi) * sigma)


def gaussian_first(x, sigma):
    return -np.asarray(x) / sigma**2 * gaussian(x, sigma)


def gaussian_second(x, sigma):
    return (np.asarray(x)**2 / sigma**4 - 1 / sigma**2) * gaussian(x, sigma)


def ridge_filter(x, sigma, gamma=1.0):
    return -sigma**(2 * gamma) * gaussian_second(x, sigma)


def sampled_kernel(sigma, gamma=1., tail_sigma=5., valley=False):
    """Exact pixel-bin integral, finite truncation and tiny center DC correction.

    Discretization and reflect padding are declared engineering choices; no
    L1/amplitude normalization is applied to the paper's scale-normalized filter.
    """
    validate_sigma(sigma)
    radius = max(1, int(np.ceil(tail_sigma * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    weights = sigma**(2 * gamma) * (gaussian_first(x + .5, sigma) - gaussian_first(x - .5, sigma))
    if not valley:
        weights = -weights
    correction = float(weights.sum())
    weights[radius] -= correction
    return x, weights, correction


def response(signal, weights):
    return convolve1d(np.asarray(signal, dtype=np.float64), weights, mode='reflect')
