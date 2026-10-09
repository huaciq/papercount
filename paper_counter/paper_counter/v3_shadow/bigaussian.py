"""Chen et al. III-B equations (3)--(6), true C2 piecewise Bi-Gaussian."""
import numpy as np
from .gaussian import gaussian, gaussian_first, gaussian_second, validate_sigma


def parameters(sigma, rho):
    validate_sigma(sigma)
    if not np.isfinite(rho) or not 0 < rho <= 1:
        raise ValueError('rho must be in (0, 1]')
    return rho * sigma, rho**2


def bi_gaussian(x, sigma, rho=.3):
    sb, k = parameters(sigma, rho)
    x = np.asarray(x, dtype=np.float64)
    offset = np.exp(-.5) / np.sqrt(2 * np.pi) * (rho - 1) / sigma
    return np.where(np.abs(x) <= sigma, gaussian(x, sigma) + offset,
                    k * gaussian(np.abs(x) + sb - sigma, sb))


def bi_first(x, sigma, rho=.3):
    sb, k = parameters(sigma, rho)
    x = np.asarray(x, dtype=np.float64)
    return np.where(np.abs(x) <= sigma, gaussian_first(x, sigma),
                    k * gaussian_first(np.abs(x) + sb - sigma, sb) * np.sign(x))


def bi_second(x, sigma, rho=.3):
    sb, k = parameters(sigma, rho)
    x = np.asarray(x, dtype=np.float64)
    return np.where(np.abs(x) <= sigma, gaussian_second(x, sigma),
                    k * gaussian_second(np.abs(x) + sb - sigma, sb))


def bi_ridge_filter(x, sigma, rho=.3, gamma=1.):
    return -sigma**(2 * gamma) * bi_second(x, sigma, rho)


def sampled_kernel(sigma, rho=.3, gamma=1., tail_sigma=5.):
    sb, _ = parameters(sigma, rho)
    radius = max(1, int(np.ceil(sigma + (tail_sigma - 1) * sb)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    weights = -sigma**(2 * gamma) * (bi_first(x + .5, sigma, rho) - bi_first(x - .5, sigma, rho))
    correction = float(weights.sum())
    weights[radius] -= correction
    return x, weights, correction
