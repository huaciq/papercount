"""Independent valley filter (III-C), signed F_RV = F_B - eta F_V."""
import numpy as np
from .gaussian import sampled_kernel as gaussian_kernel, response
from .bigaussian import sampled_kernel as bi_kernel


def multiscale(signal, sigmas, kind='bi', rho=.3, gamma=1., tail_sigma=5.):
    if kind not in ('gaussian', 'bi'):
        raise ValueError(kind)
    sigmas = np.asarray(sigmas, dtype=np.float64)
    if sigmas.ndim != 1 or not len(sigmas) or not np.all(np.isfinite(sigmas)) or np.any(sigmas <= 0):
        raise ValueError('Positive finite 1D scales required')
    kernels, responses = [], []
    for s in sigmas:
        x, w, dc = (gaussian_kernel(s, gamma, tail_sigma) if kind == 'gaussian'
                     else bi_kernel(s, rho, gamma, tail_sigma))
        kernels.append(dict(sigma=float(s), x=x, weights=w, dc_correction=dc))
        responses.append(response(signal, w))
    levels = np.stack(responses)
    best_index = np.argmax(levels, axis=0)
    best = levels[best_index, np.arange(levels.shape[1])]
    return dict(levels=levels, maximum=best, best_index=best_index,
                best_sigma=sigmas[best_index], sigmas=sigmas, kernels=kernels)


def valley_likelihood(signal, sigma_v, gamma=1., tail_sigma=5.):
    x, w, dc = gaussian_kernel(sigma_v, gamma, tail_sigma, valley=True)
    return response(signal, w), dict(sigma=float(sigma_v), x=x, weights=w, dc_correction=dc)


def descriptor(f_b, f_v, eta=.8):
    if np.shape(f_b) != np.shape(f_v):
        raise ValueError('Ridge and valley coordinates must agree')
    if not np.isfinite(eta) or eta < 0:
        raise ValueError('eta must be finite and nonnegative')
    return np.asarray(f_b) - eta * np.asarray(f_v)
