"""Turning a noisy RMSE(N) curve into a defensible "how many sensors do you
need" answer: a saturation-curve fit, an N* rule that isn't just argmin
(argmin of a noisy curve is a winner's-curse-biased answer), and the
dissimilarity index used to judge whether a test location is well
represented by the network at a given N.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit


def saturation_curve(n, a, b, c):
    """RMSE(N) = a + b * N^(-c). `a` is the asymptotic floor as N -> infinity
    (compare against the variogram nugget from notebook 05: if `a` is close
    to the nugget, the network is already near the irreducible noise floor;
    if `a` is well above it, there's still exploitable spatial structure a
    bigger network could capture)."""
    return a + b * np.power(n, -c)


def fit_saturation_curve(n_values, rmse_values, weights=None):
    """Weighted least squares fit of the saturation curve. `weights` should
    be inverse-variance (e.g. 1/SE^2 per N) so noisier points (fewer
    permutations, smaller N) don't dominate the fit."""
    n_values = np.asarray(n_values, dtype=float)
    rmse_values = np.asarray(rmse_values, dtype=float)
    sigma = None if weights is None else 1.0 / np.sqrt(np.asarray(weights))
    p0 = [rmse_values.min(), rmse_values.max() - rmse_values.min(), 0.5]
    try:
        popt, pcov = curve_fit(saturation_curve, n_values, rmse_values, p0=p0,
                                sigma=sigma, maxfev=10000)
    except RuntimeError:
        return None
    return {"a": popt[0], "b": popt[1], "c": popt[2], "cov": pcov}


def estimate_n_star(n_values, rmse_values, se_values, marginal_gain_threshold=0.02,
                     se_multiplier=1.0, n_grid=None):
    """N* = the smallest N in `n_grid` (default: a fine grid spanning
    n_values) such that (1) the fitted curve's RMSE at N is within
    `se_multiplier` * SE of the asymptote `a`, AND (2) the marginal gain
    RMSE(N) - RMSE(N+1) implied by the fitted curve is below
    `marginal_gain_threshold`. Both conditions avoid picking the noisy
    curve's argmin (winner's curse) -- N* is read off the smooth fit, not
    the raw per-N medians.
    """
    fit = fit_saturation_curve(n_values, rmse_values, weights=1.0 / np.asarray(se_values) ** 2)
    if fit is None:
        return {"n_star": None, "reason": "curve fit failed"}

    a, b, c = fit["a"], fit["b"], fit["c"]
    n_grid = n_grid if n_grid is not None else np.arange(min(n_values), max(n_values) + 1)
    rmse_grid = saturation_curve(n_grid, a, b, c)
    marginal = -np.diff(rmse_grid, append=rmse_grid[-1])

    # Average SE across observed N as a flat tolerance band (conservative:
    # using the largest-N SE would be too optimistic a band for small N).
    se_at_n = np.interp(n_grid, n_values, se_values)
    within_se = np.abs(rmse_grid - a) <= se_multiplier * se_at_n
    small_marginal = marginal <= marginal_gain_threshold

    candidates = n_grid[within_se & small_marginal]
    if len(candidates) == 0:
        max_n = int(max(n_values))
        return {"n_star": None, "fit": fit, "reason": f"N* > {max_n}, cannot be determined from this range"}
    return {"n_star": int(candidates.min()), "fit": fit, "rmse_grid": rmse_grid, "n_grid": n_grid}


def dissimilarity_index(target_static: np.ndarray, network_static: np.ndarray) -> float:
    """Area-of-Applicability-style dissimilarity (Meyer & Pebesma 2021,
    simplified): the target's nearest-neighbor distance in standardized
    static-feature space to the network, divided by the typical
    nearest-neighbor distance *within* the network itself. >1 means the
    target is further from every network sensor than network sensors
    typically are from each other -- i.e. extrapolation, not interpolation.
    `target_static` / `network_static` are already z-scored using the
    network's own mean/std (same scaling for both).
    """
    network_static = np.asarray(network_static, dtype=float)
    n = len(network_static)
    if n < 2:
        return np.inf

    # Typical within-network nearest-neighbor distance.
    within_dists = []
    for i in range(n):
        d = np.linalg.norm(network_static - network_static[i], axis=1)
        d[i] = np.inf
        within_dists.append(d.min())
    typical_nn = np.median(within_dists)

    target_static = np.asarray(target_static, dtype=float).reshape(1, -1)
    d_to_network = np.linalg.norm(network_static - target_static, axis=1).min()
    return float(d_to_network / typical_nn) if typical_nn > 0 else np.inf
