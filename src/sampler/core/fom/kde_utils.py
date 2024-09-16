from typing import List, Tuple, Dict, Any, Callable, Union, Optional
import warnings
import numpy as np
import pandas as pd

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import RandomizedSearchCV
from sklearn.cluster import MeanShift
from scipy.optimize import brentq
from scipy.stats import qmc
from sklearn.neighbors import KernelDensity

from .term_base import BANDWIDTH_BOUNDS, RANDOM_STATE


def brentq_bandwidth(
    X: np.ndarray,
    kernel_name: str,
    threshold: float,
    previous_bandwidth: Optional[float] = None
) -> float:
    """
    Find the largest bandwidth where null density area proportion is close to
    threshold. Formally:

    best_bandwidth = max({bw | null_density_proportion > threshold})
                   = max({
        bandwidth in bw_range |
        V({X in [0, 1]^p | Density(X) < 0 + tol }) / V([0, 1]^p) > threshold
    })

    This function uses the previous bandwidth to optimize the search process.

    Note:
        - When X data is not distributed all over [0, 1]^p, the threshold must
          be adapted and high enough to have a meaning. Otherwise condition will
          be unusefully always satisfied. But in IRBS context X is always well
          distributed thanks to LHS initialization.
    """

    def density_condition(bandwidth):
        # Set the bandwidth
        kde = KernelDensity(kernel=kernel_name, bandwidth=bandwidth)
        kde.fit(X)

        # Compute area of null density
        null_density_proportion = estimate_low_density_proportion(kde, X.shape[1])

        return null_density_proportion - threshold

    # Set bw range where f(a) and f(b) have different signs
    search_range = set_brentq_range(previous_bandwidth, density_condition)

    if isinstance(search_range, float):
        # No suitable search range, returning adequate bound value
        return search_range

    # Use Brent's method to find the root (null_density_proportion - threshold = 0) 
    optimal_bandwidth, results = brentq(
        density_condition,
        search_range[0],
        search_range[1],
        xtol=1e-2,  # absolute tol over root (over null_density_proportion)
        maxiter=50,
        full_output=True,
        disp=False
    )

    # The final null_density_proportion
    null_density_proportion = results.root + threshold
    print(
        f"brentq_bandwidth -> "
        f"optimal bandwidth: {optimal_bandwidth:.3f}, "
        f"n_samples: {X.shape[0]}, "
        f"null_density_proportion: {null_density_proportion:.3f}, "
        f"function calls: {results.function_calls}"
    )

    return optimal_bandwidth


def set_brentq_range(
    previous_bandwidth: Optional[float],
    density_condition: Callable[[float], float],
    full_range_proba: float = 0.1
) -> Union[Tuple[float, float], float]:
    """ 
    Set the range for the Brent's method search.
    
    Args:
        previous_bandwidth: The bandwidth from the previous iteration, if any.
        density_condition: A function that computes the density condition for a
        given bandwidth.
        bandwidth_bounds: The overall bounds for the bandwidth search.
        full_range_proba: The probability of using the full range for search.

    Returns:
        Either a tuple representing the search range, or a float representing a
        default bandwidth.
    """

    bandwidth_bounds: Tuple[float, float] = BANDWIDTH_BOUNDS

    if np.random.rand() < full_range_proba or previous_bandwidth is None:
        # Chance to search over full range or if no previous bandwidth
        search_range = bandwidth_bounds
    else:
        # Define search range around previous bandwidth
        search_range = (
            max(bandwidth_bounds[0], previous_bandwidth / 2),
            min(bandwidth_bounds[1], previous_bandwidth * 2)
        )

    # Check if f(a) and f(b) have different signs
    fa = density_condition(search_range[0])
    fb = density_condition(search_range[1])
    
    if fa * fb > 0:  # Same sign
        # Use the full bandwidth bounds range
        search_range = bandwidth_bounds
        fa = density_condition(search_range[0])
        fb = density_condition(search_range[1])

        # If still same sign, warn and return default value
        if fa * fb > 0:
            # If null_density_proportion > threshold anyway (fa > 0) return largest bw
            # Conversely, if it can't get over the threshold return smallest bw
            default_bw = search_range[1] if fa > 0 else search_range[0]
            warnings.warn(
                f"Cannot find a bandwidth within bounds {search_range}. "
                f"Returning default value: {default_bw}."
            )
            return default_bw

    return search_range


def estimate_low_density_proportion(
    kde: KernelDensity,
    num_features: int,
    tol: float = 1e-5,
    num_samples: int = 10000
):
    """
    Estimate the proportion of feature space [0, 1]^p with density below a
    tolerance using Monte Carlo sampling.

    This function uses Latin Hypercube Sampling to generate uniform random samples
    in the feature space, which is equivalent to Monte Carlo sampling for this purpose.

    Parameters:
    kde : sklearn.neighbors.KernelDensity object
        The trained KDE object
    num_features : int
        The number of features (dimensionality of the space)
    tol : float, optional
        The density tolerance below which is considered "low" (default 1e-5)
    num_samples : int, optional
        The number of Monte Carlo samples to use (default 10000)

    Returns:
    float
        The estimated proportion of the feature space with low density
    """
    # Generate uniform random samples in [0, 1]^p space
    sampler = qmc.LatinHypercube(d=num_features)
    mc_samples = sampler.random(n=num_samples)
    
    # Evaluate the KDE at each sample point
    log_densities = kde.score_samples(mc_samples)
    densities = np.exp(log_densities)
    
    # Estimate the proportion of low-density areas
    # n_null_density_samples / n_samples
    low_density_proportion = np.mean(densities < tol)
    
    return low_density_proportion


def mean_shift_modes(X: np.ndarray, bandwidth: float):
    """ Find modes (centroids) of data clusters using Mean shift. """
    mean_shift = MeanShift(bandwidth=bandwidth)
    mean_shift.fit(X)

    # Get modes (highest density points)
    return mean_shift.cluster_centers_


def search_cv_bandwidth(X: np.ndarray, kernel_name: str) -> float:
    """ Search bandwidth that maximizes log-likelihood for cross-validation folds. """

    # Define a range of bandwidths to search over using log scale
    bandwidths = np.logspace(
        np.log10(BANDWIDTH_BOUNDS[0]), np.log10(BANDWIDTH_BOUNDS[1]), 100
    )

    # Use RandomizedSearchCV to find the best bandwidth
    random_search = RandomizedSearchCV(
        KernelDensity(kernel=kernel_name),
        {'bandwidth': bandwidths},
        n_iter=20,
        cv=5,
        random_state=RANDOM_STATE,
        # scoring=None => use KDE.score which is log-likelihood
    )
    random_search.fit(X)

    print_search_cv_results(random_search)

    # Get best bandwidth
    best_bandwidth = random_search.best_params_['bandwidth']

    return best_bandwidth


def print_search_cv_results(random_search: RandomizedSearchCV) -> None:
    """ Convert RandomizedSearchCV results to DataFrame. """
     # Create DataFrame with results
    results = random_search.cv_results_
    df = pd.DataFrame({
        'Bandwidth': results['param_bandwidth'],
        'Mean Score': results['mean_test_score'],
        'Std Score': results['std_test_score']
    })

    # Sort DataFrame by Mean Score in descending order
    df = df.sort_values('Mean Score', ascending=False).reset_index(drop=True)

    # Print results
    print("CV Search Results:")
    print(f"Best bandwidth: {random_search.best_params_['bandwidth']:.6f}")
    print(f"Best score (log-likelihood): {random_search.best_score_:.6f}")
    
    print("\nTop 5 bandwidths based on log-likelihood of test folds:")
    print(df.head().to_string(index=False, float_format='{:.6f}'.format))
