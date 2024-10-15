from typing import List, Tuple, Dict, Any, Callable, Union, Optional
import warnings
import time
import numpy as np
import pandas as pd

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import RandomizedSearchCV
from sklearn.cluster import MeanShift
from scipy.optimize import brentq
from scipy.stats import qmc

from .term_base import BANDWIDTH_BOUNDS, RANDOM_STATE


class LowDensityRatioBandwidth:
    """
    Find the largest bandwidth where low density volume ratio is close to
    target ratio. Formally:

    best_bandwidth = max({bw | low_density_ratio > target_ratio})
                   = max({
        bandwidth in bw_range |
        V({X in [0, 1]^p | Density(X) < 0 + tol }) / V([0, 1]^p) > target_ratio
    })

    This class stores the optimal bandwidth for next search to optimize the
    search process.

    Note:
        - When X data is not distributed all over [0, 1]^p, the threshold must
          be adapted and high enough to have a meaning. Otherwise condition will
          be unusefully always satisfied. But in IRBS context X is always well
          distributed thanks to LHS initialization.
    """
    def __init__(self,
        default_bounds: Tuple[float, float],  # Bandhwidth range
        threshold: float,  # Threshold for indicator function
        target_ratio: float,
    ):
        self.default_bounds = default_bounds
        self.threshold = threshold
        self.target_ratio = target_ratio
        self.a_rdecrease = 0.2  # Relative decrease of lower bound
        self.f_atol = self.target_ratio * 0.01  # Absolute tol around 0
        self.previous_root = None
        self.X_train = None
        self._first_estimation = True

        # Initialize profiling variables
        self.ncalls_log: List[int] = []
        self.cumtime_log: List[float] = []
        self.ncalls: int = 0
        self.cumtime: float = 0.0
        self.cache: Dict[float, float] = {}
        self._first_reset: bool = True

    def init_profiler(self):
        """ Resets profilier and store logs of current value. """
        if self._first_reset:
            self._first_reset = False
        else:
            # Log calls and cumulative time
            self.ncalls_log.append(self.ncalls)
            self.cumtime_log.append(self.cumtime)

        # Reset profiler variables
        self.ncalls = 0
        self.cumtime = 0.0
        self.cache.clear()

    def init_mc_samples(self, n_dim: int):
        if self._first_estimation:
            self._first_estimation = False
            # Generate uniform random samples in [0, 1]^p space for Monte Carlo
            sampler = qmc.LatinHypercube(d=n_dim)
            self.mc_samples = sampler.random(n=10000)

    def compute_ratio(self, bandwidth: float) -> float:
        # Check if the result is already in the cache
        if bandwidth in self.cache:
            return self.cache[bandwidth]

        # Mark start time
        start_time = time.perf_counter()

        # Set the bandwidth
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
        kde.fit(self.X_train)

        # Compute function for which to find root
        low_density_ratio = estimate_low_density_ratio(kde, self.threshold, self.mc_samples)

        # Update profiler
        self.ncalls += 1
        self.cumtime += time.perf_counter() - start_time

        # Store the result in the cache
        self.cache[bandwidth] = low_density_ratio

        return low_density_ratio

    def compute_ratio_discrepancy(self, bandwidth: float) -> float:
        """
        Compute the discrepancy between the computed low density ratio and the
        target ratio.
        """
        return self.compute_ratio(bandwidth) - self.target_ratio
    
    def set_search_range(self) -> Tuple[float, float]:
        """ Set the search range for the Brent's method. """
        if self.previous_root is None:
            return self.default_bounds
        else:
            a = max(self.default_bounds[0], self.previous_root * (1 - self.a_rdecrease))
            b = min(self.default_bounds[1], self.previous_root)
            return a, b

    def check_root_in_range(self, a: float, b: float) -> Optional[float]:
        """
        Check if the root is in the given range based on the Intermediate Value
        Theorem, minimizing function evaluations.
        
        Note: Low density ratio is decreasing with bandwidth. To find a root,
              bounds must verify f(a) <= 0 <= f(b).
        """
        # Evaluate f(b) first, as it's more likely to be close to the root
        fb = self.compute_ratio_discrepancy(b)

        # f is a decreasing function, f(b) must be negative
        if fb >= 0 - self.f_atol:
            # If fb is positive, then b is nearest value to the root
            return b
        
        # Only evaluate f(a) if necessary
        fa = self.compute_ratio_discrepancy(a)

        # f is a decreasing function, f(a) must be positive
        if fa <= 0 + self.f_atol:
            # If fb is negative, then a is nearest value to the root
            return a

        # If we reach here, the root is within the bounds
        return None

    def brentq_search(self, X: np.ndarray) -> float:
        """ 
        Set the range for the Brent's method search of a decreasing function. 
        
        Assumptions:
            (1) ratio_discrepancy is a decreasing function. To find a root,
                bounds must verify f(a) <= 0 <= f(b).
            (2) ratio_discrepancy is smaller that previous one, which results in
                a smaller expected root, i.e. expected_root <= previous_root
            (3) If previous_root is available, new root is expected to be smaller
                 by maximum 20%.

        Algorithm:
            1. Set search range:
                - If previous_root is not available, use full search range
                - Else, use narrowed search range with previous_root as upper
                  bound.
            2. Check if root is in search range:
                - If f changes sign, search root using brentq
                - Else, use nearest bound to root
        """
        # Reset profiling variables
        self.init_profiler()
    
        # Initialize Monte Carlo samples
        self.init_mc_samples(X.shape[1])

        # Updat training data
        self.X_train = X

        # Set search range
        a, b = self.set_search_range()

        # Check if root in search range (f(a) and f(b) have different signs)
        x_root = self.check_root_in_range(a, b)

        if x_root is None:
            # Use Brent's method to find the root
            x_root, results = brentq(
                self.compute_ratio_discrepancy,
                a, b,
                xtol=1e-3,  # absolute tol over root bandwidth
                rtol=1e-2,
                maxiter=50,
                full_output=True,
                disp=False
            )

        # Use cache to evaluate low density ratio
        low_density_ratio = self.compute_ratio(x_root)

        # Store root for next search
        self.previous_root = x_root

        return x_root, low_density_ratio

    def get_params(self) -> Dict[str, Any]:
        return {
            'default_bounds': self.default_bounds,
            'threshold': self.threshold,
            'target_ratio': self.target_ratio,
            'a_rdecrease': self.a_rdecrease,
            'f_atol': self.f_atol
        }


def smooth_lower_bound_indicator(
    X: Union[float, np.ndarray],
    threshold: float,
    sigma: float = 0.01,
    eps: float = 0.01
) -> Union[float, np.ndarray]:
    """
    Compute a smooth approximation of the indicator function I(X < threshold)
    on [0, 1].
    
    This function uses a scaled and shifted logistic function to approximate
    the step function at the threshold, with a smooth transition. It avoids
    computing large exponential values for X >> threshold.

    Args:
        X (Union[float, np.ndarray]): Input value(s) to evaluate.
        threshold (float): The threshold value for the indicator function.
        sigma (float): Controls the steepness of the sigmoid.
        eps (float): Marginal value after transition

    Note:
        The transition width is controlled by sigma, where the function
        transitions from 1-eps/2 to 0+eps/2 over an interval of 2*sigma_eps
        centered at pos.
    """
    sigma_eps = sigma * np.log((2-eps)/eps)  # Half-width of the transition region
    pos = threshold + sigma_eps  # Center of the transition region

    # Compute the centered and scaled input
    z = (X - pos) / sigma

    return np.where(
        z > 10, 0,  # Avoid large exponential computations
        1 / (1 + np.exp(z))  # Compute sigmoid for other values
    )


def estimate_low_density_ratio(
    kde: KernelDensity,
    threshold: float,
    mc_samples: Optional[np.ndarray],
) -> np.ndarray:
    """
    Estimate the volume ratio of feature space [0, 1]^p with normalized density
    below a threshold using Monte Carlo sampling.

    This function uses Latin Hypercube Sampling to generate uniform random
    samples in the feature space, which is equivalent to Monte Carlo sampling
    for this purpose.

    Parameters:
    kde : sklearn.neighbors.KernelDensity object
        The trained KDE object
    threshold : float, optional
        The density threshold below which is considered "low"
    mc_samples : np.ndarray, optional
        The Monte Carlo samples to use 
    """

    # Evaluate the KDE at each sample point
    densities = np.exp(kde.score_samples(mc_samples))
    max_density = densities.max()
    densities_scaled = densities / max_density if max_density > 0 else 0

    # Smooth indicator function for density < threshold
    low_density_mask = smooth_lower_bound_indicator(densities_scaled, threshold)

    # Estimate the ratio of low-density volume
    # n_low_density_samples / n_samples
    low_density_ratio = np.mean(low_density_mask)

    return low_density_ratio


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
