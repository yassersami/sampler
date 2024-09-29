from typing import List, Dict, Tuple, Optional, Union
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import norm
from scipy.optimize import brentq


class MixedMinMaxScaler:
    def __init__(
            self,
            features: List[str],
            targets: List[str],
            scale: Optional[Dict[str, str]] = None,
    ):
        self.features = features
        self.targets = targets
        self.all_vars = features + targets

        if scale is not None:
            # Check if all variables are present in the scale dict
            missing_vars = set(self.all_vars) - set(scale.keys())
            if missing_vars:
                raise ValueError(f"Missing scale type for variables: {missing_vars}")
            
            # Check if there are any extra variables in scale dict
            extra_vars = set(scale.keys()) - set(self.all_vars)
            if extra_vars:
                raise ValueError(f"Unexpected variables in scale dict: {extra_vars}")

            # Check if all scale values are valid
            invalid_scales = {
                var: scale_type for var, scale_type in scale.items()
                if scale_type not in ['lin', 'log']
            }
            if invalid_scales:
                raise ValueError(
                    f"Invalid scale types: {invalid_scales}. \n"
                    "Only 'lin' and 'log' are allowed."
                )

            self.scale = scale
        else:
            self.scale = {var: 'lin' for var in self.all_vars}

        # Create boolean arrays for linear and logarithmic variables
        self.lin_vars = np.array([self.scale[var] == 'lin' for var in self.all_vars])
        self.log_vars = np.array([self.scale[var] == 'log' for var in self.all_vars])

        self.lin_on = self.log_on = False
        if self.lin_vars.any():
            self.lin_on = True
            self.scaler_lin = MinMaxScaler()
        if self.log_vars.any():
            self.log_on = True
            self.scaler_log = MinMaxScaler()

    def fit(self, X):
        if self.lin_on:
            self.scaler_lin.fit(X[:, self.lin_vars])
        if self.log_on:
            self.scaler_log.fit(np.log(X[:, self.log_vars]))

    def transform(self, X: np.ndarray):
        if X.shape[0] == 0:
            return X
        if self.lin_on:
            x_lin_norm = self.scaler_lin.transform(X[:, self.lin_vars])
        else:
            x_lin_norm = np.empty(len(X))
        if self.log_on:
            # x_log_norm = self.scaler_log.transform(np.log(X[:, self.log_vars]))
            X_log_safe = np.copy(X[:, self.log_vars])
            X_log_safe[X_log_safe <= 0] = 1e-10
            x_log_norm = self.scaler_log.transform(np.log(X_log_safe))
        else:
            x_log_norm = np.empty(len(X))
        return self._merge(x_lin_norm, x_log_norm)

    def transform_with_nans(self, X_nan: np.ndarray) -> np.ndarray:
        """ Transform handling NaNs by filling temporarily with zeroes"""
        cols = self.features + self.targets
        df = pd.DataFrame(X_nan, columns=cols)
        nan_mask = df.isna()
        df = df.fillna(0)  # 0 is a dummy value
        df[cols] = self.transform(df.values)
        df = df.mask(nan_mask)
        return df.values
    
    def transform_features(self, X_feat: np.ndarray):
        """ Transforms a features only array filling the rest with ones """
        ones = np.ones((X_feat.shape[0], len(self.targets)))
        X = np.concatenate((X_feat, ones), axis=1)
        # Return only the first len(features) columns
        return self.transform(X)[:, :len(self.features)]

    def transform_targets(self, X_tar: np.ndarray):
        """ Transforms a targets only array filling the rest with ones """
        ones = np.ones((X_tar.shape[0], len(self.features)))
        X = np.concatenate((ones, X_tar), axis=1)
        # Return only the last len(targets) columns
        return self.transform(X)[:, -len(self.targets):]

    def inverse_transform(self, X: np.ndarray):
        if self.lin_on:
            x_lin = self.scaler_lin.inverse_transform(X[:, self.lin_vars])
        else:
            x_lin = np.empty(X.shape[0])
        if self.log_on:
            x_log = self.scaler_log.inverse_transform(X[:, self.log_vars])
            x_log = np.exp(x_log)
        else:
            x_log = np.empty(X.shape[0])
        return self._merge(x_lin, x_log)

    def _merge(self, x_lin_norm, x_log_norm):
        dim_col = len(self.lin_vars + self.log_vars)
        dim_row = len(x_log_norm) if self.log_on else len(x_lin_norm)
        x = np.ones((dim_row, dim_col))
        j_lin = j_log = 0
        for j in range(dim_col):
            if self.lin_vars[j]:
                x[:, j] = x_lin_norm[:, j_lin]
                j_lin += 1
            elif self.log_vars[j]:
                x[:, j] = x_log_norm[:, j_log]
                j_log += 1
        return x


def scale_interest_region(
    interest_region: Dict[str, Tuple[float, float]],
    scaler: MixedMinMaxScaler
) -> Dict[str, Tuple[float, float]]:
    """ Scale values of the region of interest"""

    lowers = [region[0] for region in interest_region.values()]
    uppers = [region[1] for region in interest_region.values()]
    scaled_bounds = scaler.transform_targets(X_tar=np.array([lowers, uppers]))

    scaled_interest_region = {
        key: list(scaled_bounds[:, i])
        for i, key in enumerate(interest_region.keys())
    }

    for key, scaled_values in scaled_interest_region.items():
        assert all(0 <= val <= 1 for val in scaled_values), (
            f"Error! Region bounds {scaled_values} for key '{key}' not in range [0, 1]!"
            "\n prep.json must contain absolute bounds! (ie. all data and interest "
            "regions values must be INSIDE those bounds)"
        )

    return scaled_interest_region


def set_scaler(
    features: List[str],
    targets: List[str],
    ranges: Dict[str, Dict[str, Union[float, str]]],
) -> MixedMinMaxScaler:
    scaler_variables = features + targets

    # Prepare input-output space description
    bounds_dict = {var_name: ranges[var_name]['bounds'] for var_name in scaler_variables}
    scale = {var_name: ranges[var_name]['scale'] for var_name in scaler_variables}

    # Fit scaler to scale all values from 0 to 1
    scaler = MixedMinMaxScaler(features=features, targets=targets, scale=scale)
    bounds = np.array(list(bounds_dict.values())).T  # dict.values() keeps order
    scaler.fit(bounds)

    return scaler


def linear_tent(
    x: np.ndarray, L: np.ndarray, U: np.ndarray, slope: float=1.0
) -> np.ndarray:
    """
    Tent function equal to 1 on interval [L, U],
    and decreasing linearly outside in both directions.

    x: shape (n, p)
    L and U: float or shape (1, p) if p > 1

    test with:
    L = array([[0.8003412 , 0.89822933]])
    U = array([[0.85116726, 0.97268397]])
    x = np.array([[0, 0], [0.8, 0.8], [0.85, 0.85], [0.9, 0.9], [1, 1]])

    Output
    ------
    y: shaped(n, p)
    """
    x = np.atleast_2d(x)
    if np.any(L >= U):
        raise ValueError(f"L should be less than U \nL: \n{L} \nU: \n{U}")

    center = (U+L)/2  # Center of interval
    half_width = (U-L)/2  # Half interval width
    dist_from_center = np.abs(x - center)  # >= 0
    # x_dist is distance from interval: =0 inside [L, U] and >0 outside
    x_dist = np.max([dist_from_center - half_width, np.zeros_like(x)], axis=0)

    y = -slope*x_dist + 1

    return y


def hypercube_linear_tent(X: np.ndarray, interest_region: List[Tuple[float, float]]) -> np.ndarray:
    """
    Hypercube tent function:
    - X on hypercube boundaries      -> y = 0 
    - X in [L, U] for each dimension -> y = 1 
    - X elsewhere                    -> y = linear

    Parameters:
    X: shape (n, p) or (p,), values in [0, 1]^p
    interest_region: list of tuples of (lower, upper)

    Returns:
    y: shape (n,) or scalar
    """
    X = np.atleast_2d(X.T).T

    y = np.ones(X.shape[0])  # Initialize scores for each sample

    for i, (lower, upper) in enumerate(interest_region):
        # Calculate linear decrease from L to 0
        mask_lower = X[:, i] < lower
        y[mask_lower] *= X[mask_lower, i] / lower

        # Calculate linear decrease from U to 1
        mask_upper = X[:, i] > upper
        y[mask_upper] *= (1 - X[mask_upper, i]) / (1 - upper)

    return y


def hypercube_exponential_tent(X: np.ndarray, interest_region: List[Tuple[float, float]]) -> np.ndarray:
    """
    Hypercube exponential tent function:
    - X in interest region -> y = 1 
    - X elsewhere          -> y = decreases exponentially from 1 to 0

    Parameters:
    X: shape (n, p) or (p,), values in [0, 1]^p
    interest_region: list of tuples of (lower, upper)

    Returns:
    y: shape (n,) or scalar
    """
    X = np.atleast_2d(X.T).T

    # Set decay dist as half the width of the interest region
    decay_dist = np.array([(upper - lower)/2 for lower, upper in interest_region])

    scores = np.ones(X.shape[0])  # Initialize scores for each sample

    for i, (low, high) in enumerate(interest_region):
        mask_lower = X[:, i] < low
        mask_upper = X[:, i] > high
        
        # Calculate distances for values outside the interest region
        distances = np.zeros(X.shape[0])
        distances[mask_lower] = low - X[mask_lower, i]
        distances[mask_upper] = X[mask_upper, i] - high
        
        # Apply exponential decay
        scores *= np.exp(-distances / decay_dist[i])
    
    return scores


def find_sigma_for_interval(interval_width: float, sigma_bounds: Tuple[float, float] =(0.005, 0.15)) -> float:
    """
    Find the largest standard deviation (sigma) that makes the probability of
    belonging to a given interval (based on its width) equal to one for a normal
    distribution.

    This function uses the Brent's method to efficiently find the optimal sigma
    value that satisfies the target probability condition.

    Parameters:
    -----------
    interval_width : float
        The width of the interval of interest. Should be a positive value between 0 and 1.

    sigma_bounds : Tuple[float, float], optional
        The lower and upper bounds for the sigma search range. Default is (0.005, 0.15).
        - Lower bound (0.005) is suitable for interval widths as small as 0.05.
        - Upper bound (0.15) is suitable for interval widths up to 1.
    """
    half_width = interval_width / 2
    target_prob = 0.999  # to avoid saturation around 1 of objective function
    mu = 0  # Center of the interval

    def objective(sigma):
        proba = norm.cdf(half_width, loc=mu, scale=sigma) - norm.cdf(-half_width, loc=mu, scale=sigma)
        return proba - target_prob

    a, b = sigma_bounds
    fa, fb = objective(a), objective(b)

    # Check if fa and b have same signs
    if np.sign(fa) == np.sign(fb):
        warnings.warn(
            f"Objective function does not change sign in interval [{a}, {b}]. "
            "The method will return sigma that generates closest probability to 1."
        )
        # ...[a.....b]... if proba when sigma = b is still < 1, return b
        return b if fb <= 0 else a

    # Use brentq to find the root
    sigma = brentq(objective, a, b, xtol=1e-2)
    return sigma


def interest_probability(y_mean: np.ndarray, y_std: np.ndarray, interest_region: List[Tuple[float, float]]) -> np.ndarray:
    """
    Computes the probability of being in the region of interest for multiple dimensions.

    Parameters:
    -----------
    y_mean : np.ndarray
        2D array of mean values. Shape: (n_samples, n_dimensions)
    y_std : np.ndarray
        2D array of standard deviation values. Shape: (n_samples, n_dimensions)
    interest_region : List[Tuple[float, float]]
        List of tuples defining the lower and upper bounds of the interest region for each dimension.
        
    Returns:
    --------
    np.ndarray
        1D array of probabilities for each sample being in the region of interest across all dimensions.
    """
    y_mean = np.atleast_2d(y_mean.T).T
    y_std = np.atleast_2d(y_std.T).T
    
    if y_mean.shape != y_std.shape:
        raise ValueError("y_mean and y_std must have the same shape")
    
    if len(interest_region) != y_mean.shape[1]:
        raise ValueError("Number of interest regions must match the number of dimensions in y_mean and y_std")
    
    n_samples, n_dimensions = y_mean.shape
    
    probabilities = np.ones((n_samples, n_dimensions))
    
    for i, (lower, upper) in enumerate(interest_region):
        # CDF: cumulative distribution function P(X <= x)
        probabilities[:, i] = (
            norm.cdf(upper, loc=y_mean[:, i], scale=y_std[:, i]) - 
            norm.cdf(lower, loc=y_mean[:, i], scale=y_std[:, i])
        )
    
    return np.prod(probabilities, axis=1)