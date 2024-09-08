from typing import List, Tuple, Dict, Callable, Optional
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import shgo
from scipy.spatial.distance import cdist

from .scalers import MixedMinMaxScaler


def linear_interest_mapping(
    y_raw: np.ndarray,
    interest_bounds: List[Tuple[float, float]], 
    y_raw_center: float = 0.5,
    y_raw_bandwidth: float = 0.15
) -> np.ndarray:
    """
    Applies a piecewise linear mapping to raw values based on interest bounds.

    This function maps the input values as follows:
    - y_raw = 0            -> y = 0
    - y_raw = y_raw_lower  -> y = lower bound of interest region
    - y_raw = y_raw_center -> y = center of interest region
    - y_raw = y_raw_upper  -> y = upper bound of interest region
    - y_raw = 1            -> y = 1

    Args:
        y_raw (np.ndarray): Input array of shape (n_samples,) in range [0, 1].
        interest_bounds (List[Tuple[float, float]]): Min and max interest values for each output dimension.
        y_raw_center (float): Center of the interest region in y_raw space. Default is 0.5.
        y_raw_bandwidth (float): Width of the interest region in y_raw space. Default is 0.1.

    Returns:
        np.ndarray: Mapped output array y of shape (n_samples, n_dim_y) in range [0, 1].
    """
    # Compute lower and upper bounds in y_raw space
    y_raw_lower = y_raw_center - y_raw_bandwidth / 2
    y_raw_upper = y_raw_center + y_raw_bandwidth / 2

    y_raw = np.atleast_1d(y_raw)  # Ensure y_raw is at least 1D
    n_samples = y_raw.shape[0]
    n_dim_y = len(interest_bounds)  # Get dimension of target space

    # Create masks for each piece of the piecewise function
    mask_lower = y_raw <= y_raw_lower
    mask_lower_center = (y_raw > y_raw_lower) & (y_raw <= y_raw_center)
    mask_center_upper = (y_raw > y_raw_center) & (y_raw <= y_raw_upper)
    mask_upper = y_raw > y_raw_upper

    # Calculate bounds and center of interest region
    interest_bounds = np.array(interest_bounds)
    interest_lower = interest_bounds[:, 0]
    interest_upper = interest_bounds[:, 1]
    interest_centers = np.mean(interest_bounds, axis=1)

    # Initialize output array
    y = np.zeros((n_samples, n_dim_y))

    # Piecewise linear mapping
    for i in range(n_dim_y):
        # 0 to lower bound
        y[mask_lower, i] = (interest_lower[i] / y_raw_lower) * y_raw[mask_lower]

        # Lower bound to center
        y[mask_lower_center, i] = interest_lower[i] + (interest_centers[i] - interest_lower[i]) * (y_raw[mask_lower_center] - y_raw_lower) / (y_raw_center - y_raw_lower)

        # Center to upper bound
        y[mask_center_upper, i] = interest_centers[i] + (interest_upper[i] - interest_centers[i]) * (y_raw[mask_center_upper] - y_raw_center) / (y_raw_upper - y_raw_center)

        # Upper bound to 1
        y[mask_upper, i] = interest_upper[i] + (1 - interest_upper[i]) * (y_raw[mask_upper] - y_raw_upper) / (1 - y_raw_upper)

    return y


def spherical_shell_interest_function(X: np.ndarray, interest_bounds: List[Tuple[float, float]]) -> np.ndarray:
    """
    Creates a spherical shell-like interest region in the design space.

    The interest values are calculated as follows:

    1. Calculate the normalized distance from each design space point to the
       center of the hypercube (0 at center, 1 at corners).
    2. Maps these normalized distances to output space using a piecewise
       linear function.
    3. The resulting interest region forms a spherical shell pattern in the
       design space.
    """
    X = np.atleast_2d(X)
    n_dim_X = X.shape[1]

    # Define the center of the hypercube
    hypercube_center = np.full(X.shape, 0.5)

    # Calculate maximal distance from center (half of hypercube diagonal)
    max_dist = np.sqrt(n_dim_X) / 2

    # Compute normalized distances from center (0 at center, 1 at corners)
    X_dist = np.linalg.norm(X - hypercube_center, axis=1).ravel() / max_dist

    # Compute interest values using piecewise linear function
    # Inner shell: linear increase from 0 to interest_center
    # Outer shell: linear increase from interest_centers to 1
    y = linear_interest_mapping(X_dist, interest_bounds)

    return y


def shubert_like_interest_function(X: np.ndarray, interest_bounds: List[Tuple[float, float]]) -> np.ndarray:
    """
    This function creates a complex, multi-modal landscape based on the Shubert
    function, and then maps it to a specified interest region. It's designed to
    create a challenging optimization landscape with multiple local optima.
    
    The function performs the following steps:
    1. Transforms the input space from [0, 1]^n_dim_X to [-1, 1]^n_dim_X.
    2. Computes a Shubert-like function for each input dimension.
    3. Averages these values across dimensions to get a single value per sample
       in [0, 1].
    4. Maps these raw values to the specified interest region using a linear
       transformation to get to [0, 1]^n_dim_y.
    """
    X = np.atleast_2d(X)
    n_samples, n_dim_X = X.shape
    
    # Transform [0, 1]^n_dim_X into [-1, 1]^n_dim_X
    X_transformed = 2 * (X - 0.5)

    # Convert interest bounds to numpy array and calculate center of interest for each dimension
    interest_bounds = np.array(interest_bounds)
    n_dim_y = len(interest_bounds)
    interest_centers = np.mean(interest_bounds, axis=1)

    # Angular frequency (omega)
    omega = 10 * np.pi

    # Compute Shubert-like function for each input dimension
    shubert_terms = 0.5 * ((1 - np.abs(X_transformed)) * np.cos(omega * X_transformed**2) + 1)
    y_raw = np.mean(shubert_terms, axis=1)

    # Apply linear mapping to interest region
    y = linear_interest_mapping(y_raw, interest_bounds)

    return y


def proxy_function(X: np.ndarray, interest_bounds: List[Tuple[float, float]]) -> np.ndarray:
    return spherical_shell_interest_function(X, interest_bounds)  # easy mode
    # return shubert_like_interest_function(X, interest_bounds)  # hard mode

class FastSimulator:
    """
    A proxy fast simulator that uses a `proxy_function` class method to compute
    a numerical function mapping input space to output space. 

    Key features:
    1. Part of its response curve lies within a specified interest region.
    2. Searches for and stores an interest region sample to ensure at least one
       known point.
    3. Incorporates artificial outliers to complexify its behavior and challenge
       the adaptive sampling process.
    4. Always includes at least one outlier, set at the origin of the scaled
       design space [0, 1]^n_dim_X.

    This simulator is designed to provide a fast, controllable environment for
    testing and developing adaptive sampling strategies, offering a balance
    between realistic behavior and computational efficiency.
    """

    def __init__(
        self,
        features: List[str],
        targets: List[str],
        additional_values: List[str],
        interest_region: Dict[str, Tuple[float, float]],
        n_outlier_regions: int = 1,
        outlier_radius: float = 0.05
    ):
        self.features = features
        self.targets = targets
        self.additional_values = additional_values
        self.n_dim_X = len(features)
        self.n_dim_y = len(targets)
        self.interest_region = np.array(list(interest_region.values()))
        self.outlier_radius = outlier_radius
        self.X_interest = self.find_interest_sample()
        self.X_outliers = self.set_outlier_regions_centers(n_outlier_regions)
        self.remove_close_outliers()

    def proxy_function(self, X: np.ndarray) -> np.ndarray:
        return proxy_function(X, self.interest_region)

    def find_interest_sample(self) -> np.ndarray:
        """
        Search for an interest sample where y is in the interest_region using
        SHGO algorithm.
        """
        print(f"{self.__class__.__name__} - > Searching for interest sample...")

        # Define objective function for SHGO
        lower_bounds, upper_bounds = self.interest_region.T

        def objective(x: np.ndarray) -> float:
            """
            Objective function for SHGO to minimize.
            
            Calculates the squared deviation from the interest region.
            Returns 0 if y is within the region, positive penalty otherwise.
            numpy.maximum compares two arrays and return a new array containing the
            element-wise maxima, like np.max([...], axis=1).
            """
            y = self.proxy_function(x.reshape(1, -1)).ravel()
            penalties = np.maximum(lower_bounds - y, 0)**2 + np.maximum(y - upper_bounds, 0)**2
            return np.sum(penalties)

        # Define bounds for each input dimension, avoiding the origin outlier region
        bounds = [(self.outlier_radius, 1)] * self.n_dim_X

        # Run SHGO optimization
        result = shgo(objective, bounds, n=1000, iters=5)

        if result.fun == 0:
            # If a valid solution is found (objective function is zero)
            return result.x.reshape(1, -1)
        else:
            raise ValueError("Could not find an interest sample within the specified region.")

    def set_outlier_regions_centers(self, n_outlier_regions: int):

        # First outlier region at the origin
        X_outliers = np.zeros((1, self.n_dim_X))

        # Add more outlier regions if needed
        if n_outlier_regions > 1:
            additional_outliers = np.random.rand(n_outlier_regions - 1,self.n_dim_X)
            X_outliers = np.vstack([X_outliers, additional_outliers])

        return X_outliers

    def remove_close_outliers(self) -> np.ndarray:
        """
        Remove outliers that are too close to the X_interest sample.
        """
        # Calculate vectors from X_interest to outliers
        vectors_to_outliers = self.X_outliers - self.X_interest

        # Calculate distances from X_interest to outliers
        distances = np.linalg.norm(vectors_to_outliers, axis=1)

        # Create a mask for outliers that are too close
        mask_too_close = distances < self.outlier_radius

        # Count how many outliers will be removed
        num_removed = np.sum(mask_too_close)

        if num_removed > 0:
            warnings.warn(f"{num_removed} outlier(s) removed due to proximity to X_interest.")

        # Return only the outliers that are not too close
        return self.X_outliers[~mask_too_close]


    def proxy_function_with_outliers(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Compute y_scaled values for given X_scaled, handling outliers.

        Args:
            X_scaled (np.ndarray): Scaled input array.

        Returns:
            np.ndarray: y_scaled values with NaNs for outliers.
        """
        n_samples, _ = X_scaled.shape

        # Initialize y_scaled with NaNs
        y_scaled = np.full((n_samples, self.n_dim_y), np.nan)

        # If there are outlier regions, compute distances and mask
        if self.X_outliers.size > 0:
            distances = cdist(X_scaled, self.X_outliers)
            non_outlier_mask = ~np.any(distances <= self.outlier_radius, axis=1)
        else:
            non_outlier_mask = np.ones(n_samples, dtype=bool)

        # Compute y values only for non-outlier points
        y_scaled[non_outlier_mask] = self.proxy_function(X_scaled[non_outlier_mask])

        return y_scaled

    def run_fast_simulation(self, X_real: np.ndarray, scaler: MixedMinMaxScaler) -> pd.DataFrame:
        """
        Run a fast simulation using a numerical function for quicker testing.

        Args:
            X_real (np.ndarray): Input array in real space.

        Returns:
            pd.DataFrame: Results dataframe with features, targets, and additional values.
        """
        X_scaled = scaler.transform_features(X_real)

        # Compute y_scaled values
        y_scaled = self.proxy_function_with_outliers(X_scaled)

        # Get targets in real space (not in scaled one)
        XY_real = scaler.inverse_transform(np.hstack([X_scaled, y_scaled]))

        df_results = pd.DataFrame(XY_real[:, -self.n_dim_y:], columns=self.targets)
        df_results[self.additional_values] = np.nan

        return df_results

    def append_spicy_data(self, data: pd.DataFrame, max_simu_time: int) -> pd.DataFrame:
        """
        Add outlier data to test outlier cleaning process and interest data to help
        FOM terms that needs interest samples to start.

        Args:
            data (pd.DataFrame): Input data in scaled space.
            max_simu_time (int): Maximum simulation time.

        Returns:
            pd.DataFrame: Data with appended spicy samples.
        """
        # Set outlier target values
        y_outliers = np.array([
            [0.5] * (self.n_dim_y-1) + [1.1],  # target out of bounds
            [0.5] * (self.n_dim_y-1) + [np.nan],  # failed simulation causing error (missing value)
            [0.5] * self.n_dim_y,  # no_interest sample but timed out
        ])
        sim_time = np.array([[0], [0], [max_simu_time]])

        # Use the first outlier region (at origin) for spicy data
        outlier_region = self.X_outliers[0]
        X_outliers = outlier_region + np.random.uniform(-self.outlier_radius, self.outlier_radius, size=(y_outliers.shape[0], self.n_dim_X))
        X_outliers = np.clip(X_outliers, 0, 1)

        # Create a DataFrame with the new data
        df_outliers = pd.DataFrame(
            np.hstack([X_outliers, y_outliers, sim_time]), 
            columns=self.features + self.targets + ['sim_time']
        )

        # Add an interest sample
        y_interest = self.proxy_function(self.X_interest)
        df_interest = pd.DataFrame(
            np.hstack([self.X_interest, y_interest]), 
            columns=self.features + self.targets
        )

        # Append new data to existing DataFrame
        return pd.concat([data, df_outliers, df_interest], ignore_index=True)
