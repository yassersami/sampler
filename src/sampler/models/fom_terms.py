from typing import List, Tuple, Dict, Union, Optional
import warnings
import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy.stats import norm

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import RandomizedSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RationalQuadratic, RBF
from sklearn.cluster import MeanShift
from scipy.optimize import shgo


RANDOM_STATE = 42


class SurrogateGPR(GaussianProcessRegressor):
    def __init__(
        self, 
        features: List[str], 
        targets: List[str],
        interest_region: Dict[str, Tuple[float, float]],
        kernel=None,
        random_state=RANDOM_STATE,
        **kwargs
    ):
        if kernel is None:
            kernel = RationalQuadratic(length_scale_bounds=(1e-5, 2))
        
        super().__init__(kernel=kernel, random_state=random_state, **kwargs)

        self.features = features
        self.targets = targets
        self.interest_region = interest_region
        
        self.lowers = [region[0] for region in interest_region.values()]
        self.uppers = [region[1] for region in interest_region.values()]

        # To normalize std of coverage function to be between 0 and 1
        self.max_std = None
    
    def predict_interest_proba(self, x: np.ndarray) -> np.ndarray:
        '''
        Computes the probability of being in the region of interest.
        CDF: cumulative distribution function P(X <= x)
        '''
        x = np.atleast_2d(x)

        y_hat, y_std = self.predict(x, return_std=True)

        point_norm = norm(loc=y_hat, scale=y_std)

        probabilities = point_norm.cdf(self.uppers) - point_norm.cdf(self.lowers)
        
        if y_hat.ndim == 1:
            return probabilities
        return np.prod(probabilities, axis=1)

    def get_std(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the combined standard deviation for a GP regressor with multiple
        targets.

        Note: In previous implementations, the mean of the Gaussian Process (GP)
        standard deviations was computed for the Figure of Merit (FOM). However,
        the maximum standard deviation was determined using only the first
        target's standard deviation (y_std[:, 1]). Then the max_std was computed
        using max of stadard deviations, but this approach was still lacking
        consistency. This function unifies the approach by consistently
        combining the standard deviations for both the determination of the
        maximum standard deviation and the application of FOM constraints.
        """
        # Predict standard deviations for each target
        _, y_std = self.predict(x, return_std=True)

        # Combine standard deviations using the mean
        y_std_combined = y_std.mean(axis=1)

        return y_std_combined

    def get_norm_std(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        if self.max_std is None:
            raise RuntimeError("max_std must be updated before predicting scores.")
        return self.get_std(x) / self.max_std
    
    def update_max_std(self, shgo_iters: int = 5, shgo_n: int = 1000):
        """
        Update maximum standard deviation of the Gaussian Process for points
        between 0 and 1.
        """
        print(
            "SurrogateGPR.update_max_std -> Searching for the maximum standard "
            "deviation of the surrogate..."
        )
        search_error = 0.01

        def get_opposite_std(x):
            """Opposite of std to be minimized."""
            x = np.atleast_2d(x)
            y_std = self.get_std(x)
            return -1 * y_std

        bounds = [(0, 1)]*len(self.features)
        result = shgo(
            get_opposite_std, bounds, iters=shgo_iters, n=shgo_n, sampling_method='simplicial'
        )

        max_std = -1 * result.fun
        max_std = min(1.0, max_std * (1 + search_error))
        self.max_std = max_std

        print(f"SurrogateGPR.update_max_std -> Maximum GP std: {max_std}")


def compute_sigmoid_local_density(
    x: np.ndarray, dataset_points: np.ndarray, decay_dist: float=0.04
) -> np.ndarray:
    """
    Compute the decay score and apply a sigmoid transformation to obtain a 
    density-like value.

    Parameters:
    - decay_dist (delta): A float representing the characteristic distance at 
    which the decay effect becomes significant. This parameter controls the 
    rate at which the influence of a reference point decreases with distance. 
    Lower values allow for the inclusion of farther points, effectively 
    shifting the sigmoid curve horizontally. Note that for x_half = delta * 
    np.log(2), we have np.exp(-x_half/delta) = 1/2, indicating the distance 
    at which the decay effect reduces to half its initial value.
    
    Note: an efficient decay with sigmoid is at decay_dist = 0.04

    Returns:
    - A float representing the transformed score after applying the decay and 
    sigmoid functions.

    Explanation:
    - Sigmoid Effect: The sigmoid function is applied to the decay score to 
    compress it into a range between 0 and 1. This transformation makes the 
    score resemble a density measure, where values close to 1 indicate a 
    densely populated area.
    - Sigmoid Parameters: The sigmoid position and sigmoid speed are fixed at 5
    and 1, respectively. The sigmoid_pos determines the midpoint of the sigmoid
    curve, while the sigmoid_speed controls its steepness. In our context, these
    parameters do not have significantly different effects compared to
    decay_dist, hence they are fixed values. Also adequate sigmoid parameters
    are very hard to find.
    """
    x = np.atleast_2d(x)

    # Compute for each x_row distances to every dataset point
    # Element at position [i, j] is d(x_i, dataset_points_j)
    distances = distance.cdist(x, dataset_points, metric="euclidean")

    # Compute the decay score weights for all distances
    # decay score =0 for big distance, =1 for small distance
    decay_scores = np.exp(-distances / decay_dist)

    # Sum the decay scores across each row (for each point)
    # At each row the sum is supported mainly by closer points having greater effect
    cumulated_scores = decay_scores.sum(axis=1)

    # Fix sigmoid parameters
    pos = 5
    speed = 1

    # Apply sigmoid to combined scores to get scores in [0, 1]
    transformed_decays = 1 / (1 + np.exp(-(cumulated_scores - pos) * speed))

    return transformed_decays


class KDEModel:
    def __init__(self, kernel='gaussian'):
        self.kernel = kernel
        self.kde = None
        self.data = None
        self.bandwidth = None
        self.max_density = None

    def search_bandwidth(self, X, log_bounds=(-4, 0), n_iter=20, cv=5, random_state=0):
        # Define a range of bandwidths to search over using log scale
        bandwidths = np.logspace(log_bounds[0], log_bounds[1], 100)

        # Use RandomizedSearchCV to find the best bandwidth
        random_search = RandomizedSearchCV(
            KernelDensity(kernel=self.kernel),
            {'bandwidth': bandwidths},
            n_iter=n_iter,
            cv=cv,
            random_state=random_state,
            # scoring=None => use KDE.score which is log-likelihood
        )
        random_search.fit(X)

        # Get best bandwidth
        best_bandwidth = random_search.best_params_['bandwidth']
        print(f"KDEModel.search_bandwidth -> Optimal bandwidth found: {best_bandwidth}")
    
        return best_bandwidth

    def fit(self, X: np.ndarray, bandwidth: Optional[float] = None, **kwargs):
        # Store the training data
        self.data = X

        # Use the specified bandwidth or the best one found
        if bandwidth is None:
            bandwidth = self.search_bandwidth(X, **kwargs)
        self.bandwidth = bandwidth
        
        # Fit the KDE model
        self.kde = KernelDensity(kernel=self.kernel, bandwidth=bandwidth)
        self.kde.fit(X)

    def update_max_density(self, use_mean_shift=True):
        if self.kde is None:
            raise RuntimeError("Model must be fitted before searching for max density.")
        
        if use_mean_shift:
            mean_shift = MeanShift(bandwidth=self.bandwidth)
            mean_shift.fit(self.data)
            
            # Get modes (highest density points)
            modes = mean_shift.cluster_centers_
            
            # Get maximum of modes densities
            self.max_density = self.predict_proba(modes).max()
        else:
            # Calculate max density over the training data
            densities = self.predict_proba(self.data)
            self.max_density = np.max(densities)

    def predict_proba(self, X):
        if self.kde is None:
            raise RuntimeError("Model must be fitted before predicting.")
        log_density = self.kde.score_samples(X)
        return np.exp(log_density)

    def predict_score(self, X):
        X = np.atleast_2d(X)
        if self.max_density is None:
            raise RuntimeError("max_density must be updated before predicting scores.")
        densities = self.predict_proba(X)
        return densities / self.max_density


class OutlierExcluder:
    def __init__(
        self, features: List[str], targets: List[str]
    ):
        
        self.features = features
        self.targets = targets
        # Points to be avoided as they result in erroneous simulations
        self.ignored_df = pd.DataFrame(columns=self.features)
    
    def update_outliers_set(self, df: pd.DataFrame):
        """ Add bad points (in feature space) to the set of ignored points. """
        # Identify rows where any target column has NaN
        ignored_rows = df[df[self.targets].isna().any(axis=1)]

        # Extract feature values from these rows
        new_ignored_df = ignored_rows[self.features]

        # Concatenate the new ignored points with the existing ones
        self.ignored_df = pd.concat([self.ignored_df, new_ignored_df]).drop_duplicates()
        self.ignored_df = self.ignored_df.reset_index(drop=True)

    def detect_outlier_proximity(
        self, x: np.ndarray, exclusion_radius: float = 1e-5
    ) -> np.ndarray:
        """
        Proximity Exclusion Condition: Determine if the given point is
        sufficiently distant from any point with erroneous simulations. Points
        located within a specified exclusion radius around known problematic
        points are excluded from further processing.
        
        This condition is necessary because surrogate GP does not update around
        failed samples.
        """
        x = np.atleast_2d(x)
        
        if self.ignored_df.empty:
            return np.zeros(x.shape[0], dtype=bool)
    
        # Compute distances based on the specified metric
        distances = distance.cdist(x, self.ignored_df.values, "euclidean")

        # Determine which points should be ignored based on tolerance
        should_ignore = np.any(distances < exclusion_radius, axis=1)

        # Log ignored points
        for i, ignore in enumerate(should_ignore):
            if ignore:
                print(f"OutlierProximityHandler.should_ignore -> Point {x[i]} was ignored.")

        return should_ignore


class InlierOutlierGPC(GaussianProcessClassifier):
    def __init__(self, kernel=None, random_state=None):
        if kernel is None:
            kernel = 1.0 * RBF(length_scale=1.0)
        super().__init__(kernel=kernel, random_state=random_state)
        self.is_trained = False

        # Map class labels to indices
        self.class_to_index = {"outlier": 0, "inlier": 1}
        self.index_to_class = {
            index: label for label, index in self.class_to_index.items()
        }

    def fit(self, X, y):
        # Determine inlier or outlier status based on NaN presence in y
        y = np.where(
            np.isnan(y).any(axis=1),
            self.class_to_index["outlier"],  # 0
            self.class_to_index["inlier"]   # 1
        )
        # Check if there are two classes in y
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            self.is_trained = False
            warnings.warn("No outliers have been detected yet. The model will not be fitted.")
            return self
        else:
            self.is_trained = True
            # Fit the Gaussian Process Classifier with the modified target values
            super().fit(X, y)
        return self
    
    def get_inlier_bstd(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate Bernoulli Standard Deviation weighed by the inlier proba for
        each input sample.
        
        This function computes the score using the formula:
        
                f(P) = alpha * P * sqrt(P*(1-P))
    
        Where P = P(x is inlier) the probability that a sample is an inlier. The
        term sqrt(P*(1-P)) quantifies the predication uncertainty in the
        predicted class outcome (0, 1), not the uncertainty provided by latent
        function values (logit space). And alpha is a scaling factor such that
        the maximum value of f over the interval [0, 1] is 1.
        With x_max = argmax[0, 1](f) = 3/4.
        """
        X = np.atleast_2d(X)
        
        # Return ones (best value) if the model is not trained as no outliers are encountered yet
        if not self.is_trained:
            return np.ones(X.shape[0])

        # Set scaling factor alpha
        alpha = ( (3/4)**3 * (1/4) )**(-0.5)

        # Predict inlier probabilities
        proba = self.predict_proba(X)[:, self.class_to_index["inlier"]]

        # Compute the score using the given formula
        score = alpha * proba * np.sqrt(proba * (1 - proba))

        return score
