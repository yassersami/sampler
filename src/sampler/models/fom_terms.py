from typing import List, Tuple, Dict, Union
import warnings
import numpy as np
import pandas as pd
from scipy.spatial import distance

from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RationalQuadratic, RBF
from scipy.optimize import shgo


RANDOM_STATE = 42


class SurrogateGP:
    def __init__(self, features: List[str], targets: List[str]):

        self.features = features
        self.targets = targets
        kernel = RationalQuadratic(length_scale_bounds=(1e-7, 100000))
        self.gp = GaussianProcessRegressor(kernel=kernel, random_state=RANDOM_STATE)

        # To normalize std of coverage function to be between 0 and 1
        self.max_std = 1

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.gp.fit(X_train, y_train)

    def predict(self, x: np.ndarray, return_std=False, return_cov=False):
        return self.gp.predict(x, return_std, return_cov)

    def get_std(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the combined standard deviation for a GP regressor with multiple targets.

        Note: In previous implementations, the mean of the Gaussian Process (GP) standard
        deviations was computed for the Figure of Merit (FOM). However, the maximum
        standard deviation was determined using only the first target's standard
        deviation (y_std[:, 1]). Then the max_std was computed using max of stadard
        deviations, but this approach was still lacking consistency.
        This function unifies the approach by consistently combining the standard deviations
        for both the determination of the maximum standard deviation and the application of
        FOM constraints.
        """
        # Predict standard deviations for each target
        _, y_std = self.gp.predict(x, return_std=True)

        # Combine standard deviations using the mean
        y_std_combined = y_std.mean(axis=1)

        # Alternative: Combine using the maximum standard deviation
        # y_std_combined = y_std.max(axis=1)

        return y_std_combined
    
    def update_max_std(self, shgo_iters: int = 5, shgo_n: int = 1000):
        """ Returns the maximum standard deviation of the Gaussian Process for points between 0 and 1."""

        print("SurrogateGP.update_max_std -> Searching for the maximum standard deviation of the surrogate...")
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

        print(f"SurrogateGP.update_max_std -> Maximum GP std: {max_std}")
        print(f"SurrogateGP.update_max_std -> Training data std per target: {self.gp.y_train_.std(axis=0)}")


def compute_space_local_density(
    x: np.ndarray, points: np.ndarray, decay_dist: float = 0.04
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

    Returns:
    - A float representing the transformed score after applying the decay and 
    sigmoid functions.

    Explanation:
    - Sigmoid Effect: The sigmoid function is applied to the decay score to 
    compress it into a range between 0 and 1. This transformation makes the 
    score resemble a density measure, where values close to 1 indicate a 
    densely populated area.
    - Sigmoid Parameters: The sigmoid position (sigmoid_pos) and sigmoid speed 
    (sigmoid_speed) are fixed at 5 and 1, respectively. The sigmoid_pos 
    determines the midpoint of the sigmoid curve, while the sigmoid_speed 
    controls its steepness. In our context, these parameters do not have 
    significantly different effects compared to decay_dist, hence their 
    fixed values.
    """
    x = np.atleast_2d(x)
    scores = []
    
    # Fix sigmoid parameters
    sigmoid_pos = 5
    sigmoid_speed = 1
        
    for x_row in x:
        x_row = x_row.reshape(1, -1)
        
        # Calculate distances from x_row to each reference point using the specified metric
        distances = distance.cdist(x_row, points, metric="euclidean").flatten()

        # Compute the decay score wheight (=0 for big distance, =1 for small distance)
        decay_scores = np.exp(-distances/decay_dist)
        
        # Sum all the scores (sum is supported mainly by closer points having greater effect)
        combined_scores = decay_scores.sum()

        # Apply the sigmoid transformation
        transformed_decays = 1 / (1 + np.exp(sigmoid_pos - sigmoid_speed * combined_scores))
        
        scores.append(transformed_decays)

    return np.array(scores)


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
        self, x: np.ndarray, threshold: float = 1e-5
    ) -> np.ndarray:
        """
        Proximity Exclusion Condition: Determine if the given point is near any point
        with erroneous simulations. Points located within a very small vicinity of
        lenght threshold around specified points are excluded from further processing.
        
        This condition is necessary because surrogate GP does not update around failed
        samples.
        """
        x = np.atleast_2d(x)
        
        if self.ignored_df.empty:
            return np.zeros(x.shape[0], dtype=bool)
    
        # Compute distances based on the specified metric
        distances = distance.cdist(x, self.ignored_df.values, "euclidean")

        # Determine which points should be ignored based on the distance threshold
        should_ignore = np.any(distances < threshold, axis=1)

        # Log ignored points
        for i, ignore in enumerate(should_ignore):
            if ignore:
                print(f"OutlierProximityHandler.should_ignore -> Point {x[i]} was ignored.")

        return should_ignore


class InlierOutlierGP:
    def __init__(self):
        # Initialize Gaussian Process Classifier with RBF kernel
        self.kernel = 1.0 * RBF(length_scale=1.0)
        self.gp = GaussianProcessClassifier(kernel=self.kernel)
        self.is_trained = False

        # Map class labels to indices
        self.class_to_index = {"outlier": 0, "inlier": 1}
        self.index_to_class = {
            index: label for label, index in self.class_to_index.items()
        }

    def fit(self, X_train, y_train):
        # Determine inlier or outlier status based on NaN presence in y_train
        y = np.where(
            np.isnan(y_train).any(axis=1),
            self.class_to_index["outlier"],  # 0
            self.class_to_index["inlier"]   # 1
        )
        # # Check if there are two classes in y
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            self.is_trained = False
            warnings.warn(f"No outliers have been detected yet")
        else:
            self.is_trained = True
            # Fit the Gaussian Process Classifier with the modified target values
            self.gp.fit(X_train, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """  Predict the class labels for the input data. """
        return self.gp.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """  Predict class probabilities for the input data. """
        return self.gp.predict_proba(X)

    def predict_inlier_proba(self, X: np.ndarray) -> np.ndarray:
        """ Predicts the probability of being an inlier. """
        return self.gp.predict_proba(X)[:, self.class_to_index["inlier"]]

    def get_std(self, X: np.ndarray) -> np.ndarray:
        """
        Compute a measure of uncertainty based on probabilities.
        For binary classification, use the probabilities of the positive class.
        Calculate the standard deviation of the probabilities as a measure of uncertainty.
        y_std_max = sqrt(0.5*(1-0.5)) = 1/2
        """
        # Predict inlier probabilities
        inlier_proba = self.predict_inlier_proba(X)

        # Calculate the standard deviation of the probabilities
        y_std = np.sqrt(inlier_proba * (1 - inlier_proba))
        return y_std
    
    def get_conditional_std(
        self, X: np.ndarray, use_threshold: bool = False, threshold: int = 0.8
    ) -> np.ndarray:
        """
        Calculate the conditional standard deviation score for each input sample.
        
        This function computes the score using the formula:
        
                f(P) = alpha * P * sqrt(P*(1-P))
    
        Where P = P(x is inlier) the probability that a sample is an inlier.
        And alpha is a scaling factor such that the maximum value of f over the
        interval [0, 1] is 1.
        Knowing that x_max = argmax[0, 1](f) = 3/4.
        
        If `use_threshold` is True, the score is set to 0 for samples where the inlier
        probability  P is below the specified threshold.
        """
        # Return zeros if the model is not trained as no outliers are encountered yet
        if not self.is_trained:
            return np.zeros(X.shape[0])

        # Calculate the scaling factor alpha
        alpha = ( (3/4)**3 * (1/4) )**(-0.5)

        # Predict inlier probabilities and standard deviations
        proba = self.predict_inlier_proba(X)
        y_std = self.get_std(X)

        # Compute the score using the given formula
        score = alpha * proba * y_std

        # Sanction points with low inlier probability if thresholding
        if use_threshold:
            low_proba_indices = proba < threshold
            score[low_proba_indices] = 0

        return score
