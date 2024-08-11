from typing import List, Tuple, Dict, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RationalQuadratic, RBF
from scipy.optimize import shgo

RANDOM_STATE = 42


class SurrogateGP:
    def __init__(self, features: List[str], targets: List[str], decimals: int=10):

        self.features = features
        self.targets = targets
        kernel = RationalQuadratic(length_scale_bounds=(1e-7, 100000))
        self.gp = GaussianProcessRegressor(kernel=kernel, random_state=RANDOM_STATE)
        
        # Points to be ignored as they result in erroneous simulations
        self.ignored = set()
        self.decimals = decimals

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.gp.fit(X_train, y_train)

    def predict(self, x: np.ndarray, return_std=False, return_cov=False):
        return self.gp.predict(x, return_std, return_cov)

    def add_ignored_points(self, df: pd.DataFrame):
        """Add bad points (in feature space) to the set of ignored points."""
        # Identify rows where any target column has NaN
        ignored_rows = df[df[self.targets].isna().any(axis=1)]

        # Extract feature values from these rows and convert to a set of tuples
        features_set = set(tuple(row) for row in ignored_rows[self.features].values)

        # Add these feature points to the set of ignored points
        self.ignored = self.ignored.union(features_set)

    def should_ignore_point(self, x: np.ndarray) -> bool:
        """
        Proximity Exclusion Condition: Determine if the given point is near any point
        with erroneous simulations. Points located within a very small hypercube of edge
        lenght 1e(-self.decimals) around specified points are excluded from further
        processing. This condition is necessary because surrogate GP does not update
        around failed samples.

        Args: x (np.ndarray): A single sample point to check.

        Returns: bool: True if the point should be ignored, False otherwise.
        """
        if len(self.ignored) == 0:
            return False
        x = np.atleast_2d(x)
        assert x.shape[0] == 1, "Input x must be a single point!"
        
        # TODO yasser: instead of a cube of why not set a sphere where r=(gp_std_relative > 20% or 5%)
        x_rounded = np.round(x.ravel(), self.decimals)
        ignored_rounded = np.round(np.array([*self.ignored]), self.decimals)

        return np.any(np.all(x_rounded == ignored_rounded, axis=1))


class InlierOutlierGP:
    def __init__(self):
        # Initialize Gaussian Process Classifier with RBF kernel
        self.kernel = 1.0 * RBF(length_scale=1.0)
        self.gp = GaussianProcessClassifier(kernel=self.kernel)

        # Map class labels to indices
        self.class_to_index = {"outlier": 0, "inlier": 1}
        self.index_to_class = {index: label for label, index in self.class_to_index.items()}

    def fit(self, X_train, y_train):
        # Determine inlier or outlier status based on NaN presence in y_train
        y = np.where(
            np.isnan(y_train).any(axis=1),
            self.class_to_index["outlier"],  # 0
            self.class_to_index["inlier"]   # 1
        )

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
        """
        # Predict inlier probabilities
        inlier_proba = self.predict_inlier_proba(X)

        # Calculate the standard deviation of the probabilities
        std = np.sqrt(inlier_proba * (1 - inlier_proba))
        return std


def search_max_std(
    gp: Union[GaussianProcessRegressor, GaussianProcessClassifier],
    features: List[str], shgo_iters: int = 5, shgo_n: int = 1000
):
    """ Returns the maximum standard deviation of the Gaussian Process for points between 0 and 1."""

    print("search_max_std -> Searching for the maximum standard deviation of the surrogate...")
    search_error = 0.01

    def get_opposite_std(x):
        """Opposite of std to be minimized."""
        x = np.atleast_2d(x)
        _, y_std = gp.predict(x, return_std=True)
        return -1 * y_std.max()

    bounds = [(0, 1)]*len(features)
    result = shgo(
        get_opposite_std, bounds, iters=shgo_iters, n=shgo_n, sampling_method='simplicial'
    )

    max_std = -1 * result.fun
    max_std = min(1.0, max_std * (1 + search_error))
    print(f"search_max_std -> Maximum GP std: {max_std:.3f}")
    print(f"search_max_std -> Training data std: (X, {gp.X_train_.std():.3f}), (y, {gp.y_train_.std():.3f})")
    return max_std


class FigureOfMerit:
    """
    The FigureOfMerit class is responsible for evaluating and selecting
    interesting candidates for simulation in the adaptive sampling process.

    This class uses a Gaussian Process model to assess the potential value
    of sampling points based on criteria such as standard deviation, interest,
    and coverage. The evaluation is guided by specified coefficients and
    regions of interest, allowing for dynamic and informed decision-making
    in the sampling strategy.

    Attributes:
        - gp_surrogate (SurrogateGP): A surrogate model used to identify unexplored areas
        by evaluating the standard deviation, focusing on interesting points for further
        exploration. This model does not train on outliers or errors, necessitating an
        alternative method for avoiding poor regions.
        - gp_classifier (InlierOutlierGP): A classifier designed to detect unexplored inlier
        regions by assessing the standard deviation. It aids in identifying promising
        areas while ensuring that regions with potential errors or outliers are avoided.
    """

    def __init__(
        self, features: List[str], targets: List[str],
        coefficients: Dict[str, Dict], interest_region: Dict[str, Tuple[float, float]],
        decimals: int
    ):
        """
        Initializes the FigureOfMerit class.

        Parameters:
        - features: List of feature column names.
        - targets: List of target column names.
        - coefficients: Coefficients for standard deviation, interest,
          and coverage calculations.
        - interest_region: Parameters defining the region of interest.
        - decimals: Number of decimal places for rounding.
        """
        # Initialize GP surrogate model
        self.gp_surrogate = SurrogateGP(features, targets, decimals)
        
        # Initialize GP fitted on 0|1 values
        self.gp_classifier = InlierOutlierGP()
        
        # Data attributes
        self.features = features
        self.targets = targets
        self.data = None

        # To normalize std of coverage function to be between 0 and 1
        self.max_std = 1

        # Store coefficients and initialize calculation methods
        self.coefficients = coefficients
        self.calc_std = self.set_std(config=coefficients["std"])
        self.calc_interest = self.set_interest(config=coefficients["interest"], interest_region=interest_region)
        self.calc_coverage = self.set_coverage(config=coefficients["coverage"])
        self.calc_std_x = self.set_std_x(config=coefficients["std_x"])
        # TODO yasser: this method will replace SurrogateGP.should_ignore point
        # * It is coded but not yet implemented. Create a separate class to store ignored points.
        self.calc_outlier_proximity = self.set_outlier_proximity(config=coefficients["outlier_proximity"])
        
    def update(self, data: pd.DataFrame, shgo_iters: int = 5, shgo_n: int = 1000):
        """
        Updates the Gaussian Process model with new data, excluding rows with NaN target values.

        Parameters:
        - data (DataFrame): Total available data to update the model with.
        """
        # Update current available dataset
        self.data = data

        # Filter out rows with NaN target values for GP training
        clean_data = data.dropna(subset=self.targets)
        self.gp_surrogate.fit(
            X_train=clean_data[self.features].values,
            y_train=clean_data[self.targets].values
        )

        # Train another GP to find unexplored inlier regions
        if self.coefficients["std_x"]["apply"]:
            self.gp_classifier.fit(
                X_train=data[self.features].values,
                y_train=data[self.targets].values
            )

        # Update max_std of current GP model(s)
        self.max_std = search_max_std(
            gp=self.gp_surrogate.gp, features=self.features, shgo_iters=shgo_iters, shgo_n=shgo_n
        )
        # # TODO yasser: include std_x or not need ? gp = concat(gp_surrogate + gp_classifier)

    def set_std(self, config: Dict):
        """
            Set the standard deviation function, which penalizes points with high uncertainty.
        """
        if config["apply"]:
            def std(x):
                """
                Computes the mean standard deviation over all target dimensions for a given input x
                and stretches it to reach the entire range of [0, 1].
                """
                x = np.atleast_2d(x)

                _, y_std = self.gp_surrogate.predict(x, return_std=True)

                y_std_mean = y_std.mean(axis=1)

                score = 1 - y_std_mean / self.max_std
                return score
        else:
            def std(x):
                x = np.atleast_2d(x)
                return np.zeros(x.shape[0])
        return std


    def set_interest(self, config: Dict, interest_region: Dict):
        if config["apply"]:
            lowers = [region[0] for region in interest_region.values()]
            uppers = [region[1] for region in interest_region.values()]
            def interest(x):
                """
                Given an n-dimensional x returns the sum of the probabilities to be in the interest region
                """
                x = np.atleast_2d(x)

                y_hat, y_std = self.gp_surrogate.predict(x, return_std=True)

                point_norm = norm(loc=y_hat, scale=y_std)

                probabilities = point_norm.cdf(uppers) - point_norm.cdf(lowers)

                # We want to minimize this function, to maximize the probability of being in the interest region
                score = 1 - np.prod(probabilities, axis=1)
                
                return score
        else:
            def interest(x):
                x = np.atleast_2d(x)
                return np.zeros(x.shape[0])

        return interest

    def set_coverage(self, config: Dict):
        """
            Set the coverage function, which penalizes near by points in the space,
            promoting the exploration (coverage) of the space.
        """
        if config["apply"]:
            def space_coverage_one(x: np.ndarray):
                x = np.atleast_2d(x)
                assert x.shape[0] == 1, "Input x must be a single point!"
                if config["include_outliers"]:
                    points = self.data[self.features].values
                else:
                    points = self.gp_surrogate.gp.X_train_
                
                distances = np.linalg.norm(points - x, axis=1)

                # Sigmoid parameters
                decay = config["decay"]  # 20 or lower for including farther points
                sigmoid_speed = config["sigmoid_speed"]  # 5 or lower to allow more points to be closer to each other
                sigmoid_pos = config["sigmoid_pos"]

                score = np.exp(-decay * distances).sum()
                # Pass sum of probs through sigmoid
                all_probs = 1 / (1 + np.exp(sigmoid_pos - sigmoid_speed * score))

                return all_probs

            def space_coverage(x: np.ndarray):
                # Launch multiple space_coverage_one
                x = np.atleast_2d(x)
                score = []
                for x_row in x:
                    score.append(space_coverage_one(x_row))
                score = np.array(score)
                return score
        else:
            def space_coverage(x: np.ndarray):
                x = np.atleast_2d(x)
                return np.zeros(x.shape[0])
        return space_coverage

    def set_std_x(self, config: Dict) -> callable:
        """
        Set the standard deviation function for the InlierOutlierGP classifier.
        This classifier predicts whether a sample is an inlier or outlier.
        It trains on all data, not just inliers like the SurrogateGP class.
        """
        if config["apply"]:
            def std_x(x):
                """
                Calculate scores based on inlier probability and standard deviation.
                Computes the standard deviation over feature dimensions for a given
                input x and scales it to the range [0, 1].
                """
                x = np.atleast_2d(x)
                proba = self.gp_classifier.predict_inlier_proba(x)
                y_std = self.gp_classifier.get_std(x)

                # Normalize the standard deviation
                normalized_std = y_std / self.max_std

                # Initialize scores to zero
                score = np.zeros_like(proba)
                
                # Probability threshold for considering a point as an inlier.
                threshold = config["threshold"]

                # Calculate scores for points with high inlier probability
                high_proba_indices = proba > threshold
                score[high_proba_indices] = proba[high_proba_indices] * normalized_std[high_proba_indices]

                # Make score an objective for minimization
                score = 1 - score
                return score
        else:
            def std_x(x):
                x = np.atleast_2d(x)
                return np.zeros(x.shape[0])

        return std_x
    
    def set_outlier_proximity(self, config: Dict) -> callable:
        if config["apply"]:
            def outlier_proximity(self, x: np.ndarray) -> np.ndarray:
                """
                Proximity Exclusion Condition: Determine if the given points are near any points
                with erroneous simulations. Points located within a very small hypercube of edge
                length 1e(-self.decimals) around specified points are excluded from further
                processing. This condition is necessary because the surrogate GP does not update
                around failed samples.

                Returns:
                - np.ndarray: An array of 1.0 (ignored) or 0.0 (not ignored) for each point.
                """
                if len(self.gp_surrogate.ignored) == 0:
                    return np.zeros(x.shape[0])

                decimals = config["decimals"]
                
                x = np.atleast_2d(x)
                x_rounded = np.round(x, decimals)
                ignored_rounded = np.round(np.array([*self.gp_surrogate.ignored]), decimals)

                # Determine which points should be ignored
                should_ignore = np.any(np.all(x_rounded[:, np.newaxis] == ignored_rounded, axis=2), axis=1)

                # Log ignored points
                for i, ignore in enumerate(should_ignore):
                    if ignore:
                        print(f"FOM.target_function -> Point {x[i]} was ignored.")

                # return should_ignore.astype(float)
                return should_ignore
        else:
            def outlier_proximity(x):
                x = np.atleast_2d(x)
                # return np.zeros(x.shape[0])
                return np.full(x.shape[0], False)

        return outlier_proximity

    def sort_by_relevance(self, mask: np.ndarray, unique_min: np.ndarray) -> List:
        n_feat = len(self.features)
        bounds = [[] for _ in range(n_feat + 1)]
        for val, cond in zip(unique_min, mask):
            cond_sum = cond.sum()
            for local_sum in range(n_feat + 1):
                if cond_sum == local_sum:
                    bounds[local_sum].append(val)
        return [item for val in bounds for item in val]

    def choose_min(self, size: int, unique_min: np.ndarray) -> np.ndarray:
        mask = (np.isclose(unique_min, 0) | np.isclose(unique_min, 1))
        res = self.sort_by_relevance(mask, unique_min)
        return np.array(res[:size])

    def choose_results(self, minimums: np.ndarray, size: int) -> np.ndarray:
        unique_min = np.unique(minimums, axis=0)
        if size == 1:
            return self.choose_min(1, unique_min)
        elif len(unique_min) <= size:
            return minimums
        else:
            return self.choose_min(size, unique_min)

    def target_function(self, x: np.ndarray):
        """
        Acquisition function.
        Each term can be independently turned on or off from conf file.
        """
        x = np.atleast_2d(x)
        assert x.shape[0] == 1, "Input x must be a single point!"

        if self.gp_surrogate.should_ignore_point(x) and self.coefficients["outlier_proximity"]["apply"]:
            # Point is excluded (ignored) from further processing
            print(f"FOM.target_function -> Point {x} was ignored.")
            score = np.array([1.0])
        else:
            score = (
                self.calc_std(x)
                + self.calc_interest(x)
                + self.calc_coverage(x)
                + self.calc_std_x(x)
            )
        return score.item()
    
    def get_scores(self, new_xs: np.ndarray) -> List:
        scores = pd.DataFrame(
            data=np.array([
                self.calc_std(new_xs),
                self.calc_interest(new_xs),
                self.calc_coverage(new_xs)
            ]).T,
            columns=["std", "interest", "coverage"]
        )
        return scores

    def optimize(self, batch_size: int = 1, shgo_iters: int = 5, shgo_n: int = 1000):
        """
        Optimize the acquisition function to find the best candidates for simulation.

        Args:
            batch_size (int): Number of points to select for simulation.
            iters (int): Number of iterations used in the construction of the simplicial
                         complex of SHGO optimizer.
            n (int): Number of sampling points for the SHGO optimizer.

        Returns:
            Tuple[np.ndarray, pd.DataFrame]: Selected points and their scores.
        """

        print("FOM.optimize -> Searching for good candidates using the acquisition function...")

        bounds = [(0, 1)]*len(self.features)
        
        result = shgo(
            self.target_function, bounds, iters=shgo_iters, n=shgo_n,
            sampling_method='simplicial'
        )
        res = result.xl if result.success else result.x.reshape(1, -1)
        new_xs = self.choose_results(minimums=res, size=batch_size)
        
        scores = self.get_scores(new_xs=new_xs)

        df_new_xs = pd.DataFrame(new_xs, columns=self.features)
        df_print = pd.concat([df_new_xs, scores], axis=1, ignore_index=False)
        print("FOM.optimize -> Selected points to be input to the simulator:\n", df_print)

        return new_xs, scores
