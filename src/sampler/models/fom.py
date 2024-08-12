from typing import List, Tuple, Dict, Union
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.spatial import distance

from scipy.stats import norm
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


class OutlierExcluder:
    def __init__(
        self, features: List[str], targets: List[str], threshold: float = 1e-5,
        distance_metric: str = "euclidean"
    ):
        
        self.features = features
        self.targets = targets
        self.threshold = threshold
        self.distance_metric = distance_metric
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

    def detect_outlier_proximity(self, x: np.ndarray) -> bool:
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
        distances = distance.cdist(x, self.ignored_df.values, self.distance_metric)

        # Determine which points should be ignored based on the distance threshold
        should_ignore = np.any(distances < self.threshold, axis=1)

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
        self.index_to_class = {index: label for label, index in self.class_to_index.items()}

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
        coefficients: Dict[str, Dict], interest_region: Dict[str, Tuple[float, float]]
    ):
        """
        Initializes the FigureOfMerit class.

        Parameters:
        - features: List of feature column names.
        - targets: List of target column names.
        - coefficients: Coefficients for standard deviation, interest,
          and coverage calculations.
        - interest_region: Parameters defining the region of interest.
        """
        # Initialize GP surrogate model
        self.gp_surrogate = SurrogateGP(features, targets)
        
        # Initialize outliers handler
        self.excluder = OutlierExcluder(
            features, targets, threshold=coefficients["outlier_proximity"]["threshold"]
        )
        
        # Initialize GP fitted on 0|1 values
        self.gp_classifier = InlierOutlierGP()
        
        # Data attributes
        self.features = features
        self.targets = targets
        self.data = None

        # Store coefficients and initialize calculation methods
        self.coefficients = coefficients
        self.calc_std = self.set_std(config=coefficients["std"])
        self.calc_interest = self.set_interest(config=coefficients["interest"], interest_region=interest_region)
        self.calc_coverage = self.set_coverage(config=coefficients["coverage"])
        self.calc_std_x = self.set_std_x(config=coefficients["std_x"])
        self.calc_outlier_proximity = self.set_outlier_proximity(config=coefficients["outlier_proximity"])
        
        # Count number of active scores
        self.count_active_scores = sum([score["apply"] for score in coefficients.values()])
        
    def update(self, data: pd.DataFrame, optimizer_kwargs: Dict):
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
        # Update max_std of current surrogate GP
        self.gp_surrogate.update_max_std(**optimizer_kwargs)

        # Train another GP to find unexplored inlier regions
        if self.coefficients["std_x"]["apply"]:
            self.gp_classifier.fit(
                X_train=data[self.features].values,
                y_train=data[self.targets].values
            )


    def set_std(self, config: Dict):
        """
        Set the standard deviation function, which penalizes points with high
        uncertainty. Computes the combined standard deviation over all target dimensions
        for a given input x and stretches it to reach the entire range of [0, 1].
        """
        if config["apply"]:
            def std(x):
                x = np.atleast_2d(x)

                y_std_combined = self.gp_surrogate.get_std(x)

                score = 1 - y_std_combined / self.gp_surrogate.max_std
                return score
        else:
            def std(x):
                x = np.atleast_2d(x)
                return np.zeros(x.shape[0])
        return std


    def set_interest(self, config: Dict, interest_region: Dict):
        """
        Given an n-dimensional x returns the sum of the probabilities to be in the
        interest region.
        """
        if config["apply"]:
            lowers = [region[0] for region in interest_region.values()]
            uppers = [region[1] for region in interest_region.values()]
            def interest(x):
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
        It calculates scores based on inlier probability and standard deviation.
        """
        if config["apply"]:
            def std_x(x):
                x = np.atleast_2d(x)
                
                use_threshold = config["use_threshold"]
                threshold = config["threshold"]
                
                y_std_conditional = self.gp_classifier.get_conditional_std(
                    x, use_threshold, threshold
                )

                # Make score an objective for minimization
                score = 1 - y_std_conditional
                return score
        else:
            def std_x(x):
                x = np.atleast_2d(x)
                return np.zeros(x.shape[0])

        return std_x
    
    def set_outlier_proximity(self, config: Dict) -> callable:
        """
        Proximity Exclusion Condition: Determine if the given points are near any points
        with erroneous simulations. Points located within a very small vicinity around
        outliers are excluded from further processing. This condition is necessary
        because the surrogate GP does not update around failed samples.
        """
        if config["apply"]:
            def outlier_proximity(x: np.ndarray) -> np.ndarray:
                x = np.atleast_2d(x)
                # Array of bools
                should_ignore = self.excluder.detect_outlier_proximity(x)
                # if sample is bad (near outlier), score is 1, and 0 if not
                score = should_ignore.astype(float)
                return score
        else:
            def outlier_proximity(x):
                x = np.atleast_2d(x)
                return np.zeros(x.shape[0])

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

        score = (
            self.calc_std(x)
            + self.calc_interest(x)
            + self.calc_coverage(x)
            + self.calc_std_x(x)
            + self.calc_outlier_proximity(x)
        )
        return score.item()
    
    def get_scores(self, new_xs: np.ndarray) -> List:
        scores = pd.DataFrame(
            data=np.array([
                self.calc_std(new_xs),
                self.calc_interest(new_xs),
                self.calc_coverage(new_xs),
                self.calc_std_x(new_xs),
                self.calc_outlier_proximity(new_xs)
            ]).T,
            columns=["std", "interest", "coverage", "std_x", "exclusion"]
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
