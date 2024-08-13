from typing import List, Tuple, Dict, Union
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import shgo

from sampler.models.fom_terms import (
    SurrogateGP, InlierOutlierGP, OutlierExcluder, compute_space_local_density
)


def zeros_like_rows(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x)
    return np.zeros(x.shape[0])


class FigureOfMerit:
    """
    The FigureOfMerit class (also called acquisition function) is responsible
    for evaluating and selecting interesting candidates for simulation in the
    adaptive sampling process.

    This class uses a Gaussian Process model to assess the potential value
    of sampling points based on criteria such as standard deviation, interest,
    and local density. The evaluation is guided by specified FOM terms and
    regions of interest, allowing for dynamic and informed decision-making
    in the sampling strategy.

    Attributes:
        - gp_surrogate (SurrogateGP): A surrogate model used to identify
        unexplored areas
        by evaluating the standard deviation, focusing on interesting pointsfor
        further exploration. This model does not train on outliers or errors,
        necessitating an alternative method for avoiding poor regions.
        - gp_classifier (InlierOutlierGP): A classifier designed to detect
        unexplored inlierregions by assessing the standard deviation. It aids in
        identifying promisingareas while ensuring that regions with potential
        errors or outliers are avoided.
    """

    def __init__(
        self, features: List[str], targets: List[str],
        terms: Dict[str, Dict], interest_region: Dict[str, Tuple[float, float]]
    ):
        """
        Initializes the FigureOfMerit class.

        Parameters:
        - features: List of feature column names.
        - targets: List of target column names.
        - terms: Terms of FOM such as for standard deviation, interest, and
        local density calculations.
        - interest_region: Parameters defining the region of interest.
        """
        # Initialize GP surrogate model
        self.gp_surrogate = SurrogateGP(features, targets)
        
        # Initialize outliers handler
        self.excluder = OutlierExcluder(features, targets)
        
        # Initialize GP fitted on 0|1 values
        self.gp_classifier = InlierOutlierGP()
        
        # Data attributes
        self.features = features
        self.targets = targets
        self.data = None

        # Set terms calculation methods
        self.calc_std = self.set_std(
            **terms["std"]
        )
        self.calc_interest = self.set_interest(
            **terms["interest"], interest_region=interest_region
        )
        self.calc_local_density = self.set_local_density(
            **terms["local_density"]
        )
        self.calc_outlier_proximity = self.set_outlier_proximity(
            **terms["outlier_proximity"]
        )
        self.calc_std_x = self.set_std_x(
            **terms["std_x"]
        )
        
        # Store terms and count active ones
        self.terms = terms
        self.count_active_terms = sum([d["apply"] for d in terms.values()])
        
    def update(self, data: pd.DataFrame, optimizer_kwargs: Dict):
        """
        Updates the Gaussian Process model with new data, excluding rows with
        NaN target values.

        Parameters:
        - data (DataFrame): Total available data.
        """
        # Update current available dataset
        self.data = data

        # Filter out rows with NaN target values for GP training
        if self.terms["std"]["apply"] or self.terms["interest"]["apply"]:
            clean_data = data.dropna(subset=self.targets)
            self.gp_surrogate.fit(
                X_train=clean_data[self.features].values,
                y_train=clean_data[self.targets].values
            )
            if self.terms["std"]["apply"]:
                # Update max_std of current surrogate GP
                self.gp_surrogate.update_max_std(**optimizer_kwargs)

        # Train another GP to find unexplored inlier regions
        if self.terms["std_x"]["apply"]:
            self.gp_classifier.fit(
                X_train=data[self.features].values,
                y_train=data[self.targets].values
            )


    def set_std(self, apply: bool) -> callable:
        """
        Set the standard deviation function, which penalizes points with high
        uncertainty. Computes the combined standard deviation over all target
        dimensions for a given input x and stretches it to reach the entire
        range of [0, 1].
        """
        if not apply:
            return zeros_like_rows
        
        def std(x):
            x = np.atleast_2d(x)

            y_std_combined = self.gp_surrogate.get_std(x)

            score = 1 - y_std_combined / self.gp_surrogate.max_std
            return score
        return std


    def set_interest(self, apply: bool, interest_region: Dict) -> callable:
        """
        Given an n-dimensional x returns the sum of the probabilities to be in
        the interest region.
        """
        if not apply:
            return zeros_like_rows
        
        lowers = [region[0] for region in interest_region.values()]
        uppers = [region[1] for region in interest_region.values()]
        def interest(x):
            x = np.atleast_2d(x)

            y_hat, y_std = self.gp_surrogate.predict(x, return_std=True)

            point_norm = norm(loc=y_hat, scale=y_std)

            probabilities = point_norm.cdf(uppers) - point_norm.cdf(lowers)

            # Minimize score to maximize probability of being interest region
            score = 1 - np.prod(probabilities, axis=1)
            
            return score
        return interest

    def set_local_density(
        self, apply: bool, include_outliers: bool, decay_dist: float
    ) -> callable:
        """
        Set the local density function, which penalizes near points in already
        crowded region, promoting the exploration (coverage) of the space.
        """
        if not apply:
            return zeros_like_rows
        
        def space_local_density(x: np.ndarray):
            x = np.atleast_2d(x)
            scores = []
            
            if include_outliers:
                dataset_points = self.data[self.features].values
            else:
                dataset_points = self.gp_surrogate.gp.X_train_
            
            scores = compute_space_local_density(
                x, dataset_points, decay_dist,
            )

            return scores
        return space_local_density
    
    def set_outlier_proximity(
        self, apply: bool, dist_threshold: float
    ) -> callable:
        """
        Proximity Exclusion Condition: Determine if the given points are near
        any pointswith erroneous simulations. Points located within a very small
        vicinity aroundoutliers are excluded from further processing. This
        condition is necessary because the surrogate GP does not update around
        failed samples.
        """
        if not apply:
            return zeros_like_rows
        
        def outlier_proximity(x: np.ndarray) -> np.ndarray:
            x = np.atleast_2d(x)
            
            # Get boolean array of candidates too close to an outlier
            should_ignore = self.excluder.detect_outlier_proximity(
                x, dist_threshold
            )
            
            # if sample is bad (near outlier), score is 1, and 0 if not
            score = should_ignore.astype(float)
            return score
        return outlier_proximity

    def set_std_x(
        self, apply: bool, use_threshold: bool, inlier_proba_threshold: float
    ) -> callable:
        """
        Set the standard deviation function for the InlierOutlierGP classifier.
        This classifier predicts whether a sample is an inlier or outlier.
        It trains on all data, not just inliers like the SurrogateGP class.
        It calculates scores based on inlier probability and standard deviation.
        """
        if not apply:
            return zeros_like_rows
        
        def std_x(x):
            x = np.atleast_2d(x)
            
            y_std_conditional = self.gp_classifier.get_conditional_std(
                x, use_threshold, inlier_proba_threshold
            )

            # Make score an objective for minimization
            score = 1 - y_std_conditional
            return score
        return std_x

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
            + self.calc_local_density(x)
            + self.calc_outlier_proximity(x)
            + self.calc_std_x(x)
        )
        return score.item()
    
    def get_scores(self, new_xs: np.ndarray) -> List:
        scores = pd.DataFrame(
            data=np.array([
                self.calc_std(new_xs),
                self.calc_interest(new_xs),
                self.calc_local_density(new_xs),
                self.calc_outlier_proximity(new_xs),
                self.calc_std_x(new_xs),
            ]).T,
            columns=["std", "interest", "local_density", "exclusion", "std_x"]
        )
        return scores

    def optimize(self, batch_size: int = 1, shgo_iters: int = 5, shgo_n: int = 1000):
        """
        Optimize the FOM to find the best candidates for simulation.

        Parameters:
        - batch_size (int): Number of points to select for simulation.
        - iters (int): Number of iterations used in the construction of the
        simplicial complex of SHGO optimizer.
        - n (int): Number of sampling points for the SHGO optimizer.

        Returns:
            Tuple[np.ndarray, pd.DataFrame]: Selected points and their scores.
        """

        print("FOM.optimize -> Searching for good candidates...")

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
        print(
            "FOM.optimize -> Selected points to be input to the simulator:\n",
            df_print
        )

        return new_xs, scores
