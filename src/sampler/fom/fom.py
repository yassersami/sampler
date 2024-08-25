from typing import List, Tuple, Dict, Union
import numpy as np
import pandas as pd
from scipy.optimize import shgo

from .surrogate import SurrogateGPR, InlierOutlierGPC
from .spatial import OutlierProximityDetector, SigmoidLocalDensity


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
        - gp_surrogate (SurrogateGPR): A surrogate model used to identify
        unexplored areas
        by evaluating the standard deviation, focusing on interesting pointsfor
        further exploration. This model does not train on outliers or errors,
        necessitating an alternative method for avoiding poor regions.
        - gp_classifier (InlierOutlierGPC): A classifier designed to detect
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
        self.gp_surrogate = SurrogateGPR(features, targets, interest_region)
        
        # Instanciate Simdmoig local density
        self.sigmoid = SigmoidLocalDensity()
        
        # Initialize outliers handler
        self.outlier_detector = OutlierProximityDetector(features, targets)
        
        # Initialize GP fitted on 0|1 values
        self.gp_classifier = InlierOutlierGPC()
        
        # Data attributes
        self.features = features
        self.targets = targets
        self.data = None
        
        # Store terms kwargs 
        self.terms = terms

        # Set terms calculation methods
        # self.set_fom()
        self.set_std(**terms["std"])
        self.set_interest(**terms["interest"])
        self.set_local_density(**terms["local_density"])
        self.set_outlier_proximity(**terms["outlier_proximity"])
        self.set_inlier_bstd(**terms["inlier_bstd"])
        
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
            clean_regr_data = data.dropna(subset=self.targets)
            self.gp_surrogate.fit(
                X=clean_regr_data[self.features].values,
                y=clean_regr_data[self.targets].values
            )
            if self.terms["std"]["apply"]:
                # Update max_std of current surrogate GP
                self.gp_surrogate.update_max_std(**optimizer_kwargs)

        # Give sigmoide dataset points
        if self.terms["sigmoid_density"]["apply"]:
            self.sigmoid.fit(X=data[self.features].values)

        # Train another GP to find unexplored inlier regions
        if self.terms["inlier_bstd"]["apply"]:
            self.gp_classifier.fit(
                X=data[self.features].values,
                y=data[self.targets].values
            )

        # Train KDE
        # if self.terms["kde"]["apply"]:
        #     self.kde.fit(data[self.features].values)
        #     # Update minimum density for normalization
        #     self.kde.update_min_density(**optimizer_kwargs)

    def set_fom(self):
        # def score_to_loss(x: np.ndarray) -> np.ndarray: return 1-x
    
        # surrogate_gpr_std (SurrogateGPR): 1 - self.surrogate_gpr.get_norm_std
        # surrogate_gpr_interest (SurrogateGPR): 1 - self.surrogate_gpr.predict_interest_proba

        # sigmoid_density: compute_sigmoid_local_density
        # outlier_proximity (OutlierProximityDetector):  self.outlier_detector.detect_outlier_proximity
        # gpc_inlier_bstd (InlierOutlierGPC): 1 - get_inlier_bstd
        # gpc_inlier_entropy (InlierOutlierGPC): 1 - get_inlier_entropy
        
        # surrogate_gpc_std SurrogateGPC
        # surrogate_gpc_std (SurrogateGPC): 1 - self.surrogate_gpc.get_norm_std
        # surrogate_gpc_interest (SurrogateGPC): 1 - self.surrogate_gpc.predict_interest_proba
        # surrogate_gpc_inlier (SurrogateGPC): 1 - self.surrogate_gpc.get_inlier_bstd
        
        # SurrogateMLPC
        # KDE.predict_score
        
        # Chose interest
        # Chose density
        # Chose optimizer
        
        # self.active_terms = {
        #     k: d["kwargs"] for k, d in self.terms.items() if d["apply"]
        # }
        pass

    def set_std(self, apply: bool) -> callable:
        """
        Set the standard deviation function, encourges to find points with high
        uncertainty. Computes the combined standard deviation over all target
        dimensions for a given input x and stretches it to reach the entire
        range of [0, 1].
        """
        def std(x):
            loss = 1 - self.gp_surrogate.get_norm_std(x)
            return loss
        
        if apply:
            self.calc_std = std
        else:
            self.calc_std = zeros_like_rows

    def set_interest(self, apply: bool) -> callable:
        """
        Given an n-dimensional x returns the sum of the probabilities to be in
        the interest region.
        """
        
        def interest(x):
            # Minimize loss to maximize probability of being interest region
            loss = 1 - self.gp_surrogate.predict_interest_proba(x)
            return loss
        
        if apply:
            self.calc_interest = interest
        else:
            self.calc_interest = zeros_like_rows

    def set_local_density(
        self, apply: bool, decay_dist: float
    ) -> callable:
        """
        Set the local density function, which penalizes near points in already
        crowded region, promoting the exploration (coverage) of the space.
        """
        
        def space_local_density(x: np.ndarray):
            dataset_points = self.data[self.features].values
            self.sigmoid.decay_dist = decay_dist
            loss = self.sigmoid.predict_score(x)
            return loss
        
        if apply:
            self.calc_local_density = space_local_density
        else:
            self.calc_local_density = zeros_like_rows

    def set_outlier_proximity(
        self, apply: bool, exclusion_radius: float
    ) -> callable:
        """
        Proximity Exclusion Condition: Determine if the given points are near
        any pointswith erroneous simulations. Points located within a very small
        vicinity around outliers are excluded from further processing. This
        condition is necessary because the surrogate GP does not update around
        failed samples.
        """

        def outlier_proximity(x: np.ndarray) -> np.ndarray:
            # Get boolean array of candidates too close to an outlier
            should_ignore = self.outlier_detector.predict_score(
                x, exclusion_radius
            )
            # if sample is bad (near outlier), loss is 1, and 0 if not
            loss = should_ignore.astype(float)

            return loss
    
        if apply:
            self.calc_outlier_proximity = outlier_proximity
        else:
            self.calc_outlier_proximity = zeros_like_rows

    def set_inlier_bstd(self, apply: bool) -> callable:
        """
        Set the loss from InlierOutlierGPC. This classifier predicts whether a
        sample is an inlier or outlier. It trains on all data, not just inliers
        like the SurrogateGPR class. It calculates losses based on inlier
        probability and Bernoulli standard deviation.
        """
        
        def inlier_bstd(x):
            loss = 1 - self.gp_classifier.get_inlier_bstd(x)
            return loss
    
        if apply:
            self.calc_inlier_bstd = inlier_bstd
        else:
            self.calc_inlier_bstd = zeros_like_rows

    def target_function(self, x: np.ndarray) -> float:
        """
        Acquisition function.
        Each term can be independently turned on or off from conf file.
        """
        x = np.atleast_2d(x)
        assert x.shape[0] == 1, "Input x must be a single point!"

        loss = (
            self.calc_std(x)
            + self.calc_interest(x)
            + self.calc_local_density(x)
            + self.calc_outlier_proximity(x)
            + self.calc_inlier_bstd(x)
        )
        return loss.item()
    
    def get_losses(self, new_xs: np.ndarray) -> pd.DataFrame:
        new_xs = np.atleast_2d(new_xs)
        losses = pd.DataFrame(
            data=np.array([
                self.calc_std(new_xs),
                self.calc_interest(new_xs),
                self.calc_local_density(new_xs),
                self.calc_outlier_proximity(new_xs),
                self.calc_inlier_bstd(new_xs),
            ]).T,
            columns=["std", "interest", "local_density", "exclusion", "inlier_bstd"]
        )
        return losses

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

    def optimize(
        self, batch_size: int = 1, shgo_iters: int = 5, shgo_n: int = 1000
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Optimize the FOM to find the best candidates for simulation.

        Parameters:
        - batch_size (int): Number of points to select for simulation.
        - iters (int): Number of iterations used in the construction of the
        simplicial complex of SHGO optimizer.
        - n (int): Number of sampling points for the SHGO optimizer.

        Returns:
            Tuple[np.ndarray, pd.DataFrame]: Selected points and their losses.
        """

        print("FOM.optimize -> Searching for good candidates...")

        bounds = [(0, 1)]*len(self.features)
        
        result = shgo(
            self.target_function, bounds, iters=shgo_iters, n=shgo_n,
            sampling_method='simplicial'
        )
        res = result.xl if result.success else result.x.reshape(1, -1)
        new_xs = self.choose_results(minimums=res, size=batch_size)
        
        losses = self.get_losses(new_xs=new_xs)

        df_new_xs = pd.DataFrame(new_xs, columns=self.features)
        df_print = pd.concat([df_new_xs, losses], axis=1, ignore_index=False)
        print(
            "FOM.optimize -> Selected points to be input to the simulator:\n",
            df_print
        )

        return new_xs, losses
