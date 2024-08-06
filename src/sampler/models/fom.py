from typing import List, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic
from scipy.optimize import shgo

RANDOM_STATE = 42


class GPSampler:
    def __init__(self, features: List[str], targets: List[str], decimals: int=10):

        self.features = features
        self.targets = targets
        kernel = RationalQuadratic(length_scale_bounds=(1e-7, 100000))
        self.model = GaussianProcessRegressor(kernel=kernel, random_state=RANDOM_STATE)
        
        # Points to be ignored as they result in erroneous simulations
        self.ignored = set()
        self.decimals = decimals
        # To normalize std of coverage function to be between 0 and 1
        self.max_std = 1

    def get_std(self, x: np.ndarray):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        _, y_std = self.model.predict(x, return_std=True)

        return y_std

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(x_train, y_train)

    def predict(self, x: np.ndarray, return_std=False, return_cov=False):
        return self.model.predict(x, return_std, return_cov)

    def update_max_std(self, iters, n):
        """ Returns the maximum standard deviation of the Gaussian Process for points between 0 and 1."""

        print("GPsampler.update_max_std -> Searching for the maximum standard deviation of the surrogate...")
        search_error = 0.01

        def get_opposite_std(x):
            """Opposite of std to be minimized."""
            y_std = self.get_std(x)[0][0]
            return -1 * y_std

        bounds = [(0, 1) for _ in self.features]
        result = shgo(
            get_opposite_std, bounds, iters=iters, n=n, sampling_method='simplicial'
        )

        max_std = -1 * result.fun
        self.max_std = min(1.0, max_std * (1 + search_error))
        print("GPsampler.update_max_std -> Maximum standard deviation found:", round(self.max_std, 3))

    def add_ignored_points(self, points: set):
        """Add points (in feature space) to the set of ignored points."""
        features_set = set(tuple(row) for row in points[self.features].values)
        self.ignored = self.ignored.union(features_set)

    def should_ignore_point(self, point: np.ndarray) -> bool:
        """
        Check if the given point is close to any point with erroneous simulations.

        Args: point (np.ndarray): A single sample point to check.

        Returns: bool: True if the point should be ignored, False otherwise.

        Note: The comparison is done up to self.decimals decimal places.
        """
        if len(self.ignored) == 0:
            return False

        point_rounded = np.round(point.ravel(), self.decimals)
        ignored_rounded = np.round(np.array([*self.ignored]), self.decimals)

        return np.any(np.all(point_rounded == ignored_rounded, axis=1))

    def plot_std(self, z_value=0.913):
        """Plot the standard deviation of the Gaussian Process for a grid of points between 0 and 1."""
        # Note: Useful for debugging calling it inside fit method with z_value = mean(fixed_feature)
        n = 100
        x = np.linspace(0, 1, n)
        y = np.linspace(0, 1, n)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                _, y_std = self.model.predict(np.array([[X[i, j], Y[i, j], z_value]]), return_std=True)
                std0, std1 = y_std[0]
                Z[i, j] = std0

        plot_type = 'surface' # surface or contour
        if plot_type == 'surface':
            fig = go.Figure(data=[go.Surface(z=Z, x=x, y=y)])

            # Add training points as scatter plot
            x_train = self.x_train[:, 0]
            y_train = self.x_train[:, 1]
            dummy_train = self.x_train.copy()
            dummy_train[:, 2] = 0
            _, y_std = self.model.predict(dummy_train, return_std=True)
            z_train = y_std[:, 0]
            fig.add_trace(go.Scatter3d(
                x=x_train, y=y_train, z=z_train, mode='markers', marker=dict(color='green', size=3),
                name='Training Points', showlegend=True
            ))

            fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Standard Deviation'))
        elif plot_type == 'contour':
            fig = go.Figure(data=go.Contour(z=Z, x=x, y=y, contours=dict(coloring='heatmap')))
            fig.update_layout(coloraxis=dict(colorscale='Viridis'))

        fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        fig.update_layout(height=800, margin=dict(l=0, r=0, b=0, t=0))
        fig.show()

        print("Done plotting.")


class FigureOfMerit(GPSampler):
    def __init__(
            self, features: List[str], targets: List[str],
            coefficients: Dict, interest_region: Dict, decimals: int
    ):
        super().__init__(features, targets, decimals)

        self.coefficients = coefficients

        self.calc_std = self.set_std(c=coefficients["std"])
        self.calc_interest = self.set_interest(c=coefficients["interest"], interest_region=interest_region)
        self.calc_coverage = self.set_coverage(c=coefficients["coverage"])

    def set_std(self, c: Dict):
        """
            Set the standard deviation function, which penalizes points with high uncertainty.
        """
        if c["apply"]:
            def std(x):
                """
                Computes the mean standard deviation over all target dimensions for a given input x
                and stretches it to reach the entire range of [0, 1].
                """
                if x.ndim == 1:
                    x = x.reshape(1, -1)

                _, y_std = self.model.predict(x, return_std=True)

                y_std_mean = y_std.mean(axis=1)

                score = 1 - y_std_mean / self.max_std
                return score
        else:
            def std(x):
                if x.ndim == 1:
                    x = x.reshape(1, -1)
                return np.zeros(x.shape[0])
        return std


    def set_interest(self, c: Dict, interest_region: Dict):
        if c["apply"]:
            lowers = [region[0] for region in interest_region.values()]
            uppers = [region[1] for region in interest_region.values()]
            def interest(x):
                """
                Given an n-dimensional x returns the sum of the probabilities to be in the interest region
                """
                if x.ndim == 1:
                    x = x.reshape(1, -1)

                y_hat, y_std = self.model.predict(x, return_std=True)

                point_norm = norm(loc=y_hat, scale=y_std)

                probabilities = point_norm.cdf(uppers) - point_norm.cdf(lowers)

                # We want to minimize this function, to maximize the probability of being in the interest region
                score = 1 - np.prod(probabilities, axis=1)
                
                return score
        else:
            def interest(x):
                if x.ndim == 1:
                    x = x.reshape(1, -1)
                return np.zeros(x.shape[0])

        return interest

    def set_coverage(self, c: Dict):
        """
            Set the coverage function, which penalizes near by points in the space,
            promoting the exploration (coverage) of the space.
        """
        if c["apply"] and c["mode"] == "euclidean":
            def space_coverage_one(x: np.ndarray):
                if x.ndim != 1:
                    assert False, "The input x must be a single point!"

                gp_x_data = self.model.X_train_
                distances = np.linalg.norm(gp_x_data - x, axis=1)

                # TODO: Add these to kedro catalog parameters
                decay = 25         # 20 or lower for including farther points
                sigmoid_speed = 1  # 5 or lower to allow more points to be closer to each other
                sigmoid_pos = 5

                score = np.exp(-decay * distances).sum()
                # Pass sum of probs through sigmoid
                all_probs = 1 / (1 + np.exp(sigmoid_pos - sigmoid_speed * score))

                return all_probs

            def space_coverage(x: np.ndarray):
                # Launch multiple space_coverage_one
                if x.ndim == 1:
                    x = x.reshape(1, -1)
                score = []
                for x_row in x:
                    score.append(space_coverage_one(x_row))
                score = np.array(score)
                return score
        else:
            def space_coverage(x: np.ndarray):
                if x.ndim == 1:
                    x = x.reshape(1, -1)
                return np.zeros(x.shape[0])
        return space_coverage

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

    def target_function(self, x):
        """
        Acquisition function.
        Each term can be independently turned on or off from conf file.
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        else:
            assert False, "The input x must be a single point!"

        if self.should_ignore_point(x):
            # This condition is necessary because GP do not update around failed samples
            print(f"FOM.target_function -> Point {x} was ignored.")
            score = np.array([1.0])
        else:
            score = self.calc_std(x) + self.calc_interest(x) + self.calc_coverage(x)
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

    def optimize(self, batch_size: int = 1, iters: int = 5, n: int = 1000):
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
        
        self.update_max_std(iters, n)

        print("FOM.optimize -> Searching for good candidates using the acquisition function...")

        bounds = [(0, 1)]*len(self.features)
        
        result = shgo(self.target_function, bounds, iters=iters, n=n, sampling_method='simplicial')
        res = result.xl if result.success else result.x.reshape(1, -1)
        new_xs = self.choose_results(minimums=res, size=batch_size)
        
        scores = self.get_scores(new_xs=new_xs)

        df_new_xs = pd.DataFrame(new_xs, columns=self.features)
        df_print = pd.concat([df_new_xs, scores], axis=1, ignore_index=False)
        print("FOM.optimize -> Selected points to be input to the simulator:\n", df_print)

        return new_xs, scores
