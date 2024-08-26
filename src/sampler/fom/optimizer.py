from typing import List, Tuple, Dict, Union, Optional, ClassVar
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from scipy.optimize import shgo
from sko.GA import GA

from .base import FOM

class MultiModalOptimizer(ABC):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.n_features = None

    @abstractmethod
    def optimize(self, fom: FOM) -> Tuple[np.ndarray, pd.DataFrame]:
        self.n_features = fom.n_features
        bounds = [(0, 1)]*self.n_features
        pass

    def objective_function(self, x: np.ndarray, fom: FOM) -> float:
        return fom.predict_score(x.reshape(1, -1)).item()


class SHGOOptimizer(MultiModalOptimizer):

    def __init__(self, batch_size, n=100, iters=1):
        super().__init__(batch_size)
        self.n = n
        self.iters = iters

    def sort_by_relevance(self, mask: np.ndarray, unique_min: np.ndarray) -> List:
        bounds = [[] for _ in range(self.n_features + 1)]
        for x_row, cond in zip(unique_min, mask):
            cond_sum = cond.sum()
            for local_sum in range(self.n_features + 1):
                if cond_sum == local_sum:
                    bounds[local_sum].append(x_row)
        return [item for val in bounds for item in val]

    def choose_min(self, size: int, unique_min: np.ndarray) -> np.ndarray:
        mask = (np.isclose(unique_min, 0) | np.isclose(unique_min, 1))
        res = self.sort_by_relevance(mask, unique_min)
        return np.array(res[:size])

    def choose_results(self, minimums: np.ndarray, size: int) -> np.ndarray:
        unique_minimums = np.unique(minimums, axis=0)
        if size == 1:
            return self.choose_min(1, unique_minimums)
        elif len(unique_minimums) <= size:
            return minimums
        else:
            return self.choose_min(size, unique_minimums)

    def optimize(self, fom: FOM) -> Tuple[np.ndarray, pd.DataFrame]:
        print(
            f"SHGOOptimizer -> n: {self.n}, iters: {self.iters} - "
            "Searching for good candidates..."
        )

        self.n_features = fom.n_features
        bounds = [(0, 1)]*self.n_features
        
        result = shgo(
            lambda x: -self.objective_function(x, fom),  # Negate here for minimization
            bounds,
            n=self.n,
            iters=self.iters,
            sampling_method='simplicial'
        )
        res = result.xl if result.success else result.x.reshape(1, -1)
        X_candidates = self.choose_results(minimums=res, size=self.batch_size)
        
        df_scores = fom.predict_scores_df(X_candidates)

        feature_cols = [f'feature_{i+1}' for i in range(fom.n_features)]
        df_candidates = pd.DataFrame(X_candidates, columns=feature_cols)
        df_print = pd.concat([df_candidates, df_scores], axis=1, ignore_index=False)
        print(
            "SHGOOptimizer -> Selected points to be input to the simulator:\n",
            df_print
        )

        return X_candidates, df_scores


class GAOptimizer(MultiModalOptimizer):
    def __init__(self, bounds, batch_size, size_pop=50, max_iter=100):
        super().__init__(bounds, batch_size)
        self.size_pop = size_pop  # Population size
        self.max_iter = max_iter  # Number of generations

    def optimize(self, fom: FOM):

        self.n_features = fom.n_features

        ga = GA(
            func=lambda x: self.objective_function(x, fom),  # No negation needed
            n_dim=self.n_features, 
            size_pop=self.size_pop,
            max_iter=self.max_iter, 
            prob_mut=0.1,
            lb=[0] * self.n_features,  # Lower bounds
            ub=[1] * self.n_features  # Upper bounds
        )
        
        print(
            f"GAOptimizer -> size_pop: {self.size_pop}, max_iter: {self.max_iter} - "
            "Searching for good candidates. size_pop"
        )

        ga.run()
        population = ga.chrom2x(ga.Chrom)
        score = np.array([fom.predict_score(ind) for ind in population])
        sorted_indices = np.argsort(score)

        best_indices = sorted_indices[:self.batch_size]
        X_candidates = population[best_indices]

        df_scores = fom.predict_scores_df(X_candidates)

        feature_cols = [f'feature_{i+1}' for i in range(fom.n_features)]
        df_candidates = pd.DataFrame(X_candidates, columns=feature_cols)
        df_print = pd.concat([df_candidates, df_scores], axis=1, ignore_index=False)
        print(
            "GAOptimizer -> Selected points to be input to the simulator:\n",
            df_print
        )

        return X_candidates, df_scores


import numpy as np
from scipy.optimize import shgo
from scipy.spatial.distance import pdist, squareform

class SHGOOptimizer_TODO(MultiModalOptimizer):
    def __init__(self, bounds, batch_size, n=100, iters=1, diversity_threshold=0.1):
        super().__init__(bounds, batch_size)
        self.n = n
        self.iters = iters
        self.diversity_threshold = diversity_threshold

    def optimize(self, fom):
        result = shgo(
            lambda x: -self.objective_function(x, fom),  # Negate for minimization
            self.bounds, 
            n=self.n, 
            iters=self.iters, 
            options={'ftol': 1e-6, 'maxev': 1000}
        )

        # Get all local minima found by SHGO
        candidates = np.array(result.xl)
        scores = np.array([self.objective_function(x, fom) for x in candidates])

        # Sort candidates by score (highest to lowest)
        sorted_indices = np.argsort(scores)[::-1]
        candidates = candidates[sorted_indices]
        scores = scores[sorted_indices]

        # Select diverse set of candidates
        selected_indices = [0]  # Always include the best point
        for i in range(1, len(candidates)):
            if len(selected_indices) == self.batch_size:
                break
            
            # Calculate distances to already selected points
            distances = pdist(np.vstack([candidates[selected_indices], candidates[i]]))
            min_distance = np.min(distances[-len(selected_indices):])
            
            # Add point if it's sufficiently different from already selected points
            if min_distance > self.diversity_threshold:
                selected_indices.append(i)

        # If we don't have enough diverse points, fill with best remaining points
        while len(selected_indices) < self.batch_size and len(selected_indices) < len(candidates):
            remaining_indices = set(range(len(candidates))) - set(selected_indices)
            selected_indices.append(min(remaining_indices, key=lambda i: scores[i]))

        selected_candidates = candidates[selected_indices]
        selected_scores = scores[selected_indices]

        return selected_candidates, selected_scores
