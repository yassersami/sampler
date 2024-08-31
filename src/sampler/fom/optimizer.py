from typing import List, Dict, Tuple, Type, Union, Optional, ClassVar
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.optimize import shgo
from sko.GA import GA

from .fom import FigureOfMerit


class MultiModalOptimizer(ABC):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.n_features = None

    @abstractmethod
    def optimize(self, fom: FigureOfMerit) -> Tuple[np.ndarray, pd.DataFrame]:
        self.n_features = fom.n_features
        bounds = [(0, 1)]*self.n_features
        pass

    def score_objective(self, x: np.ndarray, fom: FigureOfMerit) -> float:
        """ Objective for maximization with x of shape (n_dim,). """
        return fom.predict_score(x.reshape(1, -1)).item()

    def loss_objective(self, x: np.ndarray, fom: FigureOfMerit) -> float:
        """
        Objective for minimization. It is a smaller-is-better objective where
        smallest best value is 0. x of shape (n_dim,).
        """
        return fom.n_positive_scores - fom.predict_score(x.reshape(1, -1)).item()


class OptimizerFactory:
    _optimizers: Dict[str, Type[MultiModalOptimizer]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(optimizer_class: Type[MultiModalOptimizer]):
            cls._optimizers[name] = optimizer_class
            return optimizer_class
        return decorator

    @classmethod
    def create_from_config(cls, batch_size: int, optimizer_config: Dict[str, Dict]) -> MultiModalOptimizer:
         # Check if all configs have the 'apply' key
        for name, config in optimizer_config.items():
            if 'apply' not in config:
                raise KeyError(f"Optimizer '{name}' is missing the 'apply' key in its configuration.")

        applied_optimizers = [name for name, config in optimizer_config.items() if config['apply']]
        
        if len(applied_optimizers) != 1:
            raise ValueError(f"Exactly one optimizer must be applied. Found {len(applied_optimizers)}")
        
        optimizer_name = applied_optimizers[0]
        optimizer_args = optimizer_config[optimizer_name]
        optimizer_args.pop('apply')
        
        if optimizer_name not in cls._optimizers:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        return cls._optimizers[optimizer_name](batch_size=batch_size, **optimizer_args)

    @classmethod
    def list_optimizers(cls):
        return list(cls._optimizers.keys())


@OptimizerFactory.register('shgo')
class SHGOOptimizer(MultiModalOptimizer):

    def __init__(self, batch_size: int, n: int, iters: int):
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

    def optimize(self, fom: FigureOfMerit) -> Tuple[np.ndarray, pd.DataFrame]:
        print(
            f"SHGOOptimizer -> n: {self.n}, iters: {self.iters} - "
            "Searching for good candidates..."
        )

        self.n_features = fom.n_features  # Used in self.sort_by_relevance

        def obj_func(x): return self.loss_objective(x, fom)

        result = shgo(  # Minimization algorithm
            obj_func,
            bounds=[(0, 1)]*self.n_features,
            n=self.n,
            iters=self.iters,
            sampling_method='simplicial'
        )
        res = result.xl if result.success else result.x.reshape(1, -1)
        X_candidates = self.choose_results(minimums=res, size=self.batch_size)

        df_scores = fom.predict_scores_df(X_candidates)
        df_scores['shgo_obj'] = [obj_func(x) for x in X_candidates]

        feature_cols = [f'feature_{i+1}' for i in range(fom.n_features)]
        df_candidates = pd.DataFrame(X_candidates, columns=feature_cols)
        df_print = pd.concat([df_candidates, df_scores], axis=1, ignore_index=False)
        print(
            "SHGOOptimizer -> Selected points to be input to the simulator:\n",
            df_print
        )

        return X_candidates, df_scores


@OptimizerFactory.register('ga')
class GAOptimizer(MultiModalOptimizer):
    def __init__(self, batch_size: int, population_size: int, generations: int):
        super().__init__(batch_size)
        self.population_size  = population_size
        self.generations = generations

    def optimize(self, fom: FigureOfMerit):
        print(
            f"GAOptimizer -> population_size: {self.population_size}, generations: {self.generations} - "
            "Searching for good candidates..."
        )

        self.n_features = fom.n_features
        def obj_func(x): return self.loss_objective(x, fom)

        ga = GA(  # Maximization algorithm
            obj_func,
            n_dim=self.n_features,
            size_pop=self.population_size,  # Must be an even number
            max_iter=self.generations,
            prob_mut=0.1,
            lb=[0] * self.n_features,  # Lower bounds
            ub=[1] * self.n_features  # Upper bounds
        )

        ga.run()
        population = ga.chrom2x(ga.Chrom)
        score = np.array([obj_func(ind) for ind in population])
        sorted_indices = np.argsort(score)

        best_indices = sorted_indices[:self.batch_size]
        X_candidates = population[best_indices]

        df_scores = fom.predict_scores_df(X_candidates)
        df_scores['ga_obj'] = [obj_func(x) for x in X_candidates]

        feature_cols = [f'feature_{i+1}' for i in range(fom.n_features)]
        df_candidates = pd.DataFrame(X_candidates, columns=feature_cols)
        df_print = pd.concat([df_candidates, df_scores], axis=1, ignore_index=False)
        print(
            "GAOptimizer -> Selected points to be input to the simulator:\n",
            df_print
        )

        return X_candidates, df_scores
