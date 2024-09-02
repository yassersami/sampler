from typing import List, Dict, Tuple, Callable, Type, Union, Optional, ClassVar
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import shgo
from sko.GA import GA

from .base import MultiModalOptimizer


class OptimizerFactory:
    _optimizers: Dict[str, Type[MultiModalOptimizer]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(optimizer_class: Type[MultiModalOptimizer]):
            cls._optimizers[name] = optimizer_class
            return optimizer_class
        return decorator

    @classmethod
    def create_from_config(cls, n_dim: int, batch_size: int, optimizer_config: Dict[str, Dict]) -> MultiModalOptimizer:
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

        return cls._optimizers[optimizer_name](n_dim=n_dim, batch_size=batch_size, **optimizer_args)

    @classmethod
    def list_optimizers(cls):
        return list(cls._optimizers.keys())


@OptimizerFactory.register('shgo')
class SHGOOptimizer(MultiModalOptimizer):

    def __init__(self, n_dim: int, batch_size: int, n: int, iters: int):
        super().__init__(n_dim, batch_size)
        self.n = n
        self.iters = iters
        self.optimizer_config = {'n': n, 'iters': iters}

    def sort_by_edge_avoidance(
        self, X_unique: np.ndarray, edge_proximity_mask: np.ndarray
    ) -> List[np.ndarray]:
        n_dim = X_unique.shape[1]

        # Categorize samples by how many feature values are close to (0 | 1) edges
        edge_proximity_groups = [[] for _ in range(n_dim + 1)]

        for x_row, cond in zip(X_unique, edge_proximity_mask):
            cond_sum = cond.sum()
            edge_proximity_groups[cond_sum].append(x_row)

        return [
            x_row
            for group in edge_proximity_groups  # loop starting with group least close to an edge 
            for x_row in group  # loop with shgo initial order
        ]

    def select_interior_candidates(self, X_candidates: np.ndarray, batch_size: int) -> np.ndarray:
        X_unique = np.unique(X_candidates, axis=0)
        if len(X_unique) <= batch_size:
            # If not enough candidates return all available ones
            return X_candidates

        # Create edge proximity mask where true if value close to (0 | 1) edges 
        edge_proximity_mask = (np.isclose(X_unique, 0) | np.isclose(X_unique, 1))

        # Get number of values close to an edge per row
        extremeness = edge_proximity_mask.sum(axis=1)

        # Get indices to sort X where smallest value (less close to an edge) comes first
        # 'stable' sorting maintains the relative order of equal elements
        sort_indices = np.argsort(extremeness, kind='stable')

        # Use these indices to reorder X_unique
        X_sorted = X_unique[sort_indices]

        return np.array(X_sorted[:batch_size])

    def run_old_mmm(self, loss_func: Callable[[np.ndarray], float]) -> Tuple[np.ndarray, pd.DataFrame]:
        print(
            f"SHGOOptimizer -> n: {self.n}, iters: {self.iters} - "
            "Searching for good candidates..."
        )
        
        # Update FOM used for minimization objective (loss)
        self.loss_func = loss_func
        
        result = shgo(  # Minimization algorithm
            self._scalar_loss_func,
            bounds=self.bounds,
            n=self.n,
            iters=self.iters,
            sampling_method='simplicial'
        )
        # result.x is the solution array corresponding to the global minimum
        # results.xl is an ordered list of local minima solutions
        res = result.xl if result.success else result.x.reshape(1, -1)
        X_candidates = self.select_interior_candidates(res, self.selector.batch_size)

        df_scores = pd.DataFrame(loss_func(X_candidates), columns=['shgo_obj']) 

        feature_cols = [f'feature_{i+1}' for i in range(self.n_dim)]
        df_candidates = pd.DataFrame(X_candidates, columns=feature_cols)
        df_print = pd.concat([df_candidates, df_scores], axis=1, ignore_index=False)
        print(
            "SHGOOptimizer -> Selected points to be input to the simulator:\n",
            df_print
        )

        return X_candidates, df_scores

    def minimize(self) -> np.ndarray:
        result = shgo(  # Minimization algorithm
            self._scalar_loss_func,
            bounds=self.bounds,
            n=self.n,
            iters=self.iters,
            sampling_method='simplicial'
        )
        if not result.success:
            warnings.warn(
                f"SHGO optimization did not converge.\n{result.message}\n"
                f"Only one sample were found.",
                RuntimeWarning
            )
            X_res = result.x.reshape(1, -1)
        else:
            X_res = result.xl
        return X_res


@OptimizerFactory.register('ga')
class GAOptimizer(MultiModalOptimizer):
    def __init__(self, n_dim: int, batch_size: int, population_size: int, generations: int):
        super().__init__(n_dim, batch_size)
        self.population_size  = population_size
        self.generations = generations
        self.optimizer_config = {'population_size': population_size, 'generations': generations}

    def run_old_mmm(self, loss_func: Callable[[np.ndarray], float]) ->  Tuple[np.ndarray, pd.DataFrame]:
        print(
            f"GAOptimizer -> population_size: {self.population_size}, generations: {self.generations} - "
            "Searching for good candidates..."
        )

        # Update FOM used for minimization objective (loss)
        self.loss_func = loss_func

        ga = GA(  # Minimization algorithm
            self._scalar_loss_func,
            n_dim=self.n_dim,
            size_pop=self.population_size,  # Must be an even number
            max_iter=self.generations,
            prob_mut=0.1,
            lb=[val[0] for val in self.bounds],
            ub=[val[1] for val in self.bounds],
        )

        ga.run()
        population = ga.chrom2x(ga.Chrom)
        score = self.loss_func(population)
        sorted_indices = np.argsort(score)

        best_indices = sorted_indices[:self.selector.batch_size]
        X_candidates = population[best_indices]

        df_scores = pd.DataFrame(loss_func(X_candidates), columns=['shgo_obj']) 

        feature_cols = [f'feature_{i+1}' for i in range(self.n_dim)]
        df_candidates = pd.DataFrame(X_candidates, columns=feature_cols)
        df_print = pd.concat([df_candidates, df_scores], axis=1, ignore_index=False)
        print(
            "GAOptimizer -> Selected points to be input to the simulator:\n",
            df_print
        )

        return X_candidates, df_scores

    def minimize(self) -> np.ndarray:
        ga = GA(  # Minimization algorithm
            self._scalar_loss_func,
            n_dim=self.n_dim,
            size_pop=self.population_size,  # Must be an even number
            max_iter=self.generations,
            prob_mut=0.1,
            lb=[val[0] for val in self.bounds],
            ub=[val[1] for val in self.bounds],
        )
        ga.run()

        # Get the last generation population
        X_population = ga.X
        return X_population
