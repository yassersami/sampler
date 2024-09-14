from typing import List, Dict, Tuple, Callable, Type, Union, Optional, ClassVar
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import shgo
from sko.GA import GA

from .base import MultiModalSelector, MultiModalOptimizer, BaseFactory


class OptimizerFactory(BaseFactory[MultiModalOptimizer]):
    @classmethod
    def create_from_config(cls,
        n_dim: int,
        selector: MultiModalSelector,
        optimizer_config: Dict[str, Dict]
    ) -> MultiModalOptimizer:
        return super().create_from_config(
            optimizer_config,
            item_type="optimizer",
            n_dim=n_dim,
            selector=selector
        )


@OptimizerFactory.register('shgo')
class SHGOOptimizer(MultiModalOptimizer):

    def __init__(self, n_dim: int, selector: MultiModalSelector, n: int, iters: int):
        super().__init__(n_dim, selector)
        self.n = n
        self.iters = iters
        self.optimizer_config = {'n': n, 'iters': iters}

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
            # result.x is the solution array corresponding to the global minimum
            X_res = result.x.reshape(1, -1)
        else:
            # results.xl is an ordered list of local minima solutions
            X_res = result.xl
        return X_res


@OptimizerFactory.register('ga')
class GAOptimizer(MultiModalOptimizer):
    def __init__(
        self,
        n_dim: int,
        selector: MultiModalSelector,
        population_size: int,
        generations: int,
        mutation_probability: float
    ):
        super().__init__(n_dim, selector)
        self.population_size  = population_size
        self.generations = generations
        self.mutation_probability = mutation_probability
        self.optimizer_config = {
            'population_size': population_size,
            'generations': generations,
            'mutation_probability': mutation_probability
        }

    def minimize(self) -> np.ndarray:
        ga = GA(  # Minimization algorithm
            self._scalar_loss_func,
            n_dim=self.n_dim,
            size_pop=self.population_size,  # Must be an even number
            max_iter=self.generations,
            prob_mut=self.mutation_probability,
            lb=[val[0] for val in self.bounds],
            ub=[val[1] for val in self.bounds],
        )
        ga.run()

        # Get the last generation population
        X_population = ga.X
        return X_population
