from typing import List, Dict, Tuple, Callable, Type, Union, Optional, ClassVar
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import shgo
from sko.GA import GA

from .base import MultiModalSelector, MultiModalOptimizer


class OptimizerFactory:
    _optimizers: Dict[str, Type[MultiModalOptimizer]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(optimizer_class: Type[MultiModalOptimizer]):
            cls._optimizers[name] = optimizer_class
            return optimizer_class
        return decorator

    @classmethod
    def create_from_config(
        cls,
        n_dim: int,
        selector: MultiModalSelector,
        optimizer_config: Dict[str, Dict]
    ) -> MultiModalOptimizer:

         # Check if all configs have the 'apply' key
        for name, config in optimizer_config.items():
            if 'apply' not in config:
                raise KeyError(
                    f"Optimizer '{name}' is missing the 'apply' key "
                    "in its configuration."
                )

        # Gather all active optimizers
        applied_optimizers = [
            name for name, config in optimizer_config.items()
            if config['apply']
        ]

        if len(applied_optimizers) != 1:
            raise ValueError(
                "Exactly one optimizer must be applied. "
                f"Found {len(applied_optimizers)}"
            )

        # Set selected optimizer
        optimizer_name = applied_optimizers[0]
        optimizer_args = optimizer_config[optimizer_name]
        optimizer_args.pop('apply')

        if optimizer_name not in cls._optimizers:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        OptimizerClass = cls._optimizers[optimizer_name]
        return OptimizerClass(n_dim=n_dim, selector=selector, **optimizer_args)

    @classmethod
    def list_optimizers(cls):
        return list(cls._optimizers.keys())


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
