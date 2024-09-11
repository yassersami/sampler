from typing import List, Dict, Tuple, Callable, Type, Union, Optional, ClassVar
from abc import ABC, abstractmethod
import warnings
import cProfile
import pstats
from tqdm import tqdm
import numpy as np
import pandas as pd


class MultiModalSelector(ABC):

    # Multimodal progress record attributes
    _record_columns = ['dummy_score', 'dummy_loss']
    _records = {}

    def __init__(self, batch_size: int):
        self.batch_size = batch_size

    @abstractmethod
    def select_diverse_minima(self, X: np.ndarray, *args, **kwargs) -> pd.DataFrame:
        """
        Select diversified batch of candidates among explored points.
        """
        pass

    @staticmethod
    def _normalize(values: np.ndarray) -> np.ndarray:
        """Normalize values to [0, 1] range."""
        min_val, max_val = values.min(), values.max()
        return (values - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(values)

    def reset_records(self):
        self._records = {name: [] for name in self._record_columns}

    def save_records(self, **kwargs):
        # Store selection progress
        for key in self._record_columns:
            if key not in kwargs.keys():
                raise KeyError(f"Multimodal selection progess record is missing '{key}'")

            self._records[key].append(kwargs[key])

    def get_records_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._records)


class MultiModalOptimizer(ABC):
    def __init__(self, n_dim: int, selector: MultiModalSelector):
        """
        loss_func: Objective function for minimization. It is a
        smaller-is-better objective where smallest best value is 0.
        """
        self.selector = selector
        self.n_dim = n_dim
        self.bounds = [(0, 1)] * n_dim
        self.optimizer_config: Dict = {'dummy_param': None}  # minimizer parameters
        self.pbar = None
        self.activate_profiling = False

    @property
    def total_evaluations(self) -> int:
        if 'n' in self.optimizer_config and 'iters' in self.optimizer_config:
            return self.optimizer_config['n'] * self.optimizer_config['iters']
        elif 'population_size' in self.optimizer_config and 'generations' in self.optimizer_config:
            return self.optimizer_config['population_size'] * self.optimizer_config['generations']
        else:
            raise ValueError(f"Unknown optimizer configuration: {self.optimizer_config}")

    def run_multimodal_optim(self, loss_func: Callable[[np.ndarray], float]) -> np.ndarray:
        """ Run Multi Modal Minimization"""
    
        print(
            f"{self.__class__.__name__} -> {self.optimizer_config} - "
            "Searching for good candidates..."
        )

        # Update FOM used for minimization objective (loss)
        self.loss_func = loss_func

        # Clean multimodal progress record
        self.selector.reset_records()

        # Initialize progress bar
        self.pbar = tqdm(total=self.total_evaluations, desc="Multimodal Optimization")

        if self.activate_profiling:
            # Enable profiling
            profiler = cProfile.Profile()
            profiler.enable()

        # Run multi-objective optimization and return local minima
        X = self.minimize()

        if self.activate_profiling:
            # Disable profiling
            profiler.disable()
            stats = pstats.Stats(profiler).sort_stats('cumulative')
            stats.print_stats()

        # Close progress bar
        self.pbar.close()
        print('\n')  # Avoid confusion

        # Get loss objective values of detected minima during the optimization 
        loss_values = self.loss_func(X)

        # Perform the core multi-modal optimization task
        # by selecting diverse minima from the candidate solutions
        X_batch = self.selector.select_diverse_minima(X, loss_values)

        return X_batch

    def _scalar_loss_func(self, *args) -> float:
        """
        Wrapper for the loss function that ensures scalar output for minimizer
        compatibility. Parameter args is either an array of shape (n_dim,) or a
        tuple of 1 element containing array of shape (n_dim,). The tuple format
        is due weird output behavior of sko.GA when func is a method.
        """
        x = np.array(args).reshape(1, -1)
        loss_value = self.loss_func(x).item()
        self.pbar.update(1)
        return loss_value

    @abstractmethod
    def minimize(self) -> np.ndarray:
        """
        Minimzation algorithm that returns an array of most promising explored
        samples
        """
        pass
