from typing import List, Dict, Tuple, Callable, Type, Union, Optional, ClassVar
from abc import ABC, abstractmethod
import warnings
import numpy as np
import pandas as pd

from .selector import MultiModalSelector


class MultiModalOptimizer(ABC):
    def __init__(self, n_dim: int, batch_size: int):
        """
        loss_func: Objective function for minimization. It is a
        smaller-is-better objective where smallest best value is 0.
        """
        self.selector = MultiModalSelector(batch_size)
        self.n_dim = n_dim
        self.bounds = [(0, 1)] * n_dim
        self.optimizer_config: Dict = {'dummy_param': None}  # minimizer parameters

    def run_mmm(self, loss_func: Callable[[np.ndarray], float]) -> Tuple[np.ndarray, pd.DataFrame]:
        """ Run a Multi Modal Minimization"""
    
        print(
            f"{self.__class__.__name__} -> {self.optimizer_config} - "
            "Searching for good candidates..."
        )

        # Update FOM used for minimization objective (loss)
        self.loss_func = loss_func

        # Clean multimodal progress record
        self.selector.reset_mmm_scores()

        # Run multi-objective optimization and return promising explored samples
        X = self.minimize()

        # Get objective values of the optimization explored points
        loss_values = self.loss_func(X)

        # Perform the core multi-modal optimization task
        # by selecting diverse optima from the candidate solutions
        X_batch, df_mmm_scores = self.selector.select_diverse_optima(X, loss_values)

        # Print final results
        print(
            f"{self.__class__.__name__} -> Selected points to be input to "
            f"the simulator: \n{self._concat_results(X_batch, df_mmm_scores)}"
        )

        return X_batch, df_mmm_scores

    def _scalar_loss_func(self, *args) -> float:
        """
        Wrapper for the loss function that ensures scalar output for minimizer
        compatibility. Parameter args is either an array of shape (n_dim,) or a
        tuple of 1 element containing array of shape (n_dim,). The tuple format
        is due weird output behavior of sko.GA when func is a method.
        """
        x = np.array(args).reshape(1, -1)
        return self.loss_func(x).item()

    @abstractmethod
    def minimize(self) -> np.ndarray:
        """
        Minimzation algorithm that returns an array of most promising explored
        samples
        """
        pass

    def _concat_results(self, X_batch: np.ndarray, df_mmm_scores: pd.DataFrame) -> None:
        feature_cols = [f'feature_{i+1}' for i in range(X_batch.shape[1])]
        df_candidates = pd.DataFrame(X_batch, columns=feature_cols)
        return pd.concat([df_candidates, df_mmm_scores], axis=1)
