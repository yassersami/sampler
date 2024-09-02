from typing import List, Dict, Tuple, Callable, Type, Union, Optional, ClassVar
from abc import ABC, abstractmethod
import warnings
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import shgo
from sko.GA import GA


class DiversityMultiModalSelector:
    def __init__(self, batch_size: int):
        self.batch_size = batch_size

        # Multimodal progress record attributes
        self._mmm_score_names = ['_id', 'loss', 'diversity', 'extremeness', 'normal_loss', 'normal_diversity', 'mmm_loss']
        self.reset_mmm_scores()

    def reset_mmm_scores(self):
        self._mmm_scores = {name: [] for name in self._mmm_score_names}

    def select_diverse_optima(
        self, X: np.ndarray, loss_values: np.ndarray
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Select diversified batch of candidates among explored points during
        minimization.
        """
        X = np.atleast_2d(X)

        # Normalize objective values to [0, 1] range
        normalized_loss = self._normalize(loss_values)

        # Calculate pairwise distances between individuals
        distances = squareform(pdist(X))

        # Edge proximity loss (extremeness to minimize)
        edge_proximity_mask = (np.isclose(X, 0) | np.isclose(X, 1))  # Shape of X
        extremeness = edge_proximity_mask.mean(axis=1)

        # Initialize selected candidates with the best sample
        first_index = np.argmin(loss_values)  # best sample with smallest objective value
        selected_indices = [first_index]

        # Initialize list of not yet selected indices
        n_samples = X.shape[0]
        available_indices = list(range(n_samples))
        available_indices.remove(first_index)

        # Set dict to store selection progress
        self._store_mmm_progress(
            _id             =first_index,
            loss            =loss_values[first_index],
            diversity       =np.nan,
            extremeness     =extremeness[first_index],
            normal_loss     =normalized_loss[first_index],
            normal_diversity=np.nan,
            mmm_loss        =normalized_loss[first_index],
        )

        # Select diverse candidates
        for _ in range(1, min(self.batch_size, n_samples)):

            # Calculate diversity score: mean of distances to already selected candidates
            diversity_scores = distances[:, selected_indices].mean(axis=1)

            # Normalize diversity scores to [0, 1] range
            normalized_diversity = self._normalize(diversity_scores)

            # Combine normalized objective and diversity scores
            # objective is minimized while diversity is maximized
            combined_loss = normalized_loss - normalized_diversity  + 2 * extremeness  # This is the main operation

            # Remove rows of already selected indices
            available_combined_loss = [combined_loss[i] for i in available_indices]

            # Select the individual with the smallest loss
            next_index = available_indices[np.argmin(available_combined_loss)]

            selected_indices.append(next_index)
            available_indices.remove(next_index)

            # Store selection progress
            self._store_mmm_progress(
                _id             =next_index,
                loss            =loss_values[next_index],
                diversity       =diversity_scores[next_index],
                extremeness     =extremeness[next_index],
                normal_loss     =normalized_loss[next_index],
                normal_diversity=normalized_diversity[next_index],
                mmm_loss        =combined_loss[next_index],
            )

        return X[selected_indices], pd.DataFrame(self._mmm_scores)

    @staticmethod
    def _normalize(values: np.ndarray) -> np.ndarray:
        """Normalize values to [0, 1] range."""
        min_val, max_val = values.min(), values.max()
        return (values - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(values)

    def _store_mmm_progress(self, **kwargs):
        # Store selection progress
        for key in self._mmm_score_names:
            if key not in kwargs.keys():
                raise KeyError(f"Multimodal selection progess record is missing '{key}'")

            self._mmm_scores[key].append(kwargs[key])

class BoundaryMultiModalSelector():

    @classmethod
    def sort_by_relevance(cls, X_unique: np.ndarray, mask: np.ndarray) -> List:
        n_dim = X_unique.shape[1]
        bounds = [[] for _ in range(n_dim + 1)]
        for x_row, cond in zip(X_unique, mask):
            cond_sum = cond.sum()
            # TODO why not directly bounds[cond_sum].append since cond_sum is in [0, n_dim] anyway ?
            for local_sum in range(n_dim + 1):
                if cond_sum == local_sum:
                    bounds[local_sum].append(x_row)
        return [
            item
            for val in bounds
            for item in val
        ]

    @classmethod
    def select_diverse_optima(cls, X_candidates: np.ndarray, batch_size: int) -> np.ndarray:
        X_unique = np.unique(X_candidates, axis=0)
        if len(X_unique) <= batch_size:
            # if not enough candidates return all available ones
            return X_candidates

        # If a lot of candidates, select best ones
        mask = (np.isclose(X_unique, 0) | np.isclose(X_unique, 1))
        X_unique_sorted = cls.sort_by_relevance(X_unique, mask)
        return np.array(X_unique_sorted[:batch_size])


class MultiModalSelectorFactory():
    boundary_mms: BoundaryMultiModalSelector
    diversity_mms: DiversityMultiModalSelector

    def __init__(self, name):
        if name == 'boundary_selector':
            self.selector = BoundaryMultiModalSelector()
        elif name == 'diversity_selector':
            self.selector = DiversityMultiModalSelector()
        else:
            raise AttributeError(f"{name} is unknown multimodal selector")


class MultiModalOptimizer(ABC):
    def __init__(self, n_dim: int, batch_size: int):
        """
        loss_func: Objective function for minimization. It is a
        smaller-is-better objective where smallest best value is 0.
        """
        self.selector = DiversityMultiModalSelector(batch_size)
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
