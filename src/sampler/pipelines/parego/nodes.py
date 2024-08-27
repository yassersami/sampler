"""
This is a boilerplate pipeline 'parego'
generated using Kedro 0.18.5
"""
from datetime import datetime
import random
from typing import List, Dict, Tuple
from tqdm import tqdm
import itertools

import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic

from sampler.common.scalers import linear_tent
from sampler.common.data_treatment import DataTreatment, initialize_dataset
from sampler.common.storing import parse_results
from sampler.common.simulator import SimulationProcessor

from scipy.stats import norm
from sko.GA import GA

# PAPER : https://link.springer.com/chapter/10.1007/978-3-540-31880-4_13

RANDOM_STATE = 42


def run_parego(
    data: pd.DataFrame, treatment: DataTreatment,
    features: List[str], targets: List[str], additional_values: List[str],
    simulator_env: Dict, batch_size: int, run_condition: Dict,
    llambda_s: int, population_size: int, num_generations: int,
    tent_slope: float=10, experience: str='parEGO_maxIpr'
):
    dace = DACEModel(
        features=features, targets=targets,
        scaled_regions=treatment.scaled_interest_region, llambda_s=llambda_s,
        tent_slope=tent_slope, experience=experience
    )
    simulator = SimulationProcessor(
        features=features, targets=targets, additional_values=additional_values,
        treatment=treatment, n_proc=batch_size, simulator_env=simulator_env
    )
    data = simulator.adapt_targets(data)

    res = initialize_dataset(data=data, treatment=treatment)
    yield parse_results(res, current_history_size=0)

    # Set progress counting variables
    max_size = run_condition['max_size']
    n_interest_max = run_condition['n_interest_max']
    run_until_max_size = run_condition['run_until_max_size']
    
    n_total = 0  # counting all simulations
    n_inliers = 0  # counting only inliers
    n_interest = 0  # counting only interesting inliers
    iteration = 1
    should_continue = True
    
    # Initialize tqdm progress bar with estimated time remaining
    progress_bar = (
        tqdm(total=max_size, dynamic_ncols=True) if run_until_max_size else 
        tqdm(total=n_interest_max, dynamic_ncols=True)
    )
    
    while should_continue:
        print(f"\nRound {iteration:03} (start) " + "-"*62)
        clean_res = res.dropna(subset=targets)
        x_pop = clean_res[features].values
        y_pop = clean_res[targets].values

        # Prepare train data and train GP
        dace.update_model(x_pop, y_pop)
        
        # Search new candidates to add to res dataset
        new_x = EvolAlg(
            dace, population_size=population_size,
            num_generations=num_generations, batch_size=batch_size
        )
        
        # Launch time expensive simulations
        new_df = simulator.process_data(new_x, real_x=False, index=n_total, treat_output=True)

        print(f"Round {iteration:03} (continued) - simulation results " + "-"*37)
        print(f'run_parego -> New samples after simulation:\n {new_df}')

        # Add interesting informations about samples choice
        prediction = dace.model.predict(new_df[features].values)
        prediction_cols = [f'pred_{t}' for t in dace.model_targets]
        new_df[prediction_cols] = (
            prediction.reshape(-1, 1) if prediction.ndim == 1 else prediction
        )
        score = dace.get_score(new_df[features].values)
        new_df['obj_score'] = score
        new_df = treatment.classify_quality_interest(new_df, data_is_scaled=True)
        timenow = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_df['datetime'] = timenow
        new_df['iteration'] = iteration
        
        # Store final batch results
        yield parse_results(new_df, current_history_size=res.shape[0])

        # Concatenate new values to original results DataFrame
        res = pd.concat([res, new_df], axis=0, ignore_index=True)
        
        # Update stopping conditions
        n_new_samples = new_df.shape[0]
        n_new_inliers = new_df.dropna(subset=targets).shape[0]
        n_new_interest = new_df[new_df['quality'] == 'interest'].shape[0]
    
        n_total += n_new_samples
        n_inliers += n_new_inliers
        n_interest += n_new_interest
        iteration += 1

        # Print iteration details
        print(
            f"Round {iteration - 1:03} (end) - Report count: "
            f"Total: {n_total}, "
            f"Inliers: {n_inliers}, "
            f"Interest: {n_interest}"
        )

        # Determine the end condition
        should_continue = (
            (n_inliers < max_size) if run_until_max_size else
            (n_interest < n_interest_max)
        )

        # Update progress bar based on the condition
        progress_bar.update(
            n_new_inliers - max(0, n_inliers - max_size) if run_until_max_size else
            n_new_interest - max(0, n_interest - n_interest_max)
        )
    progress_bar.close()


class DACEModel:
    def __init__(
        self, features: List[str], targets: List[str], scaled_regions: dict,
        llambda_s: int, tent_slope= float, experience: str='parEGO_maxIpr'
    ):
        self.features = features
        self.targets = targets
        self.lambda_gen = LambdaGenerator(k=len(targets), s=llambda_s)
        self.llambda = None
        self.tent_slope = tent_slope
        self.y_max = None
        self.model = GaussianProcessRegressor(
            kernel=RationalQuadratic(length_scale_bounds=(1e-5, 2)),
            random_state=RANDOM_STATE
        )
        self.model_targets = None
        self.scaled_regions = scaled_regions
        self.L = np.array([scaled_regions[target][0] for target in targets])
        self.U = np.array([scaled_regions[target][1] for target in targets])
        self.set_experience(experience)

    def set_experience(self, experience):
        # Set conditions that defines parego pipeline operations 
        if experience == 'parEGO_maxIpr':  # True parEGO
            self.use_linear_tent = False
            self.use_tcheby = True
            self.use_maxIpr = True
            self.use_interest = False
        elif experience == 'parEGO_Itr':  # Almost parego just changing the EI
            self.use_linear_tent = False
            self.use_tcheby = True
            self.use_maxIpr = False
            self.use_interest = True
        elif experience == 'parEGO_Tent':  # not pareto, scalarizing with tent
            self.use_linear_tent = True
            self.use_tcheby = False
            self.use_maxIpr = True
            self.use_interest = False
        elif experience == 'parEGO_GP':  # Very different
            self.use_linear_tent = False
            self.use_tcheby = False
            self.use_maxIpr = False
            self.use_interest = True

    def update_model(self, x_pop: np.array, y_pop: np.array):

        # Prepare target to train on
        if self.use_linear_tent:
            # Scalarize targets with linear tent =1 on interest region
            y_pop = linear_tent(
                y_pop, L=self.L.reshape(1, -1), U=self.U.reshape(1, -1),
                slope=self.tent_slope
            )
            y_pop = y_pop.mean(axis=1)  # mean to ensure max = 1
            self.model_targets = ['f_tent']
        elif self.use_tcheby:
            # Scalarize using pareto chosing different weights at each iteration
            self.llambda = self.lambda_gen.choose_uniform_lambda()
            y_pop = tchebychev(y_pop, self.llambda)
            self.model_targets = ['f_pareto']
        else:
            self.model_targets = self.targets

        # Train GP model
        self.model.fit(x_pop, y_pop)

        # Set y_max if maximizing improvement
        if self.use_maxIpr:
            self.y_max = np.max(y_pop, axis=0)

        # Set lower/upper bound for interest region to use in interest calculation
        if self.use_interest:
            if self.use_tcheby:
                self.Lnorm, self.Unorm = self.get_tchebychev_region()
            else:
                self.Lnorm, self.Unorm = self.L, self.U

    def get_tchebychev_region(self):
        # Find right region to get ei_interest addapted for f_llambda (tchebychev)
        hypercube_vertices = itertools.product(*[
            self.scaled_regions[target] for target in self.targets
        ])
        hypercube_vertices = np.array(list(hypercube_vertices))
        segment_endpoints = tchebychev(hypercube_vertices, self.llambda)
        Ltcheby, Utcheby = segment_endpoints.min(), segment_endpoints.max()
        return Ltcheby, Utcheby

    def excpected_interest(self, x: np.array):
        """
        This score computes the probability of being in the region of interest.

        x.shape = (p,) => sigma float
        x.shape = (n, p) => sigma.shape = (n,)

        CDF: cumulative distribution function P(X <= x)
        """
        x = np.atleast_2d(x)

        y_hat, y_std = self.model.predict(x, return_std=True)
        point_norm = norm(loc=y_hat, scale=y_std)
        probabilities = point_norm.cdf(self.Unorm) - point_norm.cdf(self.Lnorm)
        if y_hat.ndim == 1:
            imp = probabilities
        else:
            imp = np.prod(probabilities, axis=1)
        return imp

    def expected_improvement(self, x: np.array):
        """
        This score is inspired from EGO (Efficient Global Optimization)
        x.shape = (p,) => sigma float
        x.shape = (n, p) => sigma.shape = (n,)
        """
        x = x.reshape(1, -1) if len(x.shape) == 1 else x
        if self.y_max is None:
            raise ValueError("The model must be updated before calling get_score.")
        
        mu, sigma = self.model.predict(x, return_std=True)
        imp = mu - self.y_max

        mask = sigma > 0
        Z = np.zeros_like(sigma)
        Z[mask] = imp[mask] / sigma[mask]
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        imp[mask] = ei[mask]
        return imp

    def get_score(self, x: np.array):
        if self.use_maxIpr:
            return self.expected_improvement(x)
        elif self.use_interest:
            return self.excpected_interest(x)


def tchebychev(y_pop: np.array, llambda: List[float]):
    p = 0.05
    max_llambdaf = np.array([
        np.max([llambda[j] * y_pop[i, j] for j in range(y_pop.shape[1])])
        for i in range(y_pop.shape[0])
    ])
    sum_llambdaf = np.sum(
        [llambda[j] * y_pop[:, j] for j in range(y_pop.shape[1])],
        axis=0
    )
    y_pop_tchebychev = max_llambdaf + p * sum_llambdaf
    return y_pop_tchebychev


def EvolAlg(
    dace: DACEModel, population_size: int=50, num_generations: int=1000,
    batch_size: int=1
) -> np.ndarray:
    dimensions = len(dace.features)

    def fitness_function(x):
        # Expected improvement is an increasing function of goodness of selected sample. Thus we add a minus for minization algorithm.
        x_reshaped = x.reshape(1, -1)
        score = dace.get_score(x_reshaped)
        return -score.item()

    # Initialize the Genetic Algorithm for minimzation
    # Basic GA, I don't have implemented the real algoritmh from paper (it's not important)
    ga = GA(
        func=fitness_function,
        n_dim=dimensions,
        size_pop=population_size,
        max_iter=num_generations,
        prob_mut=0.1,
        lb=[0] * dimensions,  # Lower bounds
        ub=[1] * dimensions  # Upper bounds
    )

    print(
        'EvolAlg -> Searching for good candidates using the fitness function '
        f'through {num_generations} generations of {population_size} '
        'population size...'
    )
    ga.run()
    population = ga.chrom2x(ga.Chrom)
    fitness = np.array([fitness_function(ind) for ind in population])
    sorted_indices = np.argsort(fitness)
    
    best_indices = sorted_indices[:batch_size]
    best_solutions = population[best_indices]
    
    print(
        'EvolAlg -> Selected points to be input to the simulator:\n'
        f'{best_solutions}'
    )

    return best_solutions


class LambdaGenerator:
    def __init__(self, k: int, s: int):
        # k : number of terms in lambda
        # s : sum of terms in lambda (where all terms are multiplied by s)
        self.k = k
        self.s = s
        self.lambda_set = self.gen_lambda_set(k, s)
        self.seed = RANDOM_STATE
        random.seed(self.seed)

    def gen_comb(self, comb: List[int], remaining_terms: int, s: int, l_comb: List[List[int]]) -> None:
        """
        Generate all combinations of positive integers (including 0) that sum to
        s, with k terms, with order Recursive function that generates all
        possible combinations, not optimized
        """
        sum_comb = sum(comb)
        if remaining_terms == 0:
            if sum_comb == s:
                l_comb.append(comb)
            return

        for i in range(s - sum_comb + 1):
            self.gen_comb(comb + [i], remaining_terms - 1, s, l_comb)

    def gen_lambda_set(self, k: int, s: int) -> List[Tuple[float]]:
        """Generate all possible lambda vectors with k terms that sum to 1 according to parameter s."""
        l_comb = []
        self.gen_comb([], k, s, l_comb)
        l_vec = [tuple(map(lambda x: x / s, comb)) for comb in l_comb]
        return l_vec

    def choose_uniform_lambda(self) -> Tuple[float]:
        """Select a random lambda (belongs to R^k) uniformly"""
        return random.choice(self.lambda_set)
