"""
This is a boilerplate pipeline 'parego'
generated using Kedro 0.18.5
"""
from datetime import datetime
import random
from typing import List, Dict, Tuple
from tqdm import tqdm
import warnings
import itertools

import pandas as pd
import numpy as np

from sampler.common.data_treatment import DataTreatment, initialize_dataset
from sampler.common.storing import results
from sampler.models.fom import GPSampler
from sampler.models.wrapper_for_0d import SimulationProcessor  # get_values_from_simulator

from scipy.stats import norm
from sko.GA import GA

# PAPER : https://link.springer.com/chapter/10.1007/978-3-540-31880-4_13

RANDOM_STATE = 42


def run_parego(
        data: pd.DataFrame, treatment: DataTreatment,
        features: List[str], targets: List[str], additional_values: List[str],
        simulator_env: Dict, max_size: int, batch_size: int = 1, llambda_s: int=100,
        tent_slope: float=10, experience: str="parEGO_maxIpr"
):

    res = initialize_dataset(
        data=data, features=features, targets=targets, treatment=treatment,
    )
    yield results(res, size=len(res), initialize=True)

    lambda_gen = LambdaGenerator(k=len(targets), s=llambda_s)
    dace = DACEModel(
        features=features, targets=targets,
        scaled_regions=treatment.scaled_interest_region, tent_slope=tent_slope,
        experience=experience
    )
    simulator = SimulationProcessor(
        features=features, targets=targets, additional_values=additional_values,
        treatment=treatment, n_proc=batch_size, simulator_env=simulator_env
    )

    # Initialize tqdm progress bar with estimated time remaining
    progress_bar = tqdm(total=max_size, dynamic_ncols=True)
    size = 0
    iteration = 0
    while size < max_size:
        iteration += 1
        print(f'Round {iteration:03} ' + '-'*80)
        x_pop = res[features].values
        y_pop = res[targets].values
        llambda = lambda_gen.choose_uniform_lambda()
        # Prepare train data and train GP
        dace.update_model(x_pop, y_pop, llambda)
        # Search new candidates to add to res dataset
        new_x = EvolAlg(dace, x_pop, batch_size=batch_size)
        # Launch time expensive simulations
        new_df, error_features = simulator.process_data(new_x, real_x=False, index=size)
        dace.model.add_ignored_points(error_features)

        print(f'Round {iteration:03} (continued): simulation results' + '-'*49)
        print(f'run_parego -> Got {len(new_df)} new samples after simulation:\n {new_df}')
        if len(new_df) == 0:
            warnings.warn("run_parego -> No new data was obtained from simulator !")
            continue

        # Add interesting informations about samples choice
        prediction = dace.model.predict(new_df[features].values)
        prediction_cols = [f'pred_{t}' for t in dace.model_targets]
        new_df[prediction_cols] = prediction if len(dace.model_targets) > 1 else prediction.reshape(-1, 1)
        score = dace.get_score(new_df[features].values)
        new_df["obj_score"] = score
        new_df = treatment.classify_scaled_interest(new_df)
        timenow = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_df['datetime'] = timenow
        new_df['iteration'] = iteration

        # Concatenate new values to original results DataFrame
        res = pd.concat([res, new_df], axis=0, ignore_index=True)
        size += len(new_df)

        # Print some informations
        iter_interest_count = (new_df['quality']=='interest').sum()
        total_interest_count = (res['quality']=='interest').sum()
        print(f'run_parego -> Final batch data that wil be stored:\n {new_df}')
        print(f'run_parego -> [batch  report] new points: {len(new_df)}, interesting points: {iter_interest_count}')
        print(f'run_parego -> [global report] progress: {size}/{max_size}, interesting points: {total_interest_count}')
        yield results(res, size=len(new_df))
        progress_bar.update(len(new_df))
    progress_bar.close()


class DACEModel:
    def __init__(
        self, features: List[str], targets: List[str], scaled_regions: dict,
        tent_slope= float, experience: str="parEGO_maxIpr"
    ):
        self.features = features
        self.targets = targets
        self.tent_slope = tent_slope
        self.llambda = None
        self.y_max = None
        self.model = None
        self.model_targets = None
        self.scaled_regions = scaled_regions
        self.L = np.array([scaled_regions[target][0] for target in targets])
        self.U = np.array([scaled_regions[target][1] for target in targets])
        # Set conditions that defines parego pipeline operations 
        if experience == 'parEGO_maxIpr':
            self.use_linear_tent = False
            self.use_tcheby = True
            self.use_maxIpr = True
            self.use_interest = False
        elif experience == 'parEGO_Tent':
            self.use_linear_tent = True
            self.use_tcheby = True
            self.use_maxIpr = True
            self.use_interest = False
        elif experience == 'parEGO_Itr':
            self.use_linear_tent = False
            self.use_tcheby = True
            self.use_maxIpr = False
            self.use_interest = True
        elif experience == 'parEGO_GP':
            self.use_linear_tent = False
            self.use_tcheby = False
            self.use_maxIpr = False
            self.use_interest = True

    def update_model(self, x_pop: np.array, y_pop: np.array, llambda: tuple):
        # Update llambda
        self.llambda = llambda

        # Prepare target to train on
        if self.use_linear_tent:
            y_pop = linear_tent(y_pop, L=self.L.reshape(1, -1), U=self.U.reshape(1, -1), slope=self.tent_slope)
        if self.use_tcheby:
            y_pop = tchebychev(y_pop, self.llambda)
            self.model_targets = ['f_llambda']
        else:
            self.model_targets = self.targets

        # Train GP model
        self.model = GPSampler(features=self.features, targets=self.model_targets)
        self.model.fit(x_train=x_pop, y_train=y_pop)

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
        hypercube_vertices = itertools.product(*[self.scaled_regions[target] for target in self.targets])
        hypercube_vertices = np.array(list(hypercube_vertices))
        segment_endpoints = tchebychev(hypercube_vertices, self.llambda)
        Ltcheby, Utcheby = segment_endpoints.min(), segment_endpoints.max()
        return Ltcheby, Utcheby

    def excpected_interest(self, x: np.array):
        '''
        This score computes the probability of being in the region of interest.

        x.shape = (p,) => sigma float
        x.shape = (n, p) => sigma.shape = (n,)

        CDF: cumulative distribution function P(X <= x)
        '''
        x = x.reshape(1, -1) if len(x.shape) == 1 else x
        if self.model is None:
            raise ValueError("The model must be fitted before calling get_score.")

        y_hat, y_std = self.model.predict(x, return_std=True)
        point_norm = norm(loc=y_hat, scale=y_std)
        probabilities = point_norm.cdf(self.Unorm) - point_norm.cdf(self.Lnorm)
        if y_hat.ndim == 1:
            imp = probabilities
        else:
            imp = np.prod(probabilities, axis=1)
        return imp

    def expected_improvement(self, x: np.array):
        '''
        This score is inspired from EGO (Efficient Global Optimization)
        x.shape = (p,) => sigma float
        x.shape = (n, p) => sigma.shape = (n,)
        '''
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

    
def linear_tent(x, L, U, slope: float=1.0):
    """
    Tent function equal to 1 on interval [L, U],
    and decreasing linearly outside in both directions.

    x: shape (n, p)
    L and U: float or shape (1, p) if p > 1

    test with:
    L = array([[0.8003412 , 0.89822933]])
    U = array([[0.85116726, 0.97268397]])
    x = np.array([[0, 0], [0.8, 0.8], [0.85, 0.85], [0.9, 0.9], [1, 1]])

    Output
    ------
    y: shaped(n, p)
    """
    if not isinstance(x, np.ndarray) or x.ndim != 2:
        raise ValueError(f'x should be 2D array shaped (n, p) \nx: \n{x}')
    if np.any(L >= U):
        raise ValueError(f'L should be less than U \nL: \n{L} \nU: \n{U}')

    center = (U+L)/2  # Center of interval
    half_width = (U-L)/2  # Half interval width
    dist_from_center = np.abs(x - center)  # >= 0
    # x_dist is distance from interval: =0 inside [L, U] and >0 outside
    x_dist = np.max([dist_from_center - half_width, np.zeros_like(x)], axis=0)

    y = -slope*x_dist + 1

    return y


def tchebychev(y_pop: np.array, llambda: List[float]):
    p = 0.05
    max_llambdaf = np.array([np.max([llambda[j] * y_pop[i, j] for j in range(y_pop.shape[1])]) for i in range(y_pop.shape[0])])
    sum_llambdaf = np.sum([llambda[j] * y_pop[:, j] for j in range(y_pop.shape[1])], axis=0)
    y_pop_tchebychev = max_llambdaf + p * sum_llambdaf
    return y_pop_tchebychev


def EvolAlg(dace: DACEModel, x_pop: np.array, num_generations: int=5, population_size: int=20, batch_size: int=1):
    dimensions = x_pop.shape[1]

    def fitness_function(x):
        # Expected improvement is an increasing function of goodness of selected sample. Thus we add a minus for minization algorithm.
        x_reshaped = x.reshape(1, -1)
        score = dace.get_score(x_reshaped)
        return -score.item()
        # return -np.mean(ei)


    # Initialize the Genetic Algorithm for minimzation
    # Basic GA, I don't have implemented the real algoritmh from paper (it's not important)
    ga = GA(func=fitness_function,
            n_dim=dimensions,
            size_pop=population_size,
            max_iter=num_generations,
            prob_mut=0.1,
            lb=[0] * dimensions,  # Lower bounds
            ub=[1] * dimensions)  # Upper bounds

    ga.run()
    population = ga.chrom2x(ga.Chrom)
    fitness = np.array([fitness_function(ind) for ind in population])
    sorted_indices = np.argsort(fitness)
    
    best_indices = sorted_indices[:batch_size]
    best_solutions = population[best_indices]
    
    print("EvolAlg -> Selected points to be input to the simulator:\n", best_solutions)

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
        '''Generate all combinations of positive integers (including 0) that sum to s, with k terms, with order
           Recursive function that generates all possible combinations, not optimized'''
        sum_comb = sum(comb)
        if remaining_terms == 0:
            if sum_comb == s:
                l_comb.append(comb)
            return

        for i in range(s - sum_comb + 1):
            self.gen_comb(comb + [i], remaining_terms - 1, s, l_comb)

    def gen_lambda_set(self, k: int, s: int) -> List[Tuple[float]]:
        '''Generate all possible lambda vectors with k terms that sum to 1 according to parameter s.'''
        l_comb = []
        self.gen_comb([], k, s, l_comb)
        l_vec = [tuple(map(lambda x: x / s, comb)) for comb in l_comb]
        return l_vec

    def choose_uniform_lambda(self) -> Tuple[float]:
        '''Select a random lambda (belongs to R^k) uniformly'''
        return random.choice(self.lambda_set)
