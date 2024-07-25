"""
This is a boilerplate pipeline 'sao_optim'
generated using Kedro 0.18.5
"""
from datetime import datetime
import pandas as pd
import numpy as np
from multiprocessing import Queue
from typing import List, Dict, Tuple, Callable

import optuna

from sampler.common.data_treatment import DataTreatment
from sampler.common.storing import set_history_folder
import sampler.pipelines.sao_optim.aux_func as aux


RANDOM_STATE = 42
SIZE_OPTIM = Queue()  # use FIFO queue to order multiple acess to variable
optuna.logging.set_verbosity(optuna.logging.WARNING)


def sao_optim_from_simulator(
        treatment: DataTreatment,
        features: List[str],
        targets: List[str],
        additional_values: List[str],
        max_size: int,
        batch_size: int,
        sao_history_path: str,
        simulator_env: Dict
):
    '''
    Output values of simulation function must be in 0, 1 so that it's easier
    for optimizer to search.

    '''
    # Chose the simulation function that will be adapted for optimizer
    f_sim = set_f_sim(treatment, features, targets, additional_values, batch_size, simulator_env)
    # Set filter function (score function) that will be optimizer objective
    f_filter = set_filter(treatment)
    # Set folder where to store explored samples during optimization
    set_history_folder(sao_history_path, should_rename=True)
    # Set the objective function for the optimizer
    objective = set_objective_optuna(
        f_sim, f_filter, features, targets, additional_values, treatment, sao_history_path
    )
    # Run optimization
    X_1D, y_obj, optiminfo_dic = run_optimization_optuna(
        objective, max_size, batch_size
    )
    # Parse optimization results
    result_dic = {
        "order": SIZE_OPTIM.get(),
        "X": treatment.scaler.transform_features(X_1D.reshape((1, -1))).ravel(),
        "y_obj": y_obj,
        **optiminfo_dic,
    }
    result_df = pd.DataFrame.from_records(result_dic)
    return dict(history={}, optim_res=result_df)


def set_f_sim(
    treatment: DataTreatment, features: List[str], targets: List[str],
    additional_values: List[str], batch_size: int, simulator_env: Dict
) -> Callable[[np.ndarray, float], Tuple[np.ndarray, np.ndarray]]:
    '''
    Simulation function must return y_sim (targets) and additional interesting values
    y_doi (doi = data of interest). Both 2D arrays.
    '''
    def f_sim(x, size_optim): return aux.sao_run_simulator(
        new_x=x, features=features, targets=targets,
        additional_values=additional_values, treatment=treatment, real_x=False,
        simulator_env=simulator_env, index=size_optim, n_proc=batch_size
    )
    return f_sim


def set_filter(
    treatment: DataTreatment, minimize: bool = True
) -> Callable[[np.ndarray], np.ndarray]:
    '''
    Set the filter function that computes weither the output of the simulation
    function is in the interest zone.
    Caution: If the optimization direction is minimizing the filter should be smaller
    (or negative) for better candidates.
    '''
    L_normalized = np.array([bounds[0] for bounds in treatment.scaled_interest_region.values()])
    U_normalized = np.array([bounds[1] for bounds in treatment.scaled_interest_region.values()])
    direction = -1 if minimize else 1

    def f_filter(y: np.ndarray) -> np.ndarray:
        # Best value is =1
        y_multi_obj = aux.linear_tent(y, L_normalized, U_normalized, slope=direction * 2.0)
        y_obj = y_multi_obj.sum(axis=1)/y.shape[1]
        return y_obj

    return f_filter


def set_objective_optuna(
    f_sim: Callable, f_filter: Callable, features: List[str], targets: List[str],
    additional_values: List[str], treatment: DataTreatment, sao_history_path: str
):
    '''
        Objective is the function that optuna optimizer takes as argument.
    '''
    global SIZE_OPTIM
    SIZE_OPTIM.put(0)

    def objective(trial):
        global SIZE_OPTIM
        x = np.array([
            trial.suggest_float(name=feature, low=0, high=1)
            for feature in features
        ])
        x = x.reshape((1, len(features)))
        # Get actual size and increment
        size_optim = SIZE_OPTIM.get()
        SIZE_OPTIM.put(size_optim + 1)
        print(
            f'optuna objective -> '
            + f'size_optim: {size_optim}, '
            + f'trial._trial_id: {trial._trial_id}, '
            + f'trial.study._study_id: {trial.study._study_id} |'
        )
        # Run simulation to get target values
        y_sim, y_doi = f_sim(x, size_optim)
        # Apply filter to compute optimizer objective (score)
        y_obj = f_filter(y_sim)
        # Create a DataFrame from X and Y
        new_df = pd.DataFrame(
            np.hstack((
                x, y_sim, y_doi, y_obj.reshape((1, 1))
            )).reshape((1, -1)),
            columns=features + targets + additional_values + ['y_obj']
        )
        # Add column is_interest with True if row is inside the interest region
        new_df = treatment.classify_scaled_interest(new_df)
        # Add datetime
        timenow = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_df['datetime'] = timenow
        # Save the result immediately
        print(f'sao_optim.nodes.objective -> size_optim: {size_optim}, new_df: \n{new_df}')
        file_name = (
            f'[{str(size_optim).zfill(3)}-{str(size_optim+1).zfill(3)}]'
            # + '_' + datetime.now().strftime("%m-%d_%H-%M-%S")
        )
        aux.store_df(
            df=new_df,
            history_path=sao_history_path,
            file_name=file_name
        )
        return y_obj
    
    return objective


# https://optuna.readthedocs.io/en/stable/reference/logging.html
def run_optimization_optuna(objective, max_size, batch_size):
    '''
    Argmax of figure of merit (fom)

    Return
    ------
    X: array
        X.shape == (p_x,)
        Point where objective is minimal
    y_obj: float
        Maximum obtained objective
    optiminfo_dic: dict of best trial
    '''
    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        direction='minimize',
    )
    print(f'Optimizing using optuna with max_size: {max_size}, batch_size: {batch_size}')
    print(f'Optuna study.optimize start time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    study.optimize(func=objective, n_trials=max_size, timeout=None, n_jobs=batch_size)

    print(f'Optuna study.optimize end time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    X = np.array([*study.best_params.values()])
    y_obj = study.best_value
    optiminfo_dic = {
        "optim_n_trial": len(study.trials),
        "optim_n_besttrial": study.best_trial.number,
    }
    return X, y_obj, optiminfo_dic
