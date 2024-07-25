import pandas as pd
import numpy as np
import os
from kedro.config import ConfigLoader
from typing import List, Dict, Tuple

from sampler.common.data_treatment import DataTreatment
from sampler.common.storing import set_history_folder
from sampler.models.wrapper_for_0d import run_simulation, run_fake_simulator


def sao_run_simulator(
        new_x: np.ndarray, features: List[str], targets: List[str], additional_values: List[str],
        treatment: DataTreatment, real_x: bool, simulator_env: Dict, index: int , n_proc: int = 1
) -> Tuple[np.ndarray, np.ndarray]:

    if real_x:
        x_real = new_x
    else:
        new_xy = np.column_stack((new_x, np.zeros((new_x.shape[0], len(targets)))))
        x_real = treatment.scaler.inverse_transform(new_xy)[:, :len(features)]
    real_df = pd.DataFrame(x_real[:, :len(features)], columns=features)

    # Create map_dir where the simulation files are stored
    set_history_folder(simulator_env['map_dir'], should_rename=False)

    if simulator_env['use']:
        results_df = run_simulation(
            x=real_df, n_proc=n_proc, index=index, map_dir=simulator_env['map_dir']
        )
        new_points = pd.concat([real_df, results_df], axis=1)
    else:
        new_points = run_fake_simulator(
            x_real, features, targets, additional_values, treatment.scaler
        )
    
    default_target_values = treatment.defaults  # dict of default values for targets
    new_points = new_points.fillna({k: v for k, v in default_target_values.items() if k in targets})

    y_sim = new_points[targets].values
    y_sim = treatment.scaler.transform_targets(y_sim.reshape(-1, len(targets)))

    y_doi = new_points[additional_values].values
    return y_sim, y_doi


def get_history_path() -> str:
    '''
    TODO yasser:
    This function works but only on base.
    It needs to have acces to the name of session environement 
    '''
    conf_loader = ConfigLoader(conf_source="conf")
    conf_catalog = conf_loader.get("catalog.yml")
    history_path = conf_catalog["fom_history"]["path"]
    print(f'get_history_path -> history_path: \n{history_path}')
    # from kedro.framework.session.session import get_current_session
    # session = get_current_session()
    # context = session.load_context()
    # context.params
    return history_path


def store_df(df: pd.DataFrame, history_path: str, file_name: str):
    
    # Check if file name ends with tile extension
    if not file_name.endswith('.csv'):
        file_name += '.csv'

    # Save file
    file_path = os.path.abspath(os.path.join(history_path, file_name))
    print(f'sao_optim.nodes.store_df -> file_path: \n{file_path}')
    df.to_csv(file_path)


def linear_tent(x: np.ndarray, L: np.ndarray, U: np.ndarray, slope: float=1.0):
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
    if x.ndim != 2:
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


def gaussian_tent(x, L, U, sigma=None):
    """
    Returns a function that is equal to 1 on the interval [L, U],
    and decreases towards negative infinity as x goes to +/- infinity.
    
    Args:
        x (float or np.array): Input value(s)
        L (float): Lower bound of the interval
        U (float): Upper bound of the interval
        sigma (float): Standard deviation of the Gaussian function
    
    Returns:
        float or np.array: Function values
    """
    if sigma is None:
        sigma = (U - L) / np.sqrt(8 * np.log(2))

    center = (U+L)/2  # Center of interval
    half_width = (U-L)/2  # Half interval width
    dist_from_center = np.abs(x - center)
    # x_dist is distance from interval: =0 inside [L, U] and >0 outside
    x_dist = np.max([dist_from_center - half_width, np.zeros_like(x)], axis=0)

    y = np.exp(-x_dist**2 / (2*sigma**2))

    return y


def sigmoid_tent(x, L, U, k=None):
    """
    A smooth step function that is equal to 1 on the interval [L, U] and 
    smoothly decreases to 0 outside this interval.
    
    Parameters:
    x (numpy array): The input values.
    L (float): The lower bound of the interval.
    U (float): The upper bound of the interval.
    k (float): The steepness of the transition.
    
    Returns:
    numpy array: The output of the smooth step function.
    """
    if L >= U:
        raise ValueError("L should be less than U")
    
    if k is None:
        k = 8 * np.log(2) / (U - L)

    def sigmoid(t):
        return 1 / (1 + np.exp(-t))

    center = (U+L)/2  # Center of interval
    half_width = (U-L)/2  # Half interval width
    dist_from_center = np.abs(x - center)
    # x_dist is distance from interval: =0 inside [L, U] and >0 outside
    x_dist = np.max([dist_from_center - half_width, np.zeros_like(x)], axis=0)

    y = 2*sigmoid(-k * x_dist)
    
    return y
