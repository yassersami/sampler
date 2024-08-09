import pandas as pd
import numpy as np
import os
from typing import List, Dict, Tuple


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
    x = np.atleast_2d(x)
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
