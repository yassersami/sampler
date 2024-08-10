import sys
from typing import List, Dict

import numpy as np
import pandas as pd
from scipy.stats import qmc

from sampler.common.data_treatment import DataTreatment
from sampler.models.wrapper_for_0d import SimulationProcessor

RANDOM_STATE = 42


def prepare_simulator_inputs(
    treatment: DataTreatment, features: List[str], targets: List[str],
    use_lhs: bool, num_samples: int, variables_ranges: Dict[str, Dict],
    csv_file: str, csv_is_real: bool,
) -> pd.DataFrame:
    """
    Prepare a scaled (not real) DataFrame for simulator inputs either by generating LHS
    samples or reading from a CSV file.

    Parameters:
    - use_lhs (bool): Whether to generate LHS samples. If False, data will be read from a CSV file.
    - num_samples (int): Number of LHS samples to generate. Required if use_lhs is True.
    - csv_file (str): Path to a CSV file containing existing data. Required if use_lhs is False.
    - csv_is_real (str): True if data in csv are in real (not scaled/normalized) scale.

    Returns:
    - pd.DataFrame: Only feature columns with real (not scaled/normalized) values.
    """

    if use_lhs:
        
        # Generate LHS samples using scipy.stats.qmc
        num_features = len(features)
        sampler = qmc.LatinHypercube(d=num_features)
        samples_0_1 = sampler.random(n=num_samples)
        
        # Scale the samples to the specified bounds
        l_bounds = [variables_ranges[feature]["bounds"][0] for feature in features]
        u_bounds = [variables_ranges[feature]["bounds"][1] for feature in features]
        samples_real = qmc.scale(samples_0_1, l_bounds, u_bounds)
        
        df_real = pd.DataFrame(samples_real, columns=features)
        
        # Handle discretes values 
        for feature in features:
            discrete_steps = variables_ranges[feature]['discete_steps']
            if discrete_steps:  # not 0
                df_real[feature] = df_real[feature].apply(
                    lambda x: round(x/discrete_steps)*discrete_steps
                )
    else:
        # Load data from CSV file
        df_csv = pd.read_csv(csv_file)
        # Inverse transform in case of scaled data
        if not csv_is_real:
            df_csv[targets] = 10  # Dummy value, avoid 0 in case of log scaling
            xy_real = treatment.scaler.inverse_transform(df_csv[features + targets].values)
            df_real = pd.DataFrame(xy_real[:, :len(features)], columns=features)

    return df_real


def evaluate_inputs(
    df_inputs: pd.DataFrame, treatment: DataTreatment,
    features: List['str'], targets: List['str'], additional_values: List['str'],
    simulator_env: Dict, n_proc: int, output_is_real: bool
) -> Dict[str, pd.DataFrame]:

    if "r_ext_pMeO" not in features:
        df_inputs['r_ext_pMeO'] = df_inputs['r_ext_pAl']
    
    # Set simulator environement
    simulator = SimulationProcessor(
        features=features, targets=targets, additional_values=additional_values,
        treatment=treatment, n_proc=n_proc, simulator_env=simulator_env
    )
    # Run simulation with possibility of treating output or not
    df_results = simulator.process_data(
        new_x=df_inputs.values, real_x=True, index=0, treat_output=(not output_is_real)
    )
    
    return df_results
