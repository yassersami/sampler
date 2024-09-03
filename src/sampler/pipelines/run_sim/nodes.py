"""
This is a boilerplate pipeline 'run_sim'
generated using Kedro 0.18.5
"""

from typing import List, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import qmc

from sampler.common.data_treatment import DataTreatment
from sampler.common.simulator import SimulationProcessor
from sampler.common.storing import parse_results


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
        l_bounds = [variables_ranges[feature]['bounds'][0] for feature in features]
        u_bounds = [variables_ranges[feature]['bounds'][1] for feature in features]
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
    data: pd.DataFrame, treatment: 'DataTreatment',
    features: List[str], targets: List[str], additional_values: List[str],
    simulator_config: Dict, batch_size: int, n_proc: int, output_is_real: bool
):
    # Ensure 'r_ext_pMeO' is in features by copying 'r_ext_pAl' if necessary
    if 'r_ext_pMeO' not in features:
        data['r_ext_pMeO'] = data['r_ext_pAl']
    
    # Initialize the simulation processor, n_proc to None to use cpu_count()
    simulator = SimulationProcessor(
        features=features, targets=targets, additional_values=additional_values,
        treatment=treatment, n_proc=n_proc, simulator_config=simulator_config
    )

    n_total = 0  # Total launched number of simulations
    iteration = 0
    max_size = data.shape[0]

    # Initialize tqdm progress bar
    progress_bar = tqdm(total=max_size, dynamic_ncols=True)

    while n_total < max_size:
        print(f"Iteration {iteration:03} - Total size {n_total}")

        # Run the simulation
        nex_x = data.loc[n_total: n_total + batch_size - 1, features].values
        new_df = simulator.process_data(
            nex_x, real_x=True, index=n_total, treat_output=(not output_is_real)
        )
        
        # Yield results
        yield parse_results(new_df, current_history_size=n_total)
        
        # Update counters
        n_new_samples = new_df.shape[0]
        n_total += n_new_samples
        iteration += 1

        # Update progress bar
        progress_bar.update(n_new_samples)

    progress_bar.close()
