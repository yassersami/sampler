import os
import warnings
import multiprocessing
from typing import List, Dict, Tuple, Union

import numpy as np
import pandas as pd
import time

from sampler.common.data_treatment import DataTreatment
from sampler.common.storing import set_history_folder

import simulator_0d.src.py0D.main as simulator_0d
from simulator_0d.src.py0D.functions_pp import DB_simu_pp as SimPostProc
from simulator_0d.src.py0D.map_generator import set_simu_name, map_generator
from simulator_0d.src.pypp.launch import set_inputs_dic, adapt_inputs_dic

"""
Simulator Module

Main Functions:
1. `create_input_folders`: Sets up input JSON files for the simulator.
   - Accepts a DataFrame of samples for simulation.
   - Uses `map_creation` to generate variable inputs and create subfolders
     with JSON files in `map_dir`.
   - Variable inputs are parameters typically modified between runs, unlike
     fixed inputs, which are constants or developer-adjusted settings.

2. `launch_0d`: Executes the simulation.
   - Runs using input files from a specified directory.
   - Handles timeouts and errors, returning results or error info.
"""

def map_creation(df: pd.DataFrame, max_simu_time: int, map_dir: str = 'default'):

    constants = {
        "r_ext_pAl": 100.e-9,
        "r_ext_pMeO": 100.e-09,
        "pAl_richness": 2.,
        'alpha_p': 0.3,
        'th_Al2O3': 5.12281599140251e-08,  # Converted by `adapt_inputs_dic` into Y_Al_pAl
        'heat_input_ths': 556000,
        'power_input': 20000000,
        'coeff_contact': 0.01,
        'Ea_D_Ox': 50000,
        'Ea_Al2O3_decomp': 400000,
        'Ea_MeO_decomp': 50000,
        'k0_D_Ox': 0.000008,
        'k0_Al2O3_decomp': 1520000,
        'k0_MeO_decomp': 30000000,
        'bool_kin': 'false',
        "max_simu_time": max_simu_time,  # max simulation time in seconds (2400 = 40min)
    }

    # don't set a feature constant if it's an input
    for col, val in constants.items():
        if col not in df.columns:
            df[col] = val

    if map_dir == 'default':
        df.apply(map_generator, axis=1)
    else:
        df.apply(lambda sample: map_generator(sample, map_dir), axis=1)


def create_input_folders(
    df_X: pd.DataFrame, index: int, max_simu_time: int, map_dir: str
) -> List[str]:
    """
    Create JSON files for simulator input at map_dir/simu_XXXXX and return a list of
    created folder paths.
    """
    # Generate new indices by offsetting the existing indices with the given index
    new_idx = df_X.index + index

    # Create a new DataFrame with the updated indices
    updated_df = pd.DataFrame(df_X.values, columns=df_X.columns, index=new_idx)

    # Add folders names simu_XXXXX
    updated_df['workdir'] = updated_df.apply(set_simu_name, axis=1)

    # Create simulation input files using the updated DataFrame
    map_creation(updated_df, max_simu_time, map_dir)

    # Generate and return a list of created folder paths
    folders = [os.path.join(map_dir, workdir) for workdir in updated_df['workdir']]
    return folders


def read_input_folders(folders: List[str]) -> List[Dict[str, Union[float, str]]]:
    # Define input dicts
    inputs_dics = []
    for input_dir in folders:
        inputs_dic = set_inputs_dic(input_dir)
        inputs_dic = adapt_inputs_dic(inputs_dic)
        inputs_dics.append(inputs_dic)
    return inputs_dics


def run_simulation_process(inputs_dic: Dict, output_dir: int) -> Dict[str, float]:

    # Start chronometer
    start_time = time.time()

    try:
        # Run simulation, store outputs in output_dir and get them in dict
        db_simulation = simulator_0d.main(inputs_dic)
        results = SimPostProc(**db_simulation.__dict__)
        res_dic = results.manage(output_dir=output_dir)

        # If succesfull simulation return results
        return res_dic

    except Exception as e:
        # Handle failed simulations
        elapsed_time = time.time() - start_time
        warnings.warn(
            f"Simulation error after {elapsed_time/60:.1f} min "
            f"for label '{inputs_dic['workdir']}': \n"
            f"{e.__class__.__name__}: {str(e)}\n"
            f"Returning NaN values."
        )
        return {
            'error': f"{e.__class__.__name__}: {str(e)}",
            'sim_time': elapsed_time,
            'timed_out': elapsed_time >= inputs_dic['max_simu_time'],
            'Tg_Tmax': np.NaN,
            'Pg_f': np.NaN,
            'Pg_rate': np.NaN,
            'Y_O2_f': np.NaN,
        }


def run_simulation(
    df_X: pd.DataFrame, n_proc: int, 
    index: int, max_simu_time: int, map_dir: str
) -> pd.DataFrame:

    # Create input folders and get their inputs dict
    folders = create_input_folders(df_X, index, max_simu_time, map_dir)
    inputs_dics_list = read_input_folders(folders)

    # Set a directory for each simulation to store its output
    output_dirs_list = [
        os.path.abspath(os.path.join(input_dir, 'outputs'))
        for input_dir in folders
    ]

    # Combine input dictionaries and output directories for each simulation
    task_params = list(zip(inputs_dics_list, output_dirs_list))

    # Set up multiprocessing
    with multiprocessing.Pool(processes=n_proc) as pool:
        # Step 1: Create asynchronous tasks
        async_results = [pool.apply_async(run_simulation_process, args) for args in task_params]

        results = []
        # Step 2: Collect results with timeout handling
        for i, async_result in enumerate(async_results):
            try:
                # Wait for the result with a timeout
                result = async_result.get(timeout=inputs_dics_list[i]['max_simu_time'])
                results.append(result)
            except multiprocessing.TimeoutError:
                # Handle case where simulation exceeds max time
                warnings.warn(
                    f"Simulation timed out after {inputs_dics_list[i]['max_simu_time']/60:.1f} min "
                    f"for label '{inputs_dics_list[i]['workdir']}'."
                )
                # Append a default error result for timed-out simulations
                results.append({
                    'error': "TimeoutError: Simulation timed out",
                    'sim_time': inputs_dics_list[i]['max_simu_time'],
                    'timed_out': True,
                    'Tg_Tmax': np.NaN,
                    'Pg_f': np.NaN,
                    'Pg_rate': np.NaN,
                    'Y_O2_f': np.NaN,
                })

    return pd.DataFrame(results)


def normalized_power_function(X, target_dim):
    # Compute the hypercube diagonal length
    diagonal_length = np.sqrt(X.shape[1])
    
    # Compute the normalized norm for each point
    norms = np.linalg.norm(X, axis=1) / diagonal_length
    
    # Apply powers based on the target space dimension
    return np.column_stack([norms**(i + 1) for i in range(target_dim)])


def run_fake_simulation(
    X_real: np.ndarray, features: List[str], targets: List[str],
    additional_values: List[str], treatment: DataTreatment
):
    """ Set a fake results df. All outputs are in real space (not scaled one)"""
    x_scaled = treatment.scaler.transform_features(X_real)
    # As if y_sim = ['Pg_f', 'Tg_Tmax'] with values in [0, 1]
    y_sim = normalized_power_function(x_scaled, len(targets))
    # As if y_doi = ['sim_time', 'Composition', ...]
    y_doi = np.zeros((X_real.shape[0], len(additional_values)))

    df_results = pd.DataFrame(
        data=np.concatenate([x_scaled, y_sim, y_doi], axis=1),
        columns=features + targets + additional_values
    )
    # Get targets in real space (not in sclaed one)
    df_results[features + targets] = treatment.scaler.inverse_transform(df_results[features + targets].values)

    return df_results


class SimulationProcessor:
    def __init__(
        self, features: List[str], targets: List[str], additional_values: List[str],
        treatment: DataTreatment, simulator_config: Dict, n_proc: int = 1
    ):
        self.features = features
        self.targets = targets
        self.additional_values = additional_values
        self.treatment = treatment
        self.use_simulator = simulator_config['use']
        self.max_simu_time = simulator_config['max_simu_time']
        self.map_dir = simulator_config['map_dir']
        self.n_proc = n_proc
        # Setup simulation environment
        set_history_folder(self.map_dir, should_rename=False)

    def _prepare_real_input(self, X: np.ndarray, is_real_X: bool) -> np.ndarray:
        if is_real_X:
            return X
        XY = np.column_stack((X, np.zeros((X.shape[0], len(self.targets)))))
        return self.treatment.scaler.inverse_transform(XY)[:, :len(self.features)]

    def process_data(
        self, X: np.ndarray, is_real_X: bool, index: int, treat_output=True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process simulation data, either from real or scaled input features.

        Args:
            X (np.ndarray): Input features array.
            is_real_X (bool): If True, X contains real (unscaled) values. If False, X contains scaled values.
            index (int): Index under which to store comming simulation in self.map_dir.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Processed scaled data and scaled errors.
        """
        X_real = self._prepare_real_input(X, is_real_X)
        df_X_real = pd.DataFrame(X_real, columns=self.features)

        if self.use_simulator:
            df_results = run_simulation(
                df_X=df_X_real, index=index, n_proc=self.n_proc,
                max_simu_time=self.max_simu_time, map_dir=self.map_dir
            )
            df_results = pd.concat([df_X_real, df_results], axis=1)
        else:
            df_results = run_fake_simulation(
                X_real, self.features, self.targets, self.additional_values,
                self.treatment,
            )

        if not treat_output:
            # Return data in real scale as returned by simulator
            return df_results

        # Scale and clean data
        scaled_data = self.treatment.treat_real_data(df_real=df_results)

        # Keep specific columnms
        available_values = [col for col in self.additional_values if col in df_results]
        scaled_data = scaled_data[self.features + self.targets + available_values]
        return scaled_data

    def adapt_targets(self, data: pd.DataFrame, spice_on: bool) -> pd.DataFrame:
        if self.use_simulator:
            return data
        
        # If using fake simulator change target values
        scaled_data = self.process_data(
            data[self.features].values, is_real_X=False, index=0, treat_output=True
        )
        data[self.targets] = scaled_data[self.targets].values

        # Add some spice to check how outliers and errors are handled
        if spice_on and data.shape[0] >= 4: 
            interest_region_center = [sum(values) / 2 for values in self.treatment.scaled_interest_region.values()]

            data.loc[0, 'sim_time'] = 60  # time_out
            data.loc[1, self.targets] = interest_region_center  # interest sample
            data.loc[2, self.targets[0]] = 1.1  # target out of bounds
            data.loc[3, self.targets[0]] = np.nan  # failed simulation causing error (missing value)
        return data
