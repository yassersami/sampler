import os
import warnings
from multiprocessing.pool import Pool
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from sampler.common.data_treatment import DataTreatment
from sampler.common.storing import set_history_folder

import simulator_0d.src.py0D.main as simulator_0d
from simulator_0d.src.py0D.functions_pp import DB_simu_pp as SimPostProc
from simulator_0d.src.py0D.map_generator import set_simu_name, map_generator
from simulator_0d.src.pypp.launch import set_inputs_dic, adapt_inputs_dic


def map_creation(df: pd.DataFrame, map_dir: str = 'default'):
    df['workdir'] = df.apply(set_simu_name, axis=1)

    constants = {
        'alpha_p': 0.3,
        'th_Al2O3': 5.12281599140251e-08,
        'heat_input_ths': 556000,
        'power_input': 20000000,
        'coeff_contact': 0.01,
        'Ea_D_Ox': 50000,
        'Ea_Al2O3_decomp': 400000,
        'Ea_MeO_decomp': 50000,
        'k0_D_Ox': 0.000008,
        'k0_Al2O3_decomp': 1520000,
        'k0_MeO_decomp': 30000000,
        'bool_kin': 'false'
    }
    
    # don't set a feature constant if it's an input
    for col, val in constants.items():
        if col not in df.columns:
            df[col] = val

    if map_dir == 'default':
        df.apply(map_generator, axis=1)
    else:
        df.apply(lambda sample: map_generator(sample, map_dir), axis=1)


def define_initial_params(index: int, x: pd.DataFrame, map_dir: str):
    """Create json files at map_dir/simu_# that will be input for the simulator"""
    new_idx = [index + idx for idx in x.index]
    map_creation(
        df=pd.DataFrame(data=x.values, columns=x.columns, index=new_idx),
        map_dir=map_dir
    )
    folders = [f'{map_dir}/simu_{idx:05}' for idx in new_idx]
    return folders


def define_in_out(input_dir: str):
    inputs_dic = set_inputs_dic(input_dir)
    inputs_dic = adapt_inputs_dic(inputs_dic)
    output_dir = os.path.abspath(os.path.join(input_dir, 'outputs'))
    return inputs_dic, output_dir


def launch_0d(input_dir: str) -> Dict[str, float]:
    # Define input and output directories
    inputs_dic, output_dir = define_in_out(input_dir)

    # Warn about the maximum simulation time
    max_simu_time_sec = inputs_dic.get('max_simu_time', 0)
    max_simu_time_min = max_simu_time_sec / 60
    warnings.warn(
        f"Maximum simulation time is set to {max_simu_time_sec} seconds "
        f"({max_simu_time_min:.2f} minutes). "
        "Value can be changed in src/simulator_0d/data/inputs_fix/inputs_fix.json"
    )

    # Default error response
    error_res = {
        'sim_time': np.NaN,
        'timed_out': np.NaN,
        'Tg_Tmax': np.NaN,
        'Pg_f': np.NaN,
        'Pg_rate': np.NaN,
        'Y_O2_f': np.NaN,
    }

    try:
        # Run the simulation
        db_simulation = simulator_0d.main(inputs_dic)
        
        # Post-process the simulation results
        results = SimPostProc(**db_simulation.__dict__)
        res_dict = results.manage(output_dir=output_dir)
        
    except Exception as e:
        # Handle any exceptions and return the error response
        warnings.warn(
            "Error in simulation! Returning NaNs.\n"
            f"Error: {e}. \nInput dic: {inputs_dic}"
        )
        return error_res

    return res_dict


def run_parallel_simulation(folders: List[str], n_proc: int=None):
    """
    If processes is specified as None the default value used is the value
    returned by the cpu_count() function.
    """
    pool = Pool(processes=n_proc)
    result_f = pool.map_async(launch_0d, folders,).get()
    pool.close()
    return pd.DataFrame(result_f)


def run_serial_simulation(folder: str):
    dict_result = launch_0d(folder)
    return pd.DataFrame.from_dict({folder[-4:]: dict_result}, orient='index')


def run_simulation(x: pd.DataFrame, n_proc: int, index: int, map_dir: str):
    folders = define_initial_params(index, x, map_dir)
    if n_proc > 1:
        new_df = run_parallel_simulation(folders=folders, n_proc=n_proc)
    else:
        new_df = run_serial_simulation(folder=folders[0])
    return new_df


def normalized_power_function(X, target_dim):
    # Compute the hypercube diagonal length
    diagonal_length = np.sqrt(X.shape[1])
    
    # Compute the normalized norm for each point
    norms = np.linalg.norm(X, axis=1) / diagonal_length
    
    # Apply powers based on the target space dimension
    return np.column_stack([norms**(i + 1) for i in range(target_dim)])


def run_fake_simulator(
    x_real: np.ndarray, features: List[str], targets: List[str],
    additional_values: List[str], treatment: DataTreatment
):
    """ Set a fake results df. All outputs are in real space (not scaled one)"""
    x_scaled = treatment.scaler.transform_features(x_real)
    # As if y_sim = ['Pg_f', 'Tg_Tmax'] with values in [0, 1]
    y_sim = normalized_power_function(x_scaled, len(targets))
    # As if y_doi = ['sim_time', 'Composition', ...]
    y_doi = np.zeros((x_real.shape[0], len(additional_values)))

    new_points = pd.DataFrame(
        data=np.concatenate([x_scaled, y_sim, y_doi], axis=1),
        columns=features + targets + additional_values
    )
    # Get targets in real space (not in sclaed one)
    new_points[features + targets] = treatment.scaler.inverse_transform(new_points[features + targets].values)

    return new_points


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
        self.map_dir = simulator_config['map_dir']
        self.n_proc = n_proc
        # Setup simulation environment
        set_history_folder(self.map_dir, should_rename=False)

    def _prepare_real_input(self, new_x: np.ndarray, real_x: bool) -> np.ndarray:
        if real_x:
            return new_x
        new_xy = np.column_stack((new_x, np.zeros((new_x.shape[0], len(self.targets)))))
        return self.treatment.scaler.inverse_transform(new_xy)[:, :len(self.features)]

    def process_data(
        self, new_x: np.ndarray, real_x: bool, index: int, treat_output=True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process simulation data, either from real or scaled input features.

        Args:
            new_x (np.ndarray): Input features array.
            real_x (bool): If True, new_x contains real (unscaled) values. If False, new_x contains scaled values.
            index (int): Index under which to store comming simulation in self.map_dir.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Processed scaled data and scaled errors.
        """
        x_real = self._prepare_real_input(new_x, real_x)
        real_df = pd.DataFrame(x_real, columns=self.features)

        if 'r_ext_pMeO' not in self.features: # If r_ext_pMeO lack, I replace with the same value than r_ext_pAl # ? rs_inputs
            real_df['r_ext_pMeO'] = real_df['r_ext_pAl']

        if self.use_simulator:
            results_df = run_simulation(
                x=real_df, index=index, n_proc=self.n_proc, map_dir=self.map_dir
            )
            new_points = pd.concat([real_df, results_df], axis=1)
        else:
            new_points = run_fake_simulator(
                x_real, self.features, self.targets, self.additional_values,
                self.treatment,
            )

        if not treat_output:
            # Return data in real scale as returned by simulator
            return new_points

        # Scale and clean data
        scaled_data = self.treatment.treat_real_data(df_real=new_points)
        scaled_data = scaled_data[self.features + self.targets + self.additional_values]
        return scaled_data

    def adapt_targets(self, data: pd.DataFrame, spice_on: bool) -> pd.DataFrame:
        if self.use_simulator:
            return data
        
        # If using fake simulator change target values
        scaled_data = self.process_data(
            data[self.features].values, real_x=False, index=0, treat_output=True
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
