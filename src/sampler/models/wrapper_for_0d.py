import warnings
from multiprocessing.pool import Pool
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from sampler.common.data_treatment import DataTreatment
from sampler.common.storing import set_history_folder
import sampler.models.simulator_aux_functions as sim_aux

import simulator_0d.src.py0D.main as simulator_0d
from simulator_0d.src.py0D.functions_pp import DB_simu_pp as SimPostProc


def launch_0d(input_dir: str) -> Dict[str, float]:
    # Define input and output directories
    inputs_dic, output_dir = sim_aux.define_in_out(input_dir)

    # Warn about the maximum simulation time
    max_simu_time_sec = inputs_dic.get("max_simu_time", 0)
    max_simu_time_min = max_simu_time_sec / 60
    warnings.warn(
        f"Maximum simulation time is set to {max_simu_time_sec} seconds "
        f"({max_simu_time_min:.2f} minutes). "
        "Value can be changed in src/simulator_0d/data/inputs_fix/inputs_fix.json"
    )

    # Default error response
    error_res = {
        "sim_time": np.NaN,
        "timed_out": np.NaN,
        "Tg_Tmax": np.NaN,
        "Pg_f": np.NaN,
        "Pg_rate": np.NaN,
        "Y_O2_f": np.NaN,
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


def define_initial_params(index: int, x: pd.DataFrame, map_dir: str):
    """Create json files at map_dir/simu_# that will be input for the simulator"""
    new_idx = [index + idx for idx in x.index]
    sim_aux.map_creation(
        df=pd.DataFrame(data=x.values, columns=x.columns, index=new_idx),
        map_dir=map_dir
    )
    folders = [f'{map_dir}/simu_{idx:05}' for idx in new_idx]
    return folders


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
    return pd.DataFrame.from_dict({folder[-4:]: dict_result}, orient="index")


def run_simulation(x: pd.DataFrame, n_proc: int, index: int, map_dir: str):
    folders = define_initial_params(index, x, map_dir)
    if n_proc > 1:
        new_df = run_parallel_simulation(folders=folders, n_proc=n_proc)
    else:
        new_df = run_serial_simulation(folder=folders[0])
    return new_df


def run_fake_simulator(x_real, features, targets, additional_values, scaler, spice_on):
    """ Set a fake results df. All outputs are in real space (not scaled one)"""
    x_scaled = scaler.transform_features(x_real)
    # As if y_sim = ["Pg_f", "Tg_Tmax"] with values in [0, 1]
    y_sim = np.array([x_scaled.min(axis=1), x_scaled.max(axis=1)]).T
    # As if y_doi = ["sim_time", "Composition", ...]
    y_doi = np.zeros((x_real.shape[0], len(additional_values)))

    new_points = pd.DataFrame(
        data=np.concatenate([x_scaled, y_sim, y_doi], axis=1),
        columns=features + targets + additional_values
    )
    # Get targets in real space (not in sclaed one)
    new_points[features + targets] = scaler.inverse_transform(new_points[features + targets].values)

    # Add some spice to check how outliers and errors are handled
    if spice_on and new_points.shape[0] >= 4:
        new_points.loc[0, "sim_time"] = 60  # time_out
        new_points.loc[1, targets] = [45e6, 6000]  # interest sample
        new_points.loc[2, targets[0]] = 1e20  # target out of bounds
        new_points.loc[3, targets[0]] = np.nan  # failed simulation causing error (missing value)

    return new_points


class SimulationProcessor:
    def __init__(
        self, features: List[str], targets: List[str], additional_values: List[str],
        treatment: DataTreatment, simulator_env: Dict, n_proc: int = 1
    ):
        self.features = features
        self.targets = targets
        self.additional_values = additional_values
        self.treatment = treatment
        self.use_simulator = simulator_env['use']
        self.map_dir = simulator_env['map_dir']
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

        if "r_ext_pMeO" not in self.features: # If r_ext_pMeO lack, I replace with the same value than r_ext_pAl # ? rs_inputs
            real_df['r_ext_pMeO'] = real_df['r_ext_pAl']

        if self.use_simulator:
            results_df = run_simulation(
                x=real_df, index=index, n_proc=self.n_proc, map_dir=self.map_dir
            )
            new_points = pd.concat([real_df, results_df], axis=1)
        else:
            new_points = run_fake_simulator(
                x_real, self.features, self.targets, self.additional_values,
                self.treatment.scaler, spice_on=(index==0)
            )
        # Return data as outputed from simulator
        if not treat_output:
            return new_points
        # Scale and clean data
        scaled_data = self.treatment.treat_real_data(df_real=new_points)
        scaled_data = scaled_data[self.features + self.targets + self.additional_values]
        return scaled_data

    def adapt_targets(self, data: pd.DataFrame) -> pd.DataFrame:
        # If using fake simulator change targets values
        if not self.use_simulator:
            scaled_data = self.process_data(
                data[self.features].values, real_x=False, index=0, treat_output=True
            )
            data[self.targets] = scaled_data[self.targets].values
        return data
