from typing import List, Dict, Tuple, Union
import sys
import contextlib
import os
import warnings
import multiprocessing

import numpy as np
import pandas as pd
import time
from tqdm import tqdm
import queue

from .storing import create_history_folder

import simulator_0d.src.py0D.main as simulator_0d
from simulator_0d.src.py0D.functions_pp import DB_simu_pp as SimPostProc
from simulator_0d.src.py0D.map_generator import set_simu_name, map_generator
from simulator_0d.src.pypp.launch import set_inputs_dic, adapt_inputs_dic

"""
Simulator wrapper for 0D chamber combustion simulator

Main Functions:
1. `create_input_folders`: Sets up input JSON files for the simulator.
   - Accepts a DataFrame of samples for simulation.
   - Uses `map_creation` to generate variable inputs and create subfolders
     with JSON files in `map_dir`.
   - Variable inputs are parameters typically modified between runs, unlike
     fixed inputs, which are constants or developer-adjusted settings.

2. `run_simulation_process`: Executes the simulation.
   - Runs using input files from a specified directory.
   - Handles timeouts and errors, returning results or error info.
"""

EMPTY_TARGETS_DICT = {
    'Tg_Tmax': np.NaN,
    'Pg_f': np.NaN,
    'Pg_rate': np.NaN,
    'Y_O2_f': np.NaN
}


def map_creation(df: pd.DataFrame, max_sim_time: int, map_dir: str = 'default'):

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
        "max_sim_time": max_sim_time,  # max simulation time in seconds (2400 = 40min)
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
    df_X: pd.DataFrame, index: int, max_sim_time: int, map_dir: str
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
    map_creation(updated_df, max_sim_time, map_dir)

    # Generate and return a list of created folder paths
    folders = [os.path.join(map_dir, workdir) for workdir in updated_df['workdir']]
    return folders


def read_input_folders(folders: List[str]) -> List[Dict[str, Union[float, str]]]:
    """Read JSON files from simulator input folders and return a list of input dictionaries."""
    inputs_dics = []
    for input_dir in folders:
        inputs_dic = set_inputs_dic(input_dir)
        inputs_dic = adapt_inputs_dic(inputs_dic)
        inputs_dics.append(inputs_dic)
    return inputs_dics


@contextlib.contextmanager
def redirect_stdout_stderr(stdout_path, stderr_path):
    """
    A context manager to redirect stdout and stderr to separate files.
    """
    with open(stdout_path, 'w') as stdout_file, open(stderr_path, 'w') as stderr_file:
        with contextlib.redirect_stdout(stdout_file), contextlib.redirect_stderr(stderr_file):
            yield


def run_simulation_process(inputs_dic: Dict, output_dir: str) -> Dict[str, float]:
    # Create paths for the stdout and stderr files
    stdout_path = os.path.join(output_dir, '..', 'stdout.log')
    stderr_path = os.path.join(output_dir, '..', 'stderr.log')

    # Start chronometer
    start_time = time.time()

    with redirect_stdout_stderr(stdout_path, stderr_path):
        try:
            print(f"Starting simulation for {inputs_dic['workdir']}")
            
            # Run simulation, store outputs in output_dir and get them in dict
            db_simulation = simulator_0d.main(inputs_dic)
            results = SimPostProc(**db_simulation.__dict__)
            res_dic = results.manage(output_dir=output_dir)

            print(f"Simulation completed successfully for {inputs_dic['workdir']}")
            return res_dic

        except Exception as e:
            # Handle failed simulations
            elapsed_time = time.time() - start_time
            error_message = (
                f"Simulation error after {elapsed_time/60:.1f} min "
                f"for label '{inputs_dic['workdir']}': \n"
                f"{e.__class__.__name__}: {str(e)}\n"
                f"Returning NaN values."
            )
            print(error_message, file=sys.stderr)
            warnings.warn(error_message)
            return {
                'error': f"{e.__class__.__name__}: {str(e)}",
                'sim_time': elapsed_time,
                'timed_out': elapsed_time >= inputs_dic['max_sim_time'],
                **EMPTY_TARGETS_DICT,
            }

    return res_dic


def progress_tracker(max_sim_time, progress_queue):
    start_time = time.time()
    elapsed_time = 0
    with tqdm(total=max_sim_time, desc="Simulation Progress", unit="s") as pbar:
        while elapsed_time < max_sim_time:
            elapsed_time = min(time.time() - start_time, max_sim_time)
            pbar.n = int(elapsed_time)
            pbar.refresh()

            time.sleep(1)  # Update every second

            # Check if all simulations are done
            try:
                if progress_queue.get_nowait() == "DONE":
                    
                    break
            except queue.Empty:
                pass

        # The progress bar will automatically close when exiting the 'with' block


def run_simulation(
    df_X: pd.DataFrame,
    n_proc: int, 
    index: int,
    max_sim_time: int,
    map_dir: str
) -> pd.DataFrame:

    if df_X.shape[0] > n_proc:
        raise ValueError(
            "The number of simulations should not exceed the number of cores "
            "n_proc. Possible overload of resources."
        )

    if index == 0:
        # Create simulation output directory
        create_history_folder(map_dir, should_rename=False)

    # Create input folders and get their inputs dict
    folders = create_input_folders(df_X, index, max_sim_time, map_dir)
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
        manager = multiprocessing.Manager()
        progress_queue = manager.Queue()

        # Set separate process for progress tracker 
        progress_tracker_process = pool.apply_async(progress_tracker, (max_sim_time, progress_queue))

        # Step 1: Create asynchronous tasks for simulation processes
        async_results = [pool.apply_async(run_simulation_process, params) for params in task_params]

        # Step 2: Collect results with timeout handling
        results = []
        for i, async_result in enumerate(async_results):
            try:
                # Wait for the result with a timeout
                result = async_result.get(timeout=inputs_dics_list[i]['max_sim_time'])
                results.append(result)
            except multiprocessing.TimeoutError:
                # Handle case where simulation exceeds max time
                warnings.warn(
                    f"Simulation timed out after {inputs_dics_list[i]['max_sim_time']/60:.1f} min "
                    f"for label '{inputs_dics_list[i]['workdir']}'."
                )
                # Append a default error result for timed-out simulations
                results.append({
                    'error': "TimeoutError: Simulation timed out",
                    'sim_time': inputs_dics_list[i]['max_sim_time'],
                    'timed_out': True,
                    **EMPTY_TARGETS_DICT,
                })

        # Signal that all simulations are done
        progress_queue.put("DONE")

        # Wait for progress tracker to finish
        progress_tracker_process.get()

    return pd.DataFrame(results)
