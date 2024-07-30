"""
This is a boilerplate pipeline 'irbs'
generated using Kedro 0.18.5
"""
from datetime import datetime
from typing import List, Dict
from tqdm import tqdm
import warnings

import pandas as pd

from sampler.common.data_treatment import DataTreatment, initialize_dataset
from sampler.common.storing import results
from sampler.models.wrapper_for_0d import SimulationProcessor  # get_values_from_simulator
from sampler.models.fom import FigureOfMerit

RANDOM_STATE = 42


def irbs_sampling(
        data: pd.DataFrame, treatment: DataTreatment,
        features: List[str], targets: List[str], additional_values: List[str],
        coefficients: Dict, simulator_env: Dict, run_condition: Dict,
        opt_iters: int = 5, opt_points: int = 1000,
        decimals: int = 7,
):
    max_size, n_interest_max, run_until_max_size, batch_size = run_condition['max_size'], run_condition['n_interest_max'], run_condition['run_until_max_size'], run_condition['batch_size']
    
    res = initialize_dataset(data=data, features=features, targets=targets, treatment=treatment) # Set dataset to complete with adaptive sampling
    yield results(res, size=len(res), initialize=True)

    # Set figure of merite (acquisition function)
    model = FigureOfMerit(
        features=features, targets=targets, coefficients=coefficients,
        interest_region=treatment.scaled_interest_region, decimals=decimals
    )

    # Set simulator environement
    simulator = SimulationProcessor(
        features=features, targets=targets, additional_values=additional_values,
        treatment=treatment, n_proc=batch_size, simulator_env=simulator_env
    )

    size = 0
    iteration = 0
    n_new_interest = 0
    end_condition = size < max_size if run_until_max_size else n_new_interest < n_interest_max 
    progress_bar = tqdm(total=max_size, dynamic_ncols=True) if run_until_max_size else tqdm(total=n_interest_max, dynamic_ncols=True) # Initialize tqdm progress bar with estimated time remaining
    print(f"Iteration {iteration:03} - Size {size} - New interest {n_new_interest}")
    while end_condition:
        model.fit(x_train=res[features].values, y_train=res[targets].values) # Set the new model that will be used in next iteration

        new_x, scores = model.optimize(batch_size=batch_size, iters=opt_iters, n=opt_points) # Search new candidates to add to res dataset

        new_df, error_features = simulator.process_data(new_x, real_x=False, index=size) # Launch time expensive simulations
        model.add_ignored_points(error_features)

        print(f'Round {iteration:03} (continued): simulation results' + '-'*49)
        print(f'irbs_sampling -> Got {len(new_df)} new samples after simulation:\n {new_df}')
        if len(new_df) == 0:
            warnings.warn("irbs_sampling -> No new data was obtained from simulator !")
            continue

        # % Add more data than features, targets and additional_values -----------------

        # Add multi_objective optimization scores
        # ignore_index=False to keep columns names 
        # join='inner' because scores can have more rows than new_df
        new_df = pd.concat([new_df, scores], axis=1, join='inner', ignore_index=False)

        # Add model prediction to selected (already simulated) points
        prediction = model.predict(new_df[features].values)
        prediction_cols = [f"pred_{t}" for t in targets]
        new_df[prediction_cols] = prediction if len(targets) > 1 else prediction.reshape(-1, 1)

        new_df = treatment.classify_scaled_interest(new_df) # Add column is_interest with True if row is inside the interest region
        
        # Add iteration number and datetime
        timenow = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_df['datetime'] = timenow
        new_df['iteration'] = iteration


        res = pd.concat([res, new_df], axis=0, ignore_index=True) # Concatenate new values to original results DataFrame
        size += len(new_df)
        n_new_interest += len(new_df[new_df['quality'] == 'interest'])
        iteration+=1

        if run_until_max_size:
            progress_bar.update(len(new_df))
        else:
            progress_bar.update(len(new_df[new_df['quality'] == 'interest']))

        end_condition = size < max_size if run_until_max_size else n_new_interest < n_interest_max
        print(f"Iteration {iteration} - Size {size} - New interest {n_new_interest}")
        yield results(res, size=len(new_df))

        # * Print some informations
        # iter_interest_count = (new_df['quality']=='interest').sum()
        # total_interest_count = (res['quality']=='interest').sum()
        # print(f'irbs_sampling -> Final batch data that wil be stored:\n {new_df}')
        # print(f'irbs_sampling -> [batch  report] new points: {len(new_df)}, interesting points: {iter_interest_count}')
        # print(f'irbs_sampling -> [global report] progress: {size}/{max_size}, interesting points: {total_interest_count}')
    progress_bar.close()

