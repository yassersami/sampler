"""
This is a boilerplate pipeline 'irbs'
generated using Kedro 0.18.5
"""
from datetime import datetime
from typing import List, Dict
from tqdm import tqdm

import numpy as np
import pandas as pd

from sampler.common.data_treatment import DataTreatment, initialize_dataset
from sampler.common.storing import parse_results
from sampler.common.simulator import SimulationProcessor
from sampler.fom.fom import FigureOfMerit
from sampler.fom.optimizer import OptimizerFactory


def irbs_sampling(
    data: pd.DataFrame, treatment: DataTreatment,
    features: List[str], targets: List[str], additional_values: List[str],
    simulator_env: Dict, batch_size: int, run_condition: Dict,
    fom_terms_config: Dict[str, Dict], optimizer_config: Dict[str, Dict]
):

    # Set figure of merite (acquisition function)
    model = FigureOfMerit(
        interest_region=treatment.scaled_interest_region,
        terms_config=fom_terms_config
    )

    # Set optimizer
    optimizer = OptimizerFactory.create_from_config(batch_size, optimizer_config)

    # Set simulator environement
    simulator = SimulationProcessor(
        features=features, targets=targets, additional_values=additional_values,
        treatment=treatment, n_proc=batch_size, simulator_env=simulator_env
    )
    # If fake simulator is used, adapt targets
    data = simulator.adapt_targets(data)

    # Set dataset to be completed with adaptive sampling
    res = initialize_dataset(data=data, treatment=treatment)
    yield parse_results(res, current_history_size=0)
    
    # Set progress counting variables
    max_size = run_condition['max_size']
    n_interest_max = run_condition['n_interest_max']
    run_until_max_size = run_condition['run_until_max_size']

    n_total = 0  # counting all simulations
    n_inliers = 0  # counting only inliers
    n_interest = 0  # counting only interesting inliers
    iteration = 1
    should_continue = True

    # Initialize tqdm progress bar with estimated time remaining
    progress_bar = (
        tqdm(total=max_size, dynamic_ncols=True) if run_until_max_size else 
        tqdm(total=n_interest_max, dynamic_ncols=True)
    )

    while should_continue:
        print(f"\nRound {iteration:03} (start) " + "-"*62)
        # Set the new model that will be used in next iteration
        model.fit(X=res[features].values, y=res[targets].values)

        # Search new candidates to add to res dataset
        new_x, scores = optimizer.optimize(model)

        # Launch time expensive simulations
        new_df = simulator.process_data(new_x, real_x=False, index=n_total, treat_output=True)

        print(f"Round {iteration:03} (continued) - simulation results " + "-"*37)
        print(f"irbs_sampling -> New samples after simulation:\n {new_df}")

        # ----- Add more cols than features, targets and additional_values -----

        # Add multi_objective optimization scores
        # * ignore_index=False to keep columns names 
        # * join='inner' because scores can have more rows than new_df
        new_df = pd.concat([new_df, scores], axis=1, join='inner', ignore_index=False)

        # Add maximum found value for surrogate GP combined std
        new_df['max_std'] = model.terms.surrogate_gpr.max_std

        # Add model prediction to selected (already simulated) points
        y_pred = model.terms.surrogate_gpr.predict(new_df[features].values)
        new_df[[f'pred_{t}' for t in targets]] = np.atleast_2d(y_pred.T).T

        # Add column is_interest with True if targets are inside the interest region
        new_df = treatment.classify_quality_interest(new_df, data_is_scaled=True)
        
        # Add iteration number and datetime
        timenow = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_df['datetime'] = timenow
        new_df['iteration'] = iteration
        
        # Store final batch results
        yield parse_results(new_df, current_history_size=res.shape[0])

        # Concatenate new values to original results DataFrame
        res = pd.concat([res, new_df], axis=0, ignore_index=True)
        
        # Update stopping conditions
        n_new_samples = new_df.shape[0]
        n_new_inliers = new_df.dropna(subset=targets).shape[0]
        n_new_interest = new_df[new_df['quality'] == 'interest'].shape[0]
    
        n_total += n_new_samples
        n_inliers += n_new_inliers
        n_interest += n_new_interest
        iteration += 1

        # Print iteration details
        print(
            f"Round {iteration - 1:03} (end) - Report count: "
            f"Total: {n_total}, "
            f"Inliers: {n_inliers}, "
            f"Interest: {n_interest}"
        )

        # Determine the end condition
        should_continue = (
            (n_inliers < max_size) if run_until_max_size else
            (n_interest < n_interest_max)
        )

        # Update progress bar based on the condition
        progress_bar.update(
            n_new_inliers - max(0, n_inliers - max_size) if run_until_max_size else
            n_new_interest - max(0, n_interest - n_interest_max)
        )
    progress_bar.close()

