"""
This is a boilerplate pipeline 'irbs'
generated using Kedro 0.18.5
"""
from typing import List, Dict, Tuple, Callable, Type, Union, Optional, ClassVar
from datetime import datetime
from tqdm import tqdm

import numpy as np
import pandas as pd

from sampler.common.data_treatment import DataTreatment, initialize_dataset
from sampler.common.storing import parse_results
from sampler.common.simulator import SimulationProcessor
from sampler.fom.fom import FigureOfMerit
from sampler.optimizer.selector import SelectorFactory
from sampler.optimizer.optimizer import OptimizerFactory
from sampler.optimizer.base import MultiModalOptimizer


def irbs_initialize_component(
    treatment: DataTreatment,
    features: List[str],
    targets: List[str],
    additional_values: List[str],
    batch_size: int,
    fom_terms_config: Dict[str, Dict],
    selector_config: Dict[str, Dict],
    optimizer_config: Dict[str, Dict],
    simulator_config: Dict,
) -> Dict[str, object]:

    # Set figure of merite (acquisition function)
    fom_model = FigureOfMerit(
        treatment.scaled_interest_region,
        fom_terms_config
    )

    # Set multimodal selector
    selector = SelectorFactory.create_from_config(
        batch_size,
        selector_config
    )

    # Set multimodal optimizer
    optimizer = OptimizerFactory.create_from_config(
        len(features),
        selector,
        optimizer_config
    )

    # Set simulator function
    simulator = SimulationProcessor(
        features=features,
        targets=targets,
        additional_values=additional_values,
        treatment=treatment,
        n_proc=batch_size,
        simulator_config=simulator_config
    )

    return {
        'fom_model': fom_model,
        'optimizer': optimizer,
        'simulator': simulator
    }


def irbs_sampling(
    data: pd.DataFrame,
    treatment: DataTreatment,
    features: List[str],
    targets: List[str],
    stop_condition: Dict,
    fom_model: FigureOfMerit,
    optimizer: MultiModalOptimizer,
    simulator: SimulationProcessor,
):

    # If fake simulator is used, adapt targets
    data = simulator.adapt_targets(data, spice_on=True)

    # Set dataset to be completed with adaptive sampling
    res = initialize_dataset(data=data, treatment=treatment)
    yield parse_results(res, current_history_size=0)
    
    # Set progress counting variables
    max_size = stop_condition['max_size']
    n_interest_max = stop_condition['n_interest_max']
    run_until_max_size = stop_condition['run_until_max_size']

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
        # Set the new FOM that will be used in next iteration
        fom_model.fit(X=res[features].values, y=res[targets].values)

        # Search new candidates to add to res dataset
        X_batch = optimizer.run_multimodal_optim(fom_model.predict_loss)

        df_mmo_scores = optimizer.selector.get_records_df()
        df_fom_scores = fom_model.get_scores_df(X_batch)
        print(f"Selected candidates to be input to the simulator: \n{X_batch}")
        print(f"Optimization selection records: \n{df_mmo_scores}")
        print(f"FOM scores records: \n{df_fom_scores}")

        # Launch time expensive simulations
        new_df = simulator.process_data(X_batch, real_x=False, index=n_total, treat_output=True)

        # ----- Add more cols than features, targets and additional_values -----

        # Add quality column with 'is_interest' value for samples in the interest region
        new_df = treatment.classify_quality_interest(new_df, data_is_scaled=True)

        print(f"Round {iteration:03} (continued) - simulation results " + "-"*37)
        print(f"irbs_sampling -> New samples after simulation:\n {new_df}")

        # Add multi-objective optimization scores
        # * ignore_index=False to keep columns names 
        # * join='inner' because scores can have more rows than new_df
        new_df = pd.concat(
            [new_df, df_fom_scores, df_mmo_scores],
            axis=1, join='inner', ignore_index=False
        )

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
