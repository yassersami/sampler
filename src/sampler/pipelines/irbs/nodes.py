"""
This is a boilerplate pipeline 'irbs'
generated using Kedro 0.18.5
"""
from typing import List, Dict, Tuple, Callable, Type, Union, Optional, ClassVar
from datetime import datetime
from tqdm import tqdm

import numpy as np
import pandas as pd

from sampler.common.data_treatment import (
    DataTreatment, initialize_dataset, append_hypercube_boundary_points
)
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
    simulator_config: Dict,
    simulator_map_dir: str,
    batch_size: int,
    fom_terms_config: Dict[str, Dict],
    selector_config: Dict[str, Dict],
    optimizer_config: Dict[str, Dict],
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
        simulator_config=simulator_config,
        map_dir=simulator_map_dir
    )

    return {
        'fom_model': fom_model,
        'optimizer': optimizer,
        'simulator': simulator
    }


def irbs_prepare_data(
    data: pd.DataFrame,
    treatment: DataTreatment,
    features: List[str],
    simulator: SimulationProcessor,
    boundary_outliers_n_per_dim: int
):
    # If fake simulator is used, adapt targets
    data = simulator.adapt_targets(data)

    if boundary_outliers_n_per_dim > 0:
        # Add outliers on design space boundaries to satisfy classifier curiosity 
        data = append_hypercube_boundary_points(
            data, features, boundary_outliers_n_per_dim
        )

    # Set dataset to be completed with adaptive sampling
    data = initialize_dataset(data, treatment)

    return data


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
    # Store initial data
    yield parse_results(data, current_history_size=0)
    
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
    pbar_total = max_size if run_until_max_size else n_interest_max
    progress_bar = tqdm(total=pbar_total, dynamic_ncols=True, desc="IRBS Sampling")

    while should_continue:
        print(f"\nRound {iteration:03} (start) " + "-"*62)
        start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Set the new FOM that will be used in next iteration
        fom_model.fit(X=data[features].values, y=data[targets].values)

        # Search new candidates to add to dataset
        X_batch = optimizer.run_multimodal_optim(fom_model.predict_loss)

        # Get multimodal optimization selection records
        df_mmo_scores = optimizer.selector.get_records_df()

        # Get FOM scores (that were optimization objectives)
        df_fom_scores = fom_model.get_scores_df(X_batch)

        # Get FOM models fitting report
        model_params = fom_model.get_model_params()

        # Get predictions profiling
        iteration_profile = fom_model.get_profile()

        print(f"Selected candidates to be input to the simulator: \n{X_batch}")
        print(f"Multimodal selection records: \n{df_mmo_scores}")
        print(f"FOM scores records: \n{df_fom_scores}")
        print(f"FOM models fitting report: \n{fom_model.serialize_dict(model_params)}")
        print(f"FOM terms prediction profiling: \n{fom_model.serialize_dict(iteration_profile)}")

        # Launch time expensive simulations
        new_df = simulator.process_data(X_batch, is_real_X=False, index=n_total, treat_output=True)

        # Add quality column with 'is_interest' value for samples in the interest region
        new_df = treatment.classify_quality_interest(new_df, data_is_scaled=True)

        print(f"Round {iteration:03} (continued) - simulation results " + "-"*37)
        print(f"New samples after simulation:\n {new_df}")

        # Add multi-objective optimization scores
        new_df = pd.concat(
            [new_df, df_fom_scores, df_mmo_scores], axis=1,
            ignore_index=False  # to keep columns names
        )

        # Add common information to first row only
        new_df.loc[0, 'iteration'] = iteration
        new_df.loc[0, 'datetime'] = start_time
        new_df.loc[0, model_params.keys()] = model_params.values()

        # Store final batch results
        yield parse_results(new_df, current_history_size=data.shape[0])

        # Concatenate new values to original results DataFrame
        data = pd.concat([data, new_df], axis=0, ignore_index=True)

        # Update stopping condition
        n_new_samples = new_df.shape[0]
        n_new_inliers = new_df.dropna(subset=targets).shape[0]
        n_new_interest = new_df[new_df['quality'] == 'interest'].shape[0]

        n_total += n_new_samples
        n_inliers += n_new_inliers
        n_interest += n_new_interest
        iteration += 1

        # Print iteration report
        print(
            f"Round {iteration - 1:03} (end) - Report count: "
            f"Total: {n_total}, "
            f"Inliers: {n_inliers}, "
            f"Interest: {n_interest}"
        )

        # Check stopping conditions
        should_continue = (
            (n_inliers < max_size) if run_until_max_size else
            (n_interest < n_interest_max)
        )

        # Update progress bar
        pbar_progress = (
            n_new_inliers  - max(0, n_inliers  - max_size) if run_until_max_size else
            n_new_interest - max(0, n_interest - n_interest_max)
        )
        progress_bar.update(pbar_progress)
    progress_bar.close()
