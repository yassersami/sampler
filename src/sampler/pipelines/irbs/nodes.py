"""
This is a boilerplate pipeline 'irbs'
generated using Kedro 0.18.5
"""
from typing import List, Dict, Tuple, Callable, Type, Union, Optional, ClassVar
from datetime import datetime
from tqdm import tqdm

import numpy as np
import pandas as pd

from sampler.core.data_processing.data_treatment import (
    DataTreatment, initialize_dataset, append_hypercube_boundary_points
)
from sampler.core.data_processing.storing import parse_results, parse_logs
from sampler.core.data_processing.sampling_tracker import SamplingProgressTracker
from sampler.core.simulator.simulator import SimulationProcessor
from sampler.core.fom.fom import FigureOfMerit
from sampler.core.optimizer.selector import SelectorFactory
from sampler.core.optimizer.optimizer import OptimizerFactory
from sampler.core.optimizer.base import MultiModalOptimizer

np.random.seed(42)


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
        'simulator': simulator,
    }


def irbs_prepare_data(
    data: pd.DataFrame,
    treatment: DataTreatment,
    features: List[str],
    additional_values: List[str],
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
    data = initialize_dataset(data, treatment, additional_values)

    return data

def irbs_store_config(
    data: pd.DataFrame,
    treatment: DataTreatment,
    fom_model: FigureOfMerit,
    optimizer: MultiModalOptimizer,
    simulator: SimulationProcessor,
) -> Dict:
    """ Try to store core parameters of IRBS pipeline using main instances. """
    
    outliers_mask = (data.quality != 'interest') & (data.quality != 'no_interest')

    irbs_config = {
        'n_samples': data.shape[0],
        'n_outliers': int(outliers_mask.sum()),
        'ranges': treatment.bounds,
        'interest_region': treatment.interest_region,
        'fom_model': fom_model.get_parameters(),
        f'optimizer_{optimizer._class_name}': optimizer.get_parameters(),
        'max_sim_time': simulator.max_sim_time,
    }
    return {
        'irbs_config': irbs_config,
        'data': data  # Dummy output for node order, No modification has been done
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
    # Store initial data
    yield {
        'history': parse_results(data, current_history_size=0),
        'logs': {},
    }

    # Set progress counting variables
    sampling_tracker = SamplingProgressTracker(**stop_condition, targets=targets)

    # Initialize tqdm progress bar with estimated time remaining
    progress_bar = tqdm(total=sampling_tracker.pbar_total, dynamic_ncols=True, desc="IRBS Sampling")

    while sampling_tracker.should_continue():
        print(f"\nRound {sampling_tracker.iteration:03} (start) " + "-"*62)
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
        fom_model.reset_predict_profiling()
        model_stats = fom_model.get_model_stats()

        print(f"Selected candidates to be input to the simulator: \n{X_batch}")
        print(f"Multimodal selection records: \n{df_mmo_scores}")
        print(f"FOM scores records: \n{df_fom_scores}")
        print(f"FOM models fitting report: \n{fom_model.serialize_dict(model_stats)}")

        # Launch time expensive simulations
        new_df = simulator.process_data(X_batch, is_real_X=False, index=sampling_tracker.n_total, treat_output=True)

        # Add quality column with 'is_interest' value for samples in the interest region
        new_df = treatment.classify_quality(new_df, data_is_scaled=True)

        print(f"Simulation results: \n{new_df}")

        # Concatenate simulation results and optimization scores
        new_df = pd.concat(
            [new_df, df_fom_scores, df_mmo_scores], axis=1,
            ignore_index=False  # to keep columns names
        )

        # Add common information to first row only
        new_df.loc[0, 'iteration'] = sampling_tracker.iteration
        new_df.loc[0, 'datetime'] = start_time

        # Store final batch results
        yield {
            'history': parse_results(new_df, current_history_size=data.shape[0]),
            'logs': parse_logs(model_stats, sampling_tracker.iteration),
        }

        # Add new samples to general DataFrame
        data = pd.concat([data, new_df], axis=0, ignore_index=True)

        # Update sampling progress
        progress = sampling_tracker.update_state(new_df)
        sampling_tracker.print_iteration_report()

        progress_bar.update(progress)
    progress_bar.close()

    print(  # Profile report since sampling started
        "\nFOM terms prediction profiling since sampling started: \n"
        f"{fom_model.serialize_dict(fom_model.get_profile(use_log=True))}"
    )
