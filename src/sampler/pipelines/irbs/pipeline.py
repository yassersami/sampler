"""
This is a boilerplate pipeline 'irbs'
generated using Kedro 0.18.5
"""

from kedro.pipeline import Pipeline, node, pipeline

from sampler.pipelines.prep import create_pipeline as create_pipeline_prep

from .nodes import irbs_initialize_component, irbs_prepare_data, irbs_sampling
from sampler.common.storing import join_history


def create_pipeline(**kwargs) -> Pipeline:
    pipeline_prep = create_pipeline_prep()
    pipeline_local = pipeline([
        node(
            func=irbs_initialize_component,
            inputs=dict(
                treatment='treatment',
                features='params:features',
                targets='params:targets',
                additional_values='params:additional_values',
                simulator_config='params:simulator_config',
                simulator_map_dir='path_simulator_output',
                batch_size='params:batch_size',
                fom_terms_config='params:irbs_fom_terms_config',
                selector_config='params:irbs_selector_config',
                optimizer_config='params:irbs_optimizer_config',
            ),
            outputs=dict(
                fom_model='fom_model',
                optimizer='optimizer',
                simulator='simulator',
                irbs_config='irbs_config',
            ),
            name='initialize_fom_optimizer_simulator'
        ),
        node(
            func=irbs_prepare_data,
            inputs=dict(
                data='treated_data',
                treatment='treatment',
                features='params:features',
                additional_values='params:additional_values',
                simulator='simulator',
                boundary_outliers_n_per_dim='params:irbs_boundary_outliers_n_per_dim'
            ),
            outputs='irbs_prepared_data',
            name='irbs_prepare_data'
        ),
        node(
            func=irbs_sampling,
            inputs=dict(
                data='irbs_prepared_data',
                treatment='treatment',
                features='params:features',
                targets='params:targets',
                stop_condition='params:stop_condition',
                fom_model='fom_model',
                optimizer='optimizer',
                simulator='simulator',
            ),
            outputs='irbs_history',
            name='irbs_sampling',
        ),
        node(
            func=join_history,
            inputs=dict(
                history='irbs_history',
                stop_condition='params:stop_condition'
            ),
            outputs='irbs_increased_data',
            name='irbs_retrieve_outputs',
        )
    ])
    return pipeline([pipeline_prep, pipeline_local])
