"""
This is a boilerplate pipeline 'irbs'
generated using Kedro 0.18.5
"""

from kedro.pipeline import Pipeline, node, pipeline

from sampler.pipelines.prep import create_pipeline as create_pipeline_prep

from .nodes import irbs_initialize_component, irbs_prepare_data, irbs_store_config, irbs_sampling
from sampler.core.data_processing.storing import join_history, join_logs


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
            func=irbs_store_config,
            inputs=dict(
                data='irbs_prepared_data',
                treatment='treatment',
                fom_model='fom_model',
                optimizer='optimizer',
                simulator='simulator',
            ),
            outputs=dict(
                irbs_config='irbs_config',
                data='irbs_prepared_data_1',  # Dummy output to keep order
            ),
            name='irbs_store_config'
        ),
        node(
            func=irbs_sampling,
            inputs=dict(
                data='irbs_prepared_data_1',
                treatment='treatment',
                features='params:features',
                targets='params:targets',
                stop_condition='params:stop_condition',
                fom_model='fom_model',
                optimizer='optimizer',
                simulator='simulator',
            ),
            outputs=dict(
                history='irbs_history',
                logs='irbs_logs',
            ),
            name='irbs_sampling',
        ),
        node(
            func=join_history,
            inputs=dict(
                history='irbs_history',
                stop_condition='params:stop_condition'
            ),
            outputs='irbs_increased_data',
            name='irbs_join_history',
        ),
        node(
            func=join_logs,
            inputs='irbs_logs',
            outputs='irbs_final_logs',
            name='irbs_join_logs',
        )
    ])
    return pipeline([pipeline_prep, pipeline_local])
