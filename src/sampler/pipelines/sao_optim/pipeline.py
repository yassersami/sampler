"""
This is a boilerplate pipeline 'sao_optim'
generated using Kedro 0.18.5
"""

from kedro.pipeline import Pipeline, node, pipeline

from sampler.pipelines.prep import create_pipeline as create_pipeline_prep

from .nodes import sao_optim_from_simulator
from sampler.core.data_processing.storing import join_history


def create_pipeline(**kwargs) -> Pipeline:
    pipeline_prep = create_pipeline_prep()
    pipeline_local = pipeline([
        node(
            func=sao_optim_from_simulator,
            inputs=dict(
                treatment='treatment',
                features='params:features',
                targets='params:targets',
                additional_values='params:additional_values',
                simulator_config='params:simulator_config',
                simulator_map_dir='path_simulator_output',
                batch_size='params:batch_size',
                stop_condition='params:stop_condition',
                
                sao_history_path='path_sao_history',
            ),
            outputs='sao_history',  # dummy output to run nodes in sequence but is used undirectly by storing.store_df
            name='sao_sampling',
        ),
        node(
            func=join_history,
            inputs=dict(
                history='sao_history',
                stop_condition='params:stop_condition'
            ),
            outputs='sao_increased_data',
            name='sao_retrieve_outputs',
        )
    ])
    return pipeline([pipeline_prep, pipeline_local])
