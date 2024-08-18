"""
This is a boilerplate pipeline 'sao_optim'
generated using Kedro 0.18.5
"""

from kedro.pipeline import Pipeline, node, pipeline

from sampler.pipelines.prep import create_pipeline as create_pipeline_prep

from .nodes import sao_optim_from_simulator
from sampler.common.storing import join_history


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
                simulator_env='params:simulator_env',
                batch_size='params:batch_size',
                run_condition='params:run_condition',
                
                sao_history_path='params:sao_history_path',
            ),
            outputs=dict(
                history='sao_history',  # dummy output to run nodes in sequence but is used undirectly by storing.store_df
                optim_res='sao_optim_res',
            ),
            name='sao_sampling',
        ),
        node(
            func=join_history,
            inputs=dict(
                history='sao_history',
                run_condition='params:run_condition',
                initial_size='params:initial_size'
            ),
            outputs='sao_increased_data',
            name='sao_retrieve_outputs',
        )
    ])
    return pipeline([pipeline_prep, pipeline_local])
