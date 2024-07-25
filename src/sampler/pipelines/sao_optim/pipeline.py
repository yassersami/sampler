"""
This is a boilerplate pipeline 'sao_optim'
generated using Kedro 0.18.5
"""

from kedro.pipeline import Pipeline, node, pipeline

from sampler.pipelines.sao_optim.nodes import sao_optim_from_simulator
from sampler.common.storing import join_history

# For this pipeline I need to install plotly and optuna
def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=sao_optim_from_simulator,
            inputs=dict(
                treatment='treatment',
                features='params:features',
                targets='params:targets',
                additional_values='params:additional_values',
                max_size='params:sao_max_size',
                batch_size='params:sao_batch_size',
                sao_history_path='params:sao_history_path',
                simulator_env='params:simulator_env',
            ),
            outputs=dict(
                history='sao_history',  # dummy output to run nodes in sequence but is used undirectly by aux_func.store_df
                optim_res='sao_optim_res',
            ),
            name='sao_sampling',
        ),
        node(
            func=join_history,
            inputs=dict(
                history='sao_history',
            ),
            outputs='sao_increased_data',
            name='sao_retrieve_outputs',
        )
    ])
