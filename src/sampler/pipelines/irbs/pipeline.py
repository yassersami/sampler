"""
This is a boilerplate pipeline 'irbs'
generated using Kedro 0.18.5
"""

from kedro.pipeline import Pipeline, node, pipeline

from sampler.pipelines.irbs.nodes import irbs_sampling
from sampler.common.storing import join_history


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=irbs_sampling,
            inputs=dict(
                data='treated_data',
                treatment='treatment',
                features='params:features',
                targets='params:targets',
                additional_values='params:additional_values',
                coefficients='params:irbs_coefficients',
                max_size='params:irbs_max_size',
                batch_size='params:irbs_batch_size',
                opt_iters='params:irbs_opt_iters',
                opt_points='params:irbs_opt_sampling_points',
                decimals='params:irbs_error_round_decimals',
                simulator_env='params:simulator_env',
            ),
            outputs='irbs_history',
            name='irbs_sampling',
        ),
        node(
            func=join_history,
            inputs=dict(
                history='irbs_history',
            ),
            outputs='irbs_increased_data',
            name='irbs_retrieve_outputs',
        )
    ])
