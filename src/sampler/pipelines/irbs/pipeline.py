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
                fom_terms='params:irbs_fom_terms',
                run_condition='params:run_condition',
                opt_iters='params:irbs_opt_iters',
                opt_points='params:irbs_opt_sampling_points',
                simulator_env='params:simulator_env',
            ),
            outputs='irbs_history',
            name='irbs_sampling',
        ),
        node(
            func=join_history,
            inputs=dict(
                history='irbs_history',
                run_condition='params:run_condition',
                initial_size='params:initial_size'
            ),
            outputs='irbs_increased_data',
            name='irbs_retrieve_outputs',
        )
    ])
