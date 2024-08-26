"""
This is a boilerplate pipeline 'irbs'
generated using Kedro 0.18.5
"""

from kedro.pipeline import Pipeline, node, pipeline

from sampler.pipelines.prep import create_pipeline as create_pipeline_prep

from .nodes import irbs_sampling
from sampler.common.storing import join_history


def create_pipeline(**kwargs) -> Pipeline:
    pipeline_prep = create_pipeline_prep()
    pipeline_local = pipeline([
        node(
            func=irbs_sampling,
            inputs=dict(
                data='treated_data',
                treatment='treatment',
                features='params:features',
                targets='params:targets',
                additional_values='params:additional_values',
                simulator_env='params:simulator_env',
                batch_size='params:batch_size',
                run_condition='params:run_condition',
                
                fom_terms='params:irbs_fom_terms',
                shgo_args='params:irbs_shgo',
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
    return pipeline([pipeline_prep, pipeline_local])
