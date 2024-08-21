"""
This is a boilerplate pipeline 'parego'
generated using Kedro 0.18.5
"""


from kedro.pipeline import Pipeline, node, pipeline

from sampler.pipelines.prep import create_pipeline as create_pipeline_prep

from .nodes import run_parego
from sampler.common.storing import join_history


def create_pipeline(**kwargs) -> Pipeline:
    pipeline_prep = create_pipeline_prep()
    pipeline_local = pipeline([
         node(
            func=run_parego,
            inputs=dict(
                data='treated_data',
                treatment='treatment',
                features='params:features',
                targets='params:targets',
                additional_values='params:additional_values', # 'sim_time', 'Y_O2', ...
                simulator_env='params:simulator_env',
                batch_size='params:batch_size',
                run_condition='params:run_condition',
                
                llambda_s='params:parego_llambda_s',
                population_size='params:parego_population_size',
                num_generations='params:parego_num_generations',
                tent_slope='params:parego_tent_slope',
                experience='params:parego_experience',
            ),
            outputs='parego_history',
            name='parego_sampling',
        ),
        node(
            func=join_history,
            inputs=dict(
                history='parego_history',
                run_condition='params:run_condition',
                initial_size='params:initial_size'
            ),
            outputs='parego_increased_data',
            name='parego_retrieve_outputs',
        )
    ])
    return pipeline([pipeline_prep, pipeline_local])
