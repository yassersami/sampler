"""
This is a boilerplate pipeline 'parego'
generated using Kedro 0.18.5
"""


from kedro.pipeline import Pipeline, node, pipeline

from sampler.pipelines.parego.nodes import run_parego
from sampler.common.storing import join_history


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
         node(
            func=run_parego,
            inputs=dict(
                data='treated_data',
                treatment='treatment',
                features='params:features',
                targets='params:targets',
                additional_values='params:additional_values', # 'sim_time', 'Y_O2', ...
                max_size='params:parego_max_size', # 150 (16 pour les tests)
                batch_size='params:parego_batch_size', # 4 (tourne avec un i5)
                llambda_s='params:parego_llambda_s',  # number of subdivision of [0, 1] interval to chose lambda combinations
                tent_slope='params:parego_tent_slope',
                experience='params:parego_experience',
                simulator_env='params:simulator_env',
            ),
            outputs='parego_history',
            name='parego_sampling',
        ),
        node(
            func=join_history,
            inputs=dict(
                history='parego_history',
            ),
            outputs='parego_increased_data',
            name='parego_retrieve_outputs',
        )
    ])
