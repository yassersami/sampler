"""
This is a boilerplate pipeline 'init_lhs'
generated using Kedro 0.18.5
"""

from kedro.pipeline import Pipeline, node, pipeline

from sampler.pipelines.init_lhs.nodes import generate_LHS_inputs, evaluate_LHS

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=generate_LHS_inputs,
            inputs=dict(
                n_input='params:initial_size',
                features='params:features',
                variables_ranges='params:variables_ranges',
            ),
            outputs='df_lhs',
            name='generate_LHS_inputs',
        ),
        node(
            func=evaluate_LHS,
            inputs=dict(
                df_lhs='df_lhs',
                features='params:features',
                targets='params:targets',
                additional_values='params:additional_values',
                n_proc='params:initlhs_n_proc',
            ),
            outputs='initial_data',
            name='evaluate_LHS',
        )
    ])
