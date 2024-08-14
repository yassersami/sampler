"""
This is a boilerplate pipeline 'init_lhs'
generated using Kedro 0.18.5
"""

from kedro.pipeline import Pipeline, node, pipeline

from sampler.pipelines.init_lhs.nodes import prepare_simulator_inputs, evaluate_inputs
from sampler.common.storing import join_history

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=prepare_simulator_inputs,
            inputs=dict(
                treatment='treatment',
                features='params:features',
                targets='params:targets',
                use_lhs='params:initLHS_use_lhs',
                num_samples='params:initLHS_num_lhs_samples',
                variables_ranges='params:variables_ranges',
                csv_file='params:initLHS_input_csv_file',
                csv_is_real='params:initLHS_input_csv_is_real',
            ),
            outputs='df_inputs',
            name='initLHS_prepare_inputs',
        ),
        node(
            func=evaluate_inputs,
            inputs=dict(
                data='df_inputs',
                treatment='treatment',
                features='params:features',
                targets='params:targets',
                additional_values='params:additional_values',
                simulator_env='params:simulator_env',
                batch_size='params:batch_size',
                
                n_proc='params:initLHS_n_proc',
                output_is_real='params:initLHS_output_is_real'
            ),
            # outputs="initLHS_output_data",
            outputs="initLHS_history",
            name='initLHS_evaluate',
        ),
        node(
            func=join_history,
            inputs=dict(
                history='initLHS_history',
                run_condition='params:run_condition',
                initial_size='params:initial_size'
            ),
            outputs='initLHS_increased_data',
            name='irbs_retrieve_outputs',
        )
    ])
