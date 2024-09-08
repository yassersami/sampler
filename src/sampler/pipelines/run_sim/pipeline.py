"""
This is a boilerplate pipeline 'run_sim'
generated using Kedro 0.18.5
"""

from kedro.pipeline import Pipeline, node, pipeline

from sampler.pipelines.prep import create_pipeline as create_pipeline_prep

from .nodes import prepare_simulator_inputs, evaluate_inputs
from sampler.common.storing import join_history

def create_pipeline(**kwargs) -> Pipeline:
    pipeline_prep = create_pipeline_prep()
    pipeline_local = pipeline([
        node(
            func=prepare_simulator_inputs,
            inputs=dict(
                treatment='treatment',
                features='params:features',
                targets='params:targets',
                use_lhs='params:run_sim_use_lhs',
                num_samples='params:run_sim_num_lhs_samples',
                variables_ranges='params:variables_ranges',
                csv_file='params:run_sim_input_csv_file',
                csv_is_real='params:run_sim_input_csv_is_real',
            ),
            outputs='df_inputs',
            name='run_sim_prepare_inputs',
        ),
        node(
            func=evaluate_inputs,
            inputs=dict(
                data='df_inputs',
                treatment='treatment',
                features='params:features',
                targets='params:targets',
                additional_values='params:additional_values',
                simulator_config='params:simulator_config',
                simulator_map_dir='path_simulator_output',
                batch_size='params:batch_size',
                
                n_proc='params:run_sim_n_proc',
                output_is_real='params:run_sim_output_is_real'
            ),
            # outputs="run_sim_output_data",
            outputs="run_sim_history",
            name='run_sim_evaluate',
        ),
        node(
            func=join_history,
            inputs=dict(
                history='run_sim_history',
                stop_condition='params:stop_condition'
            ),
            outputs='run_sim_increased_data',
            name='irbs_retrieve_outputs',
        )
    ])
    return pipeline([pipeline_prep, pipeline_local])
