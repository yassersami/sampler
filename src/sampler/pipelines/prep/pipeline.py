"""
This is a boilerplate pipeline 'prep'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline

from sampler.pipelines.prep.nodes import prepare_initial_data, get_scaler


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=get_scaler,
            inputs=dict(
                log_scale='params:log_scale',
                features='params:features',
                targets='params:targets',
                variables_ranges='params:variables_ranges',
            ),
            outputs='scaler',
            name='fit_scaler'
        ),
        node(
            func=prepare_initial_data,
            inputs=dict(
                initial_data='initial_data',
                features='params:features',
                targets='params:targets',
                additional_values='params:additional_values',
                variables_ranges='params:variables_ranges',
                interest_region='params:interest_region',
                outliers_filling='params:outliers_filling',
                sim_time_cutoff='params:sim_time_cutoff',
                scaler='scaler',
            ),
            outputs=dict(
                treated_data='treated_data',
                treatment='treatment',
            ),
            name='preparation',
        )
    ])
