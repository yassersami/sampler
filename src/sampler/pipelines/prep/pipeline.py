"""
This is a boilerplate pipeline 'prep'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import prepare_treatment, prepare_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=prepare_treatment,
            inputs=dict(
                features="params:features",
                targets="params:targets",
                variables_ranges="params:variables_ranges",
                interest_region="params:interest_region",
                simulator_config="params:simulator_config"
            ),
            outputs="treatment",
            name="prepare_treatment_node"
        ),
        node(
            func=prepare_data,
            inputs=dict(
                initial_data="initial_data",
                additional_values="params:additional_values",
                treatment="treatment"
            ),
            outputs="treated_data",
            name="prepare_data_node"
        )
    ])
