"""
This is a boilerplate pipeline 'metrics'
generated using Kedro 0.18.5
"""

from kedro.pipeline import Pipeline, node, pipeline

from sampler.pipelines.prep import create_pipeline as create_pipeline_prep

from .nodes import prepare_data_metrics, get_metrics, scale_data_for_plots, plot_metrics


def create_pipeline(**kwargs) -> Pipeline:
    pipeline_prep = create_pipeline_prep()
    pipeline_local = pipeline([
        node(
            func=prepare_data_metrics,
            inputs=dict(
                experiments='params:experiments',
                variable_aliases='params:variable_aliases',
                features='params:features',
                targets='params:targets',
                treatment='treatment',
            ),
            outputs=dict(
                exp_data='exp_data',
                features='features',
                targets='targets',
            ),
            # name='prepare_data_to_analyze',
        ),
        node(
            func=get_metrics,
            inputs=dict(
                data='exp_data',
                features='features',  # These features and targets are not those of data columns
                targets='targets',
                treatment='treatment',
                params_volume='params:params_volume',
                params_asvd='params:params_asvd',
                params_voronoi='params:params_voronoi',
            ),
            outputs=dict(
                volume='volume',
                total_asvd_scores='total_asvd_scores',
                interest_asvd_scores='interest_asvd_scores',
                volume_voronoi='volume_voronoi'
            ),
            # name='get_metrics'
        ),
        node(
            func=scale_data_for_plots,
            inputs=dict(
                data='exp_data',
                features='features',
                targets='targets',
                scales='params:scales',
                interest_region='params:interest_region',
            ),
            outputs=dict(
                scaled_data='scaled_exp_data',
                scaled_region='scaled_region'
            ),
            # name='scale_data_for_plots'
        ),
        node(
            func=plot_metrics,
            inputs=dict(
                data='scaled_exp_data',
                variable_aliases='params:variable_aliases',
                region='scaled_region',
                volume='volume',
                total_asvd_scores='total_asvd_scores',
                interest_asvd_scores='interest_asvd_scores',
                volume_voronoi='volume_voronoi',
                output_dir='path_metrics_output',
            ),
            outputs=None,
            # name='plot_metrics'
        )
    ])
    return pipeline([pipeline_prep, pipeline_local])
