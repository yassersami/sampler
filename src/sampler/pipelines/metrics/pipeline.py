"""
This is a boilerplate pipeline 'metrics'
generated using Kedro 0.18.5
"""

from kedro.pipeline import Pipeline, node, pipeline

from sampler.pipelines.prep import create_pipeline as create_pipeline_prep

from .nodes import (
    read_and_prepare_data, compute_metrics,
    get_variables_for_plot, scale_variables_for_plot,
    plot_metrics
)


def create_pipeline(**kwargs) -> Pipeline:
    pipeline_prep = create_pipeline_prep()
    pipeline_local = pipeline([
        node(
            func=read_and_prepare_data,
            inputs=dict(
                experiments='params:experiments',
                features='params:features',
                targets='params:targets',
                treatment='treatment',
            ),
            outputs='data',
            name='metrics_read_data',
        ),
        node(
            func=compute_metrics,
            inputs=dict(
                data='data',
                features='params:features',
                targets='params:targets',
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
            name='metrics_compute'
        ),
        node(
            func=get_variables_for_plot,
            inputs=dict(
                features='params:features',
                targets='params:targets',
                plot_variables='params:plot_variables',
            ),
            outputs=dict(
                feature_aliases='feature_aliases',
                target_aliases='target_aliases',
                latex_mapper='latex_mapper',
                alias_scales='alias_scales',
            ),
            name='metrics_read_plot_variables',
        ),
        node(
            func=scale_variables_for_plot,
            inputs=dict(
                data='data',
                features='params:features',
                targets='params:targets',
                feature_aliases='feature_aliases',
                target_aliases='target_aliases',
                alias_scales='alias_scales',
                variables_ranges="params:variables_ranges",
                interest_region='params:interest_region',
            ),
            outputs=dict(
                scaled_data='scaled_data',
                scaled_plot_ranges='scaled_plot_ranges',
                scaled_interest_region='scaled_interest_region'
            ),
            name='metrics_scale'
        ),
        node(
            func=plot_metrics,
            inputs=dict(
                data='scaled_data',
                feature_aliases='feature_aliases',
                target_aliases='target_aliases',
                latex_mapper='latex_mapper',
                plot_ranges='scaled_plot_ranges',
                interest_region='scaled_interest_region',
                volume='volume',
                total_asvd_scores='total_asvd_scores',
                interest_asvd_scores='interest_asvd_scores',
                volume_voronoi='volume_voronoi',
                output_dir='path_metrics_output',
            ),
            outputs=None,
            name='metrics_plot'
        )
    ])
    return pipeline([pipeline_prep, pipeline_local])
