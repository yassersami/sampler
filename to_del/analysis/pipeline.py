"""
This is a boilerplate pipeline 'analysis'
generated using Kedro 0.18.5
"""

from kedro.pipeline import Pipeline, node, pipeline

from sampler.pipelines.analysis.nodes import prepare_data, analyze_ignition_points,\
    plot_multi_analysis, plot_targets, make_resume, scale_data_for_plots


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=prepare_data,
            inputs=dict(
                experiments='params:experiments',
                names='params:names',
                features='params:features',
                targets='params:targets',
                additional_values='params:additional_values',
                treatment='treatment',
                n_slice='params:n_slice',
                tot_size='params:total_size'
            ),
            outputs=dict(
                exp_data='exp_data',
                features='features',
                targets='targets',
                targets_prediction='targets_prediction'
            ),
            name='prepare_data_to_analyze',
        ),
        node(
            func=make_resume,
            inputs=dict(
                data='exp_data',
                initial_size='params:initial_size',
                n_slice='params:n_slice',
                tot_size='params:total_size',
                features='features',
                treatment='treatment',
                digits='params:tol_round_decimals'
            ),
            outputs='resume_file',
            name='make_results_summary'
        ),
        node(
            func=scale_data_for_plots,
            inputs=dict(
                data='exp_data',
                features='features',
                targets='targets',
                targets_prediction='targets_prediction',
                scales='params:scales',
                interest_region='params:interest_region',
            ),
            outputs=dict(
                scaled_data='scaled_exp_data',
                scaled_region='scaled_region'
                ),
            name='scale_data_for_plots'
        ),
        node(
            func=analyze_ignition_points,
            inputs=dict(
                data='scaled_exp_data',
                features='features',
                targets='targets',
                points='params:ignition_points'
            ),
            outputs=dict(
                features_2d='features_2d',
                feat_tar_interest='feat_tar_interest',
                feat_tar_all='feat_tar_all',
            ),
            name='ignition_for_features',
        ),
        node(
            func=plot_multi_analysis,
            inputs=dict(
                data='scaled_exp_data',
                features='features',
                targets='targets',
            ),
            outputs='multi_analysis',
            name='plot_multi_analysis'
        ),
        node(
            func=plot_targets,
            inputs=dict(
                data='scaled_exp_data',
                targets='targets',
                region='scaled_region',
            ),
            outputs=dict(
                violin_plot='violin_plot',
                kde_plot='targets_kde',
            ),
            name='measure_coverage',
        )
    ])
