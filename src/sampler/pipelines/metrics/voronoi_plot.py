from typing import List, Dict, Tuple, Union
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def dist_volume_voronoi(data, volume_voronoi):
    all_features_data = []
    all_targets_data = []

    for val in volume_voronoi.values():
        all_features_data.extend(val['features'])
        all_targets_data.extend(val['targets'])

    # Check if all experiments have same number of interest samples
    first_exp_dict = next(iter(volume_voronoi.values()))['features']
    first_interest_size = first_exp_dict.shape[0]
    is_same_size_interest = all(
        exp_dict['features'].shape[0] == first_interest_size
        for exp_dict in volume_voronoi.values()
    )
    x_ratio_zoom = 0.75 if is_same_size_interest else 1.

    warnings.warn(
        "Voronoi Volume is interpretable only if stop_on_max_inliers==False "
        "(same number of interest points)"
    )

    fig, axes = plt.subplots(
        2, 2, figsize=(14, 10),
        gridspec_kw={'height_ratios': [0.5, 0.5],'width_ratios': [0.9, 0.1]}
    )

    inset_ax_features = fig.add_axes([0.17, 0.66, 0.2, 0.2])  # First subplot zoomed
    inset_ax_features_box = fig.add_axes([0.41, 0.66, 0.2, 0.2])  # Second subplot zoomed with boxplot
    inset_ax_targets = fig.add_axes([0.17, 0.24, 0.2, 0.2])   # Second subplot zoomed
    inset_ax_targets_box = fig.add_axes([0.41, 0.24, 0.2, 0.2])  # Second subplot zoomed with boxplot

    legend_info_features = []
    legend_info_targets = []

    feature_data_list = []
    target_data_list = []
    feature_labels = []
    target_labels = []

    x_limit_features, x_limit_targets = 1, 1
    y_limit_features, y_limit_targets = 1e-7, 1e-7
    max_features = max(all_features_data)
    max_targets = max(all_targets_data)
    
    offset = 0.1
    width = 0.5

    for i, (data_keys, vol_voronoi) in enumerate(zip(data.keys(), volume_voronoi.values())):
        features = vol_voronoi['features']
        targets = vol_voronoi['targets']

        sorted_features = sorted(features)
        sorted_targets = sorted(targets)

        median_features = np.median(features)
        std_features = np.std(features)
        median_targets = np.median(targets)
        std_targets = np.std(targets)

        x_positions_features = np.arange(len(sorted_features)) + i * offset
        x_positions_targets = np.arange(len(sorted_targets)) + i * offset

        axes[0, 0].bar(
            x_positions_features, sorted_features,
            width=width, color=data[data_keys]['color'],
            alpha=0.25, label=data[data_keys]['name']
        )   
        axes[1, 0].bar(
            x_positions_targets, sorted_targets,
            width=width, color=data[data_keys]['color'],
            alpha=0.25, label=data[data_keys]['name']
        )
        
        inset_ax_features.bar(
            x_positions_features, sorted_features, width=width,
            color=data[data_keys]['color'], alpha=0.25
        )  
        inset_ax_targets.bar(
            x_positions_targets, sorted_targets, width=width,
            color=data[data_keys]['color'], alpha=0.25
        )

        if max_features > 0:
            features_threshold = max_features * 0.01
            x_limit_f = max([
                i for i, v in enumerate(sorted_features) if v < features_threshold
            ] + [0])
            x_limit_features = max(x_limit_features, x_limit_f)
            y_limit_features = max(y_limit_features, sorted_features[x_limit_f+1])

        if max_targets > 0:
            targets_threshold = max_targets * 0.01
            x_limit_t = max([
                i for i, v in enumerate(sorted_targets)
                if v < targets_threshold
            ] + [0])
            x_limit_targets = max(x_limit_targets, x_limit_t)
            y_limit_targets = max(y_limit_targets, sorted_targets[x_limit_t+1])


        feature_data_list.append(features)
        target_data_list.append(targets)
        feature_labels.append(data[data_keys]['name'])
        target_labels.append(data[data_keys]['name'])

        # Calculate outliers
        q1_features = np.percentile(features, 25)
        q3_features = np.percentile(features, 75)
        iqr_features = q3_features - q1_features
        outliers_features = [
            x for x in features if (
                x < q1_features - 1.5 * iqr_features
                or x > q3_features + 1.5 * iqr_features
            )
        ]

        q1_targets = np.percentile(targets, 25)
        q3_targets = np.percentile(targets, 75)
        iqr_targets = q3_targets - q1_targets
        outliers_targets = [
            x for x in targets if (
                x < q1_targets - 1.5 * iqr_targets
                or x > q3_targets + 1.5 * iqr_targets
            )
        ]

        n_outliers_features = len(outliers_features)
        n_outliers_targets = len(outliers_targets)

        legend_info_features.append(
            f"{data[data_keys]['name']}\n"
            f"median: {median_features:.3e}\n"
            f"std: {std_features:.3e}\n"
            f"n_outliers: {n_outliers_features}"
        )
        legend_info_targets.append(
            f"{data[data_keys]['name']}\n"
            f"median: {median_targets:.3e}\n"
            f"std: {std_targets:.3e}\n"
            f"n_outliers: {n_outliers_targets}"
        )

    def customize_boxplot(ax, data_list, colors):
        boxplots = ax.boxplot(data_list, patch_artist=True, medianprops=dict(color='black'))
        for patch, color in zip(boxplots['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        ax.set_xticks([])
        ax.grid(True)

        # Determine the ylim to show IQR and highest whisker
        data_concat = np.concatenate(data_list)
        q1 = np.percentile(data_concat, 25)
        q3 = np.percentile(data_concat, 75)
        iqr = q3 - q1
        ylim_high = q3 + 1.6 * iqr # 1.6 to add a little margin (1.5 is the default value)
        ax.set_ylim([0, ylim_high])

    feature_colors = [data[data_keys]['color'] for data_keys in data.keys()]
    target_colors = [data[data_keys]['color'] for data_keys in data.keys()]

    customize_boxplot(inset_ax_features_box, feature_data_list, feature_colors)
    customize_boxplot(inset_ax_targets_box, target_data_list, target_colors)
    
    inset_ax_features.set_xlim([0, x_limit_features])
    inset_ax_targets.set_xlim([0, x_limit_targets])
    inset_ax_features.set_ylim([0, 1.05*y_limit_features])
    inset_ax_targets.set_ylim([0, 1.05*y_limit_targets])

    inset_ax_features.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    inset_ax_targets.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

    inset_ax_features_box.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    inset_ax_targets_box.ticklabel_format(axis='y', style='sci', scilimits=(0,0))


    # Set y-axis lim for inset bar plot
    if not is_same_size_interest:
        inset_ax_features.set_ylim(0, max(all_features_data) * 0.01)
        inset_ax_targets.set_ylim(0, max(all_targets_data) * 0.01)

    axes[0, 0].set_title('Volume of Voronoï Cell in Features Space (Interest Points)')
    axes[0, 0].legend().set_visible(False)

    axes[1, 0].set_title('Volume of Voronoï Cell in Features + Targets Space (Interest Points)')
    axes[1, 0].legend().set_visible(False)

    axes[0, 1].axis('off')  # Hide the empty subplot for the legend
    axes[1, 1].axis('off')  # Hide the empty subplot for the legend
    fig.legend(legend_info_features, loc='center', bbox_to_anchor=(0.85, 0.75), title='Features')
    fig.legend(legend_info_targets, loc='center', bbox_to_anchor=(0.85, 0.3), title='Targets')

    return fig
