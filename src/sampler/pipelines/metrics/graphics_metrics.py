import numpy as np
import seaborn as sns
import pandas as pd

from typing import List, Dict, Tuple, Union
from itertools import combinations

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec


def get_colors_legend(data):
    return [mlines.Line2D([], [], color=vals['color'], label=vals['name']) for vals in data.values()]


def plot_violin_distribution(data: Dict, targets: List[str], desired_region: Dict, volume: Dict = None):
    fig, axs = plt.subplots(1, len(targets), figsize=(10, 5))

    for col, target in enumerate(targets):
        sns.violinplot(
            data=[d['inliers'][target] for k, d in data.items()],
            palette=[val['color'] for val in data.values()],
            cut=0,  ax=axs[col]
        )
        axs[col].set_xticklabels([val['name'] for val in data.values()], rotation=20, ha='right')
        axs[col].add_patch(Rectangle(
            (-0.45, desired_region[target][0]),
            (len(data) - 0.1),
            desired_region[target][1] - desired_region[target][0],
            edgecolor='#B73E3E', facecolor='none', lw=2,
        ))
        axs[col].set_ylabel(target)

    axs[0].text(-0.03, 1.03, 'a)', transform=axs[0].transAxes, size=20, weight='bold', ha='right', va='bottom')
    axs[1].text(-0.03, 1.03, 'b)', transform=axs[1].transAxes, size=20, weight='bold', ha='right', va='bottom')
    
    # Add legend if volume is provided
    if volume is not None:
        handles = []
        labels = []
        for key, value in volume.items():
            handle = Rectangle((0,0), 1, 1, color=data[key]['color'])
            handles.append(handle)
            labels.append(f'{value:.2e}')
        
        # Add legend to the last subplot
        axs[-1].legend(handles, labels, loc='upper left', bbox_to_anchor=(1.1, 1), title='Area')
        
    fig.tight_layout()
    
    return fig


def pair_grid_for_all_variables(data, features, targets):
    sns.set_theme(font_scale=1.15)
    df_to_plot = pd.concat([v['inliers'].assign(identity=v['name']) for v in data.values()])
    g = sns.PairGrid(
        df_to_plot[features + targets + ['identity']],
        hue='identity', palette=[val['color'] for val in data.values()], diag_sharey=False
    )
    g.map_lower(sns.scatterplot, alpha=0.3)
    g.map_upper(sns.kdeplot, levels=4, linewidths=2)
    g.map_diag(sns.kdeplot, fill=False, linewidth=2)
    g.add_legend()
    return g.figure


def targets_kde(data: Dict, targets: List[str], region: Dict):
    n_col = len(targets)
    fig, axs = plt.subplots(1, n_col, figsize=(6*n_col, 6))

    df_to_plot = pd.concat(
        [
            v['inliers'].assign(identity=v['name'], experiment=k)
            for k, v in data.items()
        ],
        axis=0, ignore_index=True
    )

    #df_to_plot[targets] = df_to_plot[targets] / scales['targets']

    plt.ticklabel_format(axis='y', style='sci')
    bw_adjust_per_target = [0.2, 0.3] # Default: 1.0
    for col, (target, bw_adjust) in enumerate(zip(targets, bw_adjust_per_target)):
        sns.kdeplot(
            data=df_to_plot, x=target, hue='experiment',
            hue_order=[k for k in data.keys()][::-1], # Reverse drawing order of plots 
            palette=[val['color'] for val in data.values()][::-1],
            cut=0,
            ax=axs[col], legend=False, fill=True, common_norm=True,
            bw_adjust=bw_adjust
        )
        axs[col].axvline(x=region[target][0], color='red')
        axs[col].axvline(x=region[target][1], color='red')
        axs[col].set_xlabel(target)

    use_zoom = False
    if use_zoom:
        # Add inset axis
        axins = axs[0].inset_axes([0.5, 0.1, 0.45, 0.45])  # [x0, y0, width, height]

        # Hide axis labels
        axins.set_ylabel(' ')
        axins.set_xlabel(' ')
        axins.tick_params(axis='x', labelsize=10)
        axins.tick_params(axis='y', labelsize=10)

        for col, (target, bw_adjust) in enumerate(zip(targets, bw_adjust_per_target)):
            sns.kdeplot(data=df_to_plot, x=target, hue='experiment',
                        hue_order=[k for k in data.keys()][::-1], # Reverse drawing order of plots 
                        palette=[val['color'] for val in data.values()][::-1],
                        cut=0, ax=axins, legend=False, fill=True, common_norm=True,
                        bw_adjust=bw_adjust
                        
                        )

        axins.axvline(x=region[targets[0]][0], color='red')
        axins.axvline(x=region[targets[0]][1], color='red')

        axins.set_xlim(region[targets[0]][0] - 1, region[targets[0]][1] + 1)
        axins.set_ylim(0.02, 0.11) # < Adjust zoom window y-limits as needed
        axs[0].indicate_inset_zoom(axins)

    colors_legend = get_colors_legend(data)

    axs[1].set_ylabel('')
    axs[1].legend(handles=colors_legend, loc='upper left', bbox_to_anchor=(1.1, 1))
    axs[0].text(-0.03, 1.03, 'a)', transform=axs[0].transAxes, size=20, weight='bold', ha='right', va='bottom')
    axs[1].text(-0.03, 1.03, 'b)', transform=axs[1].transAxes, size=20, weight='bold', ha='right', va='bottom')
    fig.tight_layout()
    return fig


def plot_2d(data: Dict, features_dic: Dict, volume: Dict):
    features = features_dic['str']
    features_latex = features_dic['latex']
    feature_pairs = list(combinations(features, 2))
    n_exp = len(data)
    n_rows = len(feature_pairs)

    fig, axs = plt.subplots(n_rows, n_exp, sharey='row', figsize=(4 * n_exp, 4 * n_rows + 1),
                            constrained_layout=False, squeeze=False)
    
    # ? Maybe for one experiment, uncomment this (next commit)
    # if len(feature_pairs)==1: # axs is a 2D array, but it have to be treated as a 1D array, squeeze
    #     axs = axs[0]

    for n_col, (k, v) in enumerate(data.items()):
        num_no_interest = v['no_interest'].shape[0]
        num_interest = v['interest'].shape[0]
        num_outliers = v['outliers'].shape[0]
        interest_colors = plt.cm.autumn(np.linspace(1, 0, num_interest))

        for n_row, (x, y) in enumerate(feature_pairs):
            idx = (n_row, n_col) if n_exp > 1 else n_row
            axs[idx].scatter(
                x=v['no_interest'][x],
                y=v['no_interest'][y],
                c='gray', alpha=0.3, label='No interest'
            )
            axs[idx].scatter(
                x=v['outliers'][x],
                y=v['outliers'][y],
                c='black', alpha=0.7, marker='x', label='Outlier'
            )
            axs[idx].scatter(
                x=v['interest'][x],
                y=v['interest'][y],
                c=interest_colors, alpha=0.5, label='Interest'
            )
            axs[idx].set_xticks(np.arange(11))
            axs[idx].set_xticklabels(['0', '', '2', '', '4', '', '6', '', '8', '', '10'])
            axs[idx].set_xlabel(features_latex[features.index(x)].replace('/', '\\'))
            axs[idx].set_ylabel(features_latex[features.index(y)].replace('/', '\\'))


        idx_legend = (0, n_col) if n_exp > 1 else 0
        handles, _ = axs[idx_legend].get_legend_handles_labels()

        # Create a handle for the 'interest' marker
        interest_marker = plt.Line2D(
            [0], [0], marker='o', color='w', markerfacecolor='red',
            alpha=0.7, label=''
        )

        # Initialize handles and labels for the legend
        legend_handles = [interest_marker, handles[0]]
        legend_labels = [
            f'n_Itr: {num_interest}\n'
            f'V_Itr_bound: [{volume[k][0]:.2e}, {volume[k][1]:.2e}]',
            f'n_noItr: {num_no_interest}',
        ]

        # Add outliers to the legend if present
        if num_outliers != 0:
            legend_handles.append(handles[1])
            legend_labels.append(f'Outliers {num_outliers}')

        # Create the legend on the specified axis
        axs[idx_legend].legend(
            handles=legend_handles, labels=legend_labels,
            loc='lower center', bbox_to_anchor=(0.5, 1.0),
            title=v['name'], title_fontsize='large'
        )
    
    # Add a vertical color bar with custom ticks outside the subplots
    cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])  # Adjusted position
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='autumn_r'), cax=cbar_ax, orientation='vertical')
    cbar.set_ticks([0, 1])  # Set ticks at the start and end
    cbar.set_ticklabels(['first', 'last'])  # Label the ticks
    cbar.set_label('Interest order', fontsize='large')

    fig.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to make space for the color bar
    fig.subplots_adjust(top=0.85)
    return fig


def plot_feat_tar(data: Dict, features: List[str], targets: List[str], only_interest: bool = True, title_extension: str=''):
    n_col = len(features)
    n_rows = len(targets)
    fig, axs = plt.subplots(n_rows, n_col, sharey='row', sharex='col', figsize=(4 * n_col, 4 * n_rows))
    for n_row, tar, in enumerate(targets):
        for n_col, feat in enumerate(features):
            for v in data.values():
                axs[n_row, n_col].scatter(
                    x=v['interest'][feat],
                    y=v['interest'][tar],
                    c=v['color'], marker='.', alpha=1., label=v['name']
                )
                if not only_interest:
                    axs[n_row, n_col].scatter(
                        x=v['no_interest'][feat],
                        y=v['no_interest'][tar],
                        c='gray', marker='.', alpha=0.3, label='Not int'
                    )
            axs[1, n_col].set_xlabel(feat)
        axs[n_row, 0].set_ylabel(tar)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    if not only_interest and len(data) > 1:  # Remove redundant 'Not int' label
        handles = handles[::2] + [handles[-1]]
        labels = labels[::2] + [labels[-1]]
    fig.tight_layout()
    fig.legend(handles=handles,
               labels=labels,
               loc='upper center', ncol=len(labels),
               title=f'Targets(Features) {title_extension}',
               title_fontsize='large')
    fig.subplots_adjust(top=0.85)
    return fig


def plot_asvd_scores(
    data: Dict[str, Dict[str, Union[str, pd.DataFrame]]],
    experiments: Dict[str, Dict[str, float]], 
    metrics_to_plot: Union[List[str], None] = None,
    figsize: Tuple[int, int] = (15, 10)
):
    """
    Plot scores for different experiments using a grouped bar plot and display remaining
    metrics in a table.

    Parameters:
    - experiments (Dict[str, Dict[str, float]]): A dictionary where keys are experiment
    names and values are dictionaries of scores.
    - metrics_to_plot (List[str]): List of metric names to plot in the bar chart.
    If None, all metrics are plotted.
    - figsize (Tuple[int, int]): Size of the figure (width, height) in inches.
    """
    exp_names = list(experiments.keys())
    
    # Get all unique metrics while preserving order
    all_metrics = []
    for exp in experiments.values():
        all_metrics.extend(metric for metric in exp.keys() if metric not in all_metrics)

    if metrics_to_plot is None:
        metrics_to_plot = all_metrics.copy()
    else:
        metrics_to_plot = [metric for metric in metrics_to_plot if metric in all_metrics]

    metrics_for_table = [metric for metric in all_metrics if metric not in metrics_to_plot]

    n_experiments = len(exp_names)
    n_metrics_plot = len(metrics_to_plot)

    # Adjust figure size and subplot ratio based on whether we have a table
    if metrics_for_table:
        fig, (ax_bar, ax_table) = plt.subplots(nrows=2, ncols=1, figsize=figsize, 
                                               gridspec_kw={'height_ratios': [3, 1]})
    else:
        fig, ax_bar = plt.subplots(figsize=figsize)

    # Grouped Bar Plot
    bar_width = 0.8 / n_experiments
    index = np.arange(n_metrics_plot)

    for i, exp_name in enumerate(exp_names):
        values = [experiments[exp_name].get(metric, np.nan) for metric in metrics_to_plot]
        position = index + i * bar_width
        rects = ax_bar.bar(
            position, values, bar_width, label=exp_name, alpha=0.8,
            color=data[exp_name]['color']
        )

        # Add value labels on top of each bar
        for rect in rects:
            height = rect.get_height()
            if np.isfinite(height):
                ax_bar.text(rect.get_x() + rect.get_width()/2., height,
                            f'{height:.2f}', ha='center', va='bottom', rotation=90)

    ax_bar.set_ylabel('Values')
    ax_bar.set_title('Comparison of Metrics Across Experiments')
    ax_bar.set_xticks(index + bar_width * (n_experiments - 1) / 2)
    ax_bar.set_xticklabels(metrics_to_plot, rotation=45, ha='right')
    ax_bar.legend()

    # Table (if there are metrics for the table)
    if metrics_for_table:
        table_data = []
        for exp_name in exp_names:
            row = [exp_name]
            for metric in metrics_for_table:
                value = experiments[exp_name].get(metric, 'N/A')
                if isinstance(value, int):
                    row.append(f'{value}')
                elif isinstance(value, float):
                    row.append(f'{value:.2e}')
                else:
                    row.append(str(value))
            table_data.append(row)

        ax_table.axis('off')
        table = ax_table.table(cellText=table_data, 
                               colLabels=['Experiment'] + metrics_for_table,
                               cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)  # Adjust the scale to fit your needs

    plt.tight_layout()
    return fig

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

    print(
        "Voronoi Volume is interpretable only if run_until_max_size==False "
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


