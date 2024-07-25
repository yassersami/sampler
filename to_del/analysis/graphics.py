from cProfile import label
import math
from turtle import title
from typing import Dict, List
import warnings

import numpy as np
import pandas as pd
from pyparsing import line
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.lines as mlines

from itertools import combinations

from sampler.pipelines.analysis.postprocessing_functions import extract_percentage

plt.rcParams.update({'font.size': 18})


def line_plot_for_inliers_outliers(data, initial_size, n_slice, tot_size):
    fig, [ax0, ax1] = plt.subplots(1, 2, figsize=(10, 4), sharex='row')
    plt.rcParams.update({'font.size': 12})
    for vals in data.values():
        res_io = extract_percentage(initial_size, tot_size, n_slice, vals)
        ax0.plot(res_io['o%'], color=vals['color'])
        ax0.plot(res_io['in%'], marker='o', color=vals['color'], label=vals['name'])

        ax1.plot(res_io['others'], color=vals['color'])
        ax1.plot(res_io['interest'], marker='o', color=vals['color'])

    ax0.set_xlabel('Simulations performed')
    ax1.set_xlabel('Simulations performed')
    ax0.set_ylabel('Percentage over total samples')
    ax0.set_ylim([0, 1])
    ax1.set_ylabel('Total samples')
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position("right")
    fig.legend(loc='center', bbox_to_anchor=(0.245, 0.575))
    ax1.legend(handles=get_results_legends())
    fig.tight_layout()
    return fig


def get_colors_legend(data):
    return [mlines.Line2D([], [], color=vals['color'], label=vals['name']) for vals in data.values()]


def get_results_legends():
    interest = mlines.Line2D([], [], color='gray', marker='o', label='Interest')
    other = mlines.Line2D([], [], color='gray', label='Other')
    return [interest, other]


def plot_scatter_for_nn(data: Dict, features: List[str], features_nn: List[str]):
    total_rows = len(data)
    total_cols = len(features)
    fig, axs = plt.subplots(total_rows, total_cols, figsize=(5*total_cols, 4*total_rows), sharey='col', sharex='col')
    if total_rows == 1:
        vals = [val for val in data.values()][0]
        for n_col, x in enumerate(features):
            axs[n_col].scatter(
                x=vals['outliers'][features[n_col]], y=vals['outliers'][features_nn[n_col]], c='gray', alpha=0.5,
            )
            im = axs[n_col].scatter(
                x=vals['inliers'][features[n_col]], y=vals['inliers'][features_nn[n_col]], c=list(vals['inliers'].index)
            )
            axs[total_rows - 1, n_col].set_xlabel(f'{x} sampler')
            axs[n_col].set_ylabel(f'{x} nn')
        axs[0].text(0.5e-5, 0.1e-5, vals['name'])
    else:
        for n_row, vals in enumerate(data.values()):
            for n_col, x in enumerate(features):
                axs[n_row, n_col].scatter(
                    x=vals['outliers'][features[n_col]], y=vals['outliers'][features_nn[n_col]], c='gray', alpha=0.5,
                )
                im = axs[n_row, n_col].scatter(
                    x=vals['inliers'][features[n_col]], y=vals['inliers'][features_nn[n_col]], c=list(vals['inliers'].index)
                )
                axs[total_rows-1, n_col].set_xlabel(f'{x} sampler')
                axs[n_row, n_col].set_ylabel(f'{x} nn')
            axs[n_row, 0].text(0.5e-5, 0.1e-5, vals['name'])
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    return fig


def plot_histogram_for_nn(data: Dict, features: List[str], features_nn: List[str]):
    total_rows = len(data)
    total_cols = len(features)
    fig, axs = plt.subplots(total_rows, total_cols, figsize=(5*total_cols, 4*total_rows), sharey='all')
    bins = list(np.arange(0, 100, 5))
    for n_row, vals in enumerate(data.values()):
        for n_col, x in enumerate(features):
            x_var = vals['inliers'][features[n_col]]
            y_var = vals['inliers'][features_nn[n_col]]
            val_hist = [100 * math.dist((x_var[i],), (y_var[i],)) / x_var[i] for i in list(x_var.index)]
            axs[n_row, n_col].hist(val_hist, bins=bins)
            axs[total_rows-1, n_col].set_xlabel(f'{x} similarity percentage [%]')
        axs[n_row, 0].set_ylabel('Number of samples')
        axs[n_row, 0].text(50, 500, vals['name'])
    return fig


def plot_scatter_y_1d(data: Dict, fx: List[str], y: List[str]):
    total_rows = len(data)
    fig, axs = plt.subplots(
        1, total_rows, figsize=(5, 4*total_rows), sharey='col', sharex='col'
    )
    for n_row, vals in enumerate(data.values()):
        axs[n_row].scatter(
            x=vals['outliers'][y[0]], y=vals['outliers'][fx[0]], c='gray', alpha=0.5,
        )
        im = axs[n_row].scatter(
            x=vals['inliers'][y[0]], y=vals['inliers'][fx[0]], c=list(vals['inliers'].index)
        )

        axs[total_rows - 1].set_xlabel(f'{y[0]} sampler')
        axs[n_row].set_ylabel(f'{y[0]} surrogate')
        axs[n_row].text(0.8e8, 0.2e-5, vals['name'])
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    return fig


def plot_scatter_y(data: Dict, fx: List[str], y: List[str]):
    total_rows = len(data)
    total_cols = len(y)
    fig, axs = plt.subplots(
        total_rows, total_cols, figsize=(5*total_cols, 5*total_rows), sharey='col', sharex='col'
    )
    for n_row, vals in enumerate(data.values()):
        for n_col, x in enumerate(y):
            axs[n_row, n_col].scatter(
                x=vals['outliers'][y[n_col]], y=vals['outliers'][fx[n_col]], c='gray', alpha=0.5,
            )
            im = axs[n_row, n_col].scatter(
                x=vals['inliers'][y[n_col]], y=vals['inliers'][fx[n_col]], c=list(vals['inliers'].index)
            )

            axs[total_rows - 1, n_col].set_xlabel(f'{x} sampler')
            axs[n_row, n_col].set_ylabel(f'{x} surrogate')
        axs[n_row, 0].text(1e7, 5e7, vals['name'])
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    return fig


def plot_y_histogram_1d(data: Dict, fx: List[str], y: List[str]):
    total_rows = len(data)
    fig, axs = plt.subplots(
        1, total_rows, figsize=(5, 4*total_rows), sharey='col', sharex='col'
    )
    bins = list(np.arange(0, 100, 5))
    for n_row, vals in enumerate(data.values()):
        x_var = vals['inliers'][y[0]]
        y_var = vals['inliers'][fx[0]]
        val_hist = [100 * math.dist((x_var[i],), (y_var[i],)) / x_var[i] for i in list(x_var.index)]
        axs[n_row].hist(val_hist, bins=bins)
        axs[total_rows - 1].set_xlabel(f'{y[0]} similarity percentage [%]')
        axs[n_row].set_ylabel('Number of samples')
        axs[n_row].text(70, 100, vals['name'])
    return fig


def plot_y_histogram(data: Dict, fx: List[str], y: List[str]):
    total_rows = len(data)
    total_cols = len(y)
    fig, axs = plt.subplots(
        total_rows, total_cols, figsize=(5*total_cols, 4*total_rows), sharey='col', sharex='col'
    )
    bins = list(np.arange(0, 100, 5))
    for n_row, vals in enumerate(data.values()):
        for n_col, x in enumerate(y):
            x_var = vals['inliers'][y[n_col]]
            y_var = vals['inliers'][fx[n_col]]
            val_hist = [100 * math.dist((x_var[i],), (y_var[i],)) / x_var[i] for i in list(x_var.index)]
            axs[n_row, n_col].hist(val_hist, bins=bins)
            axs[total_rows - 1, n_col].set_xlabel(f'{x} similarity percentage [%]')
        axs[n_row, 0].set_ylabel('Number of samples')
        axs[n_row, 0].text(50, 100, vals['name'])
    return fig


def targets_kde(data: Dict, targets: List[str], region: Dict):
    n_col = len(targets)
    fig, axs = plt.subplots(1, n_col, figsize=(6*n_col, 6))

    df_to_plot = pd.concat([v['inliers'].assign(identity=v['name'], experiment=k) for k, v in data.items()])

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
    axs[0].legend(handles=colors_legend, loc='upper right')
    axs[0].text(-0.28, 0.95, 'a)', transform=axs[0].transAxes, size=20, weight='bold')
    axs[1].text(-0.35, 0.95, 'b)', transform=axs[1].transAxes, size=20, weight='bold')
    fig.tight_layout()
    return fig


def pair_grid_for_all_variables(data, features, targets):
    sns.set_theme(font_scale=1.15)
    df_to_plot = pd.concat([v['inliers'].assign(identity=v['name']) for v in data.values()])
    g = sns.PairGrid(
        df_to_plot[features + targets + ['identity']],
        hue="identity", palette=[val['color'] for val in data.values()], diag_sharey=False
    )
    g.map_lower(sns.scatterplot, alpha=0.3)
    g.map_upper(sns.kdeplot, levels=4, linewidths=2)
    g.map_diag(sns.kdeplot, fill=False, linewidth=2)
    g.add_legend()
    return g.figure


def draw_rectangle(axs, n_col, n_row, region, vals, x):
    x0 = np.min(vals['df'][x])
    xn = np.max(vals['df'][x])
    axs[n_row, n_col].add_patch(
        Rectangle(
            (x0, region[0]),
            xn - x0,
            region[1] - region[0],
            edgecolor='#B73E3E', facecolor='none', lw=1.5, label='Region of interest'
        )
    )


def plot_violin_distribution(data: Dict, targets: List[str], desired_region: Dict):
    fig, axs = plt.subplots(1, len(targets), figsize=(10, 5))

    #targets_scales = scales["targets"]
    
    for col, target in enumerate(targets):
        sns.violinplot(
            data=[d['inliers'][target] for k, d in data.items()],
            palette=[val['color'] for val in data.values()],
            cut=0,  ax=axs[col]
        )
        axs[col].set_xticklabels([val['name'] for val in data.values()], rotation=15)
        axs[col].add_patch(Rectangle(
            (-0.45, desired_region[target][0]),
            (len(data) - 0.1),
            desired_region[target][1] - desired_region[target][0],
            edgecolor='#B73E3E', facecolor='none', lw=2,
        ))
        axs[col].set_ylabel(target)
    axs[1].yaxis.tick_right()
    axs[1].yaxis.set_label_position("right")
    axs[0].text(-0.15, 0.95, 'a)', transform=axs[0].transAxes, size=20, weight='bold')
    axs[1].text(-0.15, 0.95, 'b)', transform=axs[1].transAxes, size=20, weight='bold')
    fig.tight_layout()
    return fig


def plot_horizontal_bars(
        data: Dict, target: str, ranges: np.ndarray, time_colors: List[str],
        initial_size: int, n_slice: int, tot_size: int,
):
    n_ranges = len(ranges) - 1
    fig, axs = plt.subplots(1, len(data.keys()), figsize=(20, 6), sharex='all', sharey='all')
    list_ranges = [f'[{ll:.1E}-{ul:.1E}]' for ll, ul in zip(ranges[:-1], ranges[1:])]
    for n_col, (identity, v) in enumerate(data.items()):
        left_lim = np.zeros(n_ranges)
        l_lim = 0
        for n_sample, lim in enumerate(np.append([initial_size], np.arange(n_slice, tot_size+1, n_slice))):
            df = v['inliers'].loc[l_lim:lim, target]
            counts = df.groupby(pd.cut(df, ranges)).count().values
            axs[n_col].barh(y=list_ranges, width=counts, left=left_lim, color=time_colors[n_sample], label=lim)
            axs[n_col].title.set_text(v['name'])
            left_lim += counts
            l_lim = lim
    fig.suptitle('Population of region of interest')
    fig.supxlabel('Number of samples')
    fig.supylabel(target)
    plt.legend(bbox_to_anchor=(1.4, 1.02), loc='upper right', title='Nº simulations')
    return plt.gcf()


def plot_2d(data: Dict, features: List[str], points: Dict):
    colors = points["colors"]

    n_exp = len(data)
    n_rows = len(features)
    fig, axs = plt.subplots(n_rows, n_exp, sharey='row', figsize=(4*n_exp, 4*n_rows),
                        constrained_layout=False)

    feature_pairs = list(combinations(features, 2))
    
    for n_col, v in enumerate(data.values()):
        num_not_interesting = v['not_interesting'].shape[0]
        num_interest = v['interest'].shape[0]
        num_outliers = v['outliers'].shape[0]

        # Divide features columns using the scales["features"] array
        # v['interest'][features] = v['interest'][features] / scales["features"]
        # v['not_interesting'][features] = v['not_interesting'][features] / scales["features"]
        # v['outliers'][features] = v['outliers'][features] / scales["features"]

        for n_row, (x, y) in enumerate(feature_pairs):
            idx = (n_row, n_col) if len(data.keys()) > 1 else n_row
            axs[idx].scatter(
                x=v['not_interesting'][x],
                y=v['not_interesting'][y],
                c=colors['not_interesting'], alpha=0.3, label=f"Not interesting",
                # Add logscale
                
            )
            axs[idx].scatter(
                x=v['outliers'][x],
                y=v['outliers'][y],
                c=colors['outliers'], marker='x', label=f"Outliers"
            )
            axs[idx].scatter(
                x=v['interest'][x],
                y=v['interest'][y],
                c=colors['interest'], alpha=0.3, label=f"Interest"
            )
            axs[idx].set_xticks(np.arange(11))
            axs[idx].set_xticklabels(['0', '', '2', '', '4', '', '6', '', '8', '', '10'])
            if x == 'Al radius [micron]':
                axs[idx].set_xlabel(r'$r_{Al} \;[\mu m]$')
            else:
                axs[idx].set_xlabel(r'$r_{CuO} \;[\mu m]$')

            idx_label = (n_row, 0) if len(data.keys()) > 1 else n_row
            if y == 'CuO radius [micron]':
                axs[idx_label].set_ylabel(r'$r_{CuO} \;[\mu m]$')
            else:
                axs[idx_label].set_ylabel(r'$\phi$')
                
        # Sort labels as Interest, Not int and Outliers
        idx_legend = (0, n_col) if len(data.keys()) > 1 else 0
        hs, _ = axs[idx_legend].get_legend_handles_labels()

        handles = [hs[2], hs[0]] if num_outliers == 0 else [hs[2], hs[0], hs[1]]
        labels = [f'Interest {num_interest}', f'Not int {num_not_interesting}'] if num_outliers == 0 \
                  else \
                 [f'Interest {num_interest}', f'Not int {num_not_interesting}', f'Outliers {num_outliers}']
        
        axs[idx_legend].legend(handles=handles, labels=labels,
                               loc='lower center', bbox_to_anchor=(0.5, 1.0),
                               title=f"{v['name']}", title_fontsize='large')

    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    return fig


def plot_feat_tar(data: Dict, features: List[str], targets: List[str], only_interest: bool = True):
    n_col = len(features)
    n_rows = len(targets)
    fig, axs = plt.subplots(n_rows, n_col, sharey='row', sharex='col', figsize=(4 * n_col, 4 * n_rows))
    for n_row, tar, in enumerate(targets):
        for n_col, feat in enumerate(features):
            for v in data.values():
                axs[n_row, n_col].scatter(
                    x=v['interest'][feat],
                    y=v['interest'][tar],
                    c=v['color'], marker='*', alpha=0.2, label=v['name']
                )
                title = "Features vs Targets (interest points)"
                if not only_interest:
                    axs[n_row, n_col].scatter(
                        x=v['not_interesting'][feat],
                        y=v['not_interesting'][tar],
                        c=v['color'], marker='.', alpha=0.3
                    )
                    title = "Features vs Targets (all points)"
            axs[1, n_col].set_xlabel(feat)
        axs[n_row, 0].set_ylabel(tar)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(handles=handles,
               labels=labels,
               loc='upper center', ncol=len(labels),
               title=title, title_fontsize='large')
    fig.subplots_adjust(top=0.85)
    return fig

def plot_interest_points_per_iteration(data: Dict, initial_size, n_slice, tot_size):
    # TODO: Test this function
    warnings.warn("This function has not been tested yet. It may not work as expected.")
    n_rows = len(data)
    fig, axs = plt.subplots(n_rows, 1, sharex='col', figsize=(1, n_rows))

    all_counts = {}
    for name, exp in data.items():
        df = exp['df']
        c = df[df['quality'] == 'interest'].groupby('iteration').size()
        all_counts[name] = c.values()
    
    for n_row, (name, count) in enumerate(all_counts.items()):
        # Plot barplot for each count
        axs[n_row].bar(range(len(count)), count)
        axs[n_row].set_ylabel(name)
        axs[n_row].set_xlabel('Iteration')
        axs[n_row].set_ylim(0, 1.1 * max([max(c) for c in all_counts.values()]))
    fig.tight_layout()
    fig.legend()
    
    return fig