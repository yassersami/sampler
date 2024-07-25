import numpy as np
import seaborn as sns
import pandas as pd

from typing import List, Dict
from itertools import combinations

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec


def get_colors_legend(data):
    return [mlines.Line2D([], [], color=vals['color'], label=vals['name']) for vals in data.values()]


def plot_violin_distribution(data: Dict, targets: List[str], desired_region: Dict, area: Dict = None):
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
    
    # Add legend if area is provided
    if area is not None:
        handles = []
        labels = []
        for key, value in area.items():
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
        hue="identity", palette=[val['color'] for val in data.values()], diag_sharey=False
    )
    g.map_lower(sns.scatterplot, alpha=0.3)
    g.map_upper(sns.kdeplot, levels=4, linewidths=2)
    g.map_diag(sns.kdeplot, fill=False, linewidth=2)
    g.add_legend()
    return g.figure


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
    axs[1].legend(handles=colors_legend, loc='upper left', bbox_to_anchor=(1.1, 1))
    axs[0].text(-0.03, 1.03, 'a)', transform=axs[0].transAxes, size=20, weight='bold', ha='right', va='bottom')
    axs[1].text(-0.03, 1.03, 'b)', transform=axs[1].transAxes, size=20, weight='bold', ha='right', va='bottom')
    fig.tight_layout()
    return fig



def plot_2d(data: Dict, features: List[str], points: Dict, volume: Dict):
    colors = points["colors"]

    n_exp = len(data)
    n_rows = len(features)
    fig, axs = plt.subplots(
        n_rows, n_exp, sharey='row', figsize=(4*n_exp, 4*n_rows),
        constrained_layout=False
    )

    feature_pairs = list(combinations(features, 2))
    
    for n_col, (k, v) in enumerate(data.items()):
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
        labels = [f'n_Itr: {num_interest} \nV_Itr: {volume[k]:.3e}', f'n_noItr: {num_not_interesting}']
        if num_outliers != 0:
            labels += [f'Outliers {num_outliers}']
        
        axs[idx_legend].legend(handles=handles, labels=labels,
                               loc='lower center', bbox_to_anchor=(0.5, 1.0),
                               title=f'{v["name"]}', title_fontsize='large')

    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    return fig

def r2_bar_plot(data: Dict, targets: List[str], r2: Dict, title: str, all_targets=False):
    n_groups = len(data)
    fig, ax = plt.subplots()
    index = range(n_groups)
    bar_width = 0.2

    for i, target in enumerate(targets):
        scores = [r2[key][target] for key in data]
        bar_position = [j + i * bar_width for j in index]
        ax.bar(bar_position, scores, bar_width, label=target)

    bar_position = [j + len(targets) * bar_width for j in index]
    if all_targets:
        all_target_scores = [r2[key]['all_targets'] for key in data]
        ax.bar(bar_position, all_target_scores, bar_width, label='all_targets')
    
    ax.set_ylabel(title)

    ax.set_xticks([r + bar_width for r in range(n_groups)])
    ax.set_xticklabels(list([data[key]['name'] for key in data.keys()]), rotation=15)
    
    ax.legend(bbox_to_anchor=(1.05, 0.5), loc="center left", borderaxespad=0)
    plt.tight_layout()
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
                        x=v['not_interesting'][feat],
                        y=v['not_interesting'][tar],
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
    labels = [*data]