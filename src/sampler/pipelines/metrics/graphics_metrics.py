from typing import List, Dict, Tuple, Union
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

from .postprocessing_functions import set_scaled_kde
from .asvd import ASVD, get_cum_vol
from sampler.core.data_processing.sampling_tracker import get_first_iteration_index


def plot_initial_data(
    data: Dict[str, Dict],
    features: List[str],
    targets: List[str],
    latex_mapper: Dict[str, str],
    plot_ranges: Dict[str, Tuple[float, float]],
    only_first_exp: bool = False
):

    if only_first_exp:
        # Keep only the first experiment
        first_exp = next(iter(data.keys()))
        data = {first_exp: data[first_exp]}

    # Get first iteration index of each experiment
    first_iter_index = {
        exp_key: get_first_iteration_index(exp_dic['df'])
        for exp_key, exp_dic in data.items()
    }

    # Get initial data for each experiment
    initial_data = {
        exp_dic['name']: exp_dic['df'].loc[:first_iter_index[exp_key]-1]
        for exp_key, exp_dic in data.items()
    }

    # Concatenate all initial dataframes
    df_to_plot = pd.concat([df.assign(exp_name=exp_name) for exp_name, df in initial_data.items()])

    # Count samples and outliers for each label
    sample_counts = df_to_plot['exp_name'].value_counts()
    outlier_counts = df_to_plot[df_to_plot[targets].isna().any(axis=1)]['exp_name'].value_counts()

    # Create color palette
    original_palette = {exp_dic['name']: exp_dic['color'] for exp_dic in data.values()}

    # Create a mapping from original labels to labels with counts
    label_mapping = {
        label: f'{label}\nTotal: {count}\nOutliers: {outlier_counts.get(label, 0)}' 
        for label, count in sample_counts.items()
    }

    # Update the 'exp_name' column in df_to_plot
    df_to_plot['exp_name'] = df_to_plot['exp_name'].map(label_mapping)

    # Create an updated palette that matches the new labels
    palette = {label_mapping[k]: v for k, v in original_palette.items()}

    # Create PairGrid
    g = sns.PairGrid(
        df_to_plot,
        vars=features + targets,
        hue='exp_name', palette=palette, diag_sharey=False
    )

    # Define custom scatter plot function to highlight NaN values
    def scatter_with_nan_highlight(x, y, **kwargs):
        ax = plt.gca()
        
        # Regular scatter plot
        sns.scatterplot(x=x, y=y, **kwargs)
        
        # Highlight NaN values with black crosses
        nan_mask = df_to_plot[targets].isna().any(axis=1)
        if nan_mask.any():
            ax.scatter(x[nan_mask], y[nan_mask], color='black', marker='x', s=30, zorder=1, linewidth=1, label='NaN')

    g.map_lower(scatter_with_nan_highlight, alpha=0.3)
    
    # For diagonal, use a separate function
    def diagonal_kdeplot(x, **kwargs):
        try:
            sns.kdeplot(x=x, **kwargs)
        except (ValueError, IndexError):
            # If KDE plot fails, fall back to a histogram
            sns.histplot(x=x, **kwargs)

    g.map_diag(diagonal_kdeplot, linewidth=2)

    # Define custom KDE plot function with error handling
    def safe_kdeplot(x, y, **kwargs):
        try:
            sns.kdeplot(x=x, y=y, **kwargs)
        except (ValueError, IndexError):
            # If KDE plot fails, fall back to a scatter plot
            sns.scatterplot(x=x, y=y, **kwargs)

    g.map_upper(safe_kdeplot, levels=4, linewidth=2)

    # Set axis limits and update labels
    for ax in g.axes.flatten():
        var_x, var_y = ax.get_xlabel(), ax.get_ylabel()
        if var_x != '':
            ax.set_xlabel(latex_mapper[var_x])
            ax.set_xlim(plot_ranges[var_x])
        if var_y != '':
            ax.set_ylabel(latex_mapper[var_y])
            ax.set_ylim(plot_ranges[var_y])

    # Add legend without title
    g.add_legend(title='')

    return g.figure


def plot_feature_pairs(
    data: Dict[str, Dict[str, Union[str, pd.DataFrame]]],
    feature_pair: Tuple[str, str],
    latex_mapper: Dict[str, str],
    plot_ranges: Dict[str, Tuple[float, float]],
    only_new: bool = False,
    font_size: int = 14
):
    n_exp = len(data)

    # Use the specified feature pair
    x, y = feature_pair

    # Adjust figure size
    fig_width = 4 * n_exp + 1
    fig_height = 6  # Fixed height since we're only plotting one pair
    fig, axs = plt.subplots(
        1, n_exp, sharey='row', figsize=(fig_width, fig_height),
        constrained_layout=False, squeeze=False
    )

    for n_col, exp_dic in enumerate(data.values()):
        exp_dic_copy = exp_dic.copy()  # Shallow copy for non-new data
        if only_new:
            first_iter_index = get_first_iteration_index(exp_dic_copy['df'])  # samples have consistent indices over sub-dfs
            exp_dic_copy.update({  # Update dataframes
                df_name: df.loc[first_iter_index:] for df_name, df in exp_dic.items()
                if isinstance(df, pd.DataFrame)
            })
        num_no_interest = exp_dic_copy['no_interest'].shape[0]
        num_interest = exp_dic_copy['interest'].shape[0]
        num_outliers = exp_dic_copy['outliers'].shape[0]
        interest_colors = plt.cm.autumn_r(np.linspace(1, 0, num_interest))

        ax = axs[0, n_col]
        ax.scatter(
            x=exp_dic_copy['no_interest'][x],
            y=exp_dic_copy['no_interest'][y],
            c='gray', alpha=0.3, label='No interest'
        )
        ax.scatter(
            x=exp_dic_copy['outliers'][x],
            y=exp_dic_copy['outliers'][y],
            c='black', alpha=0.7, marker='x', label='Outlier'
        )
        ax.scatter(
            x=exp_dic_copy['interest'][x],
            y=exp_dic_copy['interest'][y],
            c=interest_colors, alpha=0.5, label='Interest'
        )

        # Create legend for each column
        handles, _ = ax.get_legend_handles_labels()

        # Create a handle for the 'interest' marker
        interest_marker = plt.Line2D(
            [0], [0], marker='o', color='w', markerfacecolor='red',
            alpha=0.7, label=''
        )

        # Initialize handles and labels for the legend
        legend_handles = [interest_marker, handles[0]]
        legend_labels = [
            f'Interest: {num_interest}',
            f'No Interest: {num_no_interest}',
        ]

        # Add outliers to the legend if present
        if num_outliers != 0:
            legend_handles.append(handles[1])
            legend_labels.append(f'Outlier: {num_outliers}')

        # Create the legend on the specified axis
        ax.legend(
            handles=legend_handles, labels=legend_labels,
            loc='lower center', bbox_to_anchor=(0.5, 1.05),
            title=exp_dic['name'], title_fontsize=font_size
        )

        # Set axis labels
        ax.set_xlabel(latex_mapper[x], fontsize=font_size)
        ax.set_xlim(plot_ranges[x])
        if n_col == 0:
            ax.set_ylabel(latex_mapper[y], fontsize=font_size)
            ax.set_ylim(plot_ranges[y])

        # Set tick label font sizes
        ax.tick_params(axis='both', which='major', labelsize=font_size-2)

    # Add a vertical color bar with custom ticks outside the subplots
    cbar_width = 0.2 / fig_width
    pos = ax.get_position()
    cbar_ax = fig.add_axes([0.88, pos.y0, cbar_width, 0.6])  # [left, bottom, width, height]
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='autumn_r'), cax=cbar_ax, orientation='vertical')
    cbar.set_ticks([0, 1])  # Set ticks at the start and end
    cbar.set_ticklabels(['first', 'last'])  # Label the ticks
    cbar.set_label('Interest order', fontsize=font_size)
    cbar.ax.tick_params(labelsize=font_size-2)

    fig.tight_layout(rect=[0, 0, 0.85, 0.95])  # Adjust subplots layout, not global figure

    return fig


def plot_violin_distribution(
    data: Dict,
    targets: List[str],
    latex_mapper: Dict[str, str],
    interest_region: Dict[str, Tuple[float, float]],
    plot_ranges: Dict[str, Tuple[float, float]],
    font_size: int = 14
):
    fig, axs = plt.subplots(1, len(targets), figsize=(5 * len(targets), 5))

    # Ensure axs is always a list, even for a single subplot
    axs = [axs] if len(targets) == 1 else axs

    # Get first iteration index of each experiment
    first_iter_index = {
        exp_key: get_first_iteration_index(exp_dic['df'])
        for exp_key, exp_dic in data.items()
    }

    for col, target in enumerate(targets):
        sns.violinplot(
            data=[
                exp_dic['inliers'].loc[first_iter_index[exp_key]:, target]
                for exp_key, exp_dic in data.items()
            ],
            palette=[val['color'] for val in data.values()],
            cut=0,  # limit to data range
            bw=0.1,
            ax=axs[col],
        )
        axs[col].set_xticklabels([val['name'] for val in data.values()], 
                                 rotation=20, ha='right', fontsize=font_size-2)
        axs[col].add_patch(Rectangle(
            (-0.45, interest_region[target][0]),  # x, y
            (len(data) - 0.1),  # width
            interest_region[target][1] - interest_region[target][0],  # height
            edgecolor='#B73E3E', facecolor='none', lw=2, linestyle='--',
        ))
        axs[col].set_ylabel(latex_mapper[target], fontsize=font_size)
        axs[col].set_ylim(plot_ranges[target])
        
        # Set tick label font sizes
        axs[col].tick_params(axis='both', which='major', labelsize=font_size-2)


    # Add labels only if there are multiple subplots
    if len(targets) > 1:
        axs[0].text(-0.03, 1.03, 'a)', transform=axs[0].transAxes, 
                    ha='right', va='bottom', size=font_size+6)
        axs[1].text(-0.03, 1.03, 'b)', transform=axs[1].transAxes, 
                    ha='right', va='bottom', size=font_size+6)

    fig.tight_layout()

    return fig


def targets_kde(
    data: Dict,
    asvd: Dict[str, ASVD],
    targets: List[str],
    latex_mapper: Dict[str, str],
    interest_region: Dict[str, Tuple[float, float]],
    plot_ranges: Dict[str, Tuple[float, float]],
    bins: int = 20,
    bandwidth: float = 0.1,
    font_size: int = 14,
):
    n_col = len(targets)
    fig, axs = plt.subplots(1, n_col, figsize=(6*n_col, 6), squeeze=False)
    axs = axs.flatten()  # This ensures axs is always a 1D array

    max_count = 0
    for col, target in enumerate(targets):
        ax = axs[col]
        for exp_key, exp_data in data.items():
            first_iter_index = get_first_iteration_index(exp_data['df'])
            values = exp_data['inliers'].loc[first_iter_index:, target]

            # Count samples per bin
            count, bin_edges = np.histogram(values, bins=bins, density=False)
            
            # Scale a KDE to match histogram max count height
            x_values, kde_counts = set_scaled_kde(values, height=np.max(count), bandwidth=bandwidth)

            # Set label
            num_interest = exp_data['interest'].loc[first_iter_index:].shape[0]
            q3, cumsum_q3 = get_cum_vol(asvd[exp_key].stars_volumes_x, 75)
            augmentation = asvd[exp_key].get_augmentation(use_star=True)
            exp_stats = (
                f"Interest: {num_interest}\n"
                f"Eff. Vol.: {cumsum_q3:.4f}\n"
                f"Augmentat°: {augmentation:.2f}\n"
            )
            # Plot KDE
            ax.fill_between(x_values, kde_counts, 0, alpha=0.3, color=exp_data['color'])
            ax.plot(x_values, kde_counts, label=exp_data['name'], color=exp_data['color'], linewidth=2)
            ax.plot([], [], ' ', label=exp_stats)

            # Update max count for y-axis limit
            max_count = max(max_count, np.max(count))

        ax.axvline(x=interest_region[target][0], color='red', linestyle='--')
        ax.axvline(x=interest_region[target][1], color='red', linestyle='--')
        ax.set_xlabel(latex_mapper[target], fontsize=font_size)
        ax.set_xlim(plot_ranges[target])
        
        # Set tick label font sizes
        ax.tick_params(axis='both', which='major', labelsize=font_size-2)

    if len(targets) > 1:
        axs[1].set_ylabel('')
        axs[1].set_yticklabels([])
        axs[0].text(-0.03, 1.03, 'a)', transform=axs[0].transAxes, 
                    ha='right', va='bottom', size=font_size+6)
        axs[1].text(-0.03, 1.03, 'b)', transform=axs[1].transAxes, 
                    ha='right', va='bottom', size=font_size+6)

    axs[0].set_ylabel(f'Count per {100/bins:.0f}% of target range', fontsize=font_size)
    legend = axs[-1].legend(loc='upper left', bbox_to_anchor=(1.1, 1))
    legend.set_title('Experiments\n', prop={'size': font_size})
    for text in legend.get_texts():
        text.set_fontsize(font_size-2)

    # Set y-axis limit
    for ax in axs:
        ax.set_ylim(0, max_count * 1.1)

    fig.tight_layout()

    return fig

def plot_feat_tar(
    exp_dic: Dict[str, Union[str, pd.DataFrame]],
    features: List[str],
    targets: List[str],
    latex_mapper: Dict[str, str],
    plot_ranges: Dict[str, Tuple[float, float]],
    font_size: int = 14
):
    n_col = len(features)
    n_rows = len(targets)
    fig, axs = plt.subplots(n_rows, n_col, figsize=(4 * n_col, 4 * n_rows), sharex='col', sharey='row', squeeze=False)
    
    # Get initial data for each experiment
    first_index = get_first_iteration_index(exp_dic['df'])
    initial_data = exp_dic['df'].loc[:first_index-1]
    interest_data = exp_dic['interest'].loc[first_index:]
    no_interest_data = exp_dic['no_interest'].loc[first_index:]

    for n_row, target in enumerate(targets):
        for n_col, feature in enumerate(features):
            ax = axs[n_row, n_col]
            ax.scatter(
                x=initial_data[feature],
                y=initial_data[target],
                c='grey', marker='.', s=20, alpha=0.3, label='Initial'
            )
            ax.scatter(
                x=no_interest_data[feature],
                y=no_interest_data[target],
                c='steelblue', marker='.', s=30, alpha=1., label='No interest'
            )
            ax.scatter(
                x=interest_data[feature],
                y=interest_data[target],
                c='red', marker='v', s=10, alpha=.7, label='Interest'
            )
            if n_row == n_rows - 1:  # Only set xlabel for the bottom row
                ax.set_xlabel(latex_mapper[feature], fontsize=font_size)
                ax.set_xlim(plot_ranges[feature])
            if n_col == 0:  # Only set ylabel for the leftmost column
                ax.set_ylabel(latex_mapper[target], fontsize=font_size)
                ax.set_ylim(plot_ranges[target])
            
            # Set tick label font sizes
            ax.tick_params(axis='both', which='major', labelsize=font_size-2)

    # Get legend handles and labels from the first subplot
    handles, labels = axs[0, 0].get_legend_handles_labels()

    fig.tight_layout()
    fig.legend(handles=handles,
               labels=labels,
               loc='upper center', ncol=len(labels),
               title=exp_dic['name'],
               title_fontsize=font_size,
               fontsize=font_size-2)
    fig.subplots_adjust(top=0.85)
    
    return fig
