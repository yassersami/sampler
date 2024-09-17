from typing import List, Dict, Tuple, Union
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

from .asvd import ASVD

# x and y bounds: max values encoutred accross all experiments
MAX_STAR_VOLUME = 0.1 
MAX_BIN_VOLUME = 0.2
MAX_BIN_COUNT = 150


def plot_asvd_scores(
    experiments_asvd: Dict[str, ASVD],
    exp_config: Dict[str, Dict[str, str]],
    figsize: Tuple[int, int] = (15, 10),
    font_size: int = 14,
):
    """
    Plot scores for different experiments using a grouped bar plot and display remaining
    metrics in a table.

    Variables:
    - data and experiments_asvd have same keys.
    - experiments_asvd (Dict[str, ASVD]): A dictionary of ASVD instance for each
      experiment.
    - asvd_scores (Dict[str, Dict[str, float]]): A dictionary where keys are experiment
      keys and values are dictionaries of scores.
    - metrics_to_plot (List[str]): List of metric names to plot in the bar chart.
      If None, all metrics are plotted.
    - figsize (Tuple[int, int]): Size of the figure (width, height) in inches.
    """
    
    metrics_to_plot = ['sum_augm', 'rsd_x', 'rsd_xy', 'rsd_augm', 'riqr_x', 'riqr_xy']

    asvd_scores = {}
    for exp_key, exp_asvd in experiments_asvd.items():
        asvd_scores[exp_key] = exp_asvd.compute_scores()

    exp_keys = list(experiments_asvd.keys())
    exp_names = [exp_config[exp_key]['name'] for exp_key in exp_keys]

    # Get all unique metrics while preserving order
    all_metrics = []
    for exp in asvd_scores.values():
        all_metrics.extend(metric for metric in exp.keys() if metric not in all_metrics)

    if metrics_to_plot is None:
        metrics_to_plot = all_metrics.copy()
    else:
        metrics_to_plot = [metric for metric in metrics_to_plot if metric in all_metrics]

    metrics_for_table = [metric for metric in all_metrics if metric not in metrics_to_plot]

    n_experiments = len(exp_keys)
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

    for i, (exp_key, exp_name) in enumerate(zip(exp_keys, exp_names)):
        values = [asvd_scores[exp_key].get(metric, np.nan) for metric in metrics_to_plot]
        position = index + i * bar_width
        rects = ax_bar.bar(
            position, values, bar_width, label=exp_name, alpha=0.8,
            color=exp_config[exp_key]['color']
        )

        # Add value labels on top of each bar
        for rect in rects:
            height = rect.get_height()
            if np.isfinite(height):
                ax_bar.text(rect.get_x() + rect.get_width()/2., height,
                            f'{height:.2f}', ha='center', va='bottom', rotation=90, fontsize=font_size-2)

    ax_bar.set_ylabel('Values', fontsize=font_size)
    ax_bar.set_title('Comparison of Metrics Across Experiments', fontsize=font_size+2)
    ax_bar.set_xticks(index + bar_width * (n_experiments - 1) / 2)
    ax_bar.set_xticklabels(metrics_to_plot, rotation=45, ha='right', fontsize=font_size)
    ax_bar.legend(fontsize=font_size)

    # Set tick label font sizes
    ax_bar.tick_params(axis='both', which='major', labelsize=font_size)

    # Table (if there are metrics for the table)
    if metrics_for_table:
        table_data = []
        for exp_key, exp_name in zip(exp_keys, exp_names):
            row = [exp_name]
            for metric in metrics_for_table:
                value = asvd_scores[exp_key].get(metric, 'N/A')
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
        table.set_fontsize(font_size)
        table.scale(1, 1.5)  # Adjust the scale to fit your needs

    plt.tight_layout()
    return fig


def plot_stars_volumes_distribution(asvd_instance: ASVD, exp_name: str, bins=30, figsize=(12, 6), font_size=14):
    """
    Plot the distribution of stars_volumes_xy using a histogram and KDE plot.

    Parameters:
    asvd_instance (ASVD): An instance of the ASVD class
    exp_name (str): Name of the experiment
    bins (int): Number of bins for the histogram (default: 30)

    Returns:
    fig: The created figure object
    """

    # Extract stars_volumes_xy
    volumes = asvd_instance.stars_volumes_xy

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot histogram with sum of volumes
    bin_volumes_sum, bin_edges, _ = ax.hist(volumes, bins=bins, weights=volumes, alpha=0.7, color='steelblue', edgecolor='white')

    # Calculate bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Compute KDE
    kde = gaussian_kde(volumes)
    kde_counts = kde(bin_centers)  # = counts but smoother

    # Scale KDE to match histogram height
    scaling_factor = np.max(bin_volumes_sum) / np.max(kde_counts)
    scaled_kde = kde_counts * scaling_factor

    # Plot scaled KDE
    ax.plot(bin_centers, scaled_kde, color='dodgerblue', linewidth=2, label='KDE')

    # Set labels and title
    ax.set_xlabel('Star Volume', fontsize=font_size)
    ax.set_xbound(0, MAX_STAR_VOLUME)
    ax.set_ylabel('Sum of Volumes', fontsize=font_size)
    ax.set_ybound(0, MAX_BIN_VOLUME)
    ax.set_title(f"{exp_name}\nDistribution of Augmented Star Volumes", fontsize=font_size)

    # Add summary statistics
    stats = (
        f'Sum: {np.sum(volumes):.4f}\n'
        f'Mean: {np.mean(volumes):.4f}\n'
        f'Median: {np.median(volumes):.4f}\n'
        f'Q3: {np.percentile(volumes, 75):.4f}\n'
        f'Std Dev: {np.std(volumes):.4f}'
    )
    ax.text(0.95, 0.95, stats, transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=font_size)

    # Set tick label font sizes
    ax.tick_params(axis='both', which='major', labelsize=font_size)

    # Improve layout
    plt.tight_layout()

    return fig

def plot_multiple_asvd_distributions(
    experiments_asvd: Dict[str, ASVD],
    exp_config: Dict[str, Dict[str, str]],
    bins=30, figsize=(15, 8), font_size=14
):
    """
    Plot the distribution of stars_volumes_xy for multiple ASVD instances.

    Parameters:
    experiments_asvd (dict): A dictionary where keys are experiment names and values are ASVD instances
    exp_config (dict): A dictionary with experiment configurations
    bins (int): Number of bins for the histograms (default: 30)
    figsize (tuple): Figure size (default: (15, 8))

    Returns:
    fig: The created figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    max_count = 0
    for exp_key, asvd_instance in experiments_asvd.items():
        volumes = asvd_instance.stars_volumes_xy

        # Compute histogram
        count, bin_edges = np.histogram(volumes, bins=bins, density=False)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Compute KDE for counts
        kde = gaussian_kde(volumes)
        kde_counts = kde(bin_centers)

        # Plot KDE
        ax.fill_between(bin_centers, kde_counts, 0, alpha=0.3, color=exp_config[exp_key]['color'])
        ax.plot(bin_centers, kde_counts, label=exp_config[exp_key]['name'], color=exp_config[exp_key]['color'], linewidth=2)

        # Update max density for y-axis limit
        max_count = max(max_count, np.max(count))

    # Set labels and title
    ax.set_xlabel('Star Volume', fontsize=font_size)
    ax.set_xbound(0, MAX_STAR_VOLUME)
    ax.set_ylabel('Count', fontsize=font_size)
    ax.set_ylim(0, MAX_BIN_COUNT)  # max_count noticed across all experiments
    ax.set_title('Distribution of Augmented Star Volumes for Multiple Experiments', fontsize=font_size+2)

    # Move legend to bottom right and adjust position
    legend = ax.legend(loc='upper right', bbox_to_anchor=(1, 0.4), fontsize=font_size)

    # Compute ASVD scores
    asvd_scores = {}
    for exp_key, exp_asvd in experiments_asvd.items():
        volumes_x = exp_asvd.stars_volumes_x
        volumes_xy = exp_asvd.stars_volumes_xy
        asvd_scores[exp_key] = {
            'count': volumes_x.shape[0],
            'sum_x': np.sum(volumes_x),
            'sum_xy': np.sum(volumes_xy),
            'median_x': np.median(volumes_x),
            'median_xy': np.median(volumes_xy),
            'Q3_x': np.percentile(volumes_x, 75),
            'Q3_xy': np.percentile(volumes_xy, 75),
            'std_x': np.std(volumes_x),
            'std_xy': np.std(volumes_xy), 
        }

    exp_keys = list(experiments_asvd.keys())
    exp_names = [exp_config[exp_key]['name'] for exp_key in exp_keys]
    score_names = list(asvd_scores[exp_keys[0]].keys())
        
    # Prepare table data
    table_data = [['Experiment'] + [exp_name[:10] + '...' * (len(exp_name) > 10) for exp_name in exp_names]]
    for metric in score_names:
        row = [metric]
        for exp_key in exp_keys:
            value = asvd_scores[exp_key].get(metric, 'N/A')
            if isinstance(value, int):
                row.append(f'{value}')
            elif isinstance(value, float):
                row.append(f'{value:.2e}')
            else:
                row.append(str(value))
        table_data.append(row)

    # Add table to the top right of the plot
    table = ax.table(cellText=table_data, 
                     colLabels=[''] + score_names,
                     cellLoc='center', loc='upper right',
                     bbox=[0.5, 0.5, 0.5, 0.5])  # Adjust these values as needed
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, 1.5)  # Adjust the scale to fit your needs

    # Set tick label font sizes
    ax.tick_params(axis='both', which='major', labelsize=font_size)

    # Improve layout
    plt.tight_layout()

    return fig