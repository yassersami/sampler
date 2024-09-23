from typing import List, Dict, Tuple, Union
import warnings
import numpy as np
import pandas as pd
from scipy.stats import lognorm
import matplotlib.pyplot as plt

from .asvd import ASVD
from .postprocessing_functions import set_scaled_kde

# x and y bounds: max values encoutred accross all experiments
MAX_STAR_VOL = 0.05
MAX_BIN_STAR_VOL = 0.05
MAX_SAMPLES = 150
STAR_VOL_BIN_WIDTH = 5.e-4


def adjust_bin_width(volumes, initial_bin_width, min_bins=5):
    """
    Adjusts the bin width for histogram creation to ensure a minimum number of
    bins over the volume range.

    This function takes an initial bin width and adjusts it if necessary to
    ensure that there are at least 'min_bins' bins over the volume range. If the
    initial bin width would result in fewer bins than 'min_bins', it increases
    the number of bins to 'min_bins * 2' and recalculates the bin width
    accordingly.
    """
    volumes_range = volumes.max() - volumes.min()
    initial_bins = int(volumes_range / initial_bin_width)
    if initial_bins < min_bins:
        exp_bins = min_bins * 2
        bin_width = volumes_range / min_bins
        warnings.warn(
            f"Too few bins ({initial_bins}) for initial bin width of {initial_bin_width:.6f}. "
            f"Increasing number of bins to {exp_bins} using a new bin width of {bin_width:.6f}."
        )
    else:
        exp_bins = initial_bins
        bin_width = initial_bin_width
    return bin_width, exp_bins


def fit_lognormal(data):
    """
    Fit a lognormal distribution to the given data.

    The location parameter is fixed at 0 (floc=0), which constrains the
    distribution to start at 0. This is often appropriate for data that cannot
    be negative, such as volumes.
    
    Note:
    - shape: Also known as the log-scale parameter (sigma)
    - location: Fixed at 0 in this case
    - scale: Related to the median of the distribution
    """
    shape, loc, scale = lognorm.fit(data, floc=0)
    return shape


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
    - exp_config (Dict[str, Dict[str, str]]): A dictionary for experiment name and color.
    - asvd_scores (Dict[str, Dict[str, float]]): A dictionary where keys are experiment
      keys and values are dictionaries of scores.
    - metrics_to_plot (List[str]): List of metric names to plot in the bar chart.
      If None, all metrics are plotted.
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
            row = [exp_name[:10] + '...' * (len(exp_name) > 10)]
            for metric in metrics_for_table:
                value = asvd_scores[exp_key].get(metric, 'N/A')
                if isinstance(value, int):
                    row.append(f'{value}')
                elif isinstance(value, float):
                    if value > 0.01:
                        row.append(f'{value:.2f}')
                    else:
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


def plot_stars_volumes_distribution(
    asvd_instance: ASVD, exp_name: str, bin_width: float =STAR_VOL_BIN_WIDTH ,
    figsize=(12, 6), font_size=14
):
    """
    Plot the distribution of stars_volumes_x using a histogram and KDE plot.

    Parameters:
    asvd_instance (ASVD): An instance of the ASVD class
    exp_name (str): Name of the experiment
    """

    # Extract stars_volumes_x
    volumes = asvd_instance.stars_volumes_x

    # PDF (KDE) and E[X] (Cum. Vol.) height
    normal_height = MAX_BIN_STAR_VOL * 0.9

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Compute number of bins for consistent bin width
    volumes_range = volumes.max() - volumes.min()
    bin_width, exp_bins = adjust_bin_width(volumes, bin_width)

    # Plot histogram with sum of volumes
    bin_volumes_sum, bin_edges, _ = ax.hist(volumes, bins=exp_bins, weights=volumes, alpha=0.7, color='steelblue', edgecolor='white', label='Volume')

    # Scale a KDE to match histogram height
    x_values, kde_scaled = set_scaled_kde(volumes, height=normal_height, bandwidth=0.1)

    # Plot scaled KDE
    ax.plot(x_values, kde_scaled, color='dodgerblue', linewidth=2, label='Density')

    # Calculate Q3 and add vertical line
    q3 = np.percentile(volumes, 75)
    ax.axvline(q3, color='goldenrod', linewidth=2, linestyle='--', label='Q3')

    # Calculate cumulative volume
    sorted_volumes = np.sort(volumes)
    cumulative_volumes = np.cumsum(sorted_volumes)

    # Scale cumulative volume to match histogram height
    scaling_factor = normal_height / cumulative_volumes[-1]
    scaled_cum_vol = cumulative_volumes * scaling_factor

    # Plot cumulative volume
    ax.plot(sorted_volumes, scaled_cum_vol, color='green', label='Cum. Vol.')

    # Set title
    ax.set_title(f"{exp_name}\nDistribution of Star Volumes", fontsize=font_size)

    # Set x and y-axis limits
    ax.set_xlim(0, MAX_STAR_VOL)
    ax.set_ylim(0, MAX_BIN_STAR_VOL)
    
    # Set tick label font sizes
    ax.tick_params(axis='both', which='major', labelsize=font_size-2)

    # Stats for volumes from 0 to Q3
    volumes_to_q3 = volumes[volumes <= q3]
    cumulated_volume_to_q3 = np.sum(volumes_to_q3)
    mean_volume_to_q3 = np.mean(volumes_to_q3)

    # Get log normal distribution parameters
    lognormal_sigma = fit_lognormal(volumes)

    # Add summary statistics
    stats = (
        f'Bin width: {bin_width:.4f}\n'
        f'Lognormal σ: {lognormal_sigma:.4f}\n'
        f'Std Dev: {np.std(volumes):.4f}\n'
        f'Mean: {np.mean(volumes):.4f}\n'
        f'Sum: {np.sum(volumes):.4f}\n'
        f'Q3 limit: {q3:.4f}\n'
        f'Early Mean: {mean_volume_to_q3:.4f}\n'
        f'Early Sum: {cumulated_volume_to_q3:.4f}'
    )
    ax.text(0.99, 0.98, stats, transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=font_size-2)

    # Move legend to bottom right and adjust position
    ax.legend(loc='upper right', bbox_to_anchor=(1, 0.6), fontsize=font_size)

    # Improve layout
    plt.tight_layout()

    return fig


def plot_multiple_asvd_distributions(
    experiments_asvd: Dict[str, ASVD],
    exp_config: Dict[str, Dict[str, str]],
    bin_width=STAR_VOL_BIN_WIDTH, figsize=(15, 8), font_size=14
):
    """
    Plot the distribution of stars_volumes_x for multiple ASVD instances.

    Parameters:
    experiments_asvd (dict): A dictionary where keys are experiment names and values are ASVD instances
    exp_config (dict): A dictionary with experiment configurations
    bins (int): Number of bins for the histograms (default: 30)
    figsize (tuple): Figure size (default: (15, 8))
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Check and adjust bin_width for all experiments if necessary
    for exp_key, asvd_instance in experiments_asvd.items():
        volumes = asvd_instance.stars_volumes_x
        bin_width, exp_bins = adjust_bin_width(volumes, bin_width)

    max_count = 0
    for exp_key, asvd_instance in experiments_asvd.items():
        volumes = asvd_instance.stars_volumes_x

        # Compute number of bins for consistent bin width
        volumes_range = volumes.max() - volumes.min()
        exp_bins = int(volumes_range / bin_width)

        # Compute count per bins
        count, bin_edges = np.histogram(volumes, bins=exp_bins, density=False)
        
        # Scale a KDE to match histogram height
        x_values, kde_counts = set_scaled_kde(volumes, height=np.max(count), bandwidth=0.1)

        # Plot KDE
        ax.fill_between(x_values, kde_counts, 0, alpha=0.3, color=exp_config[exp_key]['color'])
        ax.plot(x_values, kde_counts, label=exp_config[exp_key]['name'], color=exp_config[exp_key]['color'], linewidth=2)

        # Update max density for y-axis limit
        max_count = max(max_count, np.max(count))

    # Set labels and title
    ax.set_xlabel('Star Volume', fontsize=font_size)
    ax.set_xbound(0, MAX_STAR_VOL)
    ax.set_ylabel(f'Count per {bin_width} range of star volume', fontsize=font_size)
    ax.set_ylim(0, MAX_SAMPLES*1.05)
    ax.set_title('Distribution of Star Volumes for Multiple Experiments', fontsize=font_size+2)

    # Move legend to bottom right and adjust position
    ax.legend(loc='upper right', bbox_to_anchor=(1, 0.4), fontsize=font_size)

    # Compute ASVD scores
    asvd_scores = {}
    for exp_key, exp_asvd in experiments_asvd.items():
        volumes_x = exp_asvd.stars_volumes_x
        volumes_xy = exp_asvd.stars_volumes_xy

        # Total volume
        sum_volumes_x = volumes_x.sum()
        sum_volumes_xy = volumes_xy.sum()

        # Stats for volumes from 0 to Q3
        q3_x = np.percentile(volumes_x, 75)
        volumes_to_q3_x = volumes_x[volumes_x <= q3_x]
        sum_to_q3_x = np.sum(volumes_to_q3_x)

        # Get log normal distribution parameters
        lognormal_sigma_x = fit_lognormal(volumes_x)
        
        asvd_scores[exp_key] = {
            'count': volumes_x.shape[0],
            'sum_x': sum_volumes_x,
            'sum_xy': sum_volumes_xy,
            'Augmentat°': 1 if sum_volumes_x==0 else sum_volumes_xy / sum_volumes_x,
            'Q3': q3_x,
            'Early sum': sum_to_q3_x,
            'Std Dev': np.std(volumes_x),
            'Lognormal σ': lognormal_sigma_x,
        }

    exp_keys = list(experiments_asvd.keys())
    exp_names = [exp_config[exp_key]['name'] for exp_key in exp_keys]
    score_names = list(asvd_scores[exp_keys[0]].keys())
        
    # Prepare table data
    columns = ['Experiment'] + [exp_name[:10] + '...' * (len(exp_name) > 10) for exp_name in exp_names]
    table_data = []
    for metric in score_names:
        row = [metric]
        for exp_key in exp_keys:
            value = asvd_scores[exp_key].get(metric, 'N/A')
            if isinstance(value, int):
                row.append(f'{value}')
            elif isinstance(value, float):
                if value > 0.01:
                    row.append(f'{value:.2f}')
                else:
                    row.append(f'{value:.2e}')
            else:
                row.append(str(value))
        table_data.append(row)

    # Add table to the top right of the plot
    table = ax.table(
        cellText=table_data, 
        colLabels=columns,
        cellLoc='center', loc='upper right',
        bbox=[0.5, 0.5, 0.5, 0.5]  # (x0, y0, width, height)
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, 1.5)  # Adjust the scale to fit your needs

    # Set tick label font sizes
    ax.tick_params(axis='both', which='major', labelsize=font_size)

    # Improve layout
    plt.tight_layout()

    return fig