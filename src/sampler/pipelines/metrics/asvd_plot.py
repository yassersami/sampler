from typing import List, Dict, Tuple, Union
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .asvd import ASVD
from .postprocessing_functions import set_scaled_kde

cm = 1/2.54  # centimeters in inches
FONT_SIZE = 10

# x and y bounds: max values encoutred accross all experiments
STAR_VOL_BIN_WIDTH = 1.e-3
MAX_STAR_VOL = 0.04
MAX_BIN_STAR_VOL = 0.08
MAX_SAMPLES = 150


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
    initial_bins = int(round(volumes_range / initial_bin_width))
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


def format_value(value):
    """ Format a value for display in plots. """
    if isinstance(value, int):
        return f'{value}'
    elif isinstance(value, float):
        if abs(value) >= 0.01 or value == 0:
            return f'{value:.2f}'
        else:
            return f'{value:.0e}'
    else:
        return str(value)


def plot_stars_volumes_distribution(
    asvd_instance: ASVD, exp_name: str, bin_width: float =STAR_VOL_BIN_WIDTH,
    fig_size=(12*cm, 10*cm), font_size=FONT_SIZE
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

    # Create figure with gridspec
    fig = plt.figure(figsize=fig_size)
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])  # 1 row, 2 columns

    # Plot area
    ax = fig.add_subplot(gs[0])

    # Compute number of bins for consistent bin width
    volumes_range = volumes.max() - volumes.min()
    exp_bins = max(1, int(round(volumes_range / bin_width)))

    # Plot histogram with sum of volumes
    bin_volumes_sum, bin_edges, _ = ax.hist(volumes, bins=exp_bins, weights=volumes, alpha=0.7, color='steelblue', edgecolor='white', label='Volume')

    # Scale a KDE to match histogram height
    x_values, kde_scaled = set_scaled_kde(volumes, height=normal_height, bandwidth=0.1)

    # Plot scaled KDE
    ax.plot(x_values, kde_scaled, color='dodgerblue', linewidth=1, label='Density')

    # Calculate cumulative volume
    sorted_volumes = np.sort(volumes)
    cumulative_volumes = np.cumsum(sorted_volumes)

    # Scale cumulative volume to match histogram height
    scaling_factor = normal_height / cumulative_volumes[-1]
    scaled_cum_vol = cumulative_volumes * scaling_factor

    # Plot cumulative volume
    ax.plot(sorted_volumes, scaled_cum_vol, color='green', label='Cum. Vol.')

    # Get star volume scores
    star_scores = asvd_instance.get_scores(use_star=True)

    # Mark Q3 with vertical line
    ax.axvline(star_scores['3rd Quartile'], color='goldenrod', linewidth=1, linestyle='--', label='Q3')

    # Mark D9 with vertical line
    ax.axvline(star_scores['9th Decile'], color='red', linewidth=1, linestyle='--', label='D9')

    # Set title and labels
    ax.set_title(f"{exp_name}\nDistribution of Star Volumes", fontsize=font_size)
    ax.set_xlabel(f'Star Volume', fontsize=font_size)
    ax.set_ylabel(f'Total Volume per Bin', fontsize=font_size)

    # Set x and y-axis limits
    ax.set_xlim(0, MAX_STAR_VOL)
    ax.set_ylim(0, MAX_BIN_STAR_VOL)
    
    # Set tick label font sizes
    ax.tick_params(axis='both', which='major', labelsize=font_size)

    # Table area
    ax_table = fig.add_subplot(gs[1])
    ax_table.axis('off')  # Hide axes

    # Add summary statistics
    stats_dict = {'Bin width': bin_width, **star_scores}
    table_data = [[key, format_value(value)] for key, value in stats_dict.items()]

    table = ax_table.table(
        cellText=table_data,
        cellLoc='center',
        loc='lower center'
    )
    # Make first column cells left aligned
    for (row, col), cell in table.get_celld().items():
        if col == 0:  # First column
            cell.set_text_props(ha='left')
    table.auto_set_font_size(False)
    table.set_fontsize(font_size-2)
    table.auto_set_column_width(list(range(len(table_data[0]))))
    
    # Add plot legend on top center of table subplot
    handles, labels = ax.get_legend_handles_labels()
    ax_table.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1), fontsize=font_size)

    # Improve layout
    plt.tight_layout()

    return fig


def plot_multiple_asvd_distributions(
    experiments_asvd: Dict[str, ASVD],
    exp_config: Dict[str, Dict[str, str]],
    bin_width=STAR_VOL_BIN_WIDTH, fig_size=(12*cm, 12*cm), font_size=FONT_SIZE
):
    """
    Plot the distribution of stars_volumes_x for multiple ASVD instances.

    Parameters:
    experiments_asvd (dict): A dictionary where keys are experiment names and values are ASVD instances
    exp_config (dict): A dictionary with experiment configurations
    bins (int): Number of bins for the histograms (default: 30)
    fig_size (tuple): Figure size (default: (15, 8))
    """
    # Create figure with gridspec
    fig = plt.figure(figsize=fig_size)
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])  # 2 rows, 1 column

    # Plot area
    ax = fig.add_subplot(gs[0])

    max_count = 0
    for exp_key, asvd_instance in experiments_asvd.items():
        volumes = asvd_instance.stars_volumes_x

        # Compute number of bins for consistent bin width
        volumes_range = volumes.max() - volumes.min()
        exp_bins = max(1, int(round(volumes_range / bin_width)))

        # Compute count per bins
        count, bin_edges = np.histogram(volumes, bins=exp_bins, density=False)

        # Scale a KDE to match histogram height
        x_values, kde_counts = set_scaled_kde(volumes, height=np.max(count), bandwidth=0.1)

        # Plot KDE
        ax.fill_between(x_values, kde_counts, 0, alpha=0.3, color=exp_config[exp_key]['color'])
        ax.plot(x_values, kde_counts, label=exp_config[exp_key]['name'], color=exp_config[exp_key]['color'], linewidth=1)

        # Update max density for y-axis limit
        max_count = max(max_count, np.max(count))

    # Set labels and title
    ax.set_xlabel('Star Volume', fontsize=font_size)
    ax.set_xbound(0, MAX_STAR_VOL)
    ax.set_ylabel(f'Count per interval of width {bin_width:.0e}', fontsize=font_size)
    ax.set_ylim(0, MAX_SAMPLES*1.05)

    # Move legend to upper right
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize=font_size)

    # Set tick label font sizes
    ax.tick_params(axis='both', which='major', labelsize=font_size)

    # Compute ASVD scores and prepare table data
    asvd_scores = {
        exp_key: exp_asvd.get_scores(use_star=True)
        for exp_key, exp_asvd in experiments_asvd.items()
    }
    exp_keys = list(experiments_asvd.keys())
    exp_names = [exp_config[exp_key]['name'] for exp_key in exp_keys]
    score_names = list(asvd_scores[exp_keys[0]].keys())
    columns = ['Experiment'] + [exp_name[:10] + '...' * (len(exp_name) > 10) for exp_name in exp_names]
    table_data = [
        [metric] + [format_value(asvd_scores[exp_key].get(metric, 'N/A')) for exp_key in exp_keys]
        for metric in score_names
    ]

    # Table area
    ax_table = fig.add_subplot(gs[1])
    ax_table.axis('off')  # Hide axes
    table = ax_table.table(
        cellText=table_data,
        colLabels=columns,
        cellLoc='center',
        loc='upper center'
    )
    # Make first column cells left aligned
    for (row, col), cell in table.get_celld().items():
        if col == 0:  # First column
            cell.set_text_props(ha='left')
    table.auto_set_font_size(False)
    table.set_fontsize(font_size-2)
    table.auto_set_column_width(list(range(len(table_data[0]))))

    # Add indices on top right
    ax.text(
        -0.03, 1.03, 'a)', transform=ax.transAxes, 
        ha='right', va='bottom', size=font_size+2
    )
    ax_table.text(
        -0.03, 1.03, 'b)', transform=ax_table.transAxes, 
        ha='right', va='bottom', size=font_size+2
    )

    # Improve layout
    plt.tight_layout()

    return fig