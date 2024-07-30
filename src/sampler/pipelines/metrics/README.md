# Pipeline Metrics

> This pipeline produces plots and metrics to analyze the results obtained from experiments.

## Overview

This pipeline processes experimental data, calculates various metrics, and generates plots to analyze the distribution of points of interest with respect to features and targets. It consists of four main steps:

1. **Data Preparation**: Prepares the experimental data for analysis.
2. **Metric Calculation**: Computes various metrics based on the prepared data.
3. **Data Scaling**: Scales the data for plotting purposes.
4. **Metric Plotting**: Generates plots to visualize the metrics and data distributions.

## Pipeline Inputs

### Node: prepare_data_metrics
- `params:experiments`: Dictionary of experiments to compare, including:
  - `color`: To distinguish experiments
  - `name`: For legends
  - `path`: To retrieve information to plot
  - `scale`: To put results in a physical scale ("classify" or "real-inverse")
- `params:names`: Names of features and targets to display on plots
- `params:features`: Physical variables known, limited by the design space
- `params:targets`: Physical variables of interest, to be known after simulating
- `params:additional_values`: Other values of interest to track
- `treatment`: Class for classifying points in the space (Interest/Inliers/Outliers)

### Node: get_metrics
- `exp_data`: Prepared experimental data
- `features`: List of feature names
- `targets`: List of target names
- `treatment`: Treatment object
- `params:params_volume`: Parameters for volume calculation
- `params:initial_size`: Initial dataset size

### Node: scale_data_for_plots
- `exp_data`: Experimental data
- `features`: List of feature names
- `targets`: List of target names
- `targets_prediction`: Predicted target values
- `params:scales`: Scaling parameters
- `params:interest_region`: Boundaries defining the region of interest

### Node: plot_metrics
- `scaled_exp_data`: Scaled experimental data
- `features`: List of feature names
- `targets`: List of target names
- `scaled_region`: Scaled region of interest
- `params:ignition_points`: Ignition points for plotting
- `volume`: Calculated volume metric

## Pipeline Outputs

- `exp_data`: Prepared experimental data
- `features`: List of feature names
- `targets`: List of target names
- `targets_prediction`: Predicted target values
- `n_interest`: Number of points of interest
- `volume`: Volume metric
- `r2`: R-squared metric
- `crps`: Continuous Ranked Probability Score
- `scaled_exp_data`: Scaled experimental data
- `scaled_region`: Scaled region of interest
- `metrics_plots`: Generated plots for metric visualization

## Pipeline Functionality

1. **Data Preparation**: The `prepare_data_metrics` node prepares the experimental data for analysis, including scaling and classification of points.

2. **Metric Calculation**: The `get_metrics` node computes various metrics such as the number of points of interest, volume, R-squared, and CRPS.

3. **Data Scaling**: The `scale_data_for_plots` node scales the data for plotting purposes, ensuring consistent visualization across different experiments.

4. **Metric Plotting**: The `plot_metrics` node generates various plots to visualize the distribution of points of interest, the relationship between features and targets, and other relevant metrics.

This pipeline provides a comprehensive analysis of experimental results, allowing for easy comparison between different experiments and visualization of key metrics and data distributions.


# Pipeline analysis

> This pipeline produces plots and tables to analyze the results obtained.

## Overview

It creates a summary of the results, two plots to analyze the points of interest obtained depending on the features 
sampled, and two that analyze this distribution but taken into account targets.
There is also a plot analyzing all the variables (features and targets) with respect to each other.

## Pipeline inputs
* **Node: prepare_data_to_analyze**
  * experiments: Experiments to compare (at least one), defined in 'parameters/analysis:experiments'. 
  It's a dictionary that should include a key (for each experiment) and the following values:
    * "color", to distinguish experiment,
    *  "name", to add on legends,
    *  "path", to retrieve information to plot (this could be the path to 'history' if the run is not completed, or to 
    the final csv file)
    *  "scale", to put all the results in a physical scale. It could be "classify" to data already on a physical scale, 
    or "real-inverse" to data from the simulator that is scaled between 0 and 1.
  * names: Name of features and targets to display on the plots, defined in 'parameters/analysis:names',
  * features: Physical variables known, limited by the design space, defined in 'parameters:features',
  * targets: Physical variables of interest, which cannot be fixed but are to be known after simulating, defined in 'parameters:targets',
  * additional_values: Other values of interest that the user wishes to keep track (but not use), defined in 'parameters:additional_values',
  * treatment: Class containing all the information needed to classify points in the space (Interest/Inliers/Outliers) 
  for the values already scaled, defined in 'catalog:treatment',
  * n_slice: Value in which divide the simulation to plot, defined in 'parameters/analysis:n_slice',
  * tot_size: Final size value, including or not initial data set size, defined in 'parameters/analysis:total_size',
  * interest_region: Boundaries of the targets that will define the region of interest, defined in 'parameters:interest_region'. 
* **Node: make_results_summary**
  * data: Output dictionary from node 'prepare_data_to_analyze',
  * initial_size: from 'parameters/analysis:initial_size',
  * n_slice: from 'parameters/analysis:n_slice',
  * tot_size: from 'parameters/analysis:total_size'. 
* **Node: ignition_for_features**
  * data: Output dictionary from node 'prepare_data_to_analyze',
  * features: from 'parameters/analysis:features',
  * targets: from 'parameters/analysis:targets'.
* **Node: measure_coverage**
  * data: Output dictionary from node 'prepare_data_to_analyze',
  * targets: from 'parameters/analysis:targets', 
  * region: Output from node 'prepare_data_to_analyze',
* **Node: plot_multi_analysis**
  * data: Output dictionary from node 'prepare_data_to_analyze',
  * features: from 'parameters/analysis:features',
  * targets: from 'parameters/analysis:targets'.


## Pipeline outputs

* **Node: prepare_data_to_analyze**
  * exp_data: Dictionary with all the required information to analyze the results,
  * region: from 'parameters/interest_region' (it will need to be changed if targets and/or their units to plot change),
  * features: from 'parameters/analysis:features',
  * targets: from 'parameters/analysis:targets'.
* **Node: make_results_summary**
  * resume_file: Resume file on 'catalog:resume_file'.
* **Node: ignition_for_features**
  * features_2d: 2D plots of features, marking in interest/inliers/outliers points, on 'catalog:features_2d',
  * feat_tar: Features versus targets plot, for interest points only, on 'catalog:feat_tar'
* **Node: measure_coverage**
  * violin_plot: Violin plot of the distribution of values of targets for the results , on 'catalog:violin_plot',
  * kde_plot: KDE of the distribution of values of targets for the results, on 'catalog:targets_kde'.
* **Node: plot_multi_analysis**
  * multi_analysis: Plot containing a comparison of the distribution of one variable with respect to another 
  (features and targets), as a scatter plot, diagram plot, and density surface, on 'catalog:multi_analysis'