# Pipeline IRBS (Interest Region Bayesian Sampling)

> This is the main pipeline for selecting and calculating new points based on the region of interest. It returns a database with all calculated values.

## Overview

This pipeline implements the Interest Region Bayesian Sampling (IRBS) method for adaptive sampling in a design space. The pipeline enhances sampling efficiency by focusing on regions of interest within the design space. It consists of two main nodes:

- **irbs_sampling Node**: 
  - Sets up a surrogate model with the initial treated data.
  - Optimizes the Figure of Merits to determine which points will be simulated to obtain ground truth.
  - Stores batch result in a CSV file.
  - Adds new points to the dataset and retrains the surrogate model.

- **join_history Node**: 
  - Retrieves and combines the outputs from the sampling process in a unique database.


## Pipeline Inputs

The pipeline requires the following inputs:

### Data and Treatment
- **`treated_data`**: Pre-processed dataset
- **`treatment`**: DataTreatment class containing informations about design space, including design space boudaries, scaler, interest region...
### Variable Definitions
- **`params:features`**: List of feature names (physical variables known, limited by the design space)
- **`params:targets`**: List of target variable names (physical variables of interest, to be known after simulating)
- **`params:additional_values`**: List of additional values to consider (other values of interest to track such as sim_time)

### IRBS Algorithm Parameters
- **`params:irbs_fom_terms`**: Terms of figure of merit for the IRBS algorithm, including:
  - **`std`**: The std given by the trained GP.
  - **`interest`**: The probability of getting target values in the interest region.
  - **`coverage`**: A measure defining if the selected sample is covering an new empty region or an already densly sampled region.
- **`params:irbs_max_size`**: Maximum size of the sampling
- **`params:irbs_batch_size`**: Batch size to sample at each itereation
- **`params:irbs_opt_iters`**: Number of optimization iterations for the SHGO algorithm
- **`params:irbs_opt_sampling_points`**: Number of sampling points for optimization in SHGO

### Simulation Environment
- **`params:simulator_env`**: Simulator environment configuration

### Additional Parameters
- **`params:interest_region`**: Boundaries of the targets defining the region of interest

### Node-Specific Inputs
- **irbs_sampling Node**:
  - Uses all the above inputs
- **join_history Node**:
  - **`catalog:irbs_history`**: Incremental DataSet containing the initial data and results
  - Uses **`params:features`** and **`params:targets`**


## Pipeline Outputs

The pipeline produces the following outputs:

### Node: irbs_sampling
- **`catalog:irbs_history`**: Incremental DataSet containing the initial data and results (IRBS sampling process history) where each file is a batch.

### Node: join_history
- **`catalog:irbs_increased_data`**: Single CSV file containing all data from the final dataset

## Usage

To use this pipeline, ensure that all required inputs are provided in the configuration. The pipeline can be run using the Kedro framework, which will execute the IRBS sampling process and join the resulting history to produce an enhanced dataset.

## Note

This pipeline is part of a larger data science workflow and is designed to work within the Kedro ecosystem. the `prep` pipeline should be run first. Ensure that all dependencies are properly set up and that the input data and parameters are correctly specified in your Kedro project configuration.
