# Pipeline irbs

> *Note:* This is a `README.md` boilerplate generated using `Kedro 0.18.5`.

## Overview

<!---
Please describe your modular pipeline here.
-->

## Pipeline inputs

<!---
The list of pipeline inputs.
-->

## Pipeline outputs

<!---
The list of pipeline outputs.
-->
# Pipeline IRBS (Interest Region Bayesian Sampling)

## Overview

This pipeline implements the Interest Region Bayesian Sampling (IRBS) method for adaptive sampling in a design space. It consists of two main steps:

1. **IRBS Sampling**: Performs the sampling process using the IRBS algorithm and stores each batch result in a csv file.
2. **Join History**: Retrieves and combines the outputs from the sampling process.

The pipeline is designed to enhance the sampling efficiency by focusing on regions of interest within the design space.

## Pipeline Inputs

The pipeline requires the following inputs:

- `treated_data`: Pre-processed dataset
- `treatment`: Data treatment object or configuration
- `params:features`: List of feature names
- `params:targets`: List of target variable names
- `params:additional_values`: List of additional values to consider
- `params:irbs_coefficients`: Coefficients for the IRBS algorithm
- `params:irbs_max_size`: Maximum size of the sampling
- `params:irbs_batch_size`: Batch size for sampling
- `params:irbs_opt_iters`: Number of optimization iterations
- `params:irbs_opt_sampling_points`: Number of sampling points for optimization
- `params:irbs_error_round_decimals`: Decimal places for rounding errors
- `params:simulator_env`: Simulator environment configuration

## Pipeline Outputs

The pipeline produces the following outputs:

- `irbs_history`: History of the IRBS sampling process containing all final dataset samples where each csv file is a batch.
- `irbs_increased_data`: All the csv files of the final dataset but joined in a single csv.

## Usage

To use this pipeline, ensure that all required inputs are provided in the configuration. The pipeline can be run using the Kedro framework, which will execute the IRBS sampling process and join the resulting history to produce an enhanced dataset.

## Note

This pipeline is part of a larger data science workflow and is designed to work within the Kedro ecosystem. the `prep` pipeline should be run first. Ensure that all dependencies are properly set up and that the input data and parameters are correctly specified in your Kedro project configuration.
