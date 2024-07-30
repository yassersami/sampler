# IRBS
This file contains:
* an [overview](#overview) of the code,
* a [set-up](#set-up) guide to install it,
* a [guide](#run-project) to run the complete project.

## Overview
This code is developed in order to do a Bayesian Optimization to search for the optimal points to discover 
the physical specifications of a system that will lead to the desired results. 
From now on, the physical specifications of a system that are controlled by the user are called `features`, 
while the variables that are the outputs from the experiment are called `targets`.

Since the values from the targets cannot be known from the features before conducting an expensive experiment,
the goal of this code is to construct a `surrogate model` that will aim to predict the value of the targets 
without running the experiment. 
This estimated value will help to guide the search in order to find `interest targets points` without running the
full experiment. 
However, since we are interested in the real value of the targets (referred often as `ground truth`), after selecting
some points, we will conduct the experiments and improve our surrogate model for the next cycle.
This experiments will be conducted through the simulator `OD`, already included on this package. 
If in need to add another simulator, please, use the same logic existing inside `models`: 
* a wrapper for running the simulator inside one file,
* all the auxiliary functions necessary to run the simulator on one file.

At the end of this experiment, the user should obtain more point of interest then if only running a 
Lattice Hypercube Sampling (LHS).

## Set up
### Rules and guidelines
In order to get the best out of the template:
* Don't remove any lines from the `.gitignore` file we provide
* Don't commit data to your repository
* Don't commit any credentials or your local configuration to your repository. Keep all your credentials and local configuration in `conf/local/`

### Install dependencies
* When using **pip** (recommended)
The dependencies are declared in `src/requirements.txt`.  
To install them, we suggest to create a virtual environment and then install them.
When running with Python version than 3.6 run (from this folder):
```
python -m venv venv
. venv/bin/activate
pip install -r src/requirements.txt
```

* When using **pip** (recommended only if your OS is Windows)
```
conda create -n venv pip
conda actiate venv
pip install -r src/requirements.txt
```

# Run project
In order to run this project is necessary that you have an initial file, stored in the `data` folder inside this project,
and to add it to the file in `conf/base/catalog.yml` under the name `initial_data`, which is the first entry on the file.
You can change the location of all files, but make sure that those are inside `data`.
In order to give to the user an idea of how this file should look like, there is an initial file included in this repository,
located on the direction written on the catalog.

The code is divided into 3 steps:
1. <span style="color:yellow"> Prepare input data </span>: This pipeline will prepare the input data so that can be run in the next pipeline, while also creating the necessary classes to known all the parameters that define the regions of interest as well as outliers/inliers. 
2. <span style="color:green"> Run IRBS (Interest Region Bayesian Sampling) experiment </span>: This pipeline will conduct the Bayesian Optimization in order to increase the initial dataset with interest points, while improving the surrogate model. It will output a DB with several samples.
3. <span style="color:red"> Perform analysis on different experiments </span>: This pipeline contains plots to analyze results, it's created to compare several experiments at the same time (described on `conf/base/parameters/metrics.json`).

## Common parameters
There are some parameters shared between all pipelines, these are stored in `conf/base/parameters.json`, and they are:
* paths: Some programs need paths and store files bypassing the kedro structure,
* features: Physical variables known, that limit the design space,
* targets: Physical variables of interest, which cannot be fixed but are to be known after simulating,
* additional_values: Other values of interest that the user wishes to keep track (but not use).
* interest_region: Boundaries of the targets that will define the region of interest, 

The parameters that only belong to one pipeline are located in `conf/base/parameters/` and are named after each pipeline. The description of each parameter is contained in the README.md file located inside each pipeline in `src/sampler/pipelines/`.


## Environments
In case some parameters need to be modified, they could be stored in a new folder on `conf`.
This new folder should copy the structure on base, and should only contain the parameters or catalog entries that are modified (since all other values will be inherited from base).
This new configuration can be run by adding `--env new_env_folder_name` to the command to run the pipeline.  

## <span style="color:yellow"> 1. Prepare Initial Data and create classification classes </span>
```
kedro run --pipeline prep --env base
``` 
## <span style="color:green"> 2. Run IRBS (Interest Region Bayesian Sampling) experiment </span>
``` 
kedro run --pipeline irbs --env base
``` 
## <span style="color:red"> 3. Perform metrics </span>
``` 
kedro run --pipeline metrics --env base
```