# Pipeline prep

> This does the preprocessing of the initial data to enter the sampling algorithm. 
> At the same time, it returns the necessary elements to classify and scale the points retrieved. 

## Overview
In this pipeline a scaler is constructed to make all features and targets values go from 0 to 1, 
according to the selected boundaries.
It also constructs a class that contains all the information necessary to classify the points, according to
the region of interest.
With these tools, it also returns the initial data set, prepared to be enlarged.

## Pipeline inputs

* **Node: preparation**
  * initial_data: Initial data set to be used to train the surrogate containing values in physical scales, defined in 'catalog:initial_data',
  * features: from 'params:features',
  * targets: from 'params:targets',
  * additional_values: Other values of interest that the user wishes to keep track (but not use), defined in 'parameters:additional_values',
  * variables_ranges: Contains configurations for boundaries of the design space, defined in 'parameters_prep:variables_ranges'. The base configuration is the one use in sampling pipelines and to set metrics plot ranges. Other configuration are used during importing of scaled data in metrics pipeline to enable scaling back to physical SI units consistently. 
  * interest_region: Boundaries of the targets that will define the region of interest, defined in 'parameters:interest_region',
  * max_simu_time: Maximum time in seconds of simulation allowed to obtain ground truth of targets, defined in 'parameters:simulator_config.max_simu_time',
  * scaler: Class that contains the parameters needed to transform the design space into a space between 0 and 1, defined in 'catalog:scaler'.

## Pipeline outputs
* **Node: fit_scaler**
  * scaler: from 'catalog:scaler'.

* **Node: preparation**
  * treated_data: Initial data treated to be used to train the surrogate, defined in 'catalog:treated_data',
  * treatment: Class containing all the information needed to classify points in the space (Interest/Inliers/Outliers) 
  for the values already scaled, defined in 'catalog:treatment'. 