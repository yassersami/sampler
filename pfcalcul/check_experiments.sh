#!/bin/bash

# Check if all required arguments were provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <environment-folder>"
    echo "Example: $0 irbs conf/new_experiments"
    echo "Note: The environment folder should always start with 'conf/'"
    exit 1
fi

# Set the arguments
ENV_BASE_FOLDER="$1"

# Check if the environment folder starts with "conf/"
if [[ "$ENV_BASE_FOLDER" != conf/* ]]; then
    echo "Error: Environment folder must start with 'conf/'" >&2
    echo "Usage: $0 <environment-folder>"
    echo "Example: $0 irbs conf/new_experiments 4"
    exit 1
fi

# Check if the environment folder exists
if [ ! -d "${ENV_BASE_FOLDER}" ]; then
    echo "Error: Environment folder '${ENV_BASE_FOLDER}' not found" >&2
    exit 1
fi

# Function to check if an experiment has already been run
experiment_already_run() {
    local exp_path="$1"
    local output_path="data/07_model_output/${exp_path#conf/}"
    
    if [ -d "$output_path" ]; then
        return 0  # Directory exists, experiment has been run
    else
        return 1  # Directory doesn't exist, experiment hasn't been run
    fi
}

# Find all catalog.yml files in the given folder and its subdirectories
find "$ENV_BASE_FOLDER" -name catalog.yml | while read -r catalog_file; do
    # Get the experiment path (folder containing catalog.yml)
    exp_path=$(dirname "$catalog_file")
    
    # Check if this experiment has already been run
    if ! experiment_already_run "$exp_path"; then
        echo "New experiment: $exp_path"
    else
        echo "Done: $exp_path"
    fi
done

echo "All experiments completed."