#!/bin/bash

# Remove environment if it exists
conda env remove --name NINA_georef_seagul -y

# Create new environment from yml
conda env create -f environment.yml

# Activate environment and install package
source activate NINA_georef_seagul
pip install .