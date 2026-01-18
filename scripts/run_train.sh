#!/bin/bash

# Activate the virtual environment
source venv/bin/activate

# Set environment variables
export PYTHONPATH=$(pwd)/src

# Run the training script
python src/training/train.py --config configs/config_gru_transformer_cnn.yaml

# Deactivate the virtual environment
deactivate