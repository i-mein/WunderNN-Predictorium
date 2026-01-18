#!/bin/bash

# Activate the virtual environment if needed
# source /path/to/your/venv/bin/activate

# Run the evaluation script
python src/training/eval.py --config configs/config_gru_transformer_cnn.yaml --data datasets/valid.parquet --output_dir outputs/eval_results/