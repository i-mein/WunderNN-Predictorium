#!/bin/bash

# Prepare the submission package for the Wunder Challenge 2

# Define the source and output
SOL_DIR="solutions/exp_05"
OUTPUT_ZIP="submission_exp_05.zip"

if [ ! -d "$SOL_DIR" ]; then
    echo "Error: Solution directory $SOL_DIR not found."
    exit 1
fi

echo "Preparing submission from $SOL_DIR..."

# Navigate to solution dir and zip contents
cd "$SOL_DIR" || exit

# Zip all contents of the solution directory
# solution.py must be at root
zip -r ../../$OUTPUT_ZIP .

cd ../..

echo "Submission package prepared: $OUTPUT_ZIP"
# Check zip content listing
unzip -l $OUTPUT_ZIP