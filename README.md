# Welcome to the WunderNN-Predictorium ML Project!

This project is designed to tackle the Wunder Challenge 2, a machine learning competition focused on predicting future price movements based on Limit Order Book (LOB) data and recent trades. The goal is to build a robust model that can accurately forecast price indicators using advanced machine learning techniques.

## Project Overview

The project consists of various components, including data handling, model architecture, training routines, and inference mechanisms. The architecture combines GRU, Transformer, and CNN layers to leverage the strengths of each model type for effective feature extraction and prediction.

## Directory Structure

- **README.md**: Documentation for the project.
- **pyproject.toml**: Project metadata and dependencies.
- **requirements.txt**: List of required Python packages.
- **src/**: Source code for the project.
  - **data/**: Data loading and preprocessing.
  - **models/**: Model architectures and building blocks.
  - **training/**: Training routines and PyTorch Lightning integration.
  - **inference/**: Inference and prediction handling.
  - **utils/**: Utility functions and classes.
- **configs/**: Configuration files for different model settings.
- **nbs/**: Jupyter notebooks for exploration and experimentation.
- **datasets/**: Training and validation datasets in Parquet format.
- **scripts/**: Shell scripts for running training, evaluation, and submission preparation.

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd WunderNN-Predictorium-ml-project
   ```

2. **Install dependencies**:
   It is recommended to use a virtual environment. You can create one using `venv` or `conda`.
   ```bash
   pip install -r requirements.txt
   ```

3. **Data Preparation**:
   Ensure that the datasets are placed in the `datasets/` directory. The training and validation datasets should be in Parquet format.

4. **Training the Model**:
   You can start training the model using the provided shell script:
   ```bash
   ./scripts/run_train.sh
   ```

5. **Evaluating the Model**:
   After training, evaluate the model's performance on the validation dataset:
   ```bash
   ./scripts/run_eval.sh
   ```

6. **Preparing Submission**:
   To prepare your submission package, run:
   ```bash
   ./scripts/prepare_submission.sh
   ```

## Usage Guidelines

- The project is structured to facilitate experimentation with different model architectures and configurations. You can modify the configurations in the `configs/` directory to test various setups.
- Use the Jupyter notebooks in the `nbs/` directory for interactive exploration of the dataset and model performance.
- The `src/utils/metrics.py` file contains functions for calculating evaluation metrics, including the Weighted Pearson Correlation Coefficient, which is used for scoring predictions.

------------------------------------------

## How to make a submission

This page covers the technical requirements for your submission, including the code format, how to package your files, and the resource limits.

### What to submit
This is a code competition. You'll submit a .zip file containing all the code and artifacts needed to generate predictions.

The key requirements are:

The zip file must contain a solution.py file at its root.
Your solution.py must define a class named PredictionModel.
This class must have a `predict(self, data_point)` method.

### The PredictionModel class
Here’s the required structure for your PredictionModel class:

```python
import numpy as np
from utils import DataPoint

class PredictionModel:
    def __init__(self):
        # Initialize your model, load weights, etc.
        pass

    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        # This is where your prediction logic goes.
        if not data_point.need_prediction:
            return None

        # When a prediction is needed, return a numpy array of length 2 (for t0 and t1).
        # Replace this with your model's actual output.
        prediction = np.zeros(2)
        return prediction
```

Your predict method will receive a `DataPoint` object with the current market state. Your code should return `None` if `need_prediction` is False, and a NumPy array of shape (2,) otherwise.

The `DataPoint` object comes from the provided `utils.py` file and has the following attributes:

- `seq_ix`: int: The ID for the current sequence.
- `step_in_seq`: int: The step number within the sequence.
- `need_prediction`: bool: Whether a prediction is required for this point.
- `state`: np.ndarray: The current market state vector (containing p0..p9, v0..v9, dp0..1, dv0..1).

NOTE

> Remember to handle the model's internal state. When you encounter a new sequence (a new seq_ix), you must reset any recurrent state.

### Including other files
Most solutions will need more than just solution.py. You can include other files in your zip archive, such as:

- Model weights (e.g., .pt, .h5, .onnx files).
- Helper Python modules (.py files).
- Configuration files (e.g., .json, .yaml).
- Small data files.

> Just make sure `solution.py` is at the root of the archive.

### How to package your solution
You need to package all your files into a single .zip archive.

#### macOS/Linux

Open a terminal.

```bash
cd into the directory that contains your solution files.
Run the following command:

zip -r ../submission.zip .  
```

This creates `submission.zip` in the parent directory, containing everything from your current folder.


### Resource and time limits
The submitted code will run in an isolated environment with the following constraints:

- CPU: 1 vCPU core
- No GPU
- RAM: 16 GB
- Storage: Local SSD
- Time limit: Your code must finish generating predictions for the entire test set in 60 minutes or less.
- No internet access: The execution environment is offline.

Note

**The submission .zip file must be < 20 MB.**

------------------------------------------


## Contributing

Contributions to the project are welcome! Please feel free to submit issues or pull requests for any improvements or features you would like to add.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

Good luck, and have fun building your model! We can't wait to see what you create!