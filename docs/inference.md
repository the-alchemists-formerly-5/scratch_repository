
# Checkpoint Management and Inference Guide

## Table of Contents
1. [Checkpoint Directory Structure](#checkpoint-directory-structure)
2. [Checkpoint Files](#checkpoint-files)
3. [Updating the Checkpoint Tracker](#updating-the-checkpoint-tracker)
4. [Using the Inference API](#using-the-inference-api)
5. [Running Inference with Poetry](#running-inference-with-poetry)

## Checkpoint Directory Structure

The checkpoints are stored in the `checkpoints/` directory at the root of the project. The structure should look like this:

```
checkpoints/
├── checkpoint_tracker.json
├── v0.1/
│   ├── config.json
│   ├── model.safetensors
│   └── optimizer.bin
└── v0.2/
    ├── config.json
    ├── model.safetensors
    └── optimizer.bin
```

## Checkpoint Files

Each checkpoint version should include the following files:

1. `model.safetensors`: The main model weights file.
2. `config.json`: The model configuration file.
3. `optimizer.bin`: The optimizer state (optional, useful for resuming training).

## Updating the Checkpoint Tracker

The `checkpoint_tracker.json` file keeps track of all checkpoint versions and their details. Here's how to update it:

1. Format of `checkpoint_tracker.json`:

```json
{
  "latest": "v0.2",
  "checkpoints": [
    {
      "version": "v0.1",
      "date": "2023-10-15",
      "description": "Initial model",
      "performance": {
        "accuracy": 0.85,
        "f1_score": 0.83
      }
    },
    {
      "version": "v0.2",
      "date": "2023-11-01",
      "description": "Improved model with more training data",
      "performance": {
        "accuracy": 0.88,
        "f1_score": 0.86
      }
    }
  ]
}
```

2. To update the tracker, use the `update_checkpoint_tracker.py` script:

```bash
python src/team5/training/update_checkpoint_tracker.py --version v0.3 \
    --description "Fine-tuned on additional data" \
    --accuracy 0.90 \
    --f1_score 0.89 \
    --set_as_latest
```

This script will add a new entry to the `checkpoints` list and update the `latest` field if `--set_as_latest` is specified.

## Using the Inference API

To use the inference API, follow these steps:

1. Ensure that the checkpoint files are in the correct location as described in the [Checkpoint Directory Structure](#checkpoint-directory-structure) section.

2. Run the inference script:

```bash
poetry run infer --input_file inference_input_all_columns.parquet --n 10
```

Optional arguments:
- `--version`: Specify a particular model version (e.g., "v0.2"). If not provided, the latest version will be used.

3. The script will output the predictions based on the input SMILES, m/z values, and intensities.

For programmatic use, you can import the `infer` function from `src/team5/inference/inference.py`:

```python
from src.team5.inference.inference import load_model, infer

model = load_model()  # Loads the latest model by default
predictions = infer(model, tokenizer, smiles, mzs, intensities)
```

Remember to keep your checkpoints and `checkpoint_tracker.json` up to date for smooth operation of the inference API.

## Running Inference with Poetry

To run the inference script using Poetry, follow these steps:

1. Ensure you have Poetry installed. If not, install it following the instructions on the [official Poetry website](https://python-poetry.org/docs/#installation).

2. Navigate to the root directory of your project in the terminal.

3. If you haven't already, install the project dependencies:

```bash
poetry install
```

4. Run the inference script using Poetry:

```bash
poetry run infer --input_file inference_input_all_columns.parquet --n 10
```

Optional arguments remain the same as mentioned in the [Using the Inference API](#using-the-inference-api) section.

6. If you're using the inference function in a Python script within the Poetry environment, you can activate the environment and run your script:

```bash
poetry shell
python your_script.py
```

Remember to exit the Poetry shell when you're done:

```bash
exit
```

By using Poetry, you ensure that all dependencies are correctly managed and that the script runs in the appropriate environment.

## Adding .safetensors Files to Git Using Git LFS

Git LFS (Large File Storage) is used to manage large files in Git repositories. Follow these steps to add new .safetensors files to your project using Git LFS.

### Prerequisites

1. Ensure you have Git LFS installed. If not, install it:
   ```
   brew install git-lfs
   ```

2. Make sure Git LFS is initialized in your repository:
   ```
   git lfs install --local
   ```

### Steps to Add New .safetensors Files

1. Navigate to your project's root directory in the terminal.

2. If you haven't already, tell Git LFS to track .safetensors files:
   ```
   git lfs track "*.safetensors"
   ```

3. Add the .gitattributes file to your repository if it's not already there:
   ```
   git add .gitattributes
   ```

4. Move your new .safetensors file to the appropriate directory in your project (e.g., `checkpoints/v1.0/model.safetensors`).

5. Add the new .safetensors file to Git:
   ```
   git add path/to/your/new/model.safetensors
   ```

6. Commit the changes:
   ```
   git commit -m "Add new .safetensors file for model version X.X"
   ```

7. Push the changes to your remote repository:
   ```
   git push origin your-branch-name
   ```

### Verifying Git LFS is Working

To verify that Git LFS is correctly managing your .safetensors files:

1. Check the status of Git LFS:
   ```
   git lfs status
   ```

2. List all files being tracked by Git LFS:
   ```
   git lfs ls-files
   ```

### Pulling .safetensors Files

When cloning the repository or pulling updates, ensure you also pull the LFS files:

```
git lfs pull
```

### Troubleshooting

If you encounter issues:

1. Ensure Git LFS is installed and initialized.
2. Check that the .gitattributes file is correctly tracking .safetensors files.
3. Verify you have the necessary permissions to push large files to your remote repository.

For more information, refer to the [Git LFS documentation](https://git-lfs.github.com/).

## Sanity Check Error Calculation

This document explains how the sanity check script calculates errors when comparing the predicted mass spectra to the true (ground truth) mass spectra.

### Error Metrics

The sanity check uses two primary error metrics:

1. Relative Error for m/z values
2. Absolute Error for intensities

#### Relative Error for m/z Values

The relative error is used for m/z values because it provides a measure of the error relative to the magnitude of the true value. This is particularly useful for m/z values, which can span a wide range.

For each corresponding pair of true and predicted m/z values, the relative error is calculated as:

$\text{Relative Error (m/z)} = \frac{|\text{m/z}_\text{true} - \text{m/z}_\text{predicted}|}{\text{m/z}_\text{true}}$

Where:
- $\text{m/z}_\text{true}$ is the true m/z value
- $\text{m/z}_\text{predicted}$ is the predicted m/z value

#### Absolute Error for Intensities

The absolute error is used for intensities because intensities are typically normalized and fall within a specific range (often 0 to 1). In this case, the absolute difference provides a direct measure of the error.

For each corresponding pair of true and predicted intensities, the absolute error is calculated as:

$\text{Absolute Error (intensity)} = |\text{intensity}_\text{true} - \text{intensity}_\text{predicted}|$

Where:
- $\text{intensity}_\text{true}$ is the true intensity value
- $\text{intensity}_\text{predicted}$ is the predicted intensity value

### Error Calculation Process

1. For each spectrum in the dataset:
   a. Predict the m/z values and intensities using the model.
   b. Align the predicted values with the true values (this may involve truncation if the number of predicted peaks differs from the true peaks).
   c. Calculate the relative error for each m/z pair.
   d. Calculate the absolute error for each intensity pair.

2. Collect all errors across all spectra in the dataset.

3. Calculate summary statistics:
   - Mean m/z relative error:
     $\text{Mean m/z Relative Error} = \frac{1}{N} \sum_{i=1}^{N} \text{Relative Error (m/z)}_i$
   - Mean intensity absolute error:
     $\text{Mean Intensity Absolute Error} = \frac{1}{N} \sum_{i=1}^{N} \text{Absolute Error (intensity)}_i$

   Where $N$ is the total number of peak comparisons across all spectra.

### Handling Shape Mismatches

When the number of predicted peaks differs from the number of true peaks, the script handles this by:

1. Truncating the longer array to match the length of the shorter array.
2. Calculating errors only for the matched pairs.
3. Reporting the number of shape mismatches encountered.

This approach ensures that errors can be calculated even when the predicted spectrum has a different number of peaks than the true spectrum, while also providing information about how often such mismatches occur.

### Visualization

The script generates histograms of the m/z relative errors and intensity absolute errors. These visualizations help in understanding the distribution of errors across the dataset.

By examining these error metrics and their distributions, you can gain insights into the model's performance and identify areas for improvement in the prediction of mass spectra.
