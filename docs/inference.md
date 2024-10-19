# Checkpoint Management and Inference Guide

## Table of Contents
1. [Checkpoint Directory Structure](#checkpoint-directory-structure)
2. [Checkpoint Files](#checkpoint-files)
3. [Updating the Checkpoint Tracker](#updating-the-checkpoint-tracker)
4. [Using the Inference API](#using-the-inference-api)

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
python src/team5/inference/inference.py --smiles "C1=CC=CC=C1" \
    --mzs 100 200 300 \
    --intensities 0.5 0.7 0.9
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
