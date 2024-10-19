import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from src.team5.inference.inference import infer, load_model, preprocess_input
from src.team5.training.config import BASE_MODEL


def load_parquet_data(file_path, sample_size=1000):
    df = pd.read_parquet(file_path)
    return df.sample(n=min(sample_size, len(df)))


def calculate_errors(true_mzs, true_intensities, pred_mzs, pred_intensities):
    # Ensure all inputs are numpy arrays
    true_mzs = np.array(true_mzs)
    true_intensities = np.array(true_intensities)
    pred_mzs = np.array(pred_mzs)
    pred_intensities = np.array(pred_intensities)

    # Find the minimum length
    min_length = min(len(true_mzs), len(pred_mzs))

    # Truncate arrays to the minimum length
    true_mzs = true_mzs[:min_length]
    true_intensities = true_intensities[:min_length]
    pred_mzs = pred_mzs[:min_length]
    pred_intensities = pred_intensities[:min_length]

    # Calculate relative errors for m/z values
    mz_errors = np.abs(true_mzs - pred_mzs) / true_mzs

    # Calculate absolute errors for intensities
    intensity_errors = np.abs(true_intensities - pred_intensities)

    return mz_errors, intensity_errors


def plot_errors(mz_errors, intensity_errors):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot m/z errors
    ax1.hist(mz_errors.flatten(), bins=50, edgecolor="black")
    ax1.set_title("Distribution of m/z Relative Errors")
    ax1.set_xlabel("Relative Error")
    ax1.set_ylabel("Frequency")

    # Plot intensity errors
    ax2.hist(intensity_errors.flatten(), bins=50, edgecolor="black")
    ax2.set_title("Distribution of Intensity Absolute Errors")
    ax2.set_xlabel("Absolute Error")
    ax2.set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig("error_distribution.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Run sanity check inference on training data"
    )
    parser.add_argument(
        "--parquet_file",
        type=str,
        required=True,
        help="Path to the parquet file containing training data",
    )
    parser.add_argument(
        "--sample_size", type=int, default=1000, help="Number of samples to process"
    )
    parser.add_argument(
        "--model_version", type=str, help="Model version to use (default: latest)"
    )

    args = parser.parse_args()

    # Load data
    data = load_parquet_data(args.parquet_file, args.sample_size)

    # Load model and tokenizer
    model = load_model(args.model_version)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    all_mz_errors = []
    all_intensity_errors = []
    shape_mismatches = 0
    true_shapes = []
    pred_shapes = []

    for _, row in tqdm(data.iterrows(), total=len(data)):
        smiles = row["smiles"]
        true_mzs = row["mzs"]
        true_intensities = row["intensities"]

        # Run inference with a large top_k value (effectively "infinite")
        pred_mzs, pred_intensities = infer(
            model, tokenizer, smiles, true_mzs, true_intensities, top_k=2000
        )

        # Record shapes
        true_shapes.append(len(true_mzs))
        pred_shapes.append(len(pred_mzs))

        # Check for shape mismatches
        if len(true_mzs) != len(pred_mzs):
            shape_mismatches += 1
            print(
                f"Shape mismatch: True shape: {len(true_mzs)}, Predicted shape: {len(pred_mzs)}"
            )

        # Calculate errors
        mz_errors, intensity_errors = calculate_errors(
            true_mzs, true_intensities, pred_mzs, pred_intensities
        )

        all_mz_errors.extend(mz_errors)
        all_intensity_errors.extend(intensity_errors)

    # Convert to numpy arrays
    all_mz_errors = np.array(all_mz_errors)
    all_intensity_errors = np.array(all_intensity_errors)

    # Plot error distributions
    plot_errors(all_mz_errors, all_intensity_errors)

    print(f"Processed {len(data)} samples.")
    print(f"Shape mismatches occurred in {shape_mismatches} samples.")
    print(f"Mean true shape: {np.mean(true_shapes):.2f}")
    print(f"Mean predicted shape: {np.mean(pred_shapes):.2f}")
    print(f"Mean m/z relative error: {np.mean(all_mz_errors):.4f}")
    print(f"Mean intensity absolute error: {np.mean(all_intensity_errors):.4f}")
    print("Error distribution plot saved as 'error_distribution.png'")

    # Plot shape distributions
    plt.figure(figsize=(10, 5))
    plt.hist(true_shapes, bins=50, alpha=0.5, label="True")
    plt.hist(pred_shapes, bins=50, alpha=0.5, label="Predicted")
    plt.title("Distribution of Shapes")
    plt.xlabel("Number of peaks")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig("shape_distribution.png")
    plt.close()


if __name__ == "__main__":
    main()
