import argparse
from pathlib import Path

import numpy as np
import polars as pl
from transformers import AutoTokenizer
from transformers import logging as transformers_logging

from src.team5.training.config import BASE_MODEL

from .checkpoint_loader import CHECKPOINT_DIR, PROJECT_ROOT, TRACKER_FILE
from .infer import infer
from .load_model import load_model
from .plot_sample_results import plot_sample_results

# Set the logging level for transformers to ERROR to suppress model loading messages
transformers_logging.set_verbosity_error()


def process_predictions(pred_mzs, pred_intensities, intensity_threshold=0.01, top_k=20):
    # Convert to numpy for easier processing
    mzs = pred_mzs.squeeze().numpy()
    intensities = pred_intensities.squeeze().numpy()

    # Filter out low-intensity peaks
    mask = intensities > intensity_threshold
    mzs = mzs[mask]
    intensities = intensities[mask]

    # Sort by intensity (descending order)
    sorted_indices = np.argsort(intensities)[::-1]
    mzs = mzs[sorted_indices]
    intensities = intensities[sorted_indices]

    # Limit to top_k if specified
    if top_k is not None:
        mzs = mzs[:top_k]
        intensities = intensities[:top_k]

    return mzs, intensities


def main():
    parser = argparse.ArgumentParser(description="Run inference on the trained model")
    parser.add_argument(
        "--version", type=str, help="Model version to use (default: latest)"
    )
    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to input parquet file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Path to output file (default: input_file_PREDICTED.parquet)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Number of top peaks to return (default: 20)",
    )
    parser.add_argument(
        "--n",
        type=int,
        help="Number of elements to process from the input file (default: all)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot sample results if mzs and intensities are available",
    )

    args = parser.parse_args()
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Checkpoint directory: {CHECKPOINT_DIR}")
    print(f"Tracker file: {TRACKER_FILE}")

    # Set default output file name if not provided
    if args.output_file is None:
        input_path = Path(args.input_file)
        args.output_file = str(
            input_path.with_name(f"{input_path.stem}_PREDICTED{input_path.suffix}")
        )

    # Load the model
    model = load_model(args.version)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # Load input data
    input_data = pl.read_parquet(args.input_file)

    # Limit the number of rows if n is specified
    if args.n is not None:
        input_data = input_data.head(args.n)

    # Run inference
    pred_mzs, pred_intensities = infer(model, tokenizer, input_data)

    # Process predictions
    results = []
    for i in range(len(pred_mzs)):
        mzs, intensities = process_predictions(
            pred_mzs[i], pred_intensities[i], top_k=args.top_k
        )
        results.append(
            {
                "smiles": input_data["smiles"][i],
                "adduct": input_data["adduct"][i],
                "collision_energy": input_data["collision_energy"][i],
                "instrument_type": input_data["instrument_type"][i],
                "compound_class": input_data["compound_class"][i],
                "predicted_mzs": mzs.tolist(),
                "predicted_intensities": intensities.tolist(),
            }
        )

    # Save results
    output_df = pl.DataFrame(results)
    output_df.write_parquet(args.output_file)

    print(f"Inference completed. Results saved to {args.output_file}")

    # Plot sample results if requested and data is available
    # Plot sample results if requested and data is available
    if args.plot:
        try:
            plot_sample_results(model, input_data, tokenizer, n_samples=5)
        except Exception as e:
            print(f"Error plotting sample results: {e}")
            print("Make sure the model has the expected structure and methods.")


if __name__ == "__main__":
    main()
