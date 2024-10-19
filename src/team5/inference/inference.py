import argparse
import json
import os
from pathlib import Path

import numpy as np
import polars as pl
import torch
from safetensors.torch import load_file
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import logging as transformers_logging

from src.team5.data.prepare import ADDUCT, instrument
from src.team5.models.custom_model import CustomChemBERTaModel
from src.team5.training.config import (BASE_MODEL, MAX_FRAGMENTS,
                                       MAX_SEQ_LENGTH, SUPPLEMENTARY_DATA_DIM)
from src.team5.training.lora_config import create_peft_model

# Set the logging level for transformers to ERROR to suppress model loading messages
transformers_logging.set_verbosity_error()

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.absolute()

# Define the checkpoint directory path
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
TRACKER_FILE = CHECKPOINT_DIR / "checkpoint_tracker.json"


def load_checkpoint_tracker():
    if not TRACKER_FILE.exists():
        raise FileNotFoundError(f"Checkpoint tracker file not found at {TRACKER_FILE}")
    with open(TRACKER_FILE, "r") as f:
        return json.load(f)


def get_latest_checkpoint():
    tracker = load_checkpoint_tracker()
    latest_version = tracker["latest"]
    return CHECKPOINT_DIR / latest_version


def load_model(version=None):
    if version is None:
        checkpoint_dir = get_latest_checkpoint()
    else:
        checkpoint_dir = CHECKPOINT_DIR / version

    model_path = checkpoint_dir / "model.safetensors"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found in {checkpoint_dir}")

    # Load the base model
    base_model = AutoModelForMaskedLM.from_pretrained(BASE_MODEL)

    # Create the custom model
    custom_model = CustomChemBERTaModel(
        model=base_model,
        max_fragments=MAX_FRAGMENTS,
        max_seq_length=MAX_SEQ_LENGTH,
        supplementary_data_dim=SUPPLEMENTARY_DATA_DIM,
    )

    # Create the PEFT model
    peft_model = create_peft_model(custom_model)

    # Load the trained weights
    state_dict = load_file(str(model_path))

    # Custom loading function to handle PEFT layers
    def load_peft_state_dict(model, state_dict):
        model_state_dict = model.state_dict()
        for name, param in state_dict.items():
            if name in model_state_dict:
                if param.shape == model_state_dict[name].shape:
                    model_state_dict[name].copy_(param)
                else:
                    print(f"Skipping parameter {name} due to shape mismatch")
            else:
                print(f"Unexpected key in state_dict: {name}")

        model.load_state_dict(model_state_dict, strict=False)

    # Load the state dict using the custom function
    load_peft_state_dict(peft_model, state_dict)
    peft_model.eval()
    return peft_model


def preprocess_input(smiles, tokenizer, adduct, collision_energy, instrument_type):
    tokenized = tokenizer(
        smiles,
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        return_tensors="pt",
    )

    # Create supplementary data tensor
    supplementary_data = torch.zeros(SUPPLEMENTARY_DATA_DIM)
    supplementary_data[0] = ADDUCT.get(adduct, 0)
    supplementary_data[1] = float(collision_energy)
    instrument_vec = instrument(instrument_type)
    supplementary_data[2:7] = torch.tensor(instrument_vec)

    return tokenized, supplementary_data


def process_input_data(df, tokenizer):
    df = df.with_columns(
        [
            pl.col("collision_energy").cast(pl.Float64).fill_null(0),
        ]
    )

    tokenized_data = {"input_ids": [], "attention_mask": []}
    supplementary_data = []

    for row in df.iter_rows(named=True):
        tokenized, suppl_data = preprocess_input(
            row["smiles"],
            tokenizer,
            row["adduct"],
            row["collision_energy"],
            row["instrument_type"],
        )
        tokenized_data["input_ids"].append(tokenized["input_ids"][0])
        tokenized_data["attention_mask"].append(tokenized["attention_mask"][0])
        supplementary_data.append(suppl_data)

    tokenized_data["input_ids"] = torch.stack(tokenized_data["input_ids"])
    tokenized_data["attention_mask"] = torch.stack(tokenized_data["attention_mask"])
    supplementary_data = torch.stack(supplementary_data)

    return tokenized_data, supplementary_data


def process_model_output(output):
    # Assuming the output is a tensor of shape (batch_size, max_fragments, 2)
    # where [:, :, 0] represents m/z values and [:, :, 1] represents intensities
    pred_mzs = output[:, :, 0]
    pred_intensities = output[:, :, 1]

    # Ensure non-negative values for m/z and intensities
    pred_mzs = torch.clamp(pred_mzs, min=0)
    pred_intensities = torch.clamp(pred_intensities, min=0)

    # Normalize intensities
    pred_intensities = (
        pred_intensities / torch.max(pred_intensities, dim=1, keepdim=True).values
    )

    return pred_mzs, pred_intensities


def infer(model, tokenizer, input_data):
    tokenized_data, supplementary_data = process_input_data(input_data, tokenizer)

    with torch.no_grad():
        outputs = model(
            input_ids=tokenized_data["input_ids"],
            attention_mask=tokenized_data["attention_mask"],
            supplementary_data=supplementary_data,
        )

    pred_mzs, pred_intensities = process_model_output(outputs)
    return pred_mzs, pred_intensities


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


if __name__ == "__main__":
    main()
