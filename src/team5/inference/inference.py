import os
from pathlib import Path
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM,  logging as transformers_logging
from safetensors.torch import load_file
from src.team5.models.custom_model import CustomChemBERTaModel
from src.team5.training.config import BASE_MODEL, MAX_FRAGMENTS, MAX_SEQ_LENGTH, SUPPLEMENTARY_DATA_DIM
from src.team5.training.lora_config import create_peft_model
import json
import numpy as np


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
    with open(TRACKER_FILE, 'r') as f:
        return json.load(f)

def get_latest_checkpoint():
    tracker = load_checkpoint_tracker()
    latest_version = tracker['latest']
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
        max_mz=2000,
        delta_mz=1,
        intensity_power=0.5
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

def preprocess_input(smiles, mzs, intensities, tokenizer):
    # Tokenize SMILES
    tokenized_smiles = tokenizer(smiles, padding='max_length', truncation=True, max_length=MAX_SEQ_LENGTH, return_tensors='pt')
    
    # Prepare supplementary data (mzs and intensities)
    supplementary_data = torch.zeros((MAX_FRAGMENTS, 2))
    for i in range(min(len(mzs), MAX_FRAGMENTS)):
        supplementary_data[i, 0] = mzs[i]
        supplementary_data[i, 1] = intensities[i]
    
    print(f"Initial supplementary_data shape: {supplementary_data.shape}")
    
    # Reshape supplementary data to match the expected input shape
    supplementary_data = supplementary_data.view(-1)  # Flatten the tensor
    print(f"Flattened supplementary_data shape: {supplementary_data.shape}")
    
    # Adjust the size to match SUPPLEMENTARY_DATA_DIM
    if supplementary_data.size(0) < SUPPLEMENTARY_DATA_DIM:
        supplementary_data = torch.nn.functional.pad(supplementary_data, (0, SUPPLEMENTARY_DATA_DIM - supplementary_data.size(0)))
    elif supplementary_data.size(0) > SUPPLEMENTARY_DATA_DIM:
        supplementary_data = supplementary_data[:SUPPLEMENTARY_DATA_DIM]
    
    print(f"Final supplementary_data shape: {supplementary_data.shape}")
    print(f"SUPPLEMENTARY_DATA_DIM: {SUPPLEMENTARY_DATA_DIM}")
    
    return tokenized_smiles, supplementary_data

def process_predictions(pred_mzs, pred_intensities, top_k=10, intensity_threshold=0.01):
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
    
    # Take top-k peaks
    mzs = mzs[:top_k]
    intensities = intensities[:top_k]
    
    return mzs, intensities

def infer(model, tokenizer, smiles, mzs, intensities):
    tokenized_smiles, supplementary_data = preprocess_input(smiles, mzs, intensities, tokenizer)
    
    print(f"Input shapes:")
    print(f"  input_ids: {tokenized_smiles['input_ids'].shape}")
    print(f"  attention_mask: {tokenized_smiles['attention_mask'].shape}")
    print(f"  supplementary_data: {supplementary_data.shape}")
    
    with torch.no_grad():
        outputs = model(
            input_ids=tokenized_smiles['input_ids'],
            attention_mask=tokenized_smiles['attention_mask'],
            supplementary_data=supplementary_data.unsqueeze(0)  # Add batch dimension
        )
    
    # Process outputs
    pred_mzs, pred_intensities = model.base_model.process_predicted_output(outputs)
    return process_predictions(pred_mzs[0], pred_intensities[0])

def main():
    parser = argparse.ArgumentParser(description='Run inference on the trained model')
    parser.add_argument('--version', type=str, help='Model version to use (default: latest)')
    parser.add_argument('--smiles', type=str, required=True, help='SMILES string for inference')
    parser.add_argument('--mzs', type=float, nargs='+', required=True, help='List of m/z values')
    parser.add_argument('--intensities', type=float, nargs='+', required=True, help='List of intensity values')
    
    args = parser.parse_args()
    
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Checkpoint directory: {CHECKPOINT_DIR}")
    print(f"Tracker file: {TRACKER_FILE}")
    print(f"SUPPLEMENTARY_DATA_DIM: {SUPPLEMENTARY_DATA_DIM}")
    
    # Load the model
    model = load_model(args.version)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    
    # Run inference
    pred_mzs, pred_intensities = infer(model, tokenizer, args.smiles, args.mzs, args.intensities)
    
    print("\nPredicted Mass Spectrum:")
    print("------------------------")
    print("m/z\t\tIntensity")
    print("------------------------")
    for mz, intensity in zip(pred_mzs, pred_intensities):
        print(f"{mz:.4f}\t\t{intensity:.4f}")

if __name__ == "__main__":
    main()