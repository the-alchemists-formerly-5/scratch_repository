import polars as pl
import torch
from transformers import AutoTokenizer

from src.team5.data.prepare import prepare_data, tensorize
from src.team5.training.config import (BASE_MODEL, MAX_FRAGMENTS,
                                       MAX_SEQ_LENGTH, SUPPLEMENTARY_DATA_DIM)


def process_input_data(df, tokenizer):
    # Use prepare_data function from prepare.py
    prepared_df = prepare_data(df, max_mzs=MAX_FRAGMENTS)

    # Use tensorize function from prepare.py
    tokenized_smiles, attention_mask, _, supplementary_data = tensorize(
        prepared_df, split="inference"
    )

    return {
        "input_ids": tokenized_smiles,
        "attention_mask": attention_mask,
    }, supplementary_data


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
