import polars as pl
import torch

from src.team5.data.prepare import ADDUCT, instrument
from src.team5.training.config import (BASE_MODEL, MAX_FRAGMENTS,
                                       MAX_SEQ_LENGTH, SUPPLEMENTARY_DATA_DIM)


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
