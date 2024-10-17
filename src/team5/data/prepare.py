from __future__ import annotations

import re
from functools import partial
from pathlib import Path

import polars as pl
import torch
from transformers import AutoTokenizer

HEAD = 1000
MAX_MZS = 512


ADDUCT = {
    "[2M+C2H4N]+": 0,
    "[2M+CHO2]-": 1,
    "[2M+Ca-H]+": 2,
    "[2M+Ca]2+": 3,
    "[2M+H4N]+": 4,
    "[2M+H]+": 5,
    "[2M+K]+": 6,
    "[2M+Na-H2]-": 7,
    "[2M+Na2-H]+": 8,
    "[2M+Na]+": 9,
    "[2M-H3O2]+": 10,
    "[2M-HO]+": 11,
    "[2M-H]-": 12,
    "[3M+Ca-H]+": 13,
    "[3M+Ca]2+": 14,
    "[3M+H4N]+": 15,
    "[3M+K]+": 16,
    "[3M+Na]+": 17,
    "[4M+Ca]2+": 18,
    "[5M+Ca]2+": 19,
    "[Anion]-": 20,
    "[Cat]+": 21,
    "[M+C2H2NaO2]-": 22,
    "[M+C2H3NNa]+": 23,
    "[M+C2H3O2]-": 24,
    "[M+C2H4N]+": 25,
    "[M+C2H7N2]+": 26,
    "[M+C2H7OS]+": 27,
    "[M+CH3O2]+": 28,
    "[M+CH5O]+": 29,
    "[M+CHO2]-": 30,
    "[M+Ca-H]+": 31,
    "[M+Ca]2+": 32,
    "[M+Cl]-": 33,
    "[M+H-O3S]+": 34,
    "[M+H2]2+": 35,
    "[M+H3]3+": 36,
    "[M+H4N]+": 37,
    "[M+HNa]2+": 38,
    "[M+HO]+": 39,
    "[M+H]+": 40,
    "[M+K-H2]-": 41,
    "[M+K]+": 42,
    "[M+Li-H]+": 43,
    "[M+Li]+": 44,
    "[M+N-O2]+": 45,
    "[M+Na-H2]-": 46,
    "[M+Na-H]+": 47,
    "[M+Na2-H]+": 48,
    "[M+Na]+": 49,
    "[M-C12H19O9]+": 50,
    "[M-C2H3O]-": 51,
    "[M-C6H11O5]-": 52,
    "[M-C6H9O5]+": 53,
    "[M-C8H9O]+": 54,
    "[M-C9H9O5]+": 55,
    "[M-CH3F2O2]-": 56,
    "[M-CH3O]+": 57,
    "[M-CH3]-": 58,
    "[M-CHO2]-": 59,
    "[M-H2N]+": 60,
    "[M-H2O2]2+": 61,
    "[M-H2]-": 62,
    "[M-H2]2-": 63,
    "[M-H3O2]+": 64,
    "[M-H3O]-": 65,
    "[M-H4O3]2+": 66,
    "[M-H5O3]+": 67,
    "[M-H7O4]+": 68,
    "[M-H9O5]+": 69,
    "[M-HO]+": 70,
    "[M-H]-": 71,
}

RE_QTOF = re.compile(r"tof|synapt|impact", re.IGNORECASE)
RE_ORBI = re.compile(r"orbitrap|exactive|qehf", re.IGNORECASE)
RE_FTMS = re.compile(r"tf|ion trap|qit", re.IGNORECASE)
RE_QQQQ = re.compile(r"qq", re.IGNORECASE)

PREPARED_PARQUET = "prepared"


def _enumerize(parquet: Path, column: str) -> dict[str, int]:
    values = dict(
        enumerate(
            pl.scan_parquet(parquet).select(column).unique().collect()[column].sort()
        )
    )
    return {v: k for k, v in values.items()}


def pad(series: pl.Series, max_items: int) -> list[float]:
    """Pads the series with zeros up to the max_items length."""
    if series is None:
        return [0.0] * max_items
    items = series if isinstance(series, list) else series.to_list()
    padding = [0.0] * (max_items - len(items))
    return items + padding


def vectorize(lookup: dict[str, int], elem: str) -> list[int]:
    """One-hot encodes an element based on the lookup dictionary."""
    vec = [0] * len(lookup)
    vec[lookup[elem]] = 1
    return vec


def tokenize_smiles(df: pl.DataFrame, tokenizer: AutoTokenizer) -> pl.DataFrame:
    """Tokenizes the SMILES column and adds attention masks."""
    return df.with_columns(
        tokenized_smiles=pl.col("smiles").map_elements(
            partial(tokenizer.encode, padding="max_length"),
            return_dtype=pl.List(pl.Int64),
        ),
        attention_mask=pl.col("smiles").map_elements(
            lambda seq: tokenizer(seq, padding="max_length")["attention_mask"],
            return_dtype=pl.List(pl.Int64),
        ),
    )


def instrument(instr: str) -> list[int]:
    """Groups instruments into predefined categories."""
    if not instr:
        return [0, 0, 0, 0, 0]

    if instr.startswith("cfm-predict"):
        return [1, 0, 0, 0, 0]
    if RE_QTOF.search(instr):
        return [0, 1, 0, 0, 0]
    if RE_ORBI.search(instr):
        return [0, 0, 1, 0, 0]
    if RE_FTMS.search(instr):
        return [0, 0, 0, 1, 0]
    if RE_QQQQ.search(instr):
        return [0, 0, 0, 0, 1]

    return [0, 0, 0, 0, 0]


def prepare_data(df: pl.DataFrame, max_mzs: int = MAX_MZS, filename: str = None) -> pl.DataFrame:
    """Prepares the dataframe by tokenizing SMILES,
    padding mzs, and processing other columns."""
    tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

    # Apply transformations
    df_prepared = df.with_columns(
        # Padding mzs and intensities
        padded_mzs=pl.col("mzs").map_elements(
            partial(pad, max_items=max_mzs),
            return_dtype=pl.List(pl.Float64),
            skip_nulls=False,
        ),
        padded_intensities=pl.col("intensities").map_elements(
            partial(pad, max_items=max_mzs),
            return_dtype=pl.List(pl.Float64),
            skip_nulls=False,
        ),
        # Enum adduct
        enum_adduct=pl.col("adduct").map_elements(
            partial(vectorize, ADDUCT),
            return_dtype=pl.List(pl.Int64),
            skip_nulls=False,
        ),
        # Collision energy extraction (fill missing values here)
        ev=pl.col("collision_energy")
        .str.extract_all(r"\d+\.?\d*")
        .list.last()
        .cast(pl.Float64)
        .fill_null(0),
        # Enum in_silico (fill missing values here)
        enum_in_silico=pl.col("in_silico").cast(pl.Int64).fill_null(0),
        # Instrument type grouping
        enum_instrument=pl.col("instrument_type").map_elements(
            instrument,
            return_dtype=pl.List(pl.Int64),
            skip_nulls=False,
        ),
    )

    # Tokenize SMILES and add attention masks
    df_prepared = tokenize_smiles(df_prepared, tokenizer)

    # Combine supplementary data into a single column
    df_prepared = df_prepared.with_columns(
        supplementary_data=pl.concat_list(
            "precursor_mz",
            "precursor_charge",
            "ev",
            "enum_in_silico",
            "enum_instrument",
            "enum_adduct",
        )
    )

    # Fill any remaining missing values in supplementary_data with a default (e.g., 0)
    df_prepared = df_prepared.fill_null(0)

    # Select final columns to return
    df_prepared = df_prepared.select(
        [
            "tokenized_smiles",
            "attention_mask",
            "padded_mzs",
            "padded_intensities",
            "supplementary_data",
        ]
    )

    # If a filename is provided, save the prepared data as a Parquet file
    if filename:
        df_prepared.sink_parquet(str(filename))
    
    return df_prepared


def tensorize(
    df: pl.DataFrame, head: int = 0, split: str = "train", path = PREPARED_PARQUET
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Converts the prepared dataframe into PyTorch tensors for training/testing.

    Args:
        df: Prepared Polars DataFrame.
        head: Number of rows to process (if limited for debugging).
        split: The dataset split ("train" or "test").

    Returns:
        Tuple of tensors: tokenized_smiles, attention_mask, labels, supplementary_data.
    """

    # Define the filename for the prepared Parquet file
    filename = Path(f"{PREPARED_PARQUET}_{split}.parquet")

    # Check if the Parquet file already exists
    if filename.exists():
        print(f"Loading prepared data from {filename}")
        df = pl.read_parquet(filename)
    else:
        print(f"Preparing data for {split} split")
        df = prepare_data(df, filename=filename)

    # Optionally limit the dataset size for debugging
    if head > 0:
        df = df.head(head)

    # Concatenate mzs and intensities along the last dimension
    labels = torch.cat(
        (
            torch.tensor(df["padded_mzs"].to_list(), dtype=torch.float).unsqueeze(-1),
            torch.tensor(
                df["padded_intensities"].to_list(), dtype=torch.float
            ).unsqueeze(-1),
        ),
        dim=-1,
    )  # Shape: (batch_size, max_fragments, 2)

    # Return the tensors for model consumption
    return (
        torch.tensor(df["tokenized_smiles"].to_list(), dtype=torch.long),
        torch.tensor(df["attention_mask"].to_list(), dtype=torch.long),
        labels,  # Single tensor with mzs and intensities
        torch.tensor(df["supplementary_data"].to_list(), dtype=torch.float),
    )


# ---------------------- TESTS START HERE -------------------------------


def sample_df():
    """Create a sample dataframe for testing."""
    data = {
        "smiles": ["CCO", "CCC", "CCN"],
        "mzs": [[100.0, 200.0], [150.0, 250.0], [180.0]],
        "intensities": [[0.1, 0.2], [0.3, 0.4], [0.5]],
        "precursor_mz": [100.0, 150.0, 180.0],
        "precursor_charge": [1.0, 1.0, 1.0],
        "collision_energy": ["20", "25", None],
        "in_silico": [1, 1, 1],
        "instrument_type": ["TOF", "Orbitrap", "FTMS"],
        "adduct": ["[M+H]+", "[M+Na]+", "[M+K]+"],
    }
    return pl.DataFrame(data)


def test_prepare_data():
    """Test the prepare_data function."""
    df = sample_df()
    prepared_df = prepare_data(df)

    # Check if the necessary columns are present
    required_columns = {
        "tokenized_smiles",
        "attention_mask",
        "padded_mzs",
        "padded_intensities",
        "supplementary_data",
    }
    if set(prepared_df.columns) == required_columns:
        print("test_prepare_data: PASS (columns present)")
    else:
        print(
            f"test_prepare_data: FAIL (missing columns "
            f"{required_columns - set(prepared_df.columns)})"
        )

    # Ensure no missing values in the supplementary_data column
    if prepared_df["supplementary_data"].null_count() == 0:
        print("test_prepare_data: PASS (no missing values in supplementary_data)")
    else:
        print("test_prepare_data: FAIL (missing values in supplementary_data)")


def test_tensorize():
    """Test the tensorize function."""
    df = sample_df()
    prepared_df = prepare_data(df)
    try:
        tokenized_smiles, attention_mask, labels, supplementary_data = tensorize(
            prepared_df
        )

        # Check tensor types
        if (
            isinstance(tokenized_smiles, torch.Tensor)
            and isinstance(attention_mask, torch.Tensor)
            and isinstance(labels, torch.Tensor)
            and isinstance(supplementary_data, torch.Tensor)
        ):
            print("test_tensorize: PASS (tensor types are correct)")
        else:
            print("test_tensorize: FAIL (incorrect tensor types)")

        # Check tensor shapes
        if (
            tokenized_smiles.shape[0] == df.shape[0]
            and attention_mask.shape[0] == df.shape[0]
            and labels.shape[0] == df.shape[0]
            and supplementary_data.shape[0] == df.shape[0]
        ):
            print("test_tensorize: PASS (tensor shapes match batch size)")
        else:
            print("test_tensorize: FAIL (tensor shapes do not match batch size)")

    except Exception as e:
        print(f"test_tensorize: FAIL (error occurred: {e})")


if __name__ == "__main__":
    # Run the tests
    test_prepare_data()
    test_tensorize()

    # List of parquet chunk files
    chunk_files = [f"~/Downloads/chunk_{i+1}.parquet" for i in range(1)]

    # Read and concatenate all parquet files
    df = pl.concat([pl.read_parquet(file) for file in chunk_files])

    from src.team5.data.data_split import (sort_dataframe_by_scaffold,
                                           split_dataframe)

    # Sort by scaffold
    df_sorted = sort_dataframe_by_scaffold(df)

    # Split the dataframe into train and test
    df_train, df_test = split_dataframe(df_sorted, split_ratio=0.9)

    # Prepare the training and testing data
    # (this step creates 'padded_mzs' and other columns)
    df_train_prepared = prepare_data(df_train)
    df_test_prepared = prepare_data(df_test)

    # Check column names to ensure 'padded_mzs' is included
    print(df_train_prepared.columns)
    print(df_test_prepared.columns)

    # Run tensorization on prepared data
    (
        train_tokenized_smiles,
        train_attention_mask,
        train_labels,
        train_supplementary_data,
    ) = tensorize(df_train_prepared, split="train")
    print(
        "train_tokenized_smiles",
        train_tokenized_smiles.shape,
        train_tokenized_smiles.dtype,
    )
    print(
        "train_attention_mask", train_attention_mask.shape, train_attention_mask.dtype
    )
    print("train_labels", train_labels.shape, train_labels.dtype)
    print(
        "train_supplementary_data",
        train_supplementary_data.shape,
        train_supplementary_data.dtype,
    )
