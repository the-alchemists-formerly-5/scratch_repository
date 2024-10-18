from pathlib import Path

import polars as pl

from src.team5.data.data_loader import SMILESDataset
from src.team5.data.data_split import (sort_dataframe_by_scaffold,
                                       split_dataframe)
from src.team5.data.prepare import prepare_data, tensorize

from .config import BATCH_SIZE, DATASET, NUM_EPOCHS


def preprocess_data():
    # List of parquet chunk files
    data_dir = Path(DATASET).parent
    chunk_files = list(data_dir.glob("chunk_*.parquet"))

    print(f"Found {len(chunk_files)} chunk files.")

    print(f"Looking for chunk files in: {data_dir}")
    print(f"Found {len(chunk_files)} chunk files.")

    if not chunk_files:
        print("No chunk files found. Checking for a single parquet file.")
        single_file = Path(DATASET)
        if single_file.exists():
            print(f"Found single parquet file: {single_file}")
            df = pl.read_parquet(single_file)
        else:
            raise FileNotFoundError(
                f"No data files found in {data_dir} or at {single_file}"
            )
    else:
        # Read and concatenate all parquet files
        df = pl.concat([pl.read_parquet(file) for file in chunk_files])

    print("Data loaded. Sample:")
    print(df.head())

    # Sort by scaffold
    df_sorted = sort_dataframe_by_scaffold(df)

    # Split the dataframe into train and test
    df_train, df_test = split_dataframe(df_sorted, split_ratio=0.9)

    # Prepare the training and testing data
    df_train_prepared = prepare_data(df_train)
    df_test_prepared = prepare_data(df_test)

    print("Data prepared. Train columns:")
    print(df_train_prepared.columns)
    print("Test columns:")
    print(df_test_prepared.columns)

    # Run tensorization on prepared data
    (
        train_tokenized_smiles,
        train_attention_mask,
        train_labels,
        train_supplementary_data,
    ) = tensorize(df_train_prepared, split="train")
    (
        test_tokenized_smiles,
        test_attention_mask,
        test_labels,
        test_supplementary_data,
    ) = tensorize(df_test_prepared, split="test")

    # Create datasets
    train_dataset = SMILESDataset(
        train_tokenized_smiles,
        train_attention_mask,
        train_labels,
        train_supplementary_data,
    )
    test_dataset = SMILESDataset(
        test_tokenized_smiles, test_attention_mask, test_labels, test_supplementary_data
    )

    # Calculate total steps
    total_steps = len(train_dataset) // BATCH_SIZE * NUM_EPOCHS

    print(f"Data preprocessing completed.")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Total training steps: {total_steps}")

    return train_dataset, test_dataset, total_steps


if __name__ == "__main__":
    train_dataset, test_dataset, total_steps = preprocess_data()
