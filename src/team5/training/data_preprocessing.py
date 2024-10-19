import os
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
    train_prepared_file = os.path.join(Path(data_dir).parent, "prepared", "train.parquet")
    test_prepared_file = os.path.join(Path(data_dir).parent, "prepared", "test.parquet")

    if os.path.exists(train_prepared_file) and os.path.exists(test_prepared_file):
        df_train_prepared = pl.read_parquet(train_prepared_file)
        df_test_prepared = pl.read_parquet(test_prepared_file)
    else:
        chunk_files = list(data_dir.glob("chunk_*.parquet"))

        if not chunk_files:
            single_file = Path(DATASET)
            if single_file.exists():
                df = pl.read_parquet(single_file)
            else:
                raise FileNotFoundError(
                    f"No data files found in {data_dir} or at {single_file}"
                )
        else:
            # Read and concatenate all parquet files
            df = pl.concat([pl.read_parquet(file) for file in chunk_files])

        # Sort by scaffold
        df_sorted = sort_dataframe_by_scaffold(df)

        df_filtered = df_sorted.filter(pl.col("mzs").list.max() <= 2000)

        # Split the dataframe into train and test
        df_train, df_test = split_dataframe(df_filtered, split_ratio=0.9)

        # Prepare the training and testing data
        df_train_prepared = prepare_data(df_train, filename=train_prepared_file)
        df_test_prepared = prepare_data(df_test, filename=test_prepared_file)
    
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

    return train_dataset, test_dataset, total_steps


if __name__ == "__main__":
    train_dataset, test_dataset, total_steps = preprocess_data()
