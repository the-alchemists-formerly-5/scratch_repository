# test/train split
# Use Murcko scaffold and spectral entropy splitting for this, rather than random.
# This will ensure that similar molecules don't go into both training and test,
# causing cross contamination and over fitting.

import polars as pl


def sort_dataframe_by_scaffold(df: pl.DataFrame) -> pl.DataFrame:
    """Process the DataFrame to calculate scaffolds and sort."""
    # Sort the DataFrame by scaffold_smiles column
    df_sorted = df.sort("scaffold_smiles", descending=False)
    return df_sorted


def split_dataframe(
    df: pl.DataFrame, split_ratio: float = 0.9
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split the DataFrame into two parts based on the given ratio."""
    split_point = int(len(df) * split_ratio)
    df_train = df[:split_point]
    df_test = df[split_point:]
    return df_train, df_test
