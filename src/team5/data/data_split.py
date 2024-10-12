# test/train split
# Use Murcko scaffold and spectral entropy splitting for this, rather than random.
# This will ensure that similar molecules don't go into both training and test,
# causing cross contamination and over fitting.

import polars as pl


def sort_dataframe_by_scaffold(df: pl.DataFrame) -> pl.DataFrame:
    """Sort the DataFrame by the frequency of scaffold_smiles."""

    # Count the occurrences of each scaffold_smiles
    scaffold_freq = df.group_by("scaffold_smiles").agg(
        pl.count("scaffold_smiles").alias("frequency")
    )

    # Join the frequency counts back to the original DataFrame
    df_with_freq = df.join(scaffold_freq, on="scaffold_smiles")

    # Sort the DataFrame by frequency (descending order) and scaffold_smiles (ascending as secondary sort)
    df_sorted = df_with_freq.sort(
        ["frequency", "scaffold_smiles"], descending=[True, False]
    )

    # Return only the original columns (drop the 'frequency' column)
    return df_sorted.drop("frequency")


def split_dataframe(
    df: pl.DataFrame, split_ratio: float = 0.9
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split the DataFrame into two parts based on the given ratio."""
    split_point = int(len(df) * split_ratio)
    df_train = df[:split_point]
    df_test = df[split_point:]
    return df_train, df_test
