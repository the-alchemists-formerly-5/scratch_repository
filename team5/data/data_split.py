# test/train split
# Use Murcko scaffold and spectral entropy splitting for this, rather than random.
# This will ensure that similar molecules don't go into both training and test,
# causing cross contamination and over fitting.


def sort_dataframe_by_scaffold(df):
    """Process the DataFrame to calculate scaffolds and sort."""
    # Assuming scaffold calculation is already done in the input DataFrame
    # If not, you'd need to add that logic here

    # Sort the DataFrame by scaffold_smiles
    df_sorted = df.sort(["scaffold_smiles"], descending=[False])

    return df_sorted


def split_dataframe(df, split_ratio=0.9):
    """Split the DataFrame into two parts based on the given ratio."""
    split_point = int(len(df) * split_ratio)
    df_train = df[:split_point]
    df_test = df[split_point:]
    return df_train, df_test
