import polars as pl
from data_split import sort_dataframe_by_scaffold, split_dataframe
from prepare import tensorize

if __name__ == "__main__":
    print(f"Loading the data!.")

    parquet_file = "data/raw/enveda_library_subset_10percent.parquet"
    df = pl.read_parquet(parquet_file)

    df_sorted = sort_dataframe_by_scaffold(df)

    df_train, df_test = split_dataframe(df_sorted, 0.9)

    (tokenized_smiles, labels, supplementary_data) = tensorize(df_train, len(df_train))

    print("tokenized_smiles: ", tokenized_smiles)
    print("labels: ", labels)
    print("supplementary_data: ", supplementary_data)
