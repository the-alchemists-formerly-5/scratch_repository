import argparse

import pyarrow.parquet as pq


def get_row_data(parquet_file_path, row_number):
    # Read the parquet file using PyArrow
    parquet_file = pq.ParquetFile(parquet_file_path)

    # Create an iterator for reading the file in batches
    batch_size = 1000  # Adjust this based on your system's memory constraints

    # Get the necessary columns (smiles, mzs, intensities)
    columns = ["smiles", "mzs", "intensities"]

    # Read in batches to minimize memory use
    row_data = None
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
        # Convert the batch to a pandas DataFrame
        df = batch.to_pandas()

        # Check if the target row is within this batch
        if len(df) > row_number:
            row_data = df.iloc[row_number]
            break
        else:
            row_number -= len(df)  # Adjust row number for the next batch

    return row_data


def print_row_data(row_data):
    # Print the full SMILES string
    smiles = row_data["smiles"]
    print(f"SMILES: {smiles}")

    # Print the top 10 elements of mzs
    mzs = row_data["mzs"]
    top_mzs = mzs[:10] if mzs is not None else []
    print(f"Top 10 mzs: {top_mzs}")

    # Print the top 10 elements of intensities
    intensities = row_data["intensities"]
    top_intensities = intensities[:10] if intensities is not None else []
    print(f"Top 10 intensities: {top_intensities}")


def main():
    # Set up argparse to take file path and row number as command-line arguments
    parser = argparse.ArgumentParser(
        description="Extract a specific row from a Parquet file."
    )
    parser.add_argument("file_path", type=str, help="Path to the Parquet file.")
    parser.add_argument(
        "row_number", type=int, help="The row number to extract (0-indexed)."
    )

    # Parse the arguments
    args = parser.parse_args()

    # Get the row data
    row_data = get_row_data(args.file_path, args.row_number)

    # Print the result
    if row_data is not None:
        print(f"Row {args.row_number + 1} Data:")
        print_row_data(row_data)
    else:
        print(f"Row {args.row_number + 1} not found or out of bounds.")


if __name__ == "__main__":
    main()
