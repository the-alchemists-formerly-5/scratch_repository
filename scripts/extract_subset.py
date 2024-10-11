import pandas as pd
import pyarrow.parquet as pq
import numpy as np

# File paths
input_file = 'data/raw/enveda_library_subset.parquet'
output_file = 'data/raw/enveda_library_subset_10percent.parquet'

# Read the Parquet file metadata
parquet_file = pq.ParquetFile(input_file)
num_rows = parquet_file.metadata.num_rows

# Calculate the number of rows for 10% of the data
subset_size = int(num_rows * 0.1)

# Read the entire Parquet file
df = pd.read_parquet(input_file)

# Select a random subset
df_subset = df.sample(n=subset_size, random_state=42)

# Write the subset to a new Parquet file
df_subset.to_parquet(output_file, index=False)

print(f"Created subset with {len(df_subset)} rows out of {num_rows} total rows.")
print(f"Subset saved to {output_file}")