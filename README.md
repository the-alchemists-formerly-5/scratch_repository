# scratch_repository

In this repository I will be trying to translate the jupyter notebook to a set of python modules that can be called from a few scripts.

Information on how to set-up a virtualenv using poetry can be found [here](docs/virtual-env-instructions.md)

# Data

The original data should be downloaded to 'data/raw/enveda_library_subset.parquet'

See the data information [here](docs/about_the_data.md)

# Splitting small subset of data for experimentation

As the full data is a big file, for experimentation we can split it into a 10% subset using the following script

```bash
python3 scripts/extract_subset.py
```

This will generate the file 'data/raw/enveda_library_subset_10percent.parquet'
