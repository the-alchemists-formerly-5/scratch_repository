# scratch_repository

In this repository I will be trying to translate the jupyter notebook to a set of python modules that can be called from a few scripts.

# Building the docker image

Please copy the extracted data as 'enveda_library_subset.parquet' into the main directory of the repo.

## Build the base image with the data:

```bash
docker build -f docker/Dockerfile.base -t base_image .
```


# Data

The original data should be downloaded to 'data/raw/enveda_library_subset.parquet'



# Splitting small subset of data for experimentation

As the full data is a big file, for experimentation we can split it into a 10% subset using the following script

```bash
python3 scripts/extract_subset.py
```

This will generate the file 'data/raw/enveda_library_subset_10percent.parquet'
