# The Alchemists 🧪

Welcome to The Alchemists' project repository for the Lux Capital Bio x ML Hackathon!

## Project Overview

This repository contains our work on translating a Jupyter notebook into a set of Python modules that can be called from scripts. We're exploring the intersection of biology and machine learning to create innovative solutions.

## Running Inference with Docker

This guide explains how to use the Docker image to run inference on your data.

### Prerequisites

- Docker installed on your system
- Input data in Parquet format
- The inference Docker image built (let's assume it's named `ghiret/hackathon_bio:inference_image-latest`)

### Steps to Run Inference

1. **Prepare Your Data**

   Ensure your input Parquet file is in a directory on your local machine. For example, let's say your data is in `/path/to/your/data` and the input file is named `input.parquet`.

2. **Run the Docker Container**

   Use the following command to run the inference:

   ```bash
   docker run -v /path/to/your/data:/data ghiret/hackathon_bio:inference_image-latest --input_file /data/input.parquet --output_file /data/output.parquet
   ```

   This command does the following:
   - `-v /path/to/your/data:/data` mounts your local data directory to `/data` inside the container.
   - `ghiret/hackathon_bio:inference_image-latest` is the name of your Docker image.
   - `--input_file /data/input.parquet` specifies the input file (inside the container).
   - `--output_file /data/output.parquet` specifies where to save the output file (inside the container).

3. **Additional Options**

   You can add more options to customize the inference:

   ```bash
   docker run -v /path/to/your/data:/data ghiret/hackathon_bio:inference_image-latest \
     --input_file /data/input.parquet \
     --output_file /data/output.parquet \
     --version your_model_version \
     --top_k 30
   ```

   Replace `your_model_version` with the specific model version you want to use, and adjust `top_k` as needed.

4. **Accessing the Results**

   After the inference is complete, you'll find the `output.parquet` file in your local `/path/to/your/data` directory.

### Troubleshooting

- If you encounter permission issues, you may need to run the Docker command with `sudo`.
- Ensure that your local data directory has the correct read/write permissions.
- If the inference script seems to hang, check if there are any error messages in the Docker logs:
  ```bash
  docker logs $(docker ps -q -n 1)
  ```

### Note

The current setup runs the inference once and then exits the container. If you need to run multiple inferences or inspect the container after running, you may need to modify the Dockerfile or use Docker's exec command to interact with the container after it's started.

For any issues or further customization, please refer to the project's issue tracker or contact the maintainers.

## Repository Structure

- `docs/`: Contains project documentation
  - [About the Data](docs/about_the_data.md): Information about the dataset
  - [Using RunPod](docs/using_runpod.md): Instructions for using RunPod
  - [Virtual Environment Instructions](docs/virtual-env-instructions.md): Guide for setting up a virtual environment
- `data/`: Directory for storing project data
  - `raw/`: Subdirectory for raw data files
    - `enveda_library_subset.parquet`: The original dataset file (to be downloaded)

## Getting Started

1. Clone this repository to your local machine.
2. Set up a virtual environment using Poetry. Instructions can be found [here](docs/virtual-env-instructions.md).
3. Download the original data file to `data/raw/enveda_library_subset.parquet`.

## Data

For information about the dataset used in this project, please refer to the [data documentation](docs/about_the_data.md).

## Timeline

- October 2-9, 2023: Pre-hackathon exploration and setup
- October 10-20, 2023: Official hackathon - building and submitting projects
- October 20, 2023, 2-2:30pm PT: Awards ceremony

## Resources

- [RunPod](https://www.runpod.io/)
- [Modal](https://modal.com/)
- [Together AI](https://www.together.ai/)
- [Google Colab Tutorial](https://colab.research.google.com/github/deepchem/deepchem/blob/master/examples/tutorials/Transfer_Learning_With_ChemBERTa_Transformers.ipynb#scrollTo=teDLOtldQd2K)

For more detailed information about the project and its documentation, please check the [docs README](docs/README.md).

Good luck, and happy hacking! 🧪✨
