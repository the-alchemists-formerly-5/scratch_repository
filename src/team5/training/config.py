import os
from pathlib import Path

# Get the current file's directory
current_dir = Path(__file__).parent.resolve()

# Set the path to the data directory
data_dir = current_dir.parent.parent.parent.parent / "data" / "raw"
DATASET = os.getenv("DATASET", f"{data_dir}/enveda_library_subset_10percent.parquet")
PREPARED_PARQUET = os.getenv("PREPARED_PARQUET", f"{data_dir}/prepared")  # Add this line
BASE_MODEL = "seyonec/ChemBERTa-zinc-base-v1"
MAX_FRAGMENTS = 512  # from anton, max number of mzs/intensities
MAX_SEQ_LENGTH = 512  # base model max seq length
SUPPLEMENTARY_DATA_DIM = 81
ENABLE_PROFILING = False  # If turned on, will profile the training
BATCH_SIZE = int(
    os.getenv("BATCH_SIZE", 32)
)  # Note: if using CUDA, it'll automatically find the optimal batch size
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 3))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 1e-4))

# Set WANDB_API_KEY environment variable to enable logging to wandb
os.environ["TOKENIZERS_PARALLELISM"] = "true"


print(f"DATASET path: {DATASET}")
print(f"PREPARED_PARQUET path: {PREPARED_PARQUET}")
