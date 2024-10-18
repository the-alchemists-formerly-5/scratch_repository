import os

DATASET = os.getenv("DATASET", "../data/raw/enveda_library_subset_10percent.parquet")
BASE_MODEL = "seyonec/ChemBERTa-zinc-base-v1"
MAX_FRAGMENTS = 512  # from anton, max number of mzs/intensities
MAX_SEQ_LENGTH = 512  # base model max seq length
SUPPLEMENTARY_DATA_DIM = 75
ENABLE_PROFILING = False  # If turned on, will profile the training
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))  # Note: if using CUDA, it'll automatically find the optimal batch size
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 3))

# Set WANDB_API_KEY environment variable to enable logging to wandb
os.environ["TOKENIZERS_PARALLELISM"] = "true"