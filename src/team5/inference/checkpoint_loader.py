import json
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.absolute()

# Define the checkpoint directory path
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
TRACKER_FILE = CHECKPOINT_DIR / "checkpoint_tracker.json"


def load_checkpoint_tracker():
    if not TRACKER_FILE.exists():
        raise FileNotFoundError(f"Checkpoint tracker file not found at {TRACKER_FILE}")
    with open(TRACKER_FILE, "r") as f:
        return json.load(f)


def get_latest_checkpoint():
    tracker = load_checkpoint_tracker()
    latest_version = tracker["latest"]
    return CHECKPOINT_DIR / latest_version
