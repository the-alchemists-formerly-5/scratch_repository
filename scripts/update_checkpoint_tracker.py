import argparse
from datetime import datetime
import json
from pathlib import Path

CHECKPOINT_DIR = Path("/app/checkpoints")
LOCAL_CHECKPOINT_DIR = Path("../checkpoints")


def update_tracker(version, description, accuracy, f1_score, set_as_latest=False):

    # set checkpoint_dir to be the local_checkpoint_dir, but relative to this script file
    checkpoint_dir = Path(__file__).parent / LOCAL_CHECKPOINT_DIR
    if not checkpoint_dir.exists():
        checkpoint_dir = CHECKPOINT_DIR

    tracker_file = checkpoint_dir / "checkpoint_tracker.json"

    if tracker_file.exists():
        with open(tracker_file, "r") as f:
            tracker = json.load(f)
    else:
        tracker = {"latest": None, "checkpoints": []}

    new_checkpoint = {
        "version": version,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "description": description,
        "performance": {"accuracy": accuracy, "f1_score": f1_score},
    }

    # Update or add the new checkpoint
    for i, checkpoint in enumerate(tracker["checkpoints"]):
        if checkpoint["version"] == version:
            tracker["checkpoints"][i] = new_checkpoint
            break
    else:
        tracker["checkpoints"].append(new_checkpoint)

    # Set as latest if specified
    if set_as_latest:
        tracker["latest"] = version

    # Write updated tracker back to file
    with open(tracker_file, "w") as f:
        json.dump(tracker, f, indent=2)

    print(f"Updated checkpoint tracker with version {version}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update the checkpoint tracker")
    parser.add_argument(
        "--version", type=str, required=True, help="Version of the new checkpoint"
    )
    parser.add_argument(
        "--description",
        type=str,
        required=True,
        help="Description of the new checkpoint",
    )
    parser.add_argument(
        "--accuracy", type=float, required=True, help="Accuracy of the new model"
    )
    parser.add_argument(
        "--f1_score", type=float, required=True, help="F1 score of the new model"
    )
    parser.add_argument(
        "--set_as_latest", action="store_true", help="Set this version as the latest"
    )

    args = parser.parse_args()

    update_tracker(
        args.version, args.description, args.accuracy, args.f1_score, args.set_as_latest
    )
