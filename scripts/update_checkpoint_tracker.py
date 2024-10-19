import argparse
import json
from pathlib import Path

CHECKPOINT_DIR = Path("/app/checkpoints")
TRACKER_FILE = CHECKPOINT_DIR / "checkpoint_tracker.json"


def update_tracker(version, description, accuracy, f1_score, set_as_latest=False):
    if TRACKER_FILE.exists():
        with open(TRACKER_FILE, "r") as f:
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
    with open(TRACKER_FILE, "w") as f:
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
