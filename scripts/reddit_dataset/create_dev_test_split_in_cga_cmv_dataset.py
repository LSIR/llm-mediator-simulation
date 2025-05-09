"""Split the 177 conversations in the CMV dataset into dev and test sets."""

import os
import random

if __name__ == "__main__":
    # Set the random seed for reproducibility
    random.seed(42)

    # Path to the CMV dataset
    cmv_dataset_path = "data/reddit/cmv/"

    # List all files in the CMV dataset directory
    all_files = os.listdir(cmv_dataset_path)

    # Filter out only the .csv files
    cmv_files = [f for f in all_files if f.endswith(".csv")]

    # Shuffle the list of files
    random.shuffle(cmv_files)

    # Start with 10 files for dev and the remaining for test
    dev_files = cmv_files[:10]
    test_files = cmv_files[10:]

    # Create directories for dev and test sets if they don't exist
    os.makedirs(os.path.join(cmv_dataset_path, "dev"), exist_ok=True)
    os.makedirs(os.path.join(cmv_dataset_path, "test"), exist_ok=True)

    # Move the files to their respective directories
    for file in dev_files:
        os.rename(
            os.path.join(cmv_dataset_path, file),
            os.path.join(cmv_dataset_path, "dev", file),
        )

    for file in test_files:
        os.rename(
            os.path.join(cmv_dataset_path, file),
            os.path.join(cmv_dataset_path, "test", file),
        )

    print(
        f"Moved {len(dev_files)} files to dev set and {len(test_files)} files to test set."
    )
