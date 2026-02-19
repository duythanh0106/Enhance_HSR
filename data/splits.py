"""
Universal Train/Val/Test Split
Works for ANY hyperspectral dataset (.mat files)

- No hardcoded scene names
- Fully reproducible
- Dataset-agnostic
"""

import os
import json
import glob
import random


def generate_split(
    data_root,
    train_ratio=0.8,
    val_ratio=0.1,
    seed=42,
    save=True,
):
    """
    Generate reproducible split for any dataset.

    Args:
        data_root: folder containing .mat files
        train_ratio: training portion
        val_ratio: validation portion
        seed: random seed
        save: whether to save split.json

    Returns:
        dict with train/val/test file paths
    """

    # 1️⃣ Scan files
    files = sorted(glob.glob(os.path.join(data_root, "*.mat")))

    if len(files) == 0:
        raise ValueError(f"No .mat files found in {data_root}")

    # 2️⃣ Shuffle reproducibly
    random.seed(seed)
    random.shuffle(files)

    # 3️⃣ Split
    total = len(files)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)

    train = files[:train_size]
    val = files[train_size:train_size + val_size]
    test = files[train_size + val_size:]

    split_dict = {
        "seed": seed,
        "total": total,
        "train": train,
        "val": val,
        "test": test,
    }

    # 4️⃣ Save split
    if save:
        split_path = os.path.join(data_root, "split.json")
        with open(split_path, "w") as f:
            json.dump(split_dict, f, indent=2)
        print(f"✅ Split saved to {split_path}")

    return split_dict


def load_split(data_root):
    """
    Load existing split.json
    """

    split_path = os.path.join(data_root, "split.json")

    if not os.path.exists(split_path):
        raise ValueError(
            "split.json not found. Run generate_split() first."
        )

    with open(split_path, "r") as f:
        return json.load(f)


def get_split(data_root, split="train"):
    """
    Get file list for train/val/test
    """

    split_data = load_split(data_root)

    if split not in ["train", "val", "test"]:
        raise ValueError("split must be train/val/test")

    return split_data[split]
