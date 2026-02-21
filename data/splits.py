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


def _validate_ratios(train_ratio, val_ratio, test_ratio):
    ratios = {
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio),
        "test_ratio": float(test_ratio),
    }
    for name, value in ratios.items():
        if value < 0.0:
            raise ValueError(f"{name} must be >= 0, got {value}")
    total = ratios["train_ratio"] + ratios["val_ratio"] + ratios["test_ratio"]
    if total <= 0.0:
        raise ValueError("At least one split ratio must be > 0")
    return ratios, total


def _compute_split_sizes(total, train_ratio, val_ratio, test_ratio):
    ratios, ratio_sum = _validate_ratios(train_ratio, val_ratio, test_ratio)
    keys = ["train", "val", "test"]
    normalized = [ratios["train_ratio"] / ratio_sum, ratios["val_ratio"] / ratio_sum, ratios["test_ratio"] / ratio_sum]

    raw_sizes = [total * r for r in normalized]
    sizes = [int(v) for v in raw_sizes]
    remainder = total - sum(sizes)

    # Distribute leftovers to the largest fractional parts.
    fractions = sorted(
        ((raw_sizes[i] - sizes[i], i) for i in range(3)),
        key=lambda x: x[0],
        reverse=True,
    )
    for _, idx in fractions:
        if remainder <= 0:
            break
        sizes[idx] += 1
        remainder -= 1

    # Ensure non-zero splits for positive ratios when possible.
    positive_idxs = [i for i, r in enumerate(normalized) if r > 0.0]
    if total >= len(positive_idxs):
        need = [i for i in positive_idxs if sizes[i] == 0]
        for idx in need:
            sizes[idx] = 1
        deficit = len(need)
        while deficit > 0:
            # Borrow one sample from the largest split that still has >1 sample.
            candidates = sorted(
                [(sizes[i], i) for i in range(len(sizes)) if sizes[i] > 1],
                reverse=True,
            )
            if not candidates:
                break
            _, borrow_idx = candidates[0]
            sizes[borrow_idx] -= 1
            deficit -= 1

    return dict(zip(keys, sizes))


def generate_split(
    data_root,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42,
    save=True,
):
    """
    Generate reproducible split for any dataset.

    Args:
        data_root: folder containing .mat files
        train_ratio: training portion
        val_ratio: validation portion
        test_ratio: testing portion
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
    rng = random.Random(seed)
    rng.shuffle(files)

    # 3️⃣ Split
    total = len(files)
    split_sizes = _compute_split_sizes(total, train_ratio, val_ratio, test_ratio)
    train_size = split_sizes["train"]
    val_size = split_sizes["val"]
    test_size = split_sizes["test"]

    train = files[:train_size]
    val = files[train_size:train_size + val_size]
    test = files[train_size + val_size:train_size + val_size + test_size]

    split_dict = {
        "seed": seed,
        "total": total,
        "ratios": {
            "train": float(train_ratio),
            "val": float(val_ratio),
            "test": float(test_ratio),
        },
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
