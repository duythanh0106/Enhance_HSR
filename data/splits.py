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
import scipy.io as sio

try:
    import h5py
except ImportError:  # pragma: no cover - optional dependency at runtime
    h5py = None


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


def is_hyperspectral_mat(path):
    """
    Fast check whether a .mat file contains at least one 3D numeric array.
    Uses scipy.io.whosmat metadata to avoid loading large arrays into memory.
    """
    preferred_keys = {"rad", "cube", "ref", "data", "img"}

    # 1) Try standard MATLAB formats via scipy metadata.
    try:
        variables = sio.whosmat(path)
        if variables:
            for name, shape, _ in variables:
                if name in preferred_keys and len(shape) == 3:
                    return True
            for _, shape, _ in variables:
                if len(shape) == 3:
                    return True
    except Exception:
        pass

    # 2) Fallback for MATLAB v7.3 (HDF5-based) files.
    if h5py is not None:
        try:
            with h5py.File(path, "r") as f:
                # Check preferred keys first.
                for key in preferred_keys:
                    if key in f and isinstance(f[key], h5py.Dataset) and len(f[key].shape) == 3:
                        return True

                found_3d = False

                def visitor(_, obj):
                    nonlocal found_3d
                    if found_3d:
                        return
                    if isinstance(obj, h5py.Dataset) and len(obj.shape) == 3:
                        found_3d = True

                f.visititems(visitor)
                if found_3d:
                    return True
        except Exception:
            pass

    return False


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

    valid_files = [f for f in files if is_hyperspectral_mat(f)]
    valid_set = set(valid_files)
    skipped_files = [f for f in files if f not in valid_set]
    if skipped_files:
        print(f"⚠️ Skipping {len(skipped_files)} non-hyperspectral .mat file(s).")
        for skipped in skipped_files[:5]:
            print(f"   - {os.path.basename(skipped)}")
        if len(skipped_files) > 5:
            print(f"   ... and {len(skipped_files) - 5} more")

    if len(valid_files) == 0:
        raise ValueError(
            f"No valid hyperspectral .mat files found in {data_root}. "
            "Expected at least one file containing a 3D hyperspectral cube."
        )

    # 2️⃣ Shuffle reproducibly
    rng = random.Random(seed)
    rng.shuffle(valid_files)

    # 3️⃣ Split
    total = len(valid_files)
    split_sizes = _compute_split_sizes(total, train_ratio, val_ratio, test_ratio)
    train_size = split_sizes["train"]
    val_size = split_sizes["val"]
    test_size = split_sizes["test"]

    train = valid_files[:train_size]
    val = valid_files[train_size:train_size + val_size]
    test = valid_files[train_size + val_size:train_size + val_size + test_size]

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
