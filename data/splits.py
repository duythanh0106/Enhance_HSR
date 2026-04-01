"""
Universal Train/Val/Test Split
Works for .mat cubes and scene-folder band stacks (e.g. CAVE *_ms_XX.png)

- No hardcoded scene names
- Fully reproducible
- Dataset-agnostic
"""

import os
import json
import glob
import random
import scipy.io as sio
import numpy as np

try:
    import h5py
except ImportError:  # pragma: no cover - optional dependency at runtime
    h5py = None


def _validate_ratios(train_ratio, val_ratio, test_ratio):
    """Internal helper for `validate_ratios` operations.

    Args:
        train_ratio: Input parameter `train_ratio`.
        val_ratio: Input parameter `val_ratio`.
        test_ratio: Input parameter `test_ratio`.

    Returns:
        Any: Output produced by this function.
    """
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
    """Internal helper for `compute_split_sizes` operations.

    Args:
        total: Input parameter `total`.
        train_ratio: Input parameter `train_ratio`.
        val_ratio: Input parameter `val_ratio`.
        test_ratio: Input parameter `test_ratio`.

    Returns:
        Any: Output produced by this function.
    """
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
    """Execute `is_hyperspectral_mat`.

    Args:
        path: Input parameter `path`.

    Returns:
        Any: Output produced by this function.
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
                    """Execute `visitor`.

                    Args:
                        _: Input parameter `_`.
                        obj: Input parameter `obj`.

                    Returns:
                        None: This function returns no value.
                    """
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


def is_hyperspectral_scene_dir(path):
    """Check if directory contains a spectral band stack (e.g. *_ms_01.png)."""
    if not os.path.isdir(path):
        return False
    band_globs = (
        "*_ms_*.png",
        "*_ms_*.tif",
        "*_ms_*.tiff",
        "*_ms_*.bmp",
        "*_ms_*.jpg",
        "*_ms_*.jpeg",
    )
    for pattern in band_globs:
        if glob.glob(os.path.join(path, pattern)):
            return True
    return False


def is_hyperspectral_path(path):
    """Check if `path` is a supported hyperspectral sample path."""
    if os.path.isdir(path):
        return is_hyperspectral_scene_dir(path)
    if os.path.isfile(path) and path.lower().endswith(".mat"):
        return is_hyperspectral_mat(path)
    return False


def _dataset_key(data_root):
    """Normalize dataset root name for protocol-specific split rules."""
    return "".join(ch.lower() for ch in os.path.basename(os.path.normpath(str(data_root))) if ch.isalnum())


def _to_hwc_shape(shape):
    """Map a 3D array shape to HWC using the same heuristic as the loader."""
    if len(shape) != 3:
        raise ValueError(f"Expected 3D shape, got {shape}")
    smallest_axis = int(np.argmin(shape))
    if smallest_axis == 2:
        return tuple(int(v) for v in shape)
    if smallest_axis == 0:
        return int(shape[2]), int(shape[1]), int(shape[0])
    return int(shape[0]), int(shape[2]), int(shape[1])


def _extract_hwc_shape_from_mat_data(mat_data, path):
    """Return HWC shape for the largest 3D numeric array in a loaded MAT dict."""
    possible_keys = ["rad", "cube", "ref", "data", "img", "chikusei"]
    for key in possible_keys:
        if key in mat_data:
            value = mat_data[key]
            if isinstance(value, np.ndarray) and value.ndim == 3 and np.issubdtype(value.dtype, np.number):
                return _to_hwc_shape(value.shape)

    numeric_arrays = [
        value for value in mat_data.values()
        if isinstance(value, np.ndarray)
        and value.ndim == 3
        and np.issubdtype(value.dtype, np.number)
    ]
    if numeric_arrays:
        largest = max(numeric_arrays, key=lambda x: x.size)
        return _to_hwc_shape(largest.shape)

    raise ValueError(f"No hyperspectral cube found in {path}")


def _get_hyperspectral_hwc_shape(path):
    """Load only enough metadata/data to determine HWC shape."""
    try:
        mat_data = sio.loadmat(path)
        return _extract_hwc_shape_from_mat_data(mat_data, path)
    except NotImplementedError:
        if h5py is None:
            raise ValueError(
                f"{path} appears to be MATLAB v7.3, but h5py is not installed."
            )
        with h5py.File(path, "r") as f:
            possible_keys = ["rad", "cube", "ref", "data", "img", "chikusei"]
            for key in possible_keys:
                if key in f and isinstance(f[key], h5py.Dataset) and len(f[key].shape) == 3:
                    return _to_hwc_shape(f[key].shape)

            candidates = []

            def visitor(name, obj):
                if isinstance(obj, h5py.Dataset) and len(obj.shape) == 3:
                    candidates.append(name)

            f.visititems(visitor)
            if candidates:
                best_name = max(candidates, key=lambda n: np.prod(f[n].shape))
                return _to_hwc_shape(f[best_name].shape)

    raise ValueError(f"No hyperspectral cube shape found in {path}")


def _make_crop_entry(path, sample_id, top, left, height, width):
    """Create one split entry for a fixed crop region inside a raw HSI cube."""
    return {
        "path": path,
        "id": sample_id,
        "crop": [int(top), int(left), int(height), int(width)],
    }


def _generate_chikusei_protocol_split(sample_path, seed):
    """Paper-style split with an added validation subset from train patches."""
    height, width, _ = _get_hyperspectral_hwc_shape(sample_path)
    patch_h = 512
    patch_w = 512
    grid_h = height // patch_h
    grid_w = width // patch_w
    ids = [(row, col) for row in range(grid_h) for col in range(grid_w)]
    rng = random.Random(seed)
    shuffled = ids.copy()
    rng.shuffle(shuffled)
    test_ids = set(shuffled[:4])
    val_ids = set(shuffled[4:6])

    train = []
    val = []
    test = []
    for row, col in ids:
        entry = _make_crop_entry(
            sample_path,
            f"chikusei_r{row:02d}_c{col:02d}",
            row * patch_h,
            col * patch_w,
            patch_h,
            patch_w,
        )
        if (row, col) in test_ids:
            test.append(entry)
        elif (row, col) in val_ids:
            val.append(entry)
        else:
            train.append(entry)

    total = len(train) + len(val) + len(test)

    return {
        "seed": seed,
        "total": total,
        "ratios": {
            "train": len(train) / float(total),
            "val": len(val) / float(total),
            "test": len(test) / float(total),
        },
        "protocol": {
            "type": "single_scene_non_overlapping_patches",
            "dataset": "Chikusei",
            "patch_size": [patch_h, patch_w],
            "grid_shape": [grid_h, grid_w],
            "val_count": len(val),
            "test_count": len(test),
        },
        "train": train,
        "val": val,
        "test": test,
    }


def _generate_pavia_protocol_split(sample_path, seed):
    """Paper-style test split with one validation strip carved from train."""
    height, width, _ = _get_hyperspectral_hwc_shape(sample_path)
    strip_h = 120
    usable_w = min(width, 714)
    num_strips = height // strip_h
    num_test = min(3, num_strips)
    num_train = max(0, num_strips - num_test)
    num_val = 1 if num_train >= 2 else 0
    val_idx = num_train - 1 if num_val else None

    train = []
    val = []
    test = []
    for idx in range(num_strips):
        entry = _make_crop_entry(
            sample_path,
            f"pavia_strip_{idx:02d}",
            idx * strip_h,
            0,
            strip_h,
            usable_w,
        )
        if idx >= num_train:
            test.append(entry)
        elif val_idx is not None and idx == val_idx:
            val.append(entry)
        else:
            train.append(entry)

    total = len(train) + len(val) + len(test)
    return {
        "seed": seed,
        "total": total,
        "ratios": {
            "train": len(train) / float(total) if total else 0.0,
            "val": len(val) / float(total) if total else 0.0,
            "test": len(test) / float(total) if total else 0.0,
        },
        "protocol": {
            "type": "single_scene_non_overlapping_strips",
            "dataset": "Pavia",
            "patch_size": [strip_h, usable_w],
            "usable_width": usable_w,
            "val_count": len(val),
            "test_count": len(test),
        },
        "train": train,
        "val": val,
        "test": test,
    }


def _generate_protocol_split_if_needed(data_root, valid_files, seed):
    """Apply paper-style split rules for supported single-scene datasets."""
    if len(valid_files) != 1 or not os.path.isfile(valid_files[0]) or not valid_files[0].lower().endswith(".mat"):
        return None

    dataset_key = _dataset_key(data_root)
    sample_path = valid_files[0]
    if "chikusei" in dataset_key:
        return _generate_chikusei_protocol_split(sample_path, seed)
    if "pavia" in dataset_key:
        return _generate_pavia_protocol_split(sample_path, seed)
    return None


def _scan_hyperspectral_paths(data_root):
    """Scan dataset root and return supported hyperspectral sample paths.

    Scene detection strategy:
      1. .mat files directly in data_root
      2. Subdirectories of data_root that contain band images (*_ms_*.png etc.)
         — checked at ONE level of depth only to avoid duplicate paths when
           a scene folder contains a same-named subfolder (e.g. CAVE layout:
           data_root/flowers_ms/flowers_ms/*.png).

    The one-level restriction means: if data_root/flowers_ms/ itself has band
    images → it is the scene. If it does NOT but data_root/flowers_ms/flowers_ms/
    does → the inner folder is the scene. Never both.
    """
    mat_files = sorted(glob.glob(os.path.join(data_root, "*.mat")))
    scene_dirs = []

    # Only look one level deep: direct children of data_root
    try:
        top_entries = sorted(os.listdir(data_root))
    except OSError:
        return mat_files, scene_dirs

    for entry in top_entries:
        if entry.startswith('.'):
            continue
        entry_path = os.path.join(data_root, entry)
        if not os.path.isdir(entry_path):
            continue

        if is_hyperspectral_scene_dir(entry_path):
            # The top-level folder itself contains band images → use it
            scene_dirs.append(entry_path)
        else:
            # Check one level deeper (handles CAVE layout: scene/scene/*.png)
            try:
                sub_entries = sorted(os.listdir(entry_path))
            except OSError:
                continue
            for sub in sub_entries:
                if sub.startswith('.'):
                    continue
                sub_path = os.path.join(entry_path, sub)
                if os.path.isdir(sub_path) and is_hyperspectral_scene_dir(sub_path):
                    scene_dirs.append(sub_path)
                    break  # Chỉ lấy 1 scene per top-level folder

    return mat_files, sorted(set(scene_dirs))


def generate_split(
    data_root,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42,
    save=True,
):
    """Execute `generate_split`.

    Args:
        data_root: Input parameter `data_root`.
        train_ratio: Input parameter `train_ratio`.
        val_ratio: Input parameter `val_ratio`.
        test_ratio: Input parameter `test_ratio`.
        seed: Input parameter `seed`.
        save: Input parameter `save`.

    Returns:
        Any: Output produced by this function.
    """

    # 1️⃣ Scan paths
    mat_files, scene_dirs = _scan_hyperspectral_paths(data_root)
    if len(mat_files) == 0 and len(scene_dirs) == 0:
        raise ValueError(
            f"No supported hyperspectral samples found in {data_root}\n"
            "Expected either:\n"
            "  - .mat files in data_root, or\n"
            "  - scene folders containing spectral bands like '*_ms_XX.png'."
        )

    valid_mat_files = [f for f in mat_files if is_hyperspectral_mat(f)]
    valid_set = set(valid_mat_files)
    skipped_files = [f for f in mat_files if f not in valid_set]
    if skipped_files:
        print(f"⚠️ Skipping {len(skipped_files)} non-hyperspectral .mat file(s).")
        for skipped in skipped_files[:5]:
            print(f"   - {os.path.basename(skipped)}")
        if len(skipped_files) > 5:
            print(f"   ... and {len(skipped_files) - 5} more")

    valid_files = valid_mat_files + scene_dirs
    if len(valid_files) == 0:
        raise ValueError(
            f"No valid hyperspectral samples found in {data_root}. "
            "Expected at least one valid .mat cube or scene folder with band images."
        )

    protocol_split = _generate_protocol_split_if_needed(data_root, valid_files, seed)
    if protocol_split is not None:
        split_dict = protocol_split
    else:
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
    """Execute `load_split`.

    Args:
        data_root: Input parameter `data_root`.

    Returns:
        Any: Output produced by this function.
    """

    split_path = os.path.join(data_root, "split.json")

    if not os.path.exists(split_path):
        raise ValueError(
            "split.json not found. Run generate_split() first."
        )

    with open(split_path, "r") as f:
        return json.load(f)


def get_split(data_root, split="train"):
    """Execute `get_split`.

    Args:
        data_root: Input parameter `data_root`.
        split: Input parameter `split`.

    Returns:
        Any: Output produced by this function.
    """

    split_data = load_split(data_root)

    if split not in ["train", "val", "test"]:
        raise ValueError("split must be train/val/test")

    return split_data[split]
