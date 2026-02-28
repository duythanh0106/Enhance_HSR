"""
Universal Hyperspectral Dataset (Dataset-Agnostic)
Supports any hyperspectral dataset with automatic split & band detection
"""

import os
import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.io as sio
import random
from .splits import generate_split, get_split, is_hyperspectral_mat

try:
    import h5py
except ImportError:  # pragma: no cover - optional dependency at runtime
    h5py = None


def _to_hwc_cube(img):
    """
    Normalize cube layout to HxWxC by assuming the spectral dimension is the
    smallest axis (typical for hyperspectral data).
    """
    if img.ndim != 3:
        raise ValueError(f"Expected 3D hyperspectral cube, got shape={img.shape}")

    smallest_axis = int(np.argmin(img.shape))
    if smallest_axis == 2:
        return img
    if smallest_axis == 0:
        return np.transpose(img, (2, 1, 0))
    return np.transpose(img, (0, 2, 1))


def _extract_hsi_from_mat_dict(mat_data, path):
    possible_keys = ['rad', 'cube', 'ref', 'data', 'img', 'chikusei']

    for key in possible_keys:
        if key in mat_data:
            value = mat_data[key]
            if isinstance(value, np.ndarray) and value.ndim == 3 and np.issubdtype(value.dtype, np.number):
                return _to_hwc_cube(value)

    numeric_arrays = [
        value for value in mat_data.values()
        if isinstance(value, np.ndarray)
        and value.ndim == 3
        and np.issubdtype(value.dtype, np.number)
    ]
    if numeric_arrays:
        return _to_hwc_cube(max(numeric_arrays, key=lambda x: x.size))

    raise ValueError(f"No hyperspectral data found in {path}")


def _load_hdf5_hyperspectral_image(path):
    if h5py is None:
        raise ValueError(
            f"{path} appears to be MATLAB v7.3, but h5py is not installed."
        )

    possible_keys = ['rad', 'cube', 'ref', 'data', 'img', 'chikusei']
    with h5py.File(path, "r") as f:
        for key in possible_keys:
            if key in f and isinstance(f[key], h5py.Dataset) and len(f[key].shape) == 3:
                return _to_hwc_cube(f[key][()])

        candidates = []

        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset) and len(obj.shape) == 3:
                candidates.append(name)

        f.visititems(visitor)
        if candidates:
            # Read the largest candidate only.
            best_name = max(candidates, key=lambda n: np.prod(f[n].shape))
            return _to_hwc_cube(f[best_name][()])

    raise ValueError(f"No hyperspectral data found in {path}")


def load_hyperspectral_image(path):
    """Load a hyperspectral cube from .mat file."""
    try:
        mat_data = sio.loadmat(path)
        img = _extract_hsi_from_mat_dict(mat_data, path)
    except NotImplementedError:
        # MATLAB v7.3 files require HDF5 reader.
        img = _load_hdf5_hyperspectral_image(path)

    return img.astype(np.float32)


def normalize_image(img):
    """Min-max normalize cube to [0, 1]."""
    img_min, img_max = img.min(), img.max()
    if img_max - img_min > 0:
        img = (img - img_min) / (img_max - img_min)
    return img


def downsample_mean(img, scale):
    """
    Downsample by block averaging using vectorized operations.

    Keeps the same semantics as old loop implementation: truncate borders that
    do not fit into full `scale x scale` blocks.
    """
    h, w, c = img.shape
    new_h, new_w = h // scale, w // scale
    if new_h == 0 or new_w == 0:
        raise ValueError(f"Image too small for scale={scale}: shape={img.shape}")

    trimmed = img[:new_h * scale, :new_w * scale, :]
    return trimmed.reshape(new_h, scale, new_w, scale, c).mean(axis=(1, 3)).astype(np.float32)


# ==========================================================
# TRAINING DATASET
# ==========================================================

class HyperspectralDataset(Dataset):
    """
    Generic Hyperspectral Training Dataset
    - Automatic split (train)
    - Automatic spectral band detection
    - Random crop
    - Augmentation
    - Downsampling to create LR-HR pair
    """

    def __init__(self, data_root, patch_size=128,
                 upscale=4, augment=True, split='train',
                 split_seed=42, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
                 force_regenerate_split=False, virtual_samples_per_epoch=0):

        self.data_root = data_root
        self.patch_size = patch_size
        self.upscale = upscale
        self.augment = augment
        self.split = split
        self.split_seed = split_seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.force_regenerate_split = force_regenerate_split
        self.virtual_samples_per_epoch = max(0, int(virtual_samples_per_epoch or 0))

        # --------------------------------------------------
        # Create split if not exists
        # --------------------------------------------------
        split_file = os.path.join(data_root, "split.json")
        if force_regenerate_split or not os.path.exists(split_file):
            generate_split(
                data_root,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                seed=split_seed,
                save=True
            )

        if split == "trainval":
            self.image_paths = get_split(data_root, "train") + get_split(data_root, "val")
        else:
            self.image_paths = get_split(data_root, split)

        self.image_paths = self._filter_valid_paths(self.image_paths)
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found for split '{split}' in {data_root}")

        # --------------------------------------------------
        # Auto detect spectral bands
        # --------------------------------------------------
        sample = None
        for path in self.image_paths:
            sample = self._load_hyperspectral_image(path)
            if sample is not None:
                break
        if sample is None:
            raise ValueError("Cannot determine number of spectral bands.")

        self.num_bands = sample.shape[2]
        print(f"Detected {self.num_bands} spectral bands.")
        print(f"Loaded {len(self.image_paths)} images for split '{self.split}'.")
        if self.virtual_samples_per_epoch > 0:
            print(
                f"Using virtual samples per epoch for split '{self.split}': "
                f"{self.virtual_samples_per_epoch}"
            )

    # ------------------------------------------------------

    def _load_hyperspectral_image(self, path):
        try:
            return load_hyperspectral_image(path)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None

    # ------------------------------------------------------

    def _filter_valid_paths(self, paths):
        valid = []
        skipped = []
        for path in paths:
            if is_hyperspectral_mat(path):
                valid.append(path)
            else:
                skipped.append(path)

        if skipped:
            print(
                f"⚠️ Skipping {len(skipped)} invalid/non-hyperspectral file(s) "
                f"in split '{self.split}'."
            )
            for bad in skipped[:5]:
                print(f"   - {bad}")
            if len(skipped) > 5:
                print(f"   ... and {len(skipped) - 5} more")

        return valid

    # ------------------------------------------------------

    def _normalize(self, img):
        return normalize_image(img)

    # ------------------------------------------------------

    def _downsample(self, img, scale):
        return downsample_mean(img, scale)

    # ------------------------------------------------------

    def _random_crop(self, img):
        h, w, _ = img.shape

        if h >= self.patch_size and w >= self.patch_size:
            top = random.randint(0, h - self.patch_size)
            left = random.randint(0, w - self.patch_size)
            return img[top:top+self.patch_size,
                       left:left+self.patch_size, :]
        else:
            pad_h = max(0, self.patch_size - h)
            pad_w = max(0, self.patch_size - w)
            img = np.pad(img,
                         ((0, pad_h), (0, pad_w), (0, 0)),
                         mode='reflect')
            return img[:self.patch_size, :self.patch_size, :]

    # ------------------------------------------------------

    def _augment(self, lr, hr):
        if random.random() > 0.5:
            lr = np.flip(lr, axis=1).copy()
            hr = np.flip(hr, axis=1).copy()

        if random.random() > 0.5:
            lr = np.flip(lr, axis=0).copy()
            hr = np.flip(hr, axis=0).copy()

        k = random.randint(0, 3)
        lr = np.rot90(lr, k, axes=(0, 1)).copy()
        hr = np.rot90(hr, k, axes=(0, 1)).copy()

        return lr, hr

    # ------------------------------------------------------

    def __len__(self):
        if self.virtual_samples_per_epoch > 0:
            return self.virtual_samples_per_epoch
        return len(self.image_paths)

    # ------------------------------------------------------

    def __getitem__(self, idx):
        real_idx = idx % len(self.image_paths)
        img = self._load_hyperspectral_image(self.image_paths[real_idx])

        if img is None:
            bad_path = self.image_paths[real_idx]
            raise RuntimeError(
                f"Failed to load hyperspectral image: {bad_path}. "
                "Please verify the .mat file integrity and keys."
            )

        img = self._normalize(img)
        hr = self._random_crop(img)
        lr = self._downsample(hr, self.upscale)

        if self.augment:
            lr, hr = self._augment(lr, hr)

        lr = torch.from_numpy(lr.transpose(2, 0, 1))
        hr = torch.from_numpy(hr.transpose(2, 0, 1))

        return lr, hr


# ==========================================================
# TEST DATASET
# ==========================================================

class HyperspectralTestDataset(Dataset):
    """
    Full image test dataset
    """

    def __init__(self, data_root, split='test', upscale=4,
                 split_seed=42, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
                 force_regenerate_split=False):

        self.data_root = data_root
        self.split = split
        self.upscale = upscale
        self.split_seed = split_seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.force_regenerate_split = force_regenerate_split

        split_file = os.path.join(data_root, "split.json")
        if force_regenerate_split or not os.path.exists(split_file):
            generate_split(
                data_root,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                seed=split_seed,
                save=True
            )

        self.image_paths = get_split(data_root, split)
        self.image_paths = self._filter_valid_paths(self.image_paths)

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found for split '{split}'")

        sample = None
        for path in self.image_paths:
            sample = self._load_hyperspectral_image(path)
            if sample is not None:
                break
        if sample is None:
            raise ValueError("Cannot determine number of spectral bands for test dataset.")
        self.num_bands = sample.shape[2]

        print(f"Detected {self.num_bands} spectral bands.")
        print(f"Loaded {len(self.image_paths)} {split} images.")

    # ------------------------------------------------------

    def _load_hyperspectral_image(self, path):
        try:
            return load_hyperspectral_image(path)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None

    # ------------------------------------------------------

    def _filter_valid_paths(self, paths):
        valid = []
        skipped = []
        for path in paths:
            if is_hyperspectral_mat(path):
                valid.append(path)
            else:
                skipped.append(path)

        if skipped:
            print(
                f"⚠️ Skipping {len(skipped)} invalid/non-hyperspectral file(s) "
                f"in split '{self.split}'."
            )
            for bad in skipped[:5]:
                print(f"   - {bad}")
            if len(skipped) > 5:
                print(f"   ... and {len(skipped) - 5} more")

        return valid

    # ------------------------------------------------------

    def __len__(self):
        return len(self.image_paths)

    # ------------------------------------------------------

    def __getitem__(self, idx):

        img = self._load_hyperspectral_image(self.image_paths[idx])
        if img is None:
            bad_path = self.image_paths[idx]
            raise RuntimeError(
                f"Failed to load test hyperspectral image: {bad_path}. "
                "Please verify the .mat file integrity and keys."
            )

        img = normalize_image(img)
        lr = downsample_mean(img, self.upscale)
        # Match HR size to the effective region used to synthesize LR.
        # downsample_mean truncates borders not divisible by `upscale`.
        hr_h = lr.shape[0] * self.upscale
        hr_w = lr.shape[1] * self.upscale
        img = img[:hr_h, :hr_w, :]

        lr = torch.from_numpy(lr.transpose(2, 0, 1))
        hr = torch.from_numpy(img.transpose(2, 0, 1))

        return lr, hr, self.image_paths[idx]
