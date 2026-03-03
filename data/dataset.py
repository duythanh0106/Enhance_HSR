"""
Universal Hyperspectral Dataset (Dataset-Agnostic)
Supports any hyperspectral dataset with automatic split & band detection
"""

import os
import glob
import re
import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.io as sio
import random
from .splits import generate_split, get_split, is_hyperspectral_path

try:
    import h5py
except ImportError:  # pragma: no cover - optional dependency at runtime
    h5py = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional dependency at runtime
    Image = None


def _to_hwc_cube(img):
    """Internal helper for `to_hwc_cube` operations.

    Args:
        img: Input parameter `img`.

    Returns:
        Any: Output produced by this function.
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
    """Internal helper for `extract_hsi_from_mat_dict` operations.

    Args:
        mat_data: Input parameter `mat_data`.
        path: Input parameter `path`.

    Returns:
        Any: Output produced by this function.
    """
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
    """Internal helper for `load_hdf5_hyperspectral_image` operations.

    Args:
        path: Input parameter `path`.

    Returns:
        Any: Output produced by this function.
    """
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
            """Execute `visitor`.

            Args:
                name: Input parameter `name`.
                obj: Input parameter `obj`.

            Returns:
                None: This function returns no value.
            """
            if isinstance(obj, h5py.Dataset) and len(obj.shape) == 3:
                candidates.append(name)

        f.visititems(visitor)
        if candidates:
            # Read the largest candidate only.
            best_name = max(candidates, key=lambda n: np.prod(f[n].shape))
            return _to_hwc_cube(f[best_name][()])

    raise ValueError(f"No hyperspectral data found in {path}")


def _band_index_from_filename(path):
    """Extract numeric band index from filename suffix like '*_ms_31.png'."""
    match = re.search(r"_ms_(\d+)\.[^.]+$", os.path.basename(path), re.IGNORECASE)
    return int(match.group(1)) if match else -1


def _list_band_image_files(scene_dir):
    """List spectral band image files for a scene directory."""
    band_globs = (
        "*_ms_*.png",
        "*_ms_*.tif",
        "*_ms_*.tiff",
        "*_ms_*.bmp",
        "*_ms_*.jpg",
        "*_ms_*.jpeg",
    )
    band_paths = []
    for pattern in band_globs:
        band_paths.extend(glob.glob(os.path.join(scene_dir, pattern)))
    # Keep deterministic order by band index, then by filename.
    band_paths = sorted(set(band_paths), key=lambda p: (_band_index_from_filename(p), os.path.basename(p)))
    return band_paths


def _load_hyperspectral_scene_dir(scene_dir):
    """Load hyperspectral cube from scene folder containing band images."""
    if Image is None:
        raise ValueError(
            "Pillow is required to load hyperspectral scene folders from PNG/TIFF/BMP/JPEG files."
        )

    band_paths = _list_band_image_files(scene_dir)
    if not band_paths:
        raise ValueError(f"No spectral band images found in scene folder: {scene_dir}")

    bands = []
    ref_h, ref_w = None, None
    for band_path in band_paths:
        with Image.open(band_path) as band_img:
            band_arr = np.asarray(band_img)
        if band_arr.ndim == 3:
            # Safety fallback for RGB-like files; keep single-channel intensity.
            band_arr = band_arr[..., 0]
        if band_arr.ndim != 2:
            raise ValueError(f"Band image is not single-channel: {band_path} | shape={band_arr.shape}")

        if ref_h is None:
            ref_h, ref_w = band_arr.shape
        elif band_arr.shape != (ref_h, ref_w):
            raise ValueError(
                f"Inconsistent band image size in {scene_dir}: "
                f"expected {(ref_h, ref_w)}, got {band_arr.shape} at {band_path}"
            )

        bands.append(band_arr.astype(np.float32, copy=False))

    # Stack to H x W x C.
    return np.stack(bands, axis=2).astype(np.float32, copy=False)


def load_hyperspectral_image(path):
    """Execute `load_hyperspectral_image`.

    Args:
        path: Input parameter `path`.

    Returns:
        Any: Output produced by this function.
    """
    if os.path.isdir(path):
        img = _load_hyperspectral_scene_dir(path)
    else:
        try:
            mat_data = sio.loadmat(path)
            img = _extract_hsi_from_mat_dict(mat_data, path)
        except NotImplementedError:
            # MATLAB v7.3 files require HDF5 reader.
            img = _load_hdf5_hyperspectral_image(path)

    return img.astype(np.float32)


def normalize_image(img):
    """Execute `normalize_image`.

    Args:
        img: Input parameter `img`.

    Returns:
        Any: Output produced by this function.
    """
    img_min, img_max = img.min(), img.max()
    if img_max - img_min > 0:
        img = (img - img_min) / (img_max - img_min)
    return img


def build_split_kwargs(upscale=4, split_seed=42, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
                       force_regenerate_split=False):
    """Execute `build_split_kwargs`.

    Args:
        upscale: Input parameter `upscale`.
        split_seed: Input parameter `split_seed`.
        train_ratio: Input parameter `train_ratio`.
        val_ratio: Input parameter `val_ratio`.
        test_ratio: Input parameter `test_ratio`.
        force_regenerate_split: Input parameter `force_regenerate_split`.

    Returns:
        Any: Output produced by this function.
    """
    return {
        'upscale': upscale,
        'split_seed': split_seed,
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'test_ratio': test_ratio,
        'force_regenerate_split': force_regenerate_split,
    }


def load_dataset_with_fallback(dataset_cls, primary_split, fallback_split='train', log_fn=print, **kwargs):
    """Execute `load_dataset_with_fallback`.

    Args:
        dataset_cls: Input parameter `dataset_cls`.
        primary_split: Input parameter `primary_split`.
        fallback_split: Input parameter `fallback_split`.
        log_fn: Input parameter `log_fn`.
        **kwargs: Input parameter `**kwargs`.

    Returns:
        Any: Output produced by this function.
    """
    try:
        dataset = dataset_cls(split=primary_split, **kwargs)
        return dataset, primary_split
    except ValueError as exc:
        message = str(exc)
        expected = f"No images found for split '{primary_split}'"
        if expected not in message or fallback_split is None:
            raise
        if log_fn is not None:
            log_fn(
                f"⚠️ Split '{primary_split}' is empty. "
                f"Falling back to split='{fallback_split}'."
            )
        dataset = dataset_cls(split=fallback_split, **kwargs)
        return dataset, fallback_split


def downsample_mean(img, scale):
    """Execute `downsample_mean`.

    Args:
        img: Input parameter `img`.
        scale: Input parameter `scale`.

    Returns:
        Any: Output produced by this function.
    """
    h, w, c = img.shape
    new_h, new_w = h // scale, w // scale
    if new_h == 0 or new_w == 0:
        raise ValueError(f"Image too small for scale={scale}: shape={img.shape}")

    trimmed = img[:new_h * scale, :new_w * scale, :]
    return trimmed.reshape(new_h, scale, new_w, scale, c).mean(axis=(1, 3)).astype(np.float32)


def _load_hyperspectral_image_or_none(path):
    """Internal helper for `load_hyperspectral_image_or_none` operations.

    Args:
        path: Input parameter `path`.

    Returns:
        Any: Output produced by this function.
    """
    try:
        return load_hyperspectral_image(path)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def _filter_valid_hsi_paths(paths, split):
    """Internal helper for `filter_valid_hsi_paths` operations.

    Args:
        paths: Input parameter `paths`.
        split: Input parameter `split`.

    Returns:
        Any: Output produced by this function.
    """
    valid = []
    skipped = []
    for path in paths:
        if is_hyperspectral_path(path):
            valid.append(path)
        else:
            skipped.append(path)

    if skipped:
        print(
            f"⚠️ Skipping {len(skipped)} invalid/non-hyperspectral file(s) "
            f"in split '{split}'."
        )
        for bad in skipped[:5]:
            print(f"   - {bad}")
        if len(skipped) > 5:
            print(f"   ... and {len(skipped) - 5} more")

    return valid


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
                 force_regenerate_split=False, virtual_samples_per_epoch=0,
                 cache_in_memory=False):

        """Initialize the `HyperspectralDataset` instance.

        Args:
            data_root: Input parameter `data_root`.
            patch_size: Input parameter `patch_size`.
            upscale: Input parameter `upscale`.
            augment: Input parameter `augment`.
            split: Input parameter `split`.
            split_seed: Input parameter `split_seed`.
            train_ratio: Input parameter `train_ratio`.
            val_ratio: Input parameter `val_ratio`.
            test_ratio: Input parameter `test_ratio`.
            force_regenerate_split: Input parameter `force_regenerate_split`.
            virtual_samples_per_epoch: Input parameter `virtual_samples_per_epoch`.
            cache_in_memory: Input parameter `cache_in_memory`.

        Returns:
            None: This method initializes state and returns no value.
        """
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
        self.cache_in_memory = bool(cache_in_memory)
        self._image_cache = {}

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

        self.image_paths = _filter_valid_hsi_paths(self.image_paths, self.split)
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found for split '{split}' in {data_root}")

        if self.cache_in_memory:
            print(f"Caching {len(self.image_paths)} image(s) for split '{self.split}' into memory...")
            for path in self.image_paths:
                img = _load_hyperspectral_image_or_none(path)
                if img is None:
                    raise RuntimeError(
                        f"Failed to preload hyperspectral image for cache: {path}"
                    )
                self._image_cache[path] = img
            print(f"✅ Cached {len(self._image_cache)} image(s) in memory.")

        # --------------------------------------------------
        # Auto detect spectral bands
        # --------------------------------------------------
        sample = None
        if self.cache_in_memory and self.image_paths:
            sample = self._image_cache[self.image_paths[0]]
        else:
            for path in self.image_paths:
                sample = _load_hyperspectral_image_or_none(path)
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

    def _normalize(self, img):
        """Internal helper for `normalize` operations.

        Args:
            img: Input parameter `img`.

        Returns:
            Any: Output produced by this function.
        """
        return normalize_image(img)

    # ------------------------------------------------------

    def _downsample(self, img, scale):
        """Internal helper for `downsample` operations.

        Args:
            img: Input parameter `img`.
            scale: Input parameter `scale`.

        Returns:
            Any: Output produced by this function.
        """
        return downsample_mean(img, scale)

    # ------------------------------------------------------

    def _random_crop(self, img):
        """Internal helper for `random_crop` operations.

        Args:
            img: Input parameter `img`.

        Returns:
            Any: Output produced by this function.
        """
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
        """Internal helper for `augment` operations.

        Args:
            lr: Input parameter `lr`.
            hr: Input parameter `hr`.

        Returns:
            Any: Output produced by this function.
        """
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
        """Internal helper for `len__` operations.

        Args:
            None.

        Returns:
            Any: Output produced by this function.
        """
        if self.virtual_samples_per_epoch > 0:
            return self.virtual_samples_per_epoch
        return len(self.image_paths)

    # ------------------------------------------------------

    def __getitem__(self, idx):
        """Internal helper for `getitem__` operations.

        Args:
            idx: Input parameter `idx`.

        Returns:
            Any: Output produced by this function.
        """
        real_idx = idx % len(self.image_paths)
        image_path = self.image_paths[real_idx]
        if self.cache_in_memory:
            img = self._image_cache.get(image_path)
        else:
            img = _load_hyperspectral_image_or_none(image_path)

        if img is None:
            bad_path = image_path
            raise RuntimeError(
                f"Failed to load hyperspectral image: {bad_path}. "
                "Please verify sample path format and file integrity."
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
                 force_regenerate_split=False, cache_in_memory=False):

        """Initialize the `HyperspectralTestDataset` instance.

        Args:
            data_root: Input parameter `data_root`.
            split: Input parameter `split`.
            upscale: Input parameter `upscale`.
            split_seed: Input parameter `split_seed`.
            train_ratio: Input parameter `train_ratio`.
            val_ratio: Input parameter `val_ratio`.
            test_ratio: Input parameter `test_ratio`.
            force_regenerate_split: Input parameter `force_regenerate_split`.
            cache_in_memory: Input parameter `cache_in_memory`.

        Returns:
            None: This method initializes state and returns no value.
        """
        self.data_root = data_root
        self.split = split
        self.upscale = upscale
        self.split_seed = split_seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.force_regenerate_split = force_regenerate_split
        self.cache_in_memory = bool(cache_in_memory)
        self._image_cache = {}

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
        self.image_paths = _filter_valid_hsi_paths(self.image_paths, self.split)

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found for split '{split}'")

        if self.cache_in_memory:
            print(f"Caching {len(self.image_paths)} image(s) for split '{self.split}' into memory...")
            for path in self.image_paths:
                img = _load_hyperspectral_image_or_none(path)
                if img is None:
                    raise RuntimeError(
                        f"Failed to preload test hyperspectral image for cache: {path}"
                    )
                self._image_cache[path] = img
            print(f"✅ Cached {len(self._image_cache)} image(s) in memory.")

        sample = None
        if self.cache_in_memory and self.image_paths:
            sample = self._image_cache[self.image_paths[0]]
        else:
            for path in self.image_paths:
                sample = _load_hyperspectral_image_or_none(path)
                if sample is not None:
                    break
        if sample is None:
            raise ValueError("Cannot determine number of spectral bands for test dataset.")
        self.num_bands = sample.shape[2]

        print(f"Detected {self.num_bands} spectral bands.")
        print(f"Loaded {len(self.image_paths)} {split} images.")

    # ------------------------------------------------------

    def __len__(self):
        """Internal helper for `len__` operations.

        Args:
            None.

        Returns:
            Any: Output produced by this function.
        """
        return len(self.image_paths)

    # ------------------------------------------------------

    def __getitem__(self, idx):

        """Internal helper for `getitem__` operations.

        Args:
            idx: Input parameter `idx`.

        Returns:
            Any: Output produced by this function.
        """
        image_path = self.image_paths[idx]
        if self.cache_in_memory:
            img = self._image_cache.get(image_path)
        else:
            img = _load_hyperspectral_image_or_none(image_path)
        if img is None:
            bad_path = image_path
            raise RuntimeError(
                f"Failed to load test hyperspectral image: {bad_path}. "
                "Please verify sample path format and file integrity."
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

        return lr, hr, image_path
