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
from .splits import generate_split, get_split


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
                 split_seed=42, train_ratio=0.8, val_ratio=0.1,
                 force_regenerate_split=False):

        self.data_root = data_root
        self.patch_size = patch_size
        self.upscale = upscale
        self.augment = augment
        self.split = split
        self.split_seed = split_seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.force_regenerate_split = force_regenerate_split

        # --------------------------------------------------
        # Create split if not exists
        # --------------------------------------------------
        split_file = os.path.join(data_root, "split.json")
        if force_regenerate_split or not os.path.exists(split_file):
            generate_split(
                data_root,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                seed=split_seed,
                save=True
            )

        if split == "trainval":
            self.image_paths = get_split(data_root, "train") + get_split(data_root, "val")
        else:
            self.image_paths = get_split(data_root, split)

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found for split '{split}' in {data_root}")

        # --------------------------------------------------
        # Auto detect spectral bands
        # --------------------------------------------------
        sample = self._load_hyperspectral_image(self.image_paths[0])
        if sample is None:
            raise ValueError("Cannot determine number of spectral bands.")

        self.num_bands = sample.shape[2]
        print(f"Detected {self.num_bands} spectral bands.")
        print(f"Loaded {len(self.image_paths)} images for split '{self.split}'.")

    # ------------------------------------------------------

    def _load_hyperspectral_image(self, path):
        try:
            mat_data = sio.loadmat(path)
            possible_keys = ['rad', 'cube', 'ref', 'data', 'img']

            img = None
            for key in possible_keys:
                if key in mat_data:
                    img = mat_data[key]
                    break

            if img is None:
                numeric_arrays = [
                    v for v in mat_data.values()
                    if isinstance(v, np.ndarray) and v.ndim == 3
                ]
                if numeric_arrays:
                    img = max(numeric_arrays, key=lambda x: x.size)

            if img is None:
                raise ValueError(f"No hyperspectral data found in {path}")

            return img.astype(np.float32)

        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None

    # ------------------------------------------------------

    def _normalize(self, img):
        img_min, img_max = img.min(), img.max()
        if img_max - img_min > 0:
            img = (img - img_min) / (img_max - img_min)
        return img

    # ------------------------------------------------------

    def _downsample(self, img, scale):
        h, w, c = img.shape
        new_h, new_w = h // scale, w // scale
        img_down = np.zeros((new_h, new_w, c), dtype=np.float32)

        for i in range(c):
            for y in range(new_h):
                for x in range(new_w):
                    img_down[y, x, i] = img[
                        y*scale:(y+1)*scale,
                        x*scale:(x+1)*scale,
                        i
                    ].mean()

        return img_down

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
        return len(self.image_paths)

    # ------------------------------------------------------

    def __getitem__(self, idx):

        img = self._load_hyperspectral_image(self.image_paths[idx])

        if img is None:
            bad_path = self.image_paths[idx]
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
                 split_seed=42, train_ratio=0.8, val_ratio=0.1,
                 force_regenerate_split=False):

        self.data_root = data_root
        self.split = split
        self.upscale = upscale
        self.split_seed = split_seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.force_regenerate_split = force_regenerate_split

        split_file = os.path.join(data_root, "split.json")
        if force_regenerate_split or not os.path.exists(split_file):
            generate_split(
                data_root,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                seed=split_seed,
                save=True
            )

        self.image_paths = get_split(data_root, split)

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found for split '{split}'")

        sample = self._load_hyperspectral_image(self.image_paths[0])
        if sample is None:
            raise ValueError("Cannot determine number of spectral bands for test dataset.")
        self.num_bands = sample.shape[2]

        print(f"Detected {self.num_bands} spectral bands.")
        print(f"Loaded {len(self.image_paths)} {split} images.")

    # ------------------------------------------------------

    def _load_hyperspectral_image(self, path):
        try:
            mat_data = sio.loadmat(path)
            possible_keys = ['rad', 'cube', 'ref', 'data', 'img']

            img = None
            for key in possible_keys:
                if key in mat_data:
                    img = mat_data[key]
                    break

            if img is None:
                numeric_arrays = [
                    v for v in mat_data.values()
                    if isinstance(v, np.ndarray) and v.ndim == 3
                ]
                if numeric_arrays:
                    img = max(numeric_arrays, key=lambda x: x.size)

            return img.astype(np.float32)

        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None

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

        img_min, img_max = img.min(), img.max()
        if img_max - img_min > 0:
            img = (img - img_min) / (img_max - img_min)

        h, w, c = img.shape
        new_h, new_w = h // self.upscale, w // self.upscale

        lr = np.zeros((new_h, new_w, c), dtype=np.float32)

        for i in range(c):
            for y in range(new_h):
                for x in range(new_w):
                    lr[y, x, i] = img[
                        y*self.upscale:(y+1)*self.upscale,
                        x*self.upscale:(x+1)*self.upscale,
                        i
                    ].mean()

        lr = torch.from_numpy(lr.transpose(2, 0, 1))
        hr = torch.from_numpy(img.transpose(2, 0, 1))

        return lr, hr, self.image_paths[idx]
