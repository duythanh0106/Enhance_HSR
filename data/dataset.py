"""
Dataset class for CAVE and Harvard Hyperspectral Images
"""

import os
import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.io as sio
import random
from glob import glob
from .splits import get_split, filter_files_by_split


class HyperspectralDataset(Dataset):
    """
    Dataset for CAVE and Harvard hyperspectral images
    
    Supports:
    - Loading .mat files
    - Random cropping
    - Data augmentation (flip, rotation)
    - Automatic downsampling to create LR-HR pairs
    """
    
    def __init__(self, data_root, dataset_type='CAVE', patch_size=128, 
                 upscale=4, augment=True, num_bands=31):
        """
        Args:
            data_root: Path to dataset folder
            dataset_type: 'CAVE' or 'Harvard'
            patch_size: Size of HR patches
            upscale: Downsampling factor
            augment: Whether to apply data augmentation
            num_bands: Number of spectral bands (31 for CAVE/Harvard)
        """
        self.data_root = data_root
        self.dataset_type = dataset_type
        self.patch_size = patch_size
        self.upscale = upscale
        self.augment = augment
        self.num_bands = num_bands
        
        # Load all image paths
        self.image_paths = self._load_image_paths()
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No .mat files found in {data_root}")
        
        print(f"Loaded {len(self.image_paths)} hyperspectral images from {data_root}")
    
    def _load_image_paths(self):
        """Load all .mat file paths from dataset"""
        mat_files = []
        
        # Search for .mat files recursively
        for root, dirs, files in os.walk(self.data_root):
            for file in files:
                if file.endswith('.mat'):
                    mat_files.append(os.path.join(root, file))
        
        return sorted(mat_files)
    
    def _load_hyperspectral_image(self, path):
        """
        Load hyperspectral image from .mat file
        
        Returns:
            img: numpy array [H, W, C]
        """
        try:
            mat_data = sio.loadmat(path)
            
            # Different datasets use different variable names
            possible_keys = ['rad', 'cube', 'ref', 'data', 'img']
            
            img = None
            for key in possible_keys:
                if key in mat_data:
                    img = mat_data[key]
                    break
            
            # If still not found, get the largest numeric array
            if img is None:
                numeric_arrays = [v for v in mat_data.values() 
                                if isinstance(v, np.ndarray) and v.ndim == 3]
                if numeric_arrays:
                    img = max(numeric_arrays, key=lambda x: x.size)
            
            if img is None:
                raise ValueError(f"Could not find hyperspectral data in {path}")
            
            # Ensure float32
            img = img.astype(np.float32)
            
            return img
            
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None
    
    def _normalize(self, img):
        """
        Normalize image to [0, 1]
        
        Args:
            img: [H, W, C]
        Returns:
            normalized: [H, W, C] in range [0, 1]
        """
        img_min = img.min()
        img_max = img.max()
        
        if img_max - img_min > 0:
            img = (img - img_min) / (img_max - img_min)
        
        return img
    
    def _ensure_num_bands(self, img):
        """
        Ensure image has exactly num_bands channels
        
        Args:
            img: [H, W, C]
        Returns:
            img: [H, W, num_bands]
        """
        current_bands = img.shape[2]
        
        if current_bands == self.num_bands:
            return img
        elif current_bands < self.num_bands:
            # Pad with edge values
            pad_bands = self.num_bands - current_bands
            img = np.pad(img, ((0, 0), (0, 0), (0, pad_bands)), mode='edge')
        else:
            # Crop to num_bands
            img = img[:, :, :self.num_bands]
        
        return img
    
    def _downsample(self, img, scale):
        """
        Downsample image using average pooling
        
        Args:
            img: [H, W, C]
            scale: Downsampling factor
        Returns:
            img_down: [H//scale, W//scale, C]
        """
        h, w, c = img.shape
        new_h, new_w = h // scale, w // scale
        
        img_down = np.zeros((new_h, new_w, c), dtype=np.float32)
        
        for i in range(c):
            channel = img[:, :, i]
            # Average pooling
            for y in range(new_h):
                for x in range(new_w):
                    img_down[y, x, i] = channel[
                        y*scale:(y+1)*scale, 
                        x*scale:(x+1)*scale
                    ].mean()
        
        return img_down
    
    def _random_crop(self, img, patch_size):
        """
        Random crop a patch from image
        
        Args:
            img: [H, W, C]
            patch_size: Size of patch to crop
        Returns:
            patch: [patch_size, patch_size, C]
        """
        h, w, c = img.shape
        
        if h >= patch_size and w >= patch_size:
            # Random crop
            top = random.randint(0, h - patch_size)
            left = random.randint(0, w - patch_size)
            patch = img[top:top+patch_size, left:left+patch_size, :]
        else:
            # Pad if image is smaller
            pad_h = max(0, patch_size - h)
            pad_w = max(0, patch_size - w)
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            patch = img[:patch_size, :patch_size, :]
        
        return patch
    
    def _augment(self, lr, hr):
        """
        Data augmentation: random flip and rotation
        
        Args:
            lr: [H, W, C] - Low resolution
            hr: [H, W, C] - High resolution
        Returns:
            lr_aug, hr_aug: Augmented images
        """
        # Random horizontal flip
        if random.random() > 0.5:
            lr = np.flip(lr, axis=1).copy()
            hr = np.flip(hr, axis=1).copy()
        
        # Random vertical flip
        if random.random() > 0.5:
            lr = np.flip(lr, axis=0).copy()
            hr = np.flip(hr, axis=0).copy()
        
        # Random rotation (0, 90, 180, 270 degrees)
        k = random.randint(0, 3)
        lr = np.rot90(lr, k, axes=(0, 1)).copy()
        hr = np.rot90(hr, k, axes=(0, 1)).copy()
        
        return lr, hr
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Get a training sample
        
        Returns:
            lr: [C, H, W] - Low resolution input
            hr: [C, H, W] - High resolution target
        """
        # Load hyperspectral image
        img = self._load_hyperspectral_image(self.image_paths[idx])
        
        if img is None:
            # Return dummy data if loading fails
            hr = np.zeros((self.patch_size, self.patch_size, self.num_bands), 
                         dtype=np.float32)
            lr = np.zeros((self.patch_size // self.upscale, 
                          self.patch_size // self.upscale, self.num_bands), 
                         dtype=np.float32)
        else:
            # Normalize
            img = self._normalize(img)
            
            # Ensure correct number of bands
            img = self._ensure_num_bands(img)
            
            # Random crop HR patch
            hr = self._random_crop(img, self.patch_size)
            
            # Create LR by downsampling
            lr = self._downsample(hr, self.upscale)
            
            # Apply augmentation
            if self.augment:
                lr, hr = self._augment(lr, hr)
        
        # Convert to torch tensors [H, W, C] -> [C, H, W]
        lr = torch.from_numpy(lr.transpose(2, 0, 1))
        hr = torch.from_numpy(hr.transpose(2, 0, 1))
        
        return lr, hr


class HyperspectralTestDataset(Dataset):
    """
    Test dataset - loads full images without cropping
    Used for evaluation with FIXED test split
    """
    
    def __init__(self, data_root, dataset_type='CAVE', split='test', upscale=4, num_bands=31):
        """
        Args:
            data_root: Path to dataset folder
            dataset_type: 'CAVE' or 'Harvard'
            split: Usually 'test', but can be 'val' for validation
            upscale: Upscale factor
            num_bands: Number of spectral bands
        """
        self.data_root = data_root
        self.dataset_type = dataset_type
        self.split = split
        self.upscale = upscale
        self.num_bands = num_bands
        
        # Load and filter by split
        all_paths = self._load_image_paths()
        self.image_paths = filter_files_by_split(all_paths, dataset_type, split)
        
        print(f"Loaded {len(self.image_paths)} images for {dataset_type} {split}")
    
    def _load_image_paths(self):
        """Load all .mat files"""
        mat_files = []
        for root, dirs, files in os.walk(self.data_root):
            for file in files:
                if file.endswith('.mat'):
                    mat_files.append(os.path.join(root, file))
        return sorted(mat_files)
    
    def _load_hyperspectral_image(self, path):
        """Load and process image (same as training dataset)"""
        try:
            mat_data = sio.loadmat(path)
            possible_keys = ['rad', 'cube', 'ref', 'data', 'img']
            
            img = None
            for key in possible_keys:
                if key in mat_data:
                    img = mat_data[key]
                    break
            
            if img is None:
                numeric_arrays = [v for v in mat_data.values() 
                                if isinstance(v, np.ndarray) and v.ndim == 3]
                if numeric_arrays:
                    img = max(numeric_arrays, key=lambda x: x.size)
            
            if img is None:
                raise ValueError(f"Could not find data in {path}")
            
            return img.astype(np.float32)
            
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """Return full image without cropping"""
        img = self._load_hyperspectral_image(self.image_paths[idx])
        
        if img is None:
            # Return dummy
            return torch.zeros(self.num_bands, 128, 128), \
                   torch.zeros(self.num_bands, 128, 128), \
                   self.image_paths[idx]
        
        # Normalize
        img_min, img_max = img.min(), img.max()
        if img_max - img_min > 0:
            img = (img - img_min) / (img_max - img_min)
        
        # Ensure num_bands
        if img.shape[2] != self.num_bands:
            if img.shape[2] < self.num_bands:
                pad = self.num_bands - img.shape[2]
                img = np.pad(img, ((0,0), (0,0), (0,pad)), mode='edge')
            else:
                img = img[:, :, :self.num_bands]
        
        # Downsample for LR
        h, w, c = img.shape
        new_h, new_w = h // self.upscale, w // self.upscale
        
        lr = np.zeros((new_h, new_w, c), dtype=np.float32)
        for i in range(c):
            for y in range(new_h):
                for x in range(new_w):
                    lr[y,x,i] = img[y*self.upscale:(y+1)*self.upscale,
                                   x*self.upscale:(x+1)*self.upscale, i].mean()
        
        # To tensor
        lr = torch.from_numpy(lr.transpose(2, 0, 1))
        hr = torch.from_numpy(img.transpose(2, 0, 1))
        
        return lr, hr, self.image_paths[idx]


# Test code
if __name__ == '__main__':
    print("Testing HyperspectralDataset...")
    print("="*70)
    
    # You need to have actual data to test
    # This is just a template
    
    # Uncomment when you have data:
    # dataset = HyperspectralDataset(
    #     data_root='./data/CAVE',
    #     dataset_type='CAVE',
    #     patch_size=128,
    #     upscale=4,
    #     augment=True
    # )
    # 
    # print(f"Dataset size: {len(dataset)}")
    # 
    # # Get a sample
    # lr, hr = dataset[0]
    # print(f"LR shape: {lr.shape}")  # Should be [31, 32, 32]
    # print(f"HR shape: {hr.shape}")  # Should be [31, 128, 128]
    
    print("✅ Dataset module ready!")
    print("Remember to download CAVE or Harvard dataset first!")