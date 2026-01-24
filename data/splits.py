"""
Train/Validation/Test Splits cho Hyperspectral SR
Fixed splits để đảm bảo reproducibility

CAVE Dataset: 32 scenes
Harvard Dataset: 50 scenes
"""

import os
import json
import random
import numpy as np


# ============================================================================
# CAVE Dataset Splits (32 scenes total)
# ============================================================================

CAVE_TRAIN_SCENES = [
    'balloons_ms',
    'beads_ms', 
    'cd_ms',
    'chart_and_stuffed_toy_ms',
    'clay_ms',
    'cloth_ms',
    'Egyptian_statue_ms',
    'fake_and_real_beers_ms',
    'fake_and_real_food_ms',
    'fake_and_real_lemon_slices_ms',
    'fake_and_real_peppers_ms',
    'fake_and_real_sushi_ms',
    'fake_and_real_tomatoes_ms',
    'feathers_ms',
    'flowers_ms',
    'glass_tiles_ms',
    'hairs_ms',
    'jelly_beans_ms',
    'oil_painting_ms',
    'paints_ms',
    'photo_and_face_ms',
    'pompoms_ms',
    'real_and_fake_apples_ms',
    'sponges_ms',
    'stuffed_toys_ms',
]  # 25 scenes (78%)

CAVE_VAL_SCENES = [
    'superballs_ms',
    'thread_spools_ms',
    'watercolors_ms',
]  # 3 scenes (9%)

CAVE_TEST_SCENES = [
    'face_ms',
    'feathers_ms',
    'flowers_ms',
    'oil_painting_ms',
]  # 4 scenes (13%)


# ============================================================================
# Harvard Dataset Splits (50 scenes total)
# ============================================================================

HARVARD_TRAIN_SCENES = [
    'imgd1', 'imgd2', 'imgd3', 'imgd4', 'imgd5',
    'imgd6', 'imgd7', 'imgd8', 'imgd9', 'imgd10',
    'imgd11', 'imgd12', 'imgd13', 'imgd14', 'imgd15',
    'imgd16', 'imgd17', 'imgd18', 'imgd19', 'imgd20',
    'imgd21', 'imgd22', 'imgd23', 'imgd24', 'imgd25',
    'imgd26', 'imgd27', 'imgd28', 'imgd29', 'imgd30',
    'imgd31', 'imgd32', 'imgd33', 'imgd34', 'imgd35',
    'imgd36', 'imgd37', 'imgd38', 'imgd39', 'imgd40',
]  # 40 scenes (80%)

HARVARD_VAL_SCENES = [
    'imgd41', 'imgd42', 'imgd43', 'imgd44', 'imgd45',
]  # 5 scenes (10%)

HARVARD_TEST_SCENES = [
    'imgd46', 'imgd47', 'imgd48', 'imgd49', 'imgd50',
]  # 5 scenes (10%)


# ============================================================================
# Helper Functions
# ============================================================================

def get_split(dataset_type='CAVE', split='train'):
    """
    Get scene list for a specific split
    
    Args:
        dataset_type: 'CAVE' or 'Harvard'
        split: 'train', 'val', or 'test'
    
    Returns:
        list: Scene names for the split
    """
    if dataset_type.upper() == 'CAVE':
        if split == 'train':
            return CAVE_TRAIN_SCENES.copy()
        elif split == 'val' or split == 'valid':
            return CAVE_VAL_SCENES.copy()
        elif split == 'test':
            return CAVE_TEST_SCENES.copy()
        else:
            raise ValueError(f"Unknown split: {split}")
    
    elif dataset_type.upper() == 'HARVARD':
        if split == 'train':
            return HARVARD_TRAIN_SCENES.copy()
        elif split == 'val' or split == 'valid':
            return HARVARD_VAL_SCENES.copy()
        elif split == 'test':
            return HARVARD_TEST_SCENES.copy()
        else:
            raise ValueError(f"Unknown split: {split}")
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_type}")


def is_in_split(filename, dataset_type='CAVE', split='train'):
    """
    Check if a file belongs to a specific split
    
    Args:
        filename: File name or path (e.g., 'balloons_ms.mat' or 'path/to/balloons_ms.mat')
        dataset_type: 'CAVE' or 'Harvard'
        split: 'train', 'val', or 'test'
    
    Returns:
        bool: True if file is in the split
    """
    # Extract scene name from filename
    basename = os.path.basename(filename)
    scene_name = basename.replace('.mat', '').replace('_ref', '').replace('_rad', '')
    
    # Get split scenes
    split_scenes = get_split(dataset_type, split)
    
    # Check if scene is in split
    return scene_name in split_scenes


def filter_files_by_split(file_list, dataset_type='CAVE', split='train'):
    """
    Filter file list by split
    
    Args:
        file_list: List of file paths
        dataset_type: 'CAVE' or 'Harvard'
        split: 'train', 'val', or 'test'
    
    Returns:
        list: Filtered file paths
    """
    return [f for f in file_list if is_in_split(f, dataset_type, split)]


def get_split_info(dataset_type='CAVE'):
    """
    Get split information
    
    Returns:
        dict: Split statistics
    """
    train = get_split(dataset_type, 'train')
    val = get_split(dataset_type, 'val')
    test = get_split(dataset_type, 'test')
    
    total = len(train) + len(val) + len(test)
    
    return {
        'dataset': dataset_type,
        'total': total,
        'train': len(train),
        'val': len(val),
        'test': len(test),
        'train_ratio': len(train) / total,
        'val_ratio': len(val) / total,
        'test_ratio': len(test) / total,
        'train_scenes': train,
        'val_scenes': val,
        'test_scenes': test,
    }


def create_random_split(data_root, dataset_type='CAVE', train_ratio=0.8, val_ratio=0.1, 
                       seed=42, save_path=None):
    """
    Create random split from all files in data_root
    Useful if you want different splits
    
    Args:
        data_root: Path to data folder
        dataset_type: 'CAVE' or 'Harvard'
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        seed: Random seed for reproducibility
        save_path: Path to save split info (JSON)
    
    Returns:
        dict: Split information
    """
    # Get all .mat files
    mat_files = []
    for root, dirs, files in os.walk(data_root):
        for file in files:
            if file.endswith('.mat'):
                mat_files.append(file)
    
    # Sort for reproducibility
    mat_files.sort()
    
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    
    # Shuffle
    random.shuffle(mat_files)
    
    # Split
    total = len(mat_files)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    
    train_files = mat_files[:train_size]
    val_files = mat_files[train_size:train_size + val_size]
    test_files = mat_files[train_size + val_size:]
    
    split_info = {
        'dataset': dataset_type,
        'seed': seed,
        'total': total,
        'train': train_files,
        'val': val_files,
        'test': test_files,
        'train_ratio': len(train_files) / total,
        'val_ratio': len(val_files) / total,
        'test_ratio': len(test_files) / total,
    }
    
    # Save if requested
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(split_info, f, indent=2)
        print(f"Split info saved to: {save_path}")
    
    return split_info


def verify_no_overlap():
    """
    Verify that train/val/test have no overlap
    """
    print("Verifying splits...")
    
    for dataset in ['CAVE', 'HARVARD']:
        train = set(get_split(dataset, 'train'))
        val = set(get_split(dataset, 'val'))
        test = set(get_split(dataset, 'test'))
        
        # Check overlaps
        train_val = train & val
        train_test = train & test
        val_test = val & test
        
        if train_val or train_test or val_test:
            print(f"❌ {dataset}: Overlap detected!")
            if train_val:
                print(f"  Train ∩ Val: {train_val}")
            if train_test:
                print(f"  Train ∩ Test: {train_test}")
            if val_test:
                print(f"  Val ∩ Test: {val_test}")
        else:
            print(f"✅ {dataset}: No overlap. Total scenes: {len(train) + len(val) + len(test)}")


# ============================================================================
# Main - Print split info
# ============================================================================

if __name__ == '__main__':
    print("="*70)
    print("HYPERSPECTRAL SR - TRAIN/VAL/TEST SPLITS")
    print("="*70)
    
    # Print CAVE splits
    print("\n📊 CAVE Dataset:")
    print("-"*70)
    cave_info = get_split_info('CAVE')
    print(f"Total scenes: {cave_info['total']}")
    print(f"Train: {cave_info['train']} ({cave_info['train_ratio']*100:.1f}%)")
    print(f"Val:   {cave_info['val']} ({cave_info['val_ratio']*100:.1f}%)")
    print(f"Test:  {cave_info['test']} ({cave_info['test_ratio']*100:.1f}%)")
    
    print("\nTrain scenes:")
    for scene in cave_info['train_scenes']:
        print(f"  - {scene}")
    
    print("\nValidation scenes:")
    for scene in cave_info['val_scenes']:
        print(f"  - {scene}")
    
    print("\nTest scenes:")
    for scene in cave_info['test_scenes']:
        print(f"  - {scene}")
    
    # Print Harvard splits
    print("\n"+"="*70)
    print("📊 Harvard Dataset:")
    print("-"*70)
    harvard_info = get_split_info('HARVARD')
    print(f"Total scenes: {harvard_info['total']}")
    print(f"Train: {harvard_info['train']} ({harvard_info['train_ratio']*100:.1f}%)")
    print(f"Val:   {harvard_info['val']} ({harvard_info['val_ratio']*100:.1f}%)")
    print(f"Test:  {harvard_info['test']} ({harvard_info['test_ratio']*100:.1f}%)")
    
    # Verify no overlap
    print("\n"+"="*70)
    verify_no_overlap()
    
    print("\n"+"="*70)
    print("✅ Split information ready!")
    print("\nUsage in code:")
    print("  from data.splits import get_split, filter_files_by_split")
    print("  train_scenes = get_split('CAVE', 'train')")
    print("  train_files = filter_files_by_split(all_files, 'CAVE', 'train')")
    print("="*70)