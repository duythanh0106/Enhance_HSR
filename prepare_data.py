"""
Dataset Preprocessing — Paper-style split và cropping theo ESSA paper.

Script này crop raw .mat files thành patches đúng theo paper,
tạo split.json để HyperspectralDataset/HyperspectralTestDataset load được.

QUAN TRỌNG:
  - Script KHÔNG normalize — dataset.py tự normalize khi load
  - Script KHÔNG tạo LR — dataset.py tạo LR on-the-fly bằng downsample_mean
  - Output là .npy [H,W,C] float32 (sau khi thêm .npy support vào dataset.py)
  - split.json format tương thích với get_split() trong splits.py

Paper split summary:
    CAVE:      32 images   -> 24 train / 8 test  (full 512x512 images)
    Chikusei:  1 image     -> 12 train / 4 test  (512x512 non-overlapped patches)
    Pavia:     1 image     -> 6 train  / 3 test  (120x714 horizontal strips)
    Harvard:   50 images   -> 40 train / 10 test (full images, follow DCM-NET)

Usage:
    # Bước 1: inspect để biết mat key
    python prepare_data.py --inspect --dataset cave --src ./raw/CAVE

    # Bước 2: prepare
    python prepare_data.py --dataset cave     --src ./raw/CAVE     --dst ./dataset/CAVE_paper
    python prepare_data.py --dataset chikusei --src ./raw/Chikusei --dst ./dataset/Chikusei_paper
    python prepare_data.py --dataset pavia    --src ./raw/Pavia    --dst ./dataset/Pavia_paper
    python prepare_data.py --dataset harvard  --src ./raw/Harvard  --dst ./dataset/Harvard_paper

    # Nếu CAVE đã có .npy nhưng split sai (chỉ có 3 test thay vì 8):
    python prepare_data.py --update_split --dataset cave --dst ./dataset/CAVE_paper --scale 4

    # Nếu mat key không auto-detect được:
    python prepare_data.py --dataset chikusei --src ./raw --dst ./out --mat_key chikusei
"""

import argparse
import os
import json
import random
import numpy as np
import scipy.io


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def load_mat_cube(path, mat_key=None, verbose=True):
    """Load .mat file, auto-detect key, return [H,W,C] float32.
    KHÔNG normalize — dataset.py normalize khi load.
    """
    try:
        data = scipy.io.loadmat(path)
    except NotImplementedError:
        try:
            import h5py
        except ImportError:
            raise ImportError("File is MATLAB v7.3. Install h5py: pip install h5py")
        with h5py.File(path, 'r') as f:
            keys = [k for k in f.keys() if not k.startswith('#')]
            if verbose:
                for k in keys:
                    print(f"  h5 key='{k}'  shape={f[k].shape}  dtype={f[k].dtype}")
            key = mat_key or keys[0]
            cube = f[key][()]
            if cube.ndim == 3:
                cube = cube.transpose()
            return cube.astype(np.float32)

    real_keys = [k for k in data.keys() if not k.startswith('_')]
    if verbose:
        for k in real_keys:
            v = data[k]
            if isinstance(v, np.ndarray):
                print(f"  key='{k}'  shape={v.shape}  dtype={v.dtype}  "
                      f"range=[{v.min():.3f}, {v.max():.3f}]")

    candidates = ['rad', 'cube', 'ref', 'data', 'img', 'chikusei', 'pavia', 'paviaU']
    if mat_key:
        candidates = [mat_key] + candidates

    for k in candidates:
        if k in data and isinstance(data[k], np.ndarray) and data[k].ndim == 3:
            cube = data[k].astype(np.float32)
            break
    else:
        arrs = [(k, v) for k, v in data.items()
                if isinstance(v, np.ndarray) and v.ndim == 3
                and np.issubdtype(v.dtype, np.number)]
        if not arrs:
            raise ValueError(f"No 3D array in {path}. Keys: {real_keys}")
        k, cube = max(arrs, key=lambda x: x[1].size)
        cube = cube.astype(np.float32)

    # Ensure [H,W,C]: smallest dim is C (spectral)
    s = int(np.argmin(cube.shape))
    if s == 0:
        cube = np.transpose(cube, (2, 1, 0))
    elif s == 1:
        cube = np.transpose(cube, (0, 2, 1))

    return cube


def save_npy(cube, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, cube.astype(np.float32))


def save_split_json(train_paths, test_paths, val_paths, dst, scale, seed):
    n = len(train_paths) + len(test_paths) + len(val_paths)
    split = {
        "seed": seed,
        "total": n,
        "scale": scale,
        "ratios": {
            "train": round(len(train_paths) / max(1, n), 3),
            "val":   round(len(val_paths)   / max(1, n), 3),
            "test":  round(len(test_paths)  / max(1, n), 3),
        },
        "train": sorted(train_paths),
        "val":   sorted(val_paths),
        "test":  sorted(test_paths),
    }
    out = os.path.join(dst, "split.json")
    with open(out, "w") as f:
        json.dump(split, f, indent=2)
    print(f"\nSaved split.json -> {out}")
    print(f"  train={len(train_paths)}  val={len(val_paths)}  test={len(test_paths)}")


def inspect(src, dataset):
    section(f"Inspect: {dataset}  ({src})")
    entries = sorted(os.listdir(src))
    mat_files = [e for e in entries if e.lower().endswith('.mat')]
    dirs = [e for e in entries if os.path.isdir(os.path.join(src, e))]
    print(f"Files: {len(mat_files)} .mat, {len(dirs)} dirs")
    for fname in mat_files[:3]:
        print(f"\n[{fname}]")
        load_mat_cube(os.path.join(src, fname), verbose=True)
    if dirs and not mat_files:
        d = dirs[0]
        print(f"\n[dir: {d}]")
        print(sorted(os.listdir(os.path.join(src, d)))[:5])


def prepare_cave(src, dst, scale, mat_key, seed):
    """CAVE: 32 full images -> 24 train / 8 test."""
    section("CAVE")
    entries = sorted(os.listdir(src))
    mat_files = [e for e in entries if e.lower().endswith('.mat')]
    scene_dirs = [e for e in entries if os.path.isdir(os.path.join(src, e))]

    if mat_files:
        scenes, fmt = mat_files, 'mat'
    elif scene_dirs:
        scenes, fmt = scene_dirs, 'dir'
    else:
        raise ValueError(f"No .mat or scene dirs in {src}")

    print(f"Format: {fmt},  Scenes: {len(scenes)}  (paper: 32)")
    if len(scenes) < 32:
        print(f"WARNING: {len(scenes)}/32 scenes. "
              "Download: http://www.cs.columbia.edu/CAVE/databases/multispectral/")

    random.seed(seed)
    shuffled = scenes.copy()
    random.shuffle(shuffled)
    test_set = set(shuffled[:8])

    train_paths, test_paths = [], []

    for scene in scenes:
        stem = scene.replace('.mat', '').rstrip('/')
        tag = 'test' if scene in test_set else 'train'
        out_path = os.path.join(dst, tag, stem + '.npy')

        if fmt == 'mat':
            cube = load_mat_cube(os.path.join(src, scene),
                                 mat_key=mat_key, verbose=False)
        else:
            from PIL import Image as PILImage
            band_dir = os.path.join(src, scene)
            pngs = sorted([f for f in os.listdir(band_dir)
                           if f.lower().endswith('.png')])
            bands = [np.array(PILImage.open(os.path.join(band_dir, p)),
                              dtype=np.float32) for p in pngs]
            # Each band may be 2D or 3D
            bands = [b[:,:,0] if b.ndim == 3 else b for b in bands]
            cube = np.stack(bands, axis=-1)

        if cube.ndim == 3 and cube.shape[0] < cube.shape[-1]:
            cube = np.transpose(cube, (2, 1, 0))

        save_npy(cube, out_path)
        print(f"  [{tag}] {stem}  {cube.shape}")
        (test_paths if scene in test_set else train_paths).append(out_path)

    save_split_json(train_paths, test_paths, [], dst, scale, seed)


def prepare_chikusei(src, dst, scale, mat_key, seed):
    """Chikusei: 2517x2335x128 -> 16 patches 512x512. 4 test, 12 train."""
    section("Chikusei")
    mat_files = [f for f in os.listdir(src) if f.lower().endswith('.mat')]
    if not mat_files:
        raise FileNotFoundError(f"No .mat in {src}")

    print(f"Loading {mat_files[0]} ...")
    cube = load_mat_cube(os.path.join(src, mat_files[0]),
                         mat_key=mat_key, verbose=True)
    H, W, C = cube.shape
    print(f"\n[H,W,C]: {cube.shape}")

    ph = pw = 512
    nh, nw = H // ph, W // pw
    print(f"Grid {nh}x{nw} = {nh*nw} patches  "
          f"(discarding h={H%ph}px, w={W%pw}px)")

    ids = [(r, c) for r in range(nh) for c in range(nw)]
    random.seed(seed)
    shuffled = ids.copy()
    random.shuffle(shuffled)
    test_ids = set(map(tuple, shuffled[:4]))
    print(f"Test patches (row,col): {sorted(test_ids)}")

    train_paths, test_paths = [], []

    for r, c in ids:
        patch = cube[r*ph:(r+1)*ph, c*pw:(c+1)*pw, :]
        tag = 'test' if (r, c) in test_ids else 'train'
        out_path = os.path.join(dst, tag, f'chikusei_r{r:02d}_c{c:02d}.npy')
        save_npy(patch, out_path)
        print(f"  [{tag}] ({r},{c})  {patch.shape}")
        (test_paths if (r,c) in test_ids else train_paths).append(out_path)

    save_split_json(train_paths, test_paths, [], dst, scale, seed)


def prepare_pavia(src, dst, scale, mat_key, available_w):
    """Pavia: 1096x714 -> 9 strips 120x714. 3 test (last), 6 train."""
    section("Pavia")
    mat_files = [f for f in os.listdir(src) if f.lower().endswith('.mat')]
    if not mat_files:
        raise FileNotFoundError(f"No .mat in {src}")

    print(f"Loading {mat_files[0]} ...")
    cube = load_mat_cube(os.path.join(src, mat_files[0]),
                         mat_key=mat_key, verbose=True)
    H, W, C = cube.shape
    print(f"\n[H,W,C]: {cube.shape}")

    if W > available_w:
        cube = cube[:, :available_w, :]
        W = available_w
        print(f"Cropped to width {available_w}: {cube.shape}")

    strip_h = 120
    n = H // strip_h
    n_test = 3
    n_train = n - n_test
    print(f"Strips: {n} x {strip_h}x{W}  (remainder {H%strip_h}px discarded)")
    print(f"Split: {n_train} train (0..{n_train-1}) / {n_test} test ({n_train}..{n-1})")

    train_paths, test_paths = [], []

    for i in range(n):
        strip = cube[i*strip_h:(i+1)*strip_h, :, :]
        tag = 'test' if i >= n_train else 'train'
        out_path = os.path.join(dst, tag, f'pavia_strip_{i:02d}.npy')
        save_npy(strip, out_path)
        print(f"  [{tag}] strip {i:02d}  {strip.shape}")
        (test_paths if i >= n_train else train_paths).append(out_path)

    save_split_json(train_paths, test_paths, [], dst, scale, seed=0)


def prepare_harvard(src, dst, scale, mat_key, seed):
    """Harvard: 50 scenes -> 40 train / 10 test."""
    section("Harvard")
    mat_files = sorted([f for f in os.listdir(src) if f.lower().endswith('.mat')])
    print(f"Found {len(mat_files)} scenes  (expect 50)")

    random.seed(seed)
    shuffled = mat_files.copy()
    random.shuffle(shuffled)
    test_set = set(shuffled[:10])

    train_paths, test_paths = [], []

    for fname in mat_files:
        stem = fname.replace('.mat', '')
        tag = 'test' if fname in test_set else 'train'
        out_path = os.path.join(dst, tag, stem + '.npy')
        cube = load_mat_cube(os.path.join(src, fname),
                             mat_key=mat_key, verbose=False)
        save_npy(cube, out_path)
        print(f"  [{tag}] {stem}  {cube.shape}")
        (test_paths if fname in test_set else train_paths).append(out_path)

    save_split_json(train_paths, test_paths, [], dst, scale, seed)


def update_cave_split(dst, scale, seed):
    """Tạo lại split.json cho CAVE (fix 3 test -> 8 test)."""
    section("Update CAVE split")
    all_npy = []
    for d in ('train', 'test', 'val'):
        dd = os.path.join(dst, d)
        if os.path.isdir(dd):
            for f in sorted(os.listdir(dd)):
                if f.endswith('.npy'):
                    all_npy.append(os.path.join(dd, f))

    print(f"Found {len(all_npy)} .npy files")
    random.seed(seed)
    shuffled = all_npy.copy()
    random.shuffle(shuffled)
    new_test = set(shuffled[:8])

    final_train, final_test = [], []
    moved = 0
    for old_path in all_npy:
        fname = os.path.basename(old_path)
        tag = 'test' if old_path in new_test else 'train'
        new_path = os.path.join(dst, tag, fname)
        if old_path != new_path:
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            os.rename(old_path, new_path)
            print(f"  [{tag}] {fname}")
            moved += 1
        (final_test if old_path in new_test else final_train).append(new_path)

    print(f"Moved {moved} files")
    save_split_json(final_train, final_test, [], dst, scale, seed)


def main():
    p = argparse.ArgumentParser(
        description='Prepare HSI datasets theo ESSA paper split',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--dataset', required=True,
                   choices=['cave', 'chikusei', 'pavia', 'harvard'])
    p.add_argument('--src', default=None)
    p.add_argument('--dst', default=None)
    p.add_argument('--scale', type=int, default=4, choices=[2, 4, 8])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--mat_key', default=None)
    p.add_argument('--inspect', action='store_true')
    p.add_argument('--update_split', action='store_true')
    p.add_argument('--pavia_width', type=int, default=714)
    args = p.parse_args()

    if args.inspect:
        if not args.src:
            p.error('--inspect requires --src')
        inspect(args.src, args.dataset)
        return

    if args.update_split:
        if args.dataset != 'cave':
            p.error('--update_split is CAVE only')
        update_cave_split(args.dst, args.scale, args.seed)
        return

    if not args.src or not args.dst:
        p.error('--src and --dst required')

    os.makedirs(args.dst, exist_ok=True)
    for d in ('train', 'test'):
        os.makedirs(os.path.join(args.dst, d), exist_ok=True)

    if args.dataset == 'cave':
        prepare_cave(args.src, args.dst, args.scale, args.mat_key, args.seed)
    elif args.dataset == 'chikusei':
        prepare_chikusei(args.src, args.dst, args.scale, args.mat_key, args.seed)
    elif args.dataset == 'pavia':
        prepare_pavia(args.src, args.dst, args.scale, args.mat_key,
                      args.pavia_width)
    elif args.dataset == 'harvard':
        prepare_harvard(args.src, args.dst, args.scale, args.mat_key, args.seed)

    print("\nDone!")
    print(f"Train: python train.py --config {args.dataset} --data_root {args.dst}")
    print(f"Test:  python test_full_image.py --checkpoint .../best.pth "
          f"--data_root {args.dst} --chop_patch_size 512")


if __name__ == '__main__':
    main()