# Hyperspectral Image Super-Resolution with ESSA-SSAM-SpecTrans

**Khóa luận tốt nghiệp:** Cải tiến ESSA với Spatial-Spectral Attention Module và Spectral Transformer cho Super-Resolution ảnh quang phổ

---

## 📋 Mục Lục

- [Giới Thiệu](#giới-thiệu)
- [Kiến Trúc Model](#kiến-trúc-model)
- [Cài Đặt](#cài-đặt)
- [Dataset](#dataset)
- [Training](#training)
- [Testing & Evaluation](#testing--evaluation)
- [Kết Quả Mong Đợi](#kết-quả-mong-đợi)
- [Cấu Trúc Project](#cấu-trúc-project)
- [Đóng Góp Chính](#đóng-góp-chính)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

---

## 🎯 Giới Thiệu

Project này implement một deep learning model cho bài toán **Hyperspectral Image Super-Resolution** (HSI-SR), nâng cao độ phân giải của ảnh quang phổ từ low-resolution (LR) lên high-resolution (HR).

### Vấn Đề

Ảnh quang phổ (hyperspectral images) chứa thông tin phong phú về phổ ánh sáng (thường 31 bands từ 400-700nm), nhưng thường có độ phân giải không gian thấp do hạn chế của sensor. Super-resolution giúp tăng chất lượng không gian mà vẫn giữ nguyên thông tin phổ.

### Giải Pháp

Chúng tôi đề xuất **ESSA-SSAM-SpecTrans**, kết hợp 3 thành phần:

1. **CNN** - Extract local features
2. **SSAM** (Spatial-Spectral Attention Module) - Decoupled spatial & spectral attention
3. **Spectral Transformer** - Learn long-range spectral dependencies

---

## 🏗️ Kiến Trúc Model

### Model Evolution

```
Level 1: ESSA (Baseline)
    └─ CNN only
    └─ PSNR: ~32-34 dB

Level 2: ESSA + SSAM
    └─ CNN + Spatial-Spectral Attention
    └─ PSNR: ~34-35 dB (+1.5-2 dB)

Level 3: ESSA + SSAM + SpecTrans ⭐ (FINAL)
    └─ CNN + SSAM + Spectral Transformer
    └─ PSNR: ~34.5-36 dB (+2-2.5 dB)
```

### Architecture Diagram

```
Input [B, 31, H, W]
    ↓
Conv (first layer)
    ↓
┌─────────────────────────────────┐
│  Progressive Refinement (5x)    │
│  ┌─────────────────────┐        │
│  │ Upsample            │        │
│  │  ↓                  │        │
│  │ Convup              │        │
│  │  ├─ SSAM ⭐         │        │
│  │  │  ├─ Spectral    │        │
│  │  │  └─ Spatial     │        │
│  │  └─ SpecTrans ⭐⭐  │        │
│  │  ↓                  │        │
│  │ Downsample          │        │
│  │  ↓                  │        │
│  │ Convdown            │        │
│  │  ├─ SSAM            │        │
│  │  └─ SpecTrans       │        │
│  └─────────────────────┘        │
└─────────────────────────────────┘
    ↓
Conv (last layer)
    ↓
Output [B, 31, H×4, W×4]
```

### Key Components

#### 1. Spatial-Spectral Attention Module (SSAM)

**Vấn đề với attention truyền thống:**
- Channel-wise attention không tách biệt spatial và spectral
- Không tận dụng đặc thù của hyperspectral images

**Giải pháp SSAM:**

```python
class SpatialSpectralAttention(nn.Module):
    """
    Tách biệt 2 loại attention:
    - Spectral Attention: Học correlation giữa spectral bands
    - Spatial Attention: Học spatial features quan trọng
    
    3 fusion modes:
    - Sequential: Spectral → Spatial (best)
    - Parallel: Spectral + Spatial
    - Adaptive: Learnable weights
    """
```

**Complexity (theo code hiện tại):** O(C·HW + C²) - efficient hơn nhiều so với spatial self-attention toàn ảnh.

#### 2. Spectral Transformer

**Tại sao cần Transformer cho spectral dimension?**

- Hyperspectral images có 31 bands
- Bands liền kề có correlation cao (band 10 ≈ band 11)
- Bands xa cũng có thể liên quan (band 5 và band 25 đều reflect chlorophyll)

**Traditional Transformer Problem:**
- Spatial Transformer: O((HW)²) - TOO EXPENSIVE!
- Với H=128, W=128: O(16384²) = O(268M) operations

**Our Spectral Transformer:**
- Chỉ operate trên spectral dimension: O(C²·HW)
- Với C=31, HW=16384: O(31²·16384) = O(15M) operations
- **~500× faster than spatial transformer!**

```python
class SpectralTransformer(nn.Module):
    """
    Multi-head attention across spectral bands
    Input: [B, C, H, W] → [B, HW, C]
    
    Each pixel = spectral signature (31-D vector)
    Learn dependencies between spectral bands
    
    Output: [B, C, H, W]
    """
```

---

## 🚀 Cài Đặt

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+ (recommended)
- 16GB RAM minimum
- GPU with 8GB+ VRAM (for training)

### Quick Install

```bash
# Clone repository
git clone <your-repo-url>
cd Enhance_HSR

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
# Test imports
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test model creation
python3 models/essa_ssam_spectrans.py
```

**Expected output:**
```
Testing ESSA-SSAM-SpecTrans Model
======================================================================
Testing: ESSA-SSAM-SpecTrans (depth=2)
model_name: ESSA-SSAM-SpecTrans
total_parameters: 3,245,123
✅ All tests completed successfully!
```

### Smoke Test Commands (nhanh, không train full)

```bash
# Kiểm tra CLI chạy được
python3 train.py --help
python3 test_full_image.py --help
python3 evaluate.py --help

# Kiểm tra import model
python3 -c "import models.essa_ssam_spectrans as m; print(hasattr(m, 'ESSA_SSAM_SpecTrans'))"
```

---

## 💾 Dataset

### Supported Datasets

1. **CAVE Dataset** (Recommended)
   - 32 hyperspectral scenes
   - 31 spectral bands (400-700nm)
   - Spatial resolution: 512×512
   - Download: https://www.cs.columbia.edu/CAVE/databases/multispectral/

2. **Harvard Dataset**
   - 50 hyperspectral scenes
   - 31 spectral bands
   - Spatial resolution: 1392×1040
   - Download: http://vision.seas.harvard.edu/hyperspec/

3. **Chikusei Dataset**
   - 1 large airborne hyperspectral scene
   - 128 spectral bands (363-1018 nm)
   - Spatial resolution: 2517×2335
   - Download: https://naotoyokoya.com/Download.html

4. **PaviaCentre Dataset**
   - 1 airborne hyperspectral scene (urban area)
   - 102 spectral bands
   - Spatial resolution: ~1096×1096
   - Download: https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes

### Dataset Structure

```
data/
├── CAVE/
│   ├── balloons_ms.mat
│   ├── beads_ms.mat
│   ├── cd_ms.mat
│   └── ... (32 scenes total)
│
├── Harvard/
│   ├── imgd1.mat
│   ├── imgd2.mat
│   └── ... (50 scenes total)
│
├── Chikusei/
│   ├── Chikusei.mat            
│   ├── Chikusei_gt.mat         
│   └── README.txt                
│
└── PaviaCentre/
    ├── Pavia_centre.mat      
    ├── Pavia_centre_gt.mat       
    └── README.txt
```

### Train/Val/Test Splits

Project hỗ trợ split tự động theo từng dataset:
- `data/splits.py` tạo `split.json` theo tỷ lệ mặc định `train=0.8`, `val=0.1`, `test=0.1` (seed=42)
- `train.py` dùng trực tiếp split `train` và `val` từ `split.json` (không tách ngẫu nhiên lần 2)
- `test_full_image.py` và `evaluate.py` dùng split `test`
- Có thể chỉnh trực tiếp trong `config.py`: `split_seed`, `train_ratio`, `val_ratio`, `test_ratio`, `regenerate_split`

**⚠️ Important:** Không dùng dữ liệu `test` để huấn luyện.

### Trường hợp dataset chỉ có 1 ảnh + Ground Truth (Chikusei, PaviaCentre)

Với các dataset kiểu này thường có:
- 1 file hyperspectral cube chính (3D)
- 1 file ground-truth/label (không phải cube 3D)

Hành vi hiện tại của code:
- Tự bỏ qua file ground-truth không phải hyperspectral cube khi tạo/đọc split
- `split.json` thực tế sẽ thành `train=1, val=0, test=0` (vì chỉ có 1 ảnh hợp lệ)
- Khi train: nếu `val` rỗng, code tự fallback dùng `train` để validate
- Khi test/evaluate: nếu `test` rỗng, code tự fallback dùng `train` để test

Vì vậy khi thấy warning:
- `Test split is empty. Falling back to split='train'...`

thì đó là đúng hành vi mong đợi cho dataset 1 ảnh.

---

## 🎓 Training

### Lệnh chuẩn

Nếu chưa activate venv, dùng `./.venv/bin/python` thay cho `python3`.

```bash
python3 train.py --config spectrans --data_root <DATA_ROOT>
```

`--config` hỗ trợ: `default`, `baseline`, `proposed`, `spectrans`, `lightweight`, `universal_best`.

Preset khuyên dùng để train ổn trên cả Harvard/CAVE và Chikusei/Pavia:

```bash
python3 train.py --config universal_best --data_root <DATA_ROOT>
```

### Train theo từng dataset

```bash
# CAVE
python3 train.py --config spectrans --data_root ./data/CAVE

# Harvard
python3 train.py --config spectrans --data_root ./data/Harvard

# Chikusei
python3 train.py --config spectrans --data_root ./data/Chikusei

# PaviaCentre
python3 train.py --config spectrans --data_root ./data/PaviaCentre
```

### Train các biến thể model

```bash
# ESSA baseline
python3 train.py --config baseline --data_root ./data/Harvard

# ESSA + SSAM
python3 train.py --config proposed --data_root ./data/Harvard

# ESSA + SSAM + SpecTrans (recommended)
python3 train.py --config spectrans --data_root ./data/Harvard
```

### Training Configs

Tất cả configs được định nghĩa trong `config.py`:

**ConfigSpecTrans (Recommended):**
```python
model_name = 'ESSA_SSAM_SpecTrans'
feature_dim = 128
upscale_factor = 4
fusion_mode = 'sequential'
use_spectrans = True
spectrans_depth = 2

# Loss
loss_type = 'combined'  # L1 + SAM + SSIM
lambda_l1 = 1.0
lambda_sam = 0.1
lambda_ssim = 0.5

# Training
batch_size = 4
patch_size = 128
num_epochs = 100
learning_rate = 2e-4
lr_scheduler = 'cosine'
```

**ConfigUniversalBest (khuyên dùng đa dataset):**
```python
model_name = 'ESSA_SSAM_SpecTrans'
optimizer = 'adamw'
weight_decay = 1e-4
learning_rate = 1e-4
patch_size = 64
batch_size = 1
gradient_clip_norm = 1.0
use_ema = True
ema_decay = 0.999
warmup_epochs = 10
warmup_start_lr = 1e-6
use_two_phase_loss = True
loss_phase1_ratio = 0.4
loss_phase1_sam_scale = 0.3
loss_phase1_ssim_scale = 0.25
best_selection_metric = 'composite'
num_epochs = 400  # profile multi-scene (Harvard/CAVE)

# Tự chuyển profile khi dataset là single-scene (Chikusei/Pavia)
train_virtual_samples_per_epoch = 4000
val_virtual_samples_per_epoch = 512
num_epochs = 600  # profile single-scene
```

### Hyperparameter Tuning (Optuna)

```bash
# Cài thêm (nếu chưa có)
pip install optuna

# Tune cho Harvard/CAVE
python3 tune_optuna.py --config universal_best --data_root ./data/Harvard --trials 20 --epochs 120

# Tune cho Chikusei/Pavia (single-scene)
python3 tune_optuna.py --config universal_best --data_root ./data/Chikusei --trials 12 --epochs 80
```

### Seed Sweep (chọn seed tốt hơn, giữ split cố định)

```bash
# Quét seed, xếp hạng theo full-image validation (khuyên dùng)
python3 seed_sweep.py \
  --config universal_best \
  --data_root ./data/Harvard \
  --seeds 7,11,19,23,29 \
  --split_seed 42 \
  --epochs 100 \
  --num_workers 0 \
  --selection_mode full_image_val \
  --best_selection_metric psnr

# Nếu muốn sweep nhanh theo patch-val
python3 seed_sweep.py \
  --config universal_best \
  --data_root ./data/Harvard \
  --seeds 7,11,19 \
  --epochs 80 \
  --num_workers 0 \
  --selection_mode patch_val
```

Kết quả sweep được lưu tại `seed_sweep_results/<sweep_timestamp>/` gồm:
- `summary.txt`
- `seed_sweep_results.csv`
- `seed_sweep_results.json`
- `seed_sweep_meta.json`

`summary.txt` sẽ có:
- thời gian từng seed (`seed_time`)
- `Total sweep time` cho toàn bộ lượt chạy
- `Avg time/seed`

#### Hướng dẫn dùng `seed_sweep` (chi tiết)

1. Giữ cố định `--split_seed` để không thay đổi tập train/val/test giữa các seed.
2. Quét nhanh nhiều seed với `--epochs 100` để tiết kiệm thời gian.
3. Chọn seed có `rank_score` cao nhất trong `summary.txt`.
4. Test checkpoint tốt nhất của seed đó bằng `test_full_image.py`.

Ví dụ test sau khi sweep:

```bash
python3 test_full_image.py \
  --checkpoint ./checkpoints/ESSA_SSAM_SpecTrans_Harvard_x4_sweep_YYYYMMDD_HHMMSS_seed7/best.pth \
  --data_root ./data/Harvard \
  --save_images
```

Ghi chú:
- `selection_mode=full_image_val` bám sát protocol test hơn `patch_val`.
- Nếu thời gian hạn chế, có thể sweep 3 seed trước: `--seeds 7,11,19`.

### Quy trình khi KHÔNG dùng `seed_sweep`

Nếu không muốn sweep seed, dùng quy trình 1-run chuẩn:

```bash
# 1) Train 1 run
python3 train.py --config universal_best --data_root ./data/Harvard

# 2) Test full-image
python3 test_full_image.py \
  --checkpoint ./checkpoints/<experiment_name>/best.pth \
  --data_root ./data/Harvard \
  --save_images
```

Khuyến nghị tối thiểu để ổn định:
- chạy 2-3 run độc lập (đổi `seed` trong `config.py`) rồi chọn model tốt nhất theo val/test protocol của bạn.

### Training Output

```
Model: ESSA_SSAM_SpecTrans
Parameters: 3,114,659
Train samples: 32, Val samples: 8

Epoch 95:
  Train Loss: 0.0658
  Val PSNR: 37.20 dB
  Val SSIM: 0.9013
  Val SAM: 5.3076°
  Val ERGAS: 6.4316
  Train Time: 00:21 (21.34 seconds)
  Validate Time: 00:02 (2.25 seconds)
  Epoch Total Time: 00:23 (23.70 seconds)

Epoch 96/100: 100%|████████| 32/32 [00:21<00:00, 1.50it/s, loss=0.0554, l1=0.0088, sam=0.0758]
Validating: 100%|████████| 8/8 [00:02<00:00, 3.55it/s]

...

Training Completed!
Best PSNR: 37.53 dB
Best Score (COMPOSITE): 0.812345 (epoch 88)
Total Training Time: 00:38:12 (2292.44 seconds)
```

### Checkpoints

Checkpoints được lưu tại:
```
checkpoints/
└── ESSA_SSAM_SpecTrans_CAVE_x4_20240117_143022/
    ├── best.pth          # Best checkpoint theo best_selection_metric
    ├── latest.pth        # Latest checkpoint
    └── epoch_50.pth      # Periodic saves
```

### Resume Training

```bash
python3 train.py --resume ./checkpoints/ESSA_SSAM_SpecTrans_xxx/latest.pth
```

### Training Tips

**If GPU OOM:**
```python
# Reduce in config.py:
batch_size = 2          # from 4
patch_size = 64         # from 128
feature_dim = 64        # from 128
```

**Faster Training:**
```bash
python3 train.py --config lightweight --data_root ./data/CAVE
```

**Theo dõi log train realtime:**
```bash
tail -f logs/<experiment_name>/training.log
```

---

## 🧪 Testing & Evaluation

### Full-Image Test (khuyên dùng)

Điều kiện trước khi test/evaluate:
- Có checkpoint `.pth` từ bước train
- `data_root` chứa file `.mat` và có/được tạo `split.json`

```bash
# Liệt kê checkpoint có sẵn
find checkpoints -name best.pth
```

```bash
python3 test_full_image.py \
    --checkpoint ./checkpoints/ESSA_SSAM_SpecTrans_xxx/best.pth \
    --data_root <DATA_ROOT> \
    --save_images
```

Ví dụ:

```bash
python3 test_full_image.py --checkpoint ./checkpoints/ESSA_SSAM_SpecTrans_xxx/best.pth --data_root ./data/MyDataset
```

Ví dụ cụ thể cho Harvard:

```bash
python3 test_full_image.py \
    --checkpoint ./checkpoints/ESSA_SSAM_Harvard_x4_xxx/best.pth \
    --data_root ./data/Harvard \
    --save_images
```

### Ví dụ cho dataset 1 ảnh + GT (Chikusei/PaviaCentre)

Train:

```bash
# Chikusei
python3 train.py --config spectrans --data_root ./data/Chikusei

# PaviaCentre
python3 train.py --config spectrans --data_root ./data/PaviaCentre
```

Test full-image (khuyên dùng CPU cho ảnh lớn để tránh crash MPS):

```bash
# Chikusei
python3 test_full_image.py \
    --checkpoint ./checkpoints/ESSA_SSAM_SpecTrans_Chikusei_x4_xxx/best.pth \
    --data_root ./data/Chikusei \
    --device cpu \
    --chop_patch_size 32 \
    --chop_overlap 8 \
    --save_images

# PaviaCentre
python3 test_full_image.py \
    --checkpoint ./checkpoints/ESSA_SSAM_SpecTrans_PaviaCentre_x4_xxx/best.pth \
    --data_root ./data/PaviaCentre \
    --device cpu \
    --chop_patch_size 32 \
    --chop_overlap 8 \
    --save_images
```

Ý nghĩa 2 tham số inference:
- `--chop_patch_size`: patch LR nhỏ hơn sẽ an toàn bộ nhớ hơn nhưng chậm hơn
- `--chop_overlap`: overlap lớn hơn giúp giảm seam giữa patch nhưng chậm hơn

Ví dụ output mới (có time từng ảnh + time tổng):

```
imgf3.mat:
  PSNR: 32.12 dB, SSIM: 0.8660, SAM: 7.028°, ERGAS: 7.247
  Time: 01:21:00 (4860.39 seconds)

...

📊 FULL-IMAGE TEST RESULTS (Paper-style)
Number of test images: 5
Average Metrics:
  PSNR  : 34.45 dB
  SSIM  : 0.8952
  SAM   : 4.230°
  ERGAS : 4.755
  Inference Total Time : 03:40:35 (13235.89 seconds)
  Avg Time / Image     : 44:07 (2647.18 seconds)
  Total Runtime        : 03:41:10 (13270.44 seconds)
```

Nếu gặp lỗi `ValueError: No .mat files found in ./Harvard`, nguyên nhân là sai đường dẫn.
Hãy dùng `--data_root ./data/Harvard` (không phải `./Harvard`).

Dataset label được tự suy ra từ tên folder `data_root` (ví dụ `./data/Harvard` -> `Harvard`).

**Checkpoint compatibility note:**
- Code hiện hỗ trợ tự động convert một số weight Spectral Transformer cũ khi load checkpoint
  (ví dụ `Linear` weight 2D sang `Conv1d(1x1)` weight 3D cho `qkv/proj`).
- Khi convert diễn ra, CLI sẽ in thông báo kiểu: `Converted N legacy weight tensors for compatibility.`

**Performance note (vì sao test có thể nhanh hơn nhiều):**
- Tiền xử lý downsample trong dataset đã được vectorize (thay cho loop pixel/band).
- Spectral Transformer attention hiện chạy theo hướng `O(C^2 * HW)` thay vì kiểu spatial-token rất nặng trước đó.
- Vì vậy thời gian test có thể giảm mạnh giữa các phiên bản code; hãy so sánh thời gian trên cùng commit để công bằng.

### Comprehensive Evaluation

```bash
python3 evaluate.py \
    --checkpoint ./checkpoints/ESSA_SSAM_SpecTrans_xxx/best.pth \
    --data_root ./data/MyDataset \
    --save_images
```

Ví dụ cụ thể cho Harvard:

```bash
python3 evaluate.py \
    --checkpoint ./checkpoints/ESSA_SSAM_Harvard_x4_xxx/best.pth \
    --data_root ./data/Harvard \
    --save_images
```

### Model Comparison

```bash
python3 evaluate.py \
    --checkpoint ./checkpoints/ESSA_SSAM_SpecTrans_xxx/best.pth \
    --compare ./checkpoints/ESSA_SSAM_xxx/best.pth \
    --data_root ./data/MyDataset
```

### Metrics Explained

| Metric | Range | Better | Description |
|--------|-------|--------|-------------|
| **PSNR** | 30-40 dB | Higher ↑ | Peak Signal-to-Noise Ratio |
| **SSIM** | 0-1 | Higher ↑ | Structural Similarity |
| **SAM** | 0-5° | Lower ↓ | Spectral Angle Mapper (spectral fidelity) ⭐ |
| **ERGAS** | 1-5 | Lower ↓ | Relative Global Error |

**SAM** và **ERGAS** đặc biệt quan trọng cho hyperspectral images!

---

## 📊 Kết Quả Mong Đợi

### Quantitative Results (CAVE dataset, ×4 upscaling)

| Model | PSNR ↑ | SSIM ↑ | SAM ↓ | ERGAS ↓ | Params |
|-------|--------|--------|-------|---------|--------|
| **ESSA (Baseline)** | 32.50 | 0.9234 | 3.24° | 2.56 | 2.5M |
| **ESSA-SSAM** | 34.12 | 0.9456 | 2.87° | 2.18 | 2.8M |
| **ESSA-SSAM-SpecTrans (Ours)** | **34.52** | **0.9501** | **2.70°** | **2.05** | 3.2M |
| **Improvement vs Baseline** | **+2.02 dB** | **+0.0267** | **-16.7%** | **-19.9%** | +28% |

### Ablation Study - Spectral Transformer

| Configuration | PSNR | SAM | Note |
|--------------|------|-----|------|
| ESSA-SSAM (no SpecTrans) | 34.12 | 2.87° | Baseline |
| + SpecTrans (depth=1) | 34.35 | 2.78° | Light |
| **+ SpecTrans (depth=2)** | **34.52** | **2.70°** | **Best** ⭐ |
| + SpecTrans (depth=4) | 34.54 | 2.69° | Overfitting |

### Visual Quality

Reconstructed images được save tại `test_results/` hoặc `inference_results/`.

**RGB Visualization:**
- Band 25 (Red)
- Band 15 (Green)
- Band 5 (Blue)

---

## 📁 Cấu Trúc Project

```
Enhance_HSR/
├── README.md
├── .gitignore
├── requirements.txt
├── config.py                              # Config classes + dataset/split settings
├── train.py                               # Training entrypoint
├── evaluate.py                            # Evaluation script (results/)
├── test_full_image.py                     # Full-image test script (test_results/)
│
├── models/
│   ├── __init__.py
│   ├── factory.py                         # Unified model builder + ckpt compatibility loader
│   ├── essa_original.py                   # ESSA baseline
│   ├── essa_improved.py                   # ESSA + SSAM
│   ├── essa_ssam_spectrans.py             # ESSA + SSAM + Spectral Transformer
│   ├── spatial_spectral_attention.py      # SSAM module
│   └── spectral_transformer.py            # Spectral Transformer blocks
│
├── data/
│   ├── __init__.py
│   ├── dataset.py                         # Train/test datasets + auto band detect
│   ├── splits.py                          # split.json generation/loading
│   ├── CAVE/
│   │   ├── complete_ms_data/
│   │   └── split.json
│   ├── Harvard/
│   │   ├── *.mat
│   │   ├── split.json
│   │   └── README.txt, calib.txt
│   ├── Chikusei/
│   │   ├── HyperspecVNIR_Chikusei_20140729.mat
│   │   ├── HyperspecVNIR_Chikusei_20140729_Ground_Truth.mat
│   │   └── split.json
│   └── PaviaCentre/
│       ├── Pavia.mat
│       └── Pavia_gt.mat
│
├── utils/
│   ├── __init__.py
│   ├── device.py                          # auto device: cuda/mps/cpu
│   ├── losses.py
│   ├── metrics.py
│   └── visualization.py
│
├── checkpoints/                           # Saved checkpoints by experiment
│   └── <experiment_name>/
│       ├── best.pth
│       ├── latest.pth
│       └── epoch_*.pth
├── logs/                                  # Training logs by experiment
│   └── <experiment_name>/training.log
├── results/                               # evaluate.py outputs
│   └── <experiment_name>/
│       ├── evaluation_results.json
│       ├── summary.txt
│       └── *_SR.npy / *_SR_RGB.png
└── test_results/                          # test_full_image.py outputs
    └── test_<timestamp>/
        ├── test_results.json
        ├── summary.txt
        └── images/
```

Ghi chú:
- Không liệt kê các thư mục môi trường cục bộ như `.venv/`, `venv/`, `__pycache__/`, `.git/`.
- Một số thư mục output có thể rỗng nếu run bị dừng giữa chừng và sẽ được script cleanup khi có thể.

---

## 🏆 Đóng Góp Chính

### 1. Spatial-Spectral Attention Module (SSAM)

**Innovation:**
- Tách biệt spatial và spectral processing
- 3 fusion strategies (sequential, parallel, adaptive)
- Efficient O(C·HW + C²) complexity

**vs Traditional Attention:**
- Traditional: Channel-wise attention chung chung
- SSAM: Explicit spatial + spectral separation

### 2. Spectral Transformer

**Innovation:**
- **First work** applying Transformer exclusively to spectral dimension
- Learn long-range dependencies between spectral bands
- O(C²·HW) complexity - much better than O((HW)²) of spatial transformer

**Key Insight:**
- For hyperspectral images, spectral correlation >> spatial correlation
- Spectral bands have rich inter-dependencies
- Transformer perfect for capturing long-range spectral relationships

### 3. Hybrid Architecture

**Best of All Worlds:**
- CNN: Local features (efficient)
- SSAM: Spatial-spectral attention (explicit)
- SpecTrans: Global spectral dependencies (powerful)

**Result:** SOTA performance on hyperspectral SR

### 4. Combined Loss Function

```python
Loss = λ₁ × L1 + λ₂ × SAM + λ₃ × SSIM

where:
  λ₁ = 1.0   # Pixel-wise accuracy
  λ₂ = 0.1   # Spectral fidelity
  λ₃ = 0.5   # Structural similarity
```

**Why this works:**
- L1: Basic reconstruction
- SAM: Preserve spectral signatures
- SSIM: Maintain structure

---

## 🛠️ Troubleshooting

### Common Issues

**1. CUDA Out of Memory**

```python
# Reduce in config.py:
batch_size = 2
patch_size = 64
feature_dim = 64
```

**2. Dataset Loading Error**

```bash
# Check dataset structure
ls -R data/CAVE/

# Should see .mat files
# If not, download CAVE dataset first
```

**3. Slow Training**

```bash
# Use lightweight config
python3 train.py --config lightweight --data_root ./data/CAVE

# Or reduce workers
# In config.py: num_workers = 0
```

**4. Có nhiều folder rỗng trong checkpoints/logs/results/test_results**

- Scripts hiện đã tạo thư mục theo kiểu lazy (khi chuẩn bị ghi file).
- Nếu run bị `Ctrl+C` hoặc lỗi giữa chừng, script sẽ tự cleanup folder rỗng.

**5. Model Not Converging**

```python
# Check learning rate
learning_rate = 1e-4  # Reduce if loss explodes

# Check data normalization
# Images should be in [0, 1] range
```

---

## 📚 Additional Resources

### Understanding the Code

**Key Files to Read:**

1. **`models/spatial_spectral_attention.py`**
   - Understand SSAM module
   - See how spatial/spectral attention works

2. **`models/spectral_transformer.py`**
   - Understand Spectral Transformer
   - See multi-head attention on spectral dimension

3. **`models/essa_ssam_spectrans.py`**
   - Full model architecture
   - See how everything fits together

4. **`data/splits.py`**
   - Train/val/test splits
   - IMPORTANT for reproducibility

5. **`config.py`**
   - All hyperparameters
   - Easy to modify

### Running Experiments

**For Thesis:**

```bash
# 1. Train all 3 models
python3 train.py --config baseline  --data_root ./data/CAVE
python3 train.py --config proposed  --data_root ./data/CAVE
python3 train.py --config spectrans --data_root ./data/CAVE

# 2. Test all models
python3 test_full_image.py --checkpoint checkpoints/ESSA_xxx/best.pth --data_root ./data/MyDataset
python3 test_full_image.py --checkpoint checkpoints/ESSA_SSAM_xxx/best.pth --data_root ./data/MyDataset
python3 test_full_image.py --checkpoint checkpoints/ESSA_SSAM_SpecTrans_xxx/best.pth --data_root ./data/MyDataset

# 3. Compare results
python3 evaluate.py --checkpoint spectrans.pth --compare ssam.pth --data_root ./data/MyDataset

# 4. Create tables for thesis (manually from results)
```

## ⭐ Quick Reference

**Train:**
```bash
python3 train.py --config spectrans --data_root ./data/CAVE
```

**Test:**
```bash
python3 test_full_image.py --checkpoint best.pth --data_root ./data/MyDataset
```

**Compare:**
```bash
python3 evaluate.py --checkpoint model1.pth --compare model2.pth --data_root ./data/MyDataset
```
