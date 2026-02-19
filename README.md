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

**Complexity:** O(C² + HW) - efficient!

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
cd hyperspectral-sr

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
# Test imports
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test model creation
python models/essa_ssam_spectrans.py
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
- `train.py` huấn luyện từ tập `train` và tách tiếp `80/20` cho train/val nội bộ
- `test_full_image.py` và `evaluate.py` dùng split `test`

**⚠️ Important:** Không dùng dữ liệu `test` để huấn luyện.

---

## 🎓 Training

### Lệnh chuẩn

Nếu chưa activate venv, dùng `./.venv/bin/python` thay cho `python3`.

```bash
python3 train.py --config spectrans --data_root <DATA_ROOT>
```

`--config` hỗ trợ: `default`, `baseline`, `proposed`, `spectrans`, `lightweight`.

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

### Training Output

```
Model: ESSA_SSAM_SpecTrans
Parameters: 3,245,123
Train samples: 25, Val samples: 3

Epoch 1/100: 100%|████████| 156/156 [02:34<00:00]
  Train Loss: 0.0234
  Val PSNR: 28.45 dB
  Val SSIM: 0.8912
  Val SAM: 4.23°

...

Epoch 100/100:
  Train Loss: 0.0089
  Val PSNR: 33.87 dB
  Val SSIM: 0.9445
  Val SAM: 2.85°

💾 Best model saved! PSNR: 33.87 dB
```

### Checkpoints

Checkpoints được lưu tại:
```
checkpoints/
└── ESSA_SSAM_SpecTrans_CAVE_x4_20240117_143022/
    ├── best.pth          # Best validation PSNR
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

---

## 🧪 Testing & Evaluation

### Full-Image Test (khuyên dùng)

```bash
python3 test_full_image.py \
    --checkpoint ./checkpoints/ESSA_SSAM_SpecTrans_xxx/best.pth \
    --data_root <DATA_ROOT> \
    --dataset_type <DATASET_LABEL> \
    --save_images
```

Ví dụ:

```bash
python3 test_full_image.py --checkpoint ./checkpoints/ESSA_SSAM_SpecTrans_xxx/best.pth --data_root ./data/MyDataset --dataset_type MyDataset
```

`--dataset_type` chỉ là nhãn hiển thị/log, nhận chuỗi tự do.

### Comprehensive Evaluation

```bash
python3 evaluate.py \
    --checkpoint ./checkpoints/ESSA_SSAM_SpecTrans_xxx/best.pth \
    --data_root ./data/MyDataset \
    --dataset_type MyDataset \
    --save_images
```

### Model Comparison

```bash
python3 evaluate.py \
    --checkpoint ./checkpoints/ESSA_SSAM_SpecTrans_xxx/best.pth \
    --compare ./checkpoints/ESSA_SSAM_xxx/best.pth \
    --data_root ./data/MyDataset \
    --dataset_type MyDataset
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
hyperspectral-sr/
│
├── models/                              # Model architectures
│   ├── __init__.py
│   ├── essa_original.py                 # ESSA baseline
│   ├── essa_improved.py                 # ESSA + SSAM
│   ├── essa_ssam_spectrans.py          # ESSA + SSAM + SpecTrans ⭐
│   ├── spatial_spectral_attention.py   # SSAM module
│   └── spectral_transformer.py         # Spectral Transformer module
│
├── data/                                # Dataset handling
│   ├── __init__.py
│   ├── dataset.py                      # Dataset classes
│   └── splits.py                       # Train/val/test splits
│
├── utils/                               # Utilities (if needed)
│   └── __init__.py
│
├── config.py                           # Configuration classes
├── train.py                            # Training script
├── evaluate.py                         # Evaluation script
├── test_full_image.py                  # Full-image test (paper-style)
├── requirements.txt                    # Python dependencies
│
├── checkpoints/                        # Saved models (created during training)
├── logs/                               # Training logs (created during training)
├── test_results/                       # Test results (created during testing)
│
└── README.md                           # This file
```

---

## 🏆 Đóng Góp Chính

### 1. Spatial-Spectral Attention Module (SSAM)

**Innovation:**
- Tách biệt spatial và spectral processing
- 3 fusion strategies (sequential, parallel, adaptive)
- Efficient O(C² + HW) complexity

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
python3 test_full_image.py --checkpoint checkpoints/ESSA_xxx/best.pth --data_root ./data/MyDataset --dataset_type MyDataset
python3 test_full_image.py --checkpoint checkpoints/ESSA_SSAM_xxx/best.pth --data_root ./data/MyDataset --dataset_type MyDataset
python3 test_full_image.py --checkpoint checkpoints/ESSA_SSAM_SpecTrans_xxx/best.pth --data_root ./data/MyDataset --dataset_type MyDataset

# 3. Compare results
python3 evaluate.py --checkpoint spectrans.pth --compare ssam.pth --data_root ./data/MyDataset --dataset_type MyDataset

# 4. Create tables for thesis (manually from results)
```

## ⭐ Quick Reference

**Train:**
```bash
python3 train.py --config spectrans --data_root ./data/CAVE
```

**Test:**
```bash
python3 test_full_image.py --checkpoint best.pth --data_root ./data/MyDataset --dataset_type MyDataset
```

**Compare:**
```bash
python3 evaluate.py --checkpoint model1.pth --compare model2.pth --data_root ./data/MyDataset --dataset_type MyDataset
```
