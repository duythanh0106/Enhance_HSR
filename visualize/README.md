# Visualization Code — ESSA-SSAM-SpecTrans

## Cài đặt
```bash
pip install numpy matplotlib scipy pillow
```

---

## Cấu trúc thực tế của bạn

```
/Users/thanh.pd/Enhance_HSR2/
├── dataset/
│   └── CAVE/
│       └── real_and_fake_apples_ms/
│           └── real_and_fake_apples_ms/
│               ├── real_and_fake_apples_ms_01.png   ← band 1 (GT)
│               └── ...real_and_fake_apples_ms_31.png
└── test_results/
    ├── best_cave_x2/images/real_and_fake_apples_ms/
    │   ├── real_and_fake_apples_ms.npy    ← SR đề xuất (31,508,508)
    │   ├── real_and_fake_apples_ms_HR.png
    │   └── real_and_fake_apples_ms_LR.png
    └── cave_baseline_x2/images/real_and_fake_apples_ms/
        └── real_and_fake_apples_ms.npy    ← SR ESSA gốc
```

---

## Bước 1 — Chỉnh utils.py

Mở `utils.py`, chỉnh 2 dòng:
```python
DATASET_ROOT = Path("/Users/thanh.pd/Enhance_HSR2/dataset")
RESULTS_ROOT = Path("/Users/thanh.pd/Enhance_HSR2/test_results")
```

Và điền tên folder kết quả vào `RESULT_FOLDERS` cho đủ dataset/scale của bạn.

---

## Bước 2 — Chạy

### 01 — Spectral Signature Plot
```bash
python 01_spectral_signature.py --dataset CAVE --scale 2 --scene 0
python 01_spectral_signature.py --dataset CAVE --scale 2 --scene real_and_fake_apples_ms
python 01_spectral_signature.py --all
```

### 02 — False Color RGB
```bash
python 02_false_color_rgb.py --dataset CAVE --scale 2 --scene 0
python 02_false_color_rgb.py --dataset CAVE --scale 2 --bands 20,10,2
python 02_false_color_rgb.py --all
```

Band RGB gợi ý: CAVE/Harvard → 20,10,2 | Chikusei → 70,40,10 | Pavia → 60,35,10

### 03 — Ablation Bar Chart
```bash
# Điền số liệu "ESSA + SSAM" vào ABLATION_DATA trước
python 03_ablation_chart.py --dataset CAVE --scale 2
python 03_ablation_chart.py --multi --metric PSNR --scale 2
python 03_ablation_chart.py --multi --metric SAM  --scale 2
python 03_ablation_chart.py --all
```

---

## Output → thư mục figures/
- PDF: chèn vào khoá luận không mất nét
- PNG: xem nhanh

## Lưu ý
- Nếu .npy shape là (H,W,C): thêm `arr = arr.transpose(2,0,1)` trong load_sr()
- Nếu GT không có subfolder lồng nhau: chỉnh load_gt() trong utils.py
