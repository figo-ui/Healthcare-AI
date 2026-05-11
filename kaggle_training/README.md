# Kaggle Training Package — Skin Lesion Classifier

## What's in this folder

```
kaggle_training/
├── data/                          ← Upload this as a Kaggle Dataset
│   ├── train_images.npy           (7007, 64, 64, 3) float32 [0,1]
│   ├── train_labels.npy           (7007,) int64
│   ├── val_images.npy             (1003, 64, 64, 3)
│   ├── val_labels.npy             (1003,)
│   ├── test_images.npy            (2005, 64, 64, 3)
│   ├── test_labels.npy            (2005,)
│   ├── label_map.json             7 class names
│   └── metadata.json              class distribution + normalization stats
└── train_skin_lesion.ipynb        ← Run this on Kaggle GPU
```

## Classes (7)

| ID | Label | Train count |
|----|-------|-------------|
| 0 | actinic keratoses and intraepithelial carcinoma | 228 |
| 1 | basal cell carcinoma | 359 |
| 2 | benign keratosis-like lesions | 769 |
| 3 | dermatofibroma | 80 |
| 4 | melanoma | 779 |
| 5 | melanocytic nevi | 4,693 |
| 6 | vascular lesions | 99 |

**Imbalance ratio: 58.7x** — handled by WeightedRandomSampler + class-weighted loss.

---

## Step-by-step: Upload & Train on Kaggle

### Step 1 — Upload the dataset

1. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets)
2. Click **New Dataset**
3. Upload the entire `kaggle_training/data/` folder
4. Name it exactly: **`dermamnist-64px`**
5. Set visibility to **Private**
6. Click **Create**

### Step 2 — Create a new notebook

1. Go to [kaggle.com/code](https://www.kaggle.com/code)
2. Click **New Notebook**
3. Click **File → Import Notebook** → upload `train_skin_lesion.ipynb`

### Step 3 — Add the dataset

1. In the notebook, click **Add Data** (right panel)
2. Search for `dermamnist-64px` under **Your Datasets**
3. Click **Add** — it will mount at `/kaggle/input/dermamnist-64px/`

### Step 4 — Enable GPU

1. Click **Settings** (right panel) → **Accelerator**
2. Select **GPU T4 x2** (free tier) or **P100**
3. Click **Save**

### Step 5 — Run the notebook

1. Click **Run All** (or Shift+Enter through each cell)
2. Expected runtime: ~25–40 min on T4 GPU
3. Expected results:
   - Phase 1 (5 epochs): Val F1 ~0.35–0.45
   - Phase 2 (up to 30 epochs): Val F1 ~0.60–0.70
   - **Target: Test macro-F1 ≥ 0.65** (baseline was 0.449)

### Step 6 — Download the output files

From the **Output** tab, download:
- `skin_cnn_torch.pt`
- `image_labels.json`
- `image_training_metrics.json`

### Step 7 — Deploy locally

Copy the 3 downloaded files to `backend/models/`:

```bash
# Windows
copy skin_cnn_torch.pt "C:\Users\hp\Desktop\AI assistant\backend\models\"
copy image_labels.json "C:\Users\hp\Desktop\AI assistant\backend\models\"
copy image_training_metrics.json "C:\Users\hp\Desktop\AI assistant\backend\models\"
```

Then restart Django:
```bash
cd "C:\Users\hp\Desktop\AI assistant\backend"
python manage.py runserver
```

The model loads automatically — no code changes needed.

---

## What the notebook does

### Architecture: EfficientNet-B3 (ImageNet pretrained)
- Input: 64×64 images upsampled to 224×224
- 2-phase fine-tuning:
  - **Phase 1** (5 epochs): Freeze backbone, train head only
  - **Phase 2** (up to 30 epochs): Unfreeze last 3 feature blocks, fine-tune with differential LRs

### Balancing strategy
- `WeightedRandomSampler` — each class gets equal expected frequency per batch
- `CrossEntropyLoss(weight=class_weights)` — additional loss weighting
- `label_smoothing=0.1` — prevents overconfidence on majority class

### Augmentation (training only)
- Random horizontal + vertical flip
- Random rotation ±20°
- Color jitter (brightness, contrast, saturation, hue)
- Random affine (translate + scale)

### Regularization
- Dropout 0.4 + 0.3 in classifier head
- Weight decay 1e-4
- Gradient clipping (max norm 1.0)
- Early stopping (patience=8)

---

## Expected output quality

| Metric | Baseline (DermaCNN 28×28) | Expected (EfficientNet-B3 224×224) |
|--------|--------------------------|-------------------------------------|
| Test macro-F1 | 0.449 | **0.60–0.70** |
| Test accuracy | 0.5796 | **0.70–0.80** |
| Train-Val gap | unknown | **< 0.10** |
| ECE | unknown | **< 0.15** |

The biggest gains come from:
1. 224×224 input vs 28×28 — EfficientNet can actually see the lesion texture
2. WeightedRandomSampler — minority classes (dermatofibroma: 80 samples) get proper training signal
3. 2-phase fine-tuning — avoids catastrophic forgetting of ImageNet features
