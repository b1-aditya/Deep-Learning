# 😷 Face Mask Detection — Deep Technical Summary

This document covers both models built for this project: a **custom CNN (Sequential)** and a **MobileNetV2 (Transfer Learning)** model — including architecture, training strategy, design decisions, and evaluation approach.

---

## 1. Problem Statement

Binary image classification: given a face image, predict whether the person is wearing a mask (`with_mask`) or not (`without_mask`).

**Dataset:** [Face Mask Dataset – Kaggle](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)
- ~7,553 images total
- Train: 6,043 images | Validation: 1,510 images (80-20 split)
- Class indices: `{'with_mask': 0, 'without_mask': 1}`
- Loaded via `kagglehub` into Google Colab

---

## 2. Data Pipeline (Common to Both Models)

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)
val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
```

- **Augmentation** (rotation, shift, shear, zoom, flip) applied only to training data — reduces overfitting and improves generalization to real-world face angles.
- **Validation data** is only rescaled — no augmentation, to get a true measure of performance.
- `flow_from_directory` automatically infers labels from folder names (`with_mask/`, `without_mask/`) — no manual labeling needed.
- `seed=42` used everywhere for reproducibility.

---

## 3. Model 1 — Custom Sequential CNN

### Architecture

| Layer | Details |
|---|---|
| Input | 128 × 128 × 3 |
| Conv Block 1 | Conv2D(32, 3×3, relu) → BatchNorm → MaxPool(2×2) |
| Conv Block 2 | Conv2D(64, 3×3, relu) → BatchNorm → MaxPool(2×2) |
| Conv Block 3 | Conv2D(128, 3×3, relu) → BatchNorm → MaxPool(2×2) |
| Flatten | — |
| Dense | Dense(128, relu) → Dropout(0.5) |
| Output | Dense(1, sigmoid) |

### Design Reasoning
- **Progressive filter increase (32→64→128):** early layers learn low-level features (edges, textures), deeper layers learn complex patterns (mask shapes, face structure).
- **BatchNormalization after each conv layer:** stabilizes training, allows higher learning rates, reduces internal covariate shift.
- **Dropout(0.5)** before the output layer: prevents overfitting by randomly disabling 50% of neurons during training.
- **Sigmoid output:** appropriate for binary classification (outputs probability 0–1).

### Training Configuration
- Optimizer: Adam (default lr)
- Loss: `binary_crossentropy`
- Epochs: up to 20 (with early stopping)
- Callbacks:
  - `EarlyStopping(patience=5, restore_best_weights=True)` — stops training if val_loss doesn't improve for 5 epochs, restores best weights
  - `ModelCheckpoint('mask_detector_best.h5', monitor='val_accuracy', save_best_only=True)` — saves only the best-performing model
  - `ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6)` — halves learning rate if val_loss plateaus for 3 epochs

### Result
- Validation accuracy: ~95%
- Output files: `mask_detector_best.h5`, `mask_detector_final.h5`

---

## 4. Model 2 — MobileNetV2 (Transfer Learning)

### Why Transfer Learning?
A custom CNN learns features from scratch, requiring lots of data and epochs. MobileNetV2 is pretrained on **ImageNet** (1.4M images, 1000 classes) — it already knows general visual features (edges, shapes, textures, faces). We **reuse this knowledge** and only retrain the parts relevant to our task — faster convergence, better accuracy, less data needed.

### Architecture

| Layer | Details |
|---|---|
| Base | MobileNetV2 (`include_top=False`, `weights='imagenet'`, input 224×224×3) |
| Pooling | GlobalAveragePooling2D |
| Normalization | BatchNormalization |
| Dense | Dense(128, relu) → Dropout(0.5) |
| Output | Dense(1, sigmoid) |

### Two-Phase Training Strategy

**Phase 1 — Feature Extraction (Frozen Base)**
- `base_model.trainable = False` — all MobileNetV2 weights frozen
- Only the custom head (Dense layers) is trained
- Optimizer: Adam, `lr = 1e-3`
- Epochs: up to 10 (with EarlyStopping)
- Goal: let the new classification head adapt to the mask/no-mask task using MobileNetV2's existing feature representations

**Phase 2 — Fine-Tuning (Partial Unfreeze)**
- `base_model.trainable = True`, but all layers except the **last 30** are re-frozen
- Optimizer: Adam, `lr = 1e-4` (10x lower — critical to avoid destroying pretrained weights)
- Epochs: up to 10 more
- Goal: let the deeper layers slightly adjust to mask-specific features (e.g., occlusion patterns around mouth/nose) without catastrophic forgetting

### Why 224×224 Input?
MobileNetV2's pretrained weights expect this resolution (ImageNet standard) — using a different size would mismatch the architecture and discard pretrained knowledge benefits.

### Why Lower LR in Phase 2?
Pretrained weights are already close to optimal for general features. A large LR during fine-tuning would cause large weight updates that destroy this knowledge ("catastrophic forgetting"). A small LR (1e-4) makes gentle adjustments.

### Result
- Validation accuracy: ~98%+
- Output files: `mask_mobilenet_best.h5`, `mask_mobilenet_final.h5`

---

## 5. Model Comparison — Statistical Validation

Instead of comparing raw accuracy numbers, both models were evaluated on the **same validation set** and compared using **McNemar's Test** — appropriate for paired binary classification outcomes.

| Aspect | Custom CNN | MobileNetV2 |
|---|---|---|
| Input size | 128×128 | 224×224 |
| Parameters | Trained from scratch | Pretrained (ImageNet) + fine-tuned |
| Training time | Faster per epoch (smaller model) | Slower per epoch (larger backbone) |
| Validation accuracy | ~95% | ~98%+ |
| McNemar's p-value | — | < 0.05 (significant improvement) |

**Conclusion:** MobileNetV2 significantly outperforms the custom CNN — improvement is statistically validated, not just a lucky split.

---

## 6. Deployment — Real-Time Inference

### Webcam (OpenCV)
- Haar Cascade (`haarcascade_frontalface_default.xml`) detects faces in each video frame
- Each detected face is cropped, resized to model input size, normalized (`/255.0`), and passed to the model
- Bounding box drawn: 🟩 green = "Mask On", 🔴 red = "No Mask", with confidence %

### Streamlit Web App
- Two input modes: **Upload Image** and **Live Camera** (`st.camera_input`)
- **Critical fix:** Haar Cascade face detection + cropping applied **before** prediction — because training images were tightly cropped faces, while raw camera input includes full body/background. Matching the preprocessing pipeline between training and inference was essential for correct predictions.
- Output label mapping respects `class_indices = {'with_mask': 0, 'without_mask': 1}`:
  - prediction ≥ 0.5 → "No Mask"
  - prediction < 0.5 → "Mask Detected"

---

## 7. Key Engineering Lessons (Interview-Ready)

1. **Train/inference distribution mismatch** is one of the most common real-world ML bugs — a model can have 98% validation accuracy yet fail completely in production if the input preprocessing doesn't match training data (face-cropped vs full-frame images).
2. **Class index mapping** must always be explicitly checked (`train_gen.class_indices`) — assuming label order leads to silently inverted predictions.
3. **Transfer learning with staged unfreezing + LR decay** is the standard recipe for fine-tuning pretrained CNNs — freeze first, fine-tune last layers with a much smaller learning rate.
4. **Paired statistical tests (McNemar's)** are more rigorous than comparing raw accuracy when evaluating two models on the same dataset.
5. **Git LFS** is required when version-controlling large model files (`.h5`) on GitHub (25MB limit on regular files).

---

## 8. Repository Structure

```
Deep-Learning/
├── Face Mask Detection/                      # Custom CNN
│   ├── mask_detector_best.h5
│   ├── mask_detector_final.h5
│   ├── mass_detection_colab.ipynb
│   ├── cv2file.ipynb                         # webcam inference
│   └── data/ (with_mask/, without_mask/)
│
└── Face Mask Detection Using MobileNetV2/    # Transfer Learning
    ├── mask_mobilenet_best.h5
    ├── mask_mobilenet_final.h5
    ├── mask_detection_mobilenetv2.ipynb
    ├── app_mobilenet.py                      # Streamlit app
    ├── requirements.txt
    └── README.md
```
