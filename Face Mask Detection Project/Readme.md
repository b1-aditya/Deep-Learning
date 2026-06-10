# 😷 Face Mask Detection using Deep Learning

A binary image classifier built with TensorFlow/Keras to detect whether a person is wearing a face mask.

## 🚀 Tech Stack
- Python, TensorFlow / Keras
- Google Colab + KaggleHub
- ImageDataGenerator, Sequential CNN

## 🏗️ Model Architecture
- 3 Convolutional Blocks (32 → 64 → 128 filters)
- BatchNormalization + MaxPooling after each block
- Dense(128) → Dropout(0.5) → Sigmoid output

## 📊 Dataset
[Face Mask Dataset – Kaggle](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)
~12,000 images | 2 classes: `with_mask` / `without_mask`

## ⚙️ Features
- Image Augmentation (rotation, zoom, flip)
- 80-20 train-validation split
- Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

## 📁 Output
- `mask_detector_best.h5` — best model by val_accuracy
- `mask_detector_final.h5` — final trained model

## ▶️ How to Run
1. Open the notebook in Google Colab
2. Run all cells (dataset auto-downloads via KaggleHub)
3. Model saved to `.h5` after training
