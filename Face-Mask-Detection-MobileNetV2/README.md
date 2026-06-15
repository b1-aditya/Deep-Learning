# 😷 Face Mask Detection using MobileNetV2

A real-time face mask classifier built with **Transfer Learning (MobileNetV2)** and deployed using **Streamlit** on Hugging Face Spaces. The app supports both image upload and live webcam input, with automatic face detection and cropping for accurate predictions.

🔗 **Live Demo:** [face-mask-detection](https://huggingface.co/spaces/b1-aditya/face-mask-detection)

---

## 🚀 Tech Stack
- Python, TensorFlow / Keras
- MobileNetV2 (Transfer Learning)
- OpenCV (Haar Cascade Face Detection)
- Streamlit (Web App)
- Google Colab + KaggleHub
- Hugging Face Spaces (Deployment)

---

## 🏗️ Model Architecture
- **Base:** MobileNetV2 (pretrained on ImageNet, `include_top=False`)
- **Head:** GlobalAveragePooling2D → BatchNormalization → Dense(128, relu) → Dropout(0.5) → Dense(1, sigmoid)
- **Training Strategy:** 2-Phase Transfer Learning
  - Phase 1: Base model frozen, train custom head (`lr=1e-3`)
  - Phase 2: Fine-tune last 30 layers (`lr=1e-4`)

---

## 📊 Dataset
[Face Mask Dataset – Kaggle](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)
~7,553 images | 2 classes: `with_mask` / `without_mask`

| Class | Label |
|-------|-------|
| with_mask | 0 |
| without_mask | 1 |

---

## ⚙️ Features
- Image Augmentation (rotation, zoom, shift, flip)
- 80-20 train-validation split
- Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
- Real-time face detection & cropping (Haar Cascade) before inference
- Streamlit UI with **Upload Image** & **Live Camera** modes

---

## 📁 Files
- `mask_mobilenet_best.h5` — best model by val_accuracy
- `mask_mobilenet_final.h5` — final trained model
- `mask_detection_mobilenetv2.ipynb` — training notebook (Google Colab)
- `app.py` — Streamlit web app
- `requirements.txt` — Python dependencies

---

## ▶️ How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/b1-aditya/Deep-Learning.git
cd "Deep-Learning/Face-Mask-Detection-MobileNetV2"

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app.py
```

---

## 📈 Results
- Custom CNN (baseline): ~95% validation accuracy
- MobileNetV2 (transfer learning): ~98%+ validation accuracy
- Statistically validated using **McNemar's Test** (paired comparison)

---

## 🔮 Future Improvements
- Multi-face real-time video stream detection
- Add mask-type classification (cloth, surgical, N95)
- Deploy a fine-tuned, smaller model for faster inference
