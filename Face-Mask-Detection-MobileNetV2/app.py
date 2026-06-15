import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Load model
@st.cache_resource
def load_mask_model():
    model = load_model("mask_mobilenet_best.h5")
    return model

# Load face detector
@st.cache_resource
def load_face_detector():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

model = load_mask_model()
face_cascade = load_face_detector()


def predict_mask(face_img):
    """Takes a cropped PIL face image, returns prediction message + status."""
    img = face_img.resize((224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    # class_indices = {'with_mask': 0, 'without_mask': 1}
    if prediction >= 0.5:
        return f"❌ No Mask! (Confidence: {prediction*100:.1f}%)", "error"
    else:
        return f"✅ Mask Detected! (Confidence: {(1-prediction)*100:.1f}%)", "success"


def detect_and_crop_face(pil_image):
    """Detects the largest face in the image and returns cropped PIL face image."""
    img_array = np.array(pil_image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    if len(faces) == 0:
        return None, None

    # Pick the largest detected face
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    face_crop = img_array[y:y+h, x:x+w]
    return Image.fromarray(face_crop), (x, y, w, h)


# UI
st.title("😷 Face Mask Detector (MobileNetV2)")

option = st.radio("Choose Input Method:", ["📁 Upload Image", "📷 Live Camera"])

image = None

if option == "📁 Upload Image":
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

else:  # Live Camera
    camera_file = st.camera_input("Take a photo")
    if camera_file is not None:
        image = Image.open(camera_file).convert("RGB")

# Run prediction if we have an image
if image is not None:
    face_crop, box = detect_and_crop_face(image)

    if face_crop is None:
        st.warning("⚠️ No face detected. Please try again with your face clearly visible.")
    else:
        st.image(face_crop, caption="Detected Face (used for prediction)", width=200)

        message, status = predict_mask(face_crop)

        st.markdown("---")
        if status == "success":
            st.success(message)
        else:
            st.error(message)