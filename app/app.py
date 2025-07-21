import sys
import os

# Fix import path for src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from src.models.cnn import CustomCNN

# === Absolute Model Path Fix ===
APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(APP_ROOT, "models", "cnn_baseline", "best_model.pth")

# Constants
NUM_CLASSES = 15
CLASS_NAMES = [
    "Bear", "Bird", "Cat", "Cow", "Deer", "Dog", "Dolphin", "Elephant",
    "Giraffe", "Horse", "Kangaroo", "Lion", "Panda", "Tiger", "Zebra"
]
EMOJIS = {
    "Bear": "🐻", "Bird": "🐦", "Cat": "🐱", "Cow": "🐄", "Deer": "🦌",
    "Dog": "🐶", "Dolphin": "🐬", "Elephant": "🐘", "Giraffe": "🦒",
    "Horse": "🐴", "Kangaroo": "🦘", "Lion": "🦁", "Panda": "🐼",
    "Tiger": "🐯", "Zebra": "🦓"
}

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model not found at: {MODEL_PATH}")
        st.info("📌 Please train the model using `python src/training/train.py`.")
        st.stop()

    model = CustomCNN(NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# Streamlit UI setup
st.set_page_config(page_title="Animal Classifier", page_icon="🐾")
st.title("🐾 Animal Image Classifier")
st.write("Upload a `.jpg`, `.jpeg`, or `.png` image of an animal. The model will classify it automatically!")

uploaded_file = st.file_uploader("📁 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)

    # Auto-classify when image is uploaded
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)[0]
        top3 = torch.topk(probs, 3)

    predicted_idx = top3.indices[0].item()
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = probs[predicted_idx].item()

    st.subheader("🧠 Prediction")
    st.success(f"{EMOJIS[predicted_class]} **Predicted Class: {predicted_class}**")
    st.markdown(f"📊 **Confidence:** `{confidence:.2%}`")
    
    st.subheader("🔝 Top 3 Predictions")
    for i in range(3):
        idx = top3.indices[i].item()
        st.write(f"{EMOJIS[CLASS_NAMES[idx]]} {CLASS_NAMES[idx]} — `{probs[idx].item():.2%}`")

    st.subheader("📈 Class Probabilities")
    fig, ax = plt.subplots(figsize=(10, 4))
    class_probs = probs.numpy()
    ax.bar(np.arange(len(CLASS_NAMES)), class_probs, color="skyblue")
    ax.set_ylabel("Probability")
    ax.set_xticks(np.arange(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    st.pyplot(fig)
