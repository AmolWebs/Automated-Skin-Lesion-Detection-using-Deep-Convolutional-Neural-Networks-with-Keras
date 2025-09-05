import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load your model
model = load_model('skin_cancer_model.h5')

# Class labels
class_labels = [
    "Actinic keratoses (akiec)",
    "Basal cell carcinoma (bcc)",
    "Benign keratosis-like lesions (bkl)",
    "Dermatofibroma (df)",
    "Melanoma (mel)",
    "Melanocytic nevi (nv)",
    "Vascular lesions (vas)"
]

# Page config
st.set_page_config(
    page_title="Skin Cancer Classifier",
    page_icon="🧬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom styling (light orange + green + white background)
st.markdown("""
    <style>
        body {
            background-color: white;
        }
        .main-title {
            color: #ff7f50; /* light orange */
            font-size: 2.5rem;
            font-weight: bold;
            text-align: center;
            margin-bottom: 10px;
        }
        .sub-text {
            text-align: center;
            font-size: 1rem;
            color: #555;
            margin-bottom: 20px;
        }
        .result-box {
            background-color: #e8f5e9; /* light green */
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
            text-align: center;
        }
        .result-text {
            color: #2e7d32; /* dark green */
            font-size: 1.5rem;
            font-weight: 600;
        }
        .confidence-text {
            color: #333;
            font-size: 1.1rem;
            margin-top: 10px;
        }
        .footer {
            text-align: center;
            font-size: 0.9rem;
            margin-top: 30px;
            color: #888;
        }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<div class="main-title">🔬 Skin Cancer Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Upload a dermoscopic image to predict the skin lesion type using AI</div>', unsafe_allow_html=True)

# Upload image
uploaded_file = st.file_uploader("📤 Upload an image (JPG, JPEG, PNG):", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="🖼️ Uploaded Image", use_container_width=True)

    # Preprocess image
    img = img.resize((32, 32))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("🔍 Analyzing..."):
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)
        confidence = np.max(predictions)

    # Result box
    st.markdown(f"""
        <div class="result-box">
            <div class="result-text">🧠 Predicted Class: {class_labels[predicted_class]}</div>
            <div class="confidence-text">📊 Confidence: {confidence:.2%}</div>
        </div>
    """, unsafe_allow_html=True)
else:
    st.info("👈 Please upload a dermoscopic image to get started.")

# Footer
st.markdown('<div class="footer">Made with ❤️ using Streamlit | Skin Cancer Detection Project</div>', unsafe_allow_html=True)
