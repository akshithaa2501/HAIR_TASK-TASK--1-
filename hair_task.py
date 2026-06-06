import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os

# Force TensorFlow logs to quiet down to prevent console freezing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.models import load_model

# Clean Page Layout Configuration
st.set_page_config(page_title="Age, Gender & Hair Detector", layout="centered")

st.title("👤 Age, Gender & Hair Length Detector")
st.write("---")

@st.cache_resource
def load_my_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, 'Age_Sex_Detection.h5')
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model file missing! Please ensure 'Age_Sex_Detection.h5' is inside this folder: {BASE_DIR}")
        return None
    try:
        model = load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"❌ Error loading H5 model layout: {e}")
        return None

# Load the model tracking status
model = load_my_model()
if model is None:
    st.stop()

def detect_hair_length(pil_image):
    try:
        open_cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
        
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if not os.path.exists(cascade_path):
            return "Short" 
            
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return "Short"
            
        (x, y, w, h) = faces[0]
        left_side = gray[y + int(h/2) : y + h + int(h/3), max(0, x - int(w/3)) : x]
        right_side = gray[y + int(h/2) : y + h + int(h/3), x + w : min(gray.shape[1], x + w + int(w/3))]
        sides_combined = np.hstack([left_side.flatten() if left_side.size > 0 else [], 
                                   right_side.flatten() if right_side.size > 0 else []])
        
        if len(sides_combined) == 0:
            return "Short"
            
        edges = cv2.Canny(gray, 30, 100)
        left_edges = edges[y + int(h/2) : y + h + int(h/3), max(0, x - int(w/3)) : x]
        right_edges = edges[y + int(h/2) : y + h + int(h/3), x + w : min(gray.shape[1], x + w + int(w/3))]
        
        edge_density = (np.sum(left_edges == 255) + np.sum(right_edges == 255)) / sides_combined.size
        
        if edge_density > 0.04:
            return "Long"
        else:
            return "Short"
    except Exception:
        return "Short"

def preprocess_image(uploaded_image):
    if uploaded_image.mode != "RGB":
        uploaded_image = uploaded_image.convert("RGB")
    image = uploaded_image.resize((48, 48))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

st.header("Upload Target Images")
uploaded_files = st.file_uploader(
    "Choose one or multiple portrait photos...", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

if uploaded_files:
    if st.button("Run Detection Rules"):
        for i, file in enumerate(uploaded_files):
            st.subheader(f"Analyzing Image {i+1}: {file.name}")
            col1, col2 = st.columns([1, 1])
            
            image = Image.open(file)
            col1.image(image, use_container_width=True)
            
            with st.spinner("Processing deep learning pipelines..."):
                processed_array = preprocess_image(image)
                predictions = model.predict(processed_array)
                
                predicted_age = int(np.round(predictions[1][0][0]))
                gender_prob = float(predictions[0][0][0])
                model_gender = "Female" if gender_prob > 0.5 else "Male"
                
                # Fetch Hair Length Heuristic
                hair_length = detect_hair_length(image)
                
                # Custom Internship Overriding Logic Block (Ages 20 to 30)
                if 20 <= predicted_age <= 30:
                    if hair_length == "Long":
                        final_gender = "Female"
                        logic_note = "⚠️ Gender forced to Female via Long Hair Rule (Age 20-30)"
                    else:
                        final_gender = "Male"
                        logic_note = "⚠️ Gender forced to Male via Short Hair Rule (Age 20-30)"
                    confidence_display = "100.00% (Rule Enforced)"
                else:
                    final_gender = model_gender
                    logic_note = "✅ Standard Model Prediction (Outside Age 20-30 Bracket)"
                    raw_conf = gender_prob if final_gender == "Female" else 1.0 - gender_prob
                    confidence_display = f"{raw_conf:.2%}"
                
                # Visual UI Display - Explicitly displaying Age, Hair, and Gender outputs
                col2.success("📋 Analysis Output:")
                col2.markdown(f"🔹 **Predicted Age:** {predicted_age} years old")
                col2.markdown(f"🔹 **Detected Hair Structure:** **{hair_length} Hair**")
                col2.markdown(f"🔹 **Final Gender Result:** `{final_gender}`")
                col2.markdown(f"🔹 **Metric Confidence:** {confidence_display}")
                col2.info(logic_note)
            st.write("---")