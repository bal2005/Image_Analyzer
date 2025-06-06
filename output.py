import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("street_issue_classifier.h5")

model = load_model()

# Category labels - ensure order matches training
class_labels = ['Flooding', 'Road_damage', 'Street_light', 'Garbage']

# Image preprocessing
def preprocess_image(img_file, target_size=(224, 224)):
    img = Image.open(img_file).convert("RGB").resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Prediction function
def classify_image(img_file):
    img_array = preprocess_image(img_file)
    predictions = model.predict(img_array, verbose=0)
    predicted_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_index]
    predicted_label = class_labels[predicted_index]
    return predicted_label, confidence

# Save image to folder
def save_image(file, folder="accepted_uploads"):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, file.name)
    with open(filepath, "wb") as f:
        f.write(file.getbuffer())
    return filepath

# Streamlit Interface
st.title("🛠️ Smart Complaint Image Validator")

# Select type of complaint
selected_type = st.selectbox("📋 Select Complaint Type", class_labels)

# Upload image
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="🖼 Uploaded Image Preview", use_column_width=True)

    with st.spinner("🔍 Validating Image..."):
        predicted_label, confidence = classify_image(uploaded_file)

    st.write(f"🔎 **Predicted Category**: `{predicted_label}`")
    st.write(f"📊 **Confidence Score**: `{confidence:.2f}`")

    if predicted_label == selected_type and confidence >= 0.7:
        if st.button("✅ Confirm and Upload"):
            path = save_image(uploaded_file)
            st.success(f"📥 Image successfully uploaded to `{path}`.")
    else:
        st.warning("⚠️ Image does not match the selected complaint type or confidence is too low. Please recheck.")
