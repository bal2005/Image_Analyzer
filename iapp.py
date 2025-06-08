import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
import numpy as np
st.set_page_config(page_title="Complaint Redressal Analyzer", layout="centered")

def is_image_blurry(img_file, threshold=2000.0):
    img = Image.open(img_file).convert("L")  # Convert to grayscale
    img_np = np.array(img)
    laplacian_var = cv2.Laplacian(img_np, cv2.CV_64F).var()
    return laplacian_var < threshold, laplacian_var

# ---------- 🔐 Load Gemini API Key ----------
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

client = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai",
    api_key=api_key
)

# ---------- 📊 Department Info ----------
departments = [
    "Road Work", 
    "School Infrastructure Damage", 
    "Sewage and Water Stagnation", 
    "Street Light Issue", 
    "Toilet Issue", 
    "Garbage", 
    "Not Maintained Parks", 
    "Road Damage", 
    "Shop Obstructing Pathway", 
    "Tree Obstructing Road"
]

few_shot_examples = """
Complaint: "The repair work on the main road has been going on for weeks without completion."
Issue: Road Work

Complaint: "The local school building has cracked walls and damaged classrooms."
Issue: School Infrastructure Damage

Complaint: "There is stagnant water in our street that smells and breeds mosquitoes."
Issue: Sewage and Water Stagnation

Complaint: "The streetlight on our lane hasn’t been working for days, it’s completely dark at night."
Issue: Street Light Issue

Complaint: "The public toilet near the market is always locked or in a filthy condition."
Issue: Toilet Issue

Complaint: "Garbage bins are overflowing and no one has come to collect the waste."
Issue: Garbage

Complaint: "The neighborhood park is full of weeds, broken benches, and rusted play equipment."
Issue: Not Maintained Parks

Complaint: "There’s a large pothole on the road near the flyover, making it dangerous for vehicles."
Issue: Road Damage

Complaint: "A small shop is blocking the footpath, making it hard for pedestrians to walk."
Issue: Shop Obstructing Pathway

Complaint: "A tree has fallen on the road and is blocking traffic for hours."
Issue: Tree Obstructing Road
"""


# ---------- 🧠 LLM-Based Text Classifier ----------
def classify_complaint(complaint_text):
    prompt = f"""
You are an assistant that classifies civic complaints into the correct government department.

Departments:
{', '.join(departments)}

Here are some examples:
{few_shot_examples}

Now classify the following complaint:
Complaint: "{complaint_text}"
Department:"""

    response = client.chat.completions.create(
        model="gemini-1.5-flash",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    result = response.choices[0].message.content.strip()
    if result.lower().startswith("department:"):
        result = result[len("Department:"):].strip()
    return result

# ---------- 📷 Image Model Setup ----------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("issue_classifier.h5")

model = load_model()
class_labels = [
    "Road Work", 
    "School Infrastructure Damage", 
    "Sewage and Water Stagnation", 
    "Street Light Issue", 
    "Toilet Issue", 
    "Garbage", 
    "Not Maintained Parks", 
    "Road Damage", 
    "Shop Obstructing Pathway", 
    "Tree Obstructing Road"
]

def preprocess_image(img_file, target_size=(224, 224)):
    img = Image.open(img_file).convert("RGB").resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def classify_image(img_file):
    img_array = preprocess_image(img_file)
    predictions = model.predict(img_array, verbose=0)
    predicted_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_index]
    predicted_label = class_labels[predicted_index]
    return predicted_label, confidence

def save_image(file, folder="accepted_uploads"):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, file.name)
    with open(filepath, "wb") as f:
        f.write(file.getbuffer())
    return filepath

# ---------- 🎛️ Streamlit Interface ----------
st.title("📌 Complaint Redressal Analyzer (Text + Image)")

st.header("✍️ Step 1: Enter Civic Complaint")
complaint_input = st.text_area("Enter your complaint text:", height=150)

default_index = 0  # Default index for dropdown preselection
if complaint_input.strip():
    with st.spinner("Classifying text..."):
        predicted_department = classify_complaint(complaint_input)
        # Find closest matching label index for dropdown preselection
        try:
            default_index = class_labels.index(predicted_department)
        except ValueError:
            default_index = 0  # Fallback if no match found

    st.success(f"✅ Predicted Department: **{predicted_department}**")
else:
    predicted_department = None

st.markdown("---")
st.header("🖼️ Step 2: Validate with Image")

selected_type = class_labels[default_index]
st.info(f"📋 Predicted Issue Type: **{selected_type}**")


uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="🖼 Uploaded Image Preview", use_column_width=True)

    # Check image quality
    is_blur, blur_score = is_image_blurry(uploaded_file)
    st.write(f"🔍 **Blurriness Score**: `{blur_score:.2f}`")

    if is_blur:
        st.error("❌ Image is too blurry. Please upload a clearer image.")
    else:
        with st.spinner("🔍 Validating Image..."):
            predicted_label, confidence = classify_image(uploaded_file)

        st.write(f"🔎 **Predicted Category**: `{predicted_label}`")
        st.write(f"📊 **Confidence Score**: `{confidence:.2f}`")

        if predicted_label == selected_type and confidence >= 0.7:
            if st.button("✅ Confirm and Upload"):
                path = save_image(uploaded_file)
                st.success(f"📥 Image successfully uploaded to `{path}`.")
        else:
            st.warning("⚠️ Image does not match selected type or confidence too low.")


