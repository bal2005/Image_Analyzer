import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
st.set_page_config(page_title="Complaint Redressal Analyzer", layout="centered")

# ---------- 🔐 Load Gemini API Key ----------
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

client = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai",
    api_key=api_key
)

# ---------- 📊 Department Info ----------
departments = [
    "Health", "Education", "Town Planning", "Electrical", "Roads",
    "Parks", "Solid Waste Management", "Storm Water Drain", "Land and Estates"
]

few_shot_examples = """
Complaint: "Garbage has not been collected for a week and it's starting to smell."
Department: Solid Waste Management

Complaint: "The park near my house has broken swings and overgrown grass."
Department: Parks

Complaint: "There is a huge pothole near the main junction which is causing traffic."
Department: Roads

Complaint: "Street lights are not working in our neighborhood."
Department: Electrical

Complaint: "There is water logging in front of my house after yesterday's rain."
Department: Storm Water Drain

Complaint: "The government school near us has broken benches and no toilets."
Department: Education

Complaint: "There is unauthorized construction happening on a public land."
Department: Land and Estates

Complaint: "The local clinic is understaffed and lacks basic facilities."
Department: Health

Complaint: "Unauthorized structures are affecting road expansion."
Department: Town Planning
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
    return tf.keras.models.load_model("street_issue_classifier.h5")

model = load_model()
class_labels = ['Flooding', 'Road_damage', 'Street_light', 'Garbage']

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

if complaint_input.strip():
    with st.spinner("Classifying text..."):
        predicted_department = classify_complaint(complaint_input)
    st.success(f"✅ Predicted Department: **{predicted_department}**")
else:
    predicted_department = None

st.markdown("---")
st.header("🖼️ Step 2: Validate with Image (Optional)")

selected_type = st.selectbox("📋 Select Expected Image Type", class_labels)
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
        st.warning("⚠️ Image does not match selected type or confidence too low.")

