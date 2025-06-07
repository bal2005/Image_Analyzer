import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Access the API key
api_key = os.getenv("GEMINI_API_KEY")

# Initialize Gemini Client
client = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai",
    api_key=api_key
)


# Departments
departments = [
    "Health",
    "Education",
    "Town Planning",
    "Electrical",
    "Roads",
    "Parks",
    "Solid Waste Management",
    "Storm Water Drain",
    "Land and Estates"
]

# Few-shot examples
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

# Classify function
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

# Streamlit UI
st.set_page_config(page_title="Complaint Classifier", layout="centered")
st.title("🧾 Complaint Department Classifier")
st.write("Enter a civic complaint and get the appropriate department.")

complaint_input = st.text_area("✍️ Enter your complaint:", height=150)

if st.button("Classify Complaint"):
    if complaint_input.strip():
        with st.spinner("Classifying..."):
            result = classify_complaint(complaint_input)
        st.success(f"✅ Predicted Department: **{result}**")
    else:
        st.warning("Please enter a complaint to classify.")
