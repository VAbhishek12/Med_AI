import streamlit as st
from PIL import Image
import google.generativeai as genai
import io

st.set_page_config(page_title="🧠 Medical Image Analysis AI", layout="centered")

st.title("🧠 Medical Image Analysis AI")
st.write("📂 Upload a medical image and let AI analyze it.")

# ✅ Load API key
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", None)
if not GOOGLE_API_KEY:
    st.warning("⚠️ Please add your GOOGLE_API_KEY in .streamlit/secrets.toml")
else:
    genai.configure(api_key=GOOGLE_API_KEY)

# Medical analysis query
query = """
You are a highly skilled medical imaging expert. Analyze the patient's medical image and structure your response as follows:

### 1. Image Type & Region
### 2. Key Findings
### 3. Diagnostic Assessment
### 4. Patient-Friendly Explanation
### 5. Research Context
"""

uploaded_file = st.file_uploader(
    "Upload Medical Image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG"
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Medical Image", use_container_width=True)

    if st.button("🔍 Analyze Image"):
        if not GOOGLE_API_KEY:
            st.error("❌ Google API key not set.")
        else:
            with st.spinner("Analyzing image... Please wait."):
                try:
                    model = genai.GenerativeModel("gemini-1.5-flash")
                    response = model.generate_content([query, image])
                    st.subheader("📋 Analysis Results")
                    st.markdown(response.text)
                    st.caption(
                        "⚠️ Disclaimer: This analysis is AI-generated and should be reviewed by a qualified healthcare professional."
                    )
                except Exception as e:
                    st.error(f"Analysis error: {e}")
