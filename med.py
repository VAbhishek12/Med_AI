import streamlit as st
from PIL import Image
from google import genai

st.set_page_config(page_title="🧠 Medical Image Analysis AI", layout="centered")

st.title("🧠 Medical Image Analysis AI")
st.write("📂 Upload a medical image and let AI analyze it.")

# API key
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", None)

if not GOOGLE_API_KEY:
    st.warning("⚠️ Please add your GOOGLE_API_KEY in .streamlit/secrets.toml")
else:
    client = genai.Client(api_key=GOOGLE_API_KEY)

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
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Medical Image", width="stretch")

    if st.button("🔍 Analyze Image"):

        with st.spinner("Analyzing image... Please wait."):

            try:

                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=[query, image]
                )

                st.subheader("📋 Analysis Results")
                st.markdown(response.text)

                st.caption(
                    "⚠️ Disclaimer: AI-generated medical analysis. Consult a doctor."
                )

            except Exception as e:
                st.error(f"Analysis error: {e}")
