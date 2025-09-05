import os
from PIL import Image as PILImage
import streamlit as st
from agno.agent import Agent
from agno.models.gemini import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.media import Image as AgnoImage

st.set_page_config(page_title="🧠 Medical Image Analysis AI", layout="centered")

st.title("🧠 Medical Image Analysis AI")
st.write("📂 Upload a medical image and let AI analyze it.")

# ✅ Load API key from secrets.toml
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", None)

# Initialize agent
medical_agent = None
if GOOGLE_API_KEY:
    medical_agent = Agent(
        model=Gemini(id="gemini-2.0-flash", api_key=GOOGLE_API_KEY),
        tools=[DuckDuckGoTools()],
        markdown=True
    )
else:
    st.warning("⚠️ Please add your GOOGLE_API_KEY in .streamlit/secrets.toml")

# Analysis instructions
query = """
You are a highly skilled medical imaging expert. Analyze the patient's medical image and structure your response as follows:

### 1. Image Type & Region
- Modality (X-ray/MRI/CT/Ultrasound/etc.)
- Anatomical region and positioning
- Image quality & adequacy

### 2. Key Findings
- Primary observations
- Abnormalities with precise descriptions
- Location, size, shape, severity

### 3. Diagnostic Assessment
- Primary diagnosis (with confidence)
- Differential diagnoses
- Critical findings

### 4. Patient-Friendly Explanation
- Simplified explanation
- Avoid jargon
- Visual analogies if helpful

### 5. Research Context
- Recent medical literature (DuckDuckGo)
- Standard treatment protocols
- Key references with links
"""

# File uploader
uploaded_file = st.file_uploader(
    "Upload Medical Image",
    type=["jpg", "jpeg", "png", "dcm"],
    help="Supported formats: JPG, JPEG, PNG, DICOM"
)

if uploaded_file:
    # Display uploaded image
    image = PILImage.open(uploaded_file)
    st.image(image, caption="Uploaded Medical Image", use_container_width=True)

    analyze_button = st.button("🔍 Analyze Image")

    if analyze_button:
        if not medical_agent:
            st.error("❌ Agent not initialized. Check your API key and imports.")
        else:
            with st.spinner("Analyzing image... Please wait."):
                try:
                    temp_path = "temp_resized_image.png"
                    image.save(temp_path)

                    agno_image = AgnoImage(filepath=temp_path)
                    response = medical_agent.run(query, images=[agno_image])

                    st.subheader("📋 Analysis Results")
                    st.markdown(response.content)
                    st.caption(
                        "⚠️ Disclaimer: This analysis is AI-generated and should be reviewed by a qualified healthcare professional."
                    )
                except Exception as e:
                    st.error(f"Analysis error: {e}")
else:
    st.info("👆 Upload an image to begin analysis")
