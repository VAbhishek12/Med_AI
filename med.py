import streamlit as st
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from PIL import Image
import io

# ------------------- Streamlit Page Setup -------------------
st.set_page_config(page_title="🧠 Medical AI", layout="centered")
st.title("🧠 Medical Image Analysis AI")
st.write("📂 Upload a medical image and let AI analyze it.")

# ------------------- API Key Setup -------------------
agent = None
if "GOOGLE_API_KEY" in st.secrets:
    try:
        model = Gemini(id="gemini-1.5-flash", api_key=st.secrets["GOOGLE_API_KEY"])
        agent = Agent(model=model, tools=[DuckDuckGoTools()])
    except Exception as e:
        st.error(f"❌ Model initialization failed: {e}")
else:
    st.warning("⚠️ Please add your GOOGLE_API_KEY in `.streamlit/secrets.toml`")

# ------------------- File Upload -------------------
uploaded_file = st.file_uploader(
    "Upload Medical Image",
    type=["jpg", "jpeg", "png", "dcm"],
    help="Supported formats: JPG, JPEG, PNG, DICOM"
)

# ------------------- Image Display -------------------
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Medical Image", use_container_width=True)
    except Exception as e:
        st.error(f"⚠️ Error loading image: {e}")

    # ------------------- Analysis Button -------------------
    if st.button("🔍 Analyze Image"):
        if agent is None:
            st.error("❌ Agent not initialized. Check your API key and imports.")
        else:
            try:
                # Convert image to bytes for Gemini
                img_bytes = io.BytesIO()
                image.save(img_bytes, format="PNG")
                img_bytes = img_bytes.getvalue()

                # Query prompt
                query = """
                You are a highly skilled medical imaging expert. 
                Analyze this uploaded scan and provide:
                1. Image type & region
                2. Key findings (normal/abnormal observations)
                3. Possible diagnoses with confidence levels
                4. Patient-friendly explanation
                5. Recent research context (use DuckDuckGo search)
                """

                # Run analysis
                response = agent.run(query, images=[img_bytes])

                # Show result
                st.markdown("### 📋 Analysis Results")
                st.write(response)

            except Exception as e:
                st.error(f"⚠️ Analysis failed: {e}")
