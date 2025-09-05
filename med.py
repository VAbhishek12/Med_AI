import streamlit as st
from agno.agent import Agent
from agno.models.google import Gemini   # ✅ Correct import
from agno.tools.duckduckgo import DuckDuckGoTools
from PIL import Image
import io

# Set Streamlit page config
st.set_page_config(page_title="Medical AI", layout="centered")

st.title("🧠 Medical Image Analysis AI")

# Upload section
uploaded_file = st.file_uploader(
    "📂 Upload Medical Image", 
    type=["jpg", "jpeg", "png", "dcm"]
)

# Initialize Gemini model (make sure GOOGLE_API_KEY is set in Streamlit secrets)
try:
    model = Gemini(id="gemini-1.5-flash", api_key=st.secrets["GOOGLE_API_KEY"])
    agent = Agent(model=model, tools=[DuckDuckGoTools()])
except Exception as e:
    st.error(f"Model initialization failed: {e}")
    agent = None

# If file is uploaded
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Medical Image", use_column_width=True)
    except Exception as e:
        st.error(f"Error opening image: {e}")

    if st.button("🔍 Analyze Image"):
        if agent is None:
            st.error("Agent not initialized. Check your API key and imports.")
        else:
            try:
                # Convert image to bytes
                img_bytes = io.BytesIO()
                image.save(img_bytes, format="PNG")
                img_bytes = img_bytes.getvalue()

                # Ask Gemini to analyze the image
                response = agent.run(
                    f"Analyze this medical scan image and provide possible findings.",
                    images=[img_bytes]
                )
                st.success("✅ Analysis Complete")
                st.write(response)

            except Exception as e:
                st.error(f"Analysis error: {e}")
