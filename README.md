# Med AI: Medical Image Analysis

A Streamlit app that analyses an uploaded medical image with Google Gemini and returns a structured, readable report.

> Educational demo. The output is AI-generated, can be wrong, and is **not** medical advice or a diagnosis. Always consult a qualified doctor.

## What it does

Upload a JPG or PNG (X-ray, MRI slice, scan, etc.) and click **Analyze Image**. The model is prompted to act as a medical imaging expert and answers in five fixed sections:

1. **Image type and region**
2. **Key findings**
3. **Diagnostic assessment**
4. **Patient-friendly explanation**
5. **Research context**

## Repository layout

```
.
├── med.py             Streamlit app
├── requirements.txt   dependencies
└── .gitignore         keeps .streamlit/secrets.toml out of git
```

## Run locally

1. Install the dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Get a Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey) and save it in `.streamlit/secrets.toml`:

   ```toml
   GOOGLE_API_KEY = "your-key-here"
   ```

   This file is listed in `.gitignore`. Never commit it or paste a key into the code.

3. Start the app:

   ```bash
   streamlit run med.py
   ```

On Streamlit Community Cloud, add `GOOGLE_API_KEY` under **App settings → Secrets** instead of using a file.

## Notes

- The app uses the `gemini-2.0-flash` model through the `google-genai` SDK. Change the model name in `med.py` if it is retired.
- Images are sent to Google's API for analysis. Do not upload images that identify real patients.

## Tech

Python · Streamlit · Google Gemini (`google-genai`) · Pillow
