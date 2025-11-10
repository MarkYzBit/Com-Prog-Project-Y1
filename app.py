import streamlit as st
import tempfile
import os
from processing import preprocess_image_bytes, generate_summary


st.title("✨ Skincare Analyzer 🌿")
st.write("Upload a picture of the skincare product's label for analytical insights and recommendations.")
uploaded_file = st.file_uploader("Choose image file", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Save uploaded file to a temporary path so OpenCV (cv2.imread) can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        # Show only the user-uploaded image once
        st.image(tmp_path, caption='Uploaded Image', use_container_width=True)

        # Run preprocessing (used internally for OCR) but do not display the processed image
        processed = preprocess_image_bytes(tmp_path)
        st.image(processed, caption='Processed Image for OCR', use_container_width=True)

        # Analysis output appears below the image; use a placeholder so we can remove it later
        analysis_placeholder = st.empty()
        analysis_placeholder.text("Analyzing the product...")

        # Generate textual summary/recommendation
        summary = generate_summary(tmp_path)

        # Remove the 'Analyzing...' placeholder now that we have results
        analysis_placeholder.empty()
        st.markdown(summary)

    finally:
        # clean up temp file
        try:
            os.remove(tmp_path)
        except Exception:
            pass